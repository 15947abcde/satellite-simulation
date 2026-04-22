#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISL-ALNS+TS scheduler（星间互联+无速率匹配）
- 核心约束：
  1. 业务不可分片（单个任务必须在单一链路、单一连续时间段完成）
  2. 允许多星转发（任务可使用任意链路，取消归属链路限制）
  3. 链路类型：1=RF，2-5=FSO(BPSK)
  4. 无速率匹配：RF=1Gbps，FSO(BPSK)=10Gbps
"""

import math
import random
import re
import glob
import os
from collections import defaultdict, deque, namedtuple
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from functools import lru_cache  # 新增：缓存装饰器

import numpy as np
import pandas as pd
import torch


# ================= 配置类（统一管理参数）=================
class SchedulerConfigISL:
    def __init__(self):
        self.ITERATIONS = 20
        self.SEED = 42
        self.GB_IS_GIB = False
        self.RF_PENALTY_PER_GBIT = 0.0
        self.SERVICE_WINDOW_SEC = 3000 #网络规模
        # self.SERVICE_WINDOW_SEC = 36000 #业务规模

        self.EARLY_WINDOW_SEC = 0
        self.EARLY_PENALTY_COEF = 1e-11

        self.RANDOM_RATE_JITTER = 0.02
        self.REHEAT_PERIOD = 100

        # ===== 修改点：无速率匹配 =====
        # 1 = RF (1 Gbps)
        # 2–5 = FSO (BPSK, 10 Gbps)
        self.RATE_MAP_GBPS = {
            1: 1.0,
            2: 10.0,
            3: 10.0,
            4: 10.0,
            5: 10.0
        }
        self.RATE_MAP = {k: v * 1e9 for k, v in self.RATE_MAP_GBPS.items()}
        self.LINK_TYPE = {
            1: "RF",
            2: "FSO(BPSK)",
            3: "FSO(BPSK)",
            4: "FSO(BPSK)",
            5: "FSO(BPSK)"
        }

        self.MOD_SWITCH_TIME = 0.0
        self.LINK_SWITCH_TIME = 0.0


# ================= 辅助函数 =================
def mean_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    n = len(data)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(data) / n
    if n == 1:
        return (mean, mean, mean)

    var = sum((x - mean) ** 2 for x in data) / (n - 1)
    std = math.sqrt(var)
    t_table = {9: 2.262, 10: 2.228, 11: 2.201}
    t_val = t_table.get(n - 1, 1.96)
    margin = t_val * std / math.sqrt(n)
    return (mean, mean - margin, mean + margin)


def device_autoselect() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 优化：使用lru_cache缓存比特转换计算
@lru_cache(maxsize=10000)
def to_bits(size_gb: float, gb_is_gib: bool) -> float:
    if gb_is_gib:
        return size_gb * (1024 ** 3) * 8.0
    return size_gb * (1000 ** 3) * 8.0


# ================= 数据结构 =================
Segment = namedtuple("Segment", ["start", "end", "rate", "mod_level", "link_type"])


class LinkFTW:
    """链路可见时间窗口（VTW）解析（星间互联版本）"""
    __slots__ = ("name", "segments", "capacity_cache")  # 优化：添加slots减少内存开销
    def __init__(self, name: str):
        self.name = name
        self.segments: List[Segment] = []
        self.capacity_cache = {}  # 优化：缓存链路容量计算结果

    def add_segment(self, start: int, end: int, mod: int, config: SchedulerConfigISL):
        """恢复原速率映射，支持星间链路类型"""
        if end > start:
            self.segments.append(
                Segment(start, end, config.RATE_MAP[int(mod)], int(mod), config.LINK_TYPE[int(mod)])
            )

    def finalize(self):
        if not self.segments:
            return
        self.segments.sort(key=lambda s: (s.start, s.mod_level))
        merged = []
        cur = self.segments[0]
        for seg in self.segments[1:]:
            if seg.mod_level == cur.mod_level and seg.start <= cur.end:
                cur = Segment(cur.start, max(cur.end, seg.end), cur.rate, cur.mod_level, cur.link_type)
            else:
                merged.append(cur)
                cur = seg
        merged.append(cur)
        self.segments = merged

    def iter_segments_in_window(self, t0: int, t1: int):
        for s in self.segments:
            if s.end <= t0:
                continue
            if s.start >= t1:
                break
            yield Segment(max(s.start, t0), min(s.end, t1), s.rate, s.mod_level, s.link_type)

    # 优化：缓存链路容量计算
    def get_capacity_in_window(self, t0: int, t1: int) -> float:
        """获取链路在[t0, t1]内的总可用比特数（带缓存）"""
        key = (t0, t1)
        if key in self.capacity_cache:
            return self.capacity_cache[key]
        total = 0.0
        for s in self.iter_segments_in_window(t0, t1):
            total += s.rate * (s.end - s.start)
        self.capacity_cache[key] = total
        return total


class Task:
    """任务数据结构（星间互联版本，保留home_link仅作参考）"""
    __slots__ = ("task_id", "arrival", "deadline", "size_bits", "priority", "value", "home_link", "gen_time",
                 "rf_transfer_time", "fso_transfer_time", "value_density")  # 优化：添加slots
    def __init__(self, task_id, arrival, deadline, size_bits, priority, value, home_link, gen_time, config):
        self.task_id = str(task_id)
        self.arrival = int(arrival)
        self.deadline = int(deadline)
        self.size_bits = float(size_bits)
        self.priority = int(priority)
        self.value = float(value)
        self.home_link = str(home_link)  # 仅作参考，无归属链路限制
        self.gen_time = int(gen_time)
        # 优化：预计算传输时间和价值密度，避免重复计算
        self.rf_transfer_time = size_bits / config.RATE_MAP[1]
        self.fso_transfer_time = size_bits / config.RATE_MAP[2]
        self.value_density = value / (size_bits / 1e9)  # 每GB价值


# 无分片Chunk（星间互联版本）
Chunk = namedtuple(
    "Chunk",
    ["link", "start", "finish", "bits", "rf_bits", "mod_level", "link_type"]
)
Assignment = namedtuple("Assignment", ["task_id", "chunk", "total_bits", "value"])


# ================= 数据加载函数（星间互联版本）=================
def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """解析文件名提取链路名和时间（星间互联版本）"""
    name = Path(filename).stem.strip()
    m = re.search(r'(.+?)[-_]?(\d+)$', name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def load_tasks_from_dir(tasks_dir: str, config: SchedulerConfigISL) -> List[Task]:
    """加载任务数据（星间互联版本，恢复原价值量解析）"""
    tasks = []
    files = []
    for ext in ("*.csv", "*.CSV"):
        files += glob.glob(os.path.join(tasks_dir, ext))
    if not files:
        raise FileNotFoundError(f"No task CSV files in {tasks_dir}")

    # 优化：批量读取文件，减少循环内的IO操作
    for f in files:
        fname = Path(f).name
        link, arr = parse_filename(fname)
        if link is None or arr is None:
            print(f"[SKIP:name] {fname}")
            continue

        df = pd.read_csv(
            f,
            header=None,
            names=["value", "size_gb"],
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        # 清洗脏数据
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["size_gb"] = pd.to_numeric(df["size_gb"], errors="coerce")
        df = df.dropna(subset=["value", "size_gb"])

        if df.empty:
            print(f"[SKIP:empty] {fname}")
            continue

        # 优先级分层逻辑
        df = df.sort_values("value", ascending=False).reset_index(drop=True)
        n = len(df)
        hi = int(n * 0.2)
        lo = int(n * 0.8)

        # 优化：预计算deadline和to_bits，减少循环内计算
        deadline = arr + config.SERVICE_WINDOW_SEC
        for i, row in df.iterrows():
            pr = 10 if i < hi else (5 if i < lo else 1)
            size_bits = to_bits(float(row["size_gb"]), config.GB_IS_GIB)
            tid = f"{fname}_task{i}"
            tasks.append(Task(
                tid, arr, deadline, size_bits,
                pr, float(row["value"]), link, gen_time=arr, config=config  # 新增config参数
            ))

    if not tasks:
        raise ValueError("No valid tasks found (after parsing/cleaning).")
    return tasks


def load_link_ftw_csv(path: str, link_name: str, config: SchedulerConfigISL) -> LinkFTW:
    """加载链路可见性数据（星间互联版本）"""
    try:
        df = pd.read_csv(path)
        if not {"valid_time_seconds", "modulation_level"}.issubset(df.columns):
            df = pd.read_csv(path, header=None, names=["valid_time_seconds", "modulation_level"])
    except:
        df = pd.read_csv(path, header=None, names=["valid_time_seconds", "modulation_level"])

    # 清洗数据
    df["valid_time_seconds"] = pd.to_numeric(df["valid_time_seconds"], errors="coerce")
    df["modulation_level"] = pd.to_numeric(df["modulation_level"], errors="coerce")
    df = df.dropna(subset=["valid_time_seconds", "modulation_level"])
    if df.empty:
        ftw = LinkFTW(link_name)
        ftw.finalize()
        return ftw

    df["valid_time_seconds"] = df["valid_time_seconds"].astype(int)
    df["modulation_level"] = df["modulation_level"].astype(int)
    df = df.sort_values("valid_time_seconds")

    ftw = LinkFTW(link_name)
    prev_m = None
    run_s = None
    last = None

    # 优化：使用itertuples代替iterrows，提速循环
    for row in df.itertuples(index=False):
        t = int(row.valid_time_seconds)
        m = int(row.modulation_level)

        if prev_m is None:
            prev_m = m
            run_s = t
            last = t
        else:
            if m == prev_m and t == last + 1:
                last = t
            else:
                ftw.add_segment(run_s, last + 1, prev_m, config)
                prev_m = m
                run_s = t
                last = t

    if prev_m is not None:
        ftw.add_segment(run_s, last + 1, prev_m, config)

    ftw.finalize()
    return ftw


def load_all_links(visibility_dir: str, config: SchedulerConfigISL) -> Dict[str, LinkFTW]:
    """加载所有链路数据（星间互联版本，支持任意链路）"""
    link_map = {}
    csvs = []
    for ext in ("*.csv", "*.CSV"):
        csvs += glob.glob(os.path.join(visibility_dir, ext))
    if not csvs:
        raise FileNotFoundError(f"No visibility CSV files in {visibility_dir}")
    for p in csvs:
        ln = Path(p).stem
        link_map[ln] = load_link_ftw_csv(p, ln, config)
    return link_map


# ================= 核心调度器类（星间互联ISL-ALNS）=================
class SchedulerISL:
    __slots__ = ("tasks", "links", "device", "config", "rf_penalty", "assignments", 
                 "unscheduled", "slots_by_link", "best_assignments", "best_value", 
                 "cur_assignments", "cur_value", "tabu")  # 优化：添加slots
    def __init__(self, tasks: List[Task], links: Dict[str, LinkFTW], device, config: SchedulerConfigISL, seed=42):
        """初始化（星间互联版本，取消归属链路限制）"""
        self.tasks = tasks
        self.links = links  # 支持所有链路（星间转发）
        self.device = device
        self.config = config
        self.rf_penalty = config.RF_PENALTY_PER_GBIT

        # 初始化随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 调度状态
        self.assignments: List[Assignment] = []
        self.unscheduled = set(t.task_id for t in tasks)
        self.slots_by_link: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)

        # 最优解跟踪
        self.best_assignments = None
        self.best_value = -1e18
        self.cur_assignments = None
        self.cur_value = -1e18

        # 禁忌表
        self.tabu = deque(maxlen=64)

    def get_free_gaps(self, link: str, a: int, b: int) -> List[Tuple[int, int]]:
        """获取链路空闲时隙（星间互联版本）- 优化：减少循环内的属性访问"""
        used = self.slots_by_link[link]
        if not used:
            return [(a, b)] if a < b else []
        
        # 优化：提前排序并缓存到局部变量
        used_sorted = sorted(used, key=lambda x: x[0])
        gaps = []
        cur = a
        # 优化：使用局部变量缓存min/max，减少函数调用
        min_func = min
        max_func = max
        for s, e, _ in used_sorted:
            if e <= cur:
                continue
            if s > cur:
                gaps.append((cur, min_func(s, b)))
                cur = min_func(s, b)
            cur = max_func(cur, e)
            if cur >= b:
                break
        if cur < b:
            gaps.append((cur, b))
        return gaps

    def reserve(self, link: str, start: int, finish: int, task_id: str):
        """预留链路时隙（星间互联版本）"""
        self.slots_by_link[link].append((start, finish, task_id))

    def link_opportunity(self, t: Task, link: str) -> float:
        """计算任务在链路上的传输机会分数（星间互联版本）- 优化：缓存局部变量"""
        score = 0.0
        link_obj = self.links[link]
        for seg in link_obj.iter_segments_in_window(t.arrival, t.deadline):
            w = 1.0 if seg.link_type == "FSO(BPSK)" else 0.8
            score += w * seg.rate * (seg.end - seg.start)
        return score

    def candidate_heuristic_cost(self, t: Task, chunk: Chunk) -> float:
        """启发式成本计算（星间互联版本，恢复原价值量逻辑）- 优化：减少重复计算"""
        early_end = t.arrival + self.config.EARLY_WINDOW_SEC
        chunk_start = chunk.start
        chunk_finish = chunk.finish
        
        # 优化：提前计算边界值
        start_max = max(chunk_start, t.arrival)
        finish_min = min(chunk_finish, early_end)
        early_dur = max(0.0, finish_min - start_max)
        
        # 优化：避免除以0
        dur_diff = chunk_finish - chunk_start
        if dur_diff < 1e-9:
            early_bits = 0.0
        else:
            early_bits = (early_dur / dur_diff) * chunk.bits
        
        return -t.value + self.config.EARLY_PENALTY_COEF * early_bits

    def try_pack_task_on_link(self, t: Task, link: str) -> Optional[Chunk]:
        """尝试在指定链路上打包任务（新增速率抖动，与对比代码对齐）"""
        # 优化：缓存所有需要的变量到局部，减少属性访问
        need_bits = t.size_bits
        arrival = t.arrival
        deadline = t.deadline
        link_obj = self.links[link]
        config = self.config
        get_free_gaps = self.get_free_gaps
        candidate_heuristic_cost = self.candidate_heuristic_cost
        math_ceil = math.ceil
        max_func = max
        min_func = min
        
        # 优化：提前判断链路容量是否足够，避免无效计算
        if link_obj.get_capacity_in_window(arrival, deadline) < need_bits:
            return None
        
        free_gaps = get_free_gaps(link, arrival, deadline)
        if not free_gaps:
            return None

        best_chunk = None
        best_cost = float('inf')

        # 优化：遍历段时缓存局部变量
        for seg in link_obj.iter_segments_in_window(arrival, deadline):
            seg_start = seg.start
            seg_end = seg.end
            seg_rate = seg.rate
            # ===== 新增：速率抖动（与对比代码一致）=====
            jitter = 1.0 + random.uniform(-config.RANDOM_RATE_JITTER, config.RANDOM_RATE_JITTER)
            seg_rate = seg_rate * jitter
            # ==========================================
            seg_mod = seg.mod_level
            seg_type = seg.link_type
            
            # 优化：仅处理当前段内的空闲时隙
            seg_gaps = get_free_gaps(link, seg_start, seg_end)
            for gap_start, gap_end in seg_gaps:
                max_bits_in_gap = seg_rate * (gap_end - gap_start)
                if max_bits_in_gap < need_bits:
                    continue

                task_duration = need_bits / seg_rate
                start = gap_start
                finish = start + task_duration
                if finish > gap_end:
                    continue

                chunk = Chunk(
                    link=link,
                    start=start,
                    finish=finish,
                    bits=need_bits,
                    rf_bits=need_bits if seg_type == "RF" else 0.0,
                    mod_level=seg_mod,
                    link_type=seg_type
                )

                cost = candidate_heuristic_cost(t, chunk)
                if cost < best_cost:
                    best_cost = cost
                    best_chunk = chunk

        if best_chunk is None:
            return None

        self.reserve(link, int(best_chunk.start), math_ceil(best_chunk.finish), t.task_id)
        return best_chunk

    def try_pack_task_once(self, t: Task) -> Optional[Chunk]:
        """核心：遍历所有链路打包任务（星间转发，无分片）- 优化：提前过滤无效链路"""
        best_chunk = None
        best_cost = float('inf')
        snapshot = {ln: list(v) for ln, v in self.slots_by_link.items()}

        # 优化：提前过滤容量不足的链路，减少遍历次数
        eligible_links = []
        for link_name in self.links.keys():
            link_obj = self.links[link_name]
            if link_obj.get_capacity_in_window(t.arrival, t.deadline) >= t.size_bits:
                eligible_links.append(link_name)
        
        # 遍历候选链路
        for link_name in eligible_links:
            self.slots_by_link = defaultdict(list, {ln: list(v) for ln, v in snapshot.items()})
            chunk = self.try_pack_task_on_link(t, link_name)
            if chunk is None:
                continue

            cost = self.candidate_heuristic_cost(t, chunk)
            if cost < best_cost:
                best_cost = cost
                best_chunk = chunk

        self.slots_by_link = defaultdict(list, snapshot)
        if best_chunk is None:
            return None

        self.reserve(best_chunk.link, int(best_chunk.start), math.ceil(best_chunk.finish), t.task_id)
        return best_chunk

    def place_task_multicand(self, t: Task, n_cand: int = 3) -> Optional[Assignment]:
        """多候选位置放置任务（星间互联版本）"""
        best_chunk = None
        best_cost = float('inf')
        snapshot = {ln: list(v) for ln, v in self.slots_by_link.items()}

        for _ in range(n_cand):
            self.slots_by_link = defaultdict(list, {ln: list(v) for ln, v in snapshot.items()})
            chunk = self.try_pack_task_once(t)
            if chunk is None:
                continue

            cost = self.candidate_heuristic_cost(t, chunk)
            if cost < best_cost:
                best_cost = cost
                best_chunk = chunk

        if best_chunk is None:
            self.slots_by_link = defaultdict(list, snapshot)
            return None

        self.slots_by_link = defaultdict(list, snapshot)
        self.reserve(best_chunk.link, int(best_chunk.start), math.ceil(best_chunk.finish), t.task_id)

        return Assignment(
            task_id=t.task_id,
            chunk=best_chunk,
            total_bits=best_chunk.bits,
            value=t.value
        )

    def objective(self, assigns: List[Assignment]) -> float:
        """目标函数（对齐对比代码，加入RF惩罚+平均传输时间惩罚）"""
        if not assigns:
            return -1e18

        total_value = 0.0
        total_time = 0.0

        for a in assigns:
            rf_bits = a.chunk.rf_bits
            # 1. 加入RF链路使用惩罚（与对比代码一致）
            total_value += a.value - self.rf_penalty * (rf_bits / 1e9)
            # 2. 累计传输时间（单chunk场景）
            total_time += (a.chunk.finish - a.chunk.start)

        # 3. 加入平均传输时间惩罚（与对比代码一致，λ可调节）
        avg_time = total_time / len(assigns) if assigns else 0.0
        λ = 1.0  # 与对比代码的惩罚系数保持一致，可按需调整
        return total_value - λ * avg_time

    def build_initial(self):
        """构建初始解（星间互联版本）"""
        proxy = 20e9
        # 优化：预计算排序键，减少循环内计算
        task_order = sorted(
            self.tasks,
            key=lambda x: (x.value / max(1.0, x.size_bits / proxy), -x.arrival)
        )
        for t in task_order:
            if t.task_id in self.unscheduled:
                a = self.place_task_multicand(t, n_cand=3)
                if a:
                    self.assignments.append(a)
                    self.unscheduled.discard(t.task_id)
        self.cur_assignments = list(self.assignments)
        self.cur_value = self.objective(self.assignments)
        self.best_assignments = list(self.assignments)
        self.best_value = self.cur_value

    def _remove_assignments(self, removed: List[Assignment]):
        """移除调度分配（星间互联版本）- 优化：批量操作"""
        # 优化：批量添加到unscheduled
        removed_task_ids = {a.task_id for a in removed}
        self.unscheduled.update(removed_task_ids)
        
        # 优化：批量过滤时隙
        for link in self.slots_by_link:
            self.slots_by_link[link] = [
                (s, e, tid) for (s, e, tid) in self.slots_by_link[link]
                if tid not in removed_task_ids
            ]
        
        self.assignments = [x for x in self.assignments if x.task_id not in removed_task_ids]

    def remove_cluster(self, k: int) -> List[Assignment]:
        """聚类移除算子（星间互联版本）"""
        if not self.assignments:
            return []
        spans = []
        for a in self.assignments:
            st = a.chunk.start
            ft = a.chunk.finish
            spans.append((st, ft, a))
        spans.sort(key=lambda x: x[0])
        center = random.choice(spans)[0]
        spans.sort(key=lambda x: abs(x[0] - center))
        removed = []
        for _, _, a in spans:
            removed.append(a)
            if len(removed) >= min(k, len(self.assignments)):
                break
        self._remove_assignments(removed)
        return removed

    def remove_worst_rf(self, k: int) -> List[Assignment]:
        """RF优先移除算子（星间互联版本）"""
        if not self.assignments:
            return []
        order = sorted(self.assignments, key=lambda a: a.chunk.rf_bits, reverse=True)
        removed = order[:min(k, len(order))]
        self._remove_assignments(removed)
        return removed

    def remove_oldest(self, k: int) -> List[Assignment]:
        """最早任务移除算子（星间互联版本）"""
        if not self.assignments:
            return []
        order = sorted(self.assignments, key=lambda a: a.chunk.start)
        removed = order[:min(k, len(order))]
        self._remove_assignments(removed)
        return removed

    def repair_profit(self, max_trials: int = 256) -> int:
        """价值优先修复（星间互联版本）- 优化：批量过滤禁忌任务"""
        # 优化：提前过滤禁忌任务，减少循环内判断
        tabu_set = set(self.tabu)
        pending = [
            t for t in self.tasks 
            if t.task_id in self.unscheduled and t.task_id not in tabu_set
        ]
        pending.sort(key=lambda t: t.value, reverse=True)
        cnt = 0
        # 优化：缓存place_task_multicand到局部
        place_func = self.place_task_multicand
        for t in pending:
            a = place_func(t, n_cand=4)
            if a:
                self.assignments.append(a)
                self.unscheduled.discard(t.task_id)
                cnt += 1
                if cnt >= max_trials:
                    break
        return cnt

    def repair_opportunity(self, max_trials: int = 256) -> int:
        """机会优先修复（星间互联版本）- 优化：预计算机会值"""
        tabu_set = set(self.tabu)
        pending = [
            t for t in self.tasks 
            if t.task_id in self.unscheduled and t.task_id not in tabu_set
        ]
        
        # 优化：预计算所有任务的机会值，避免重复计算
        opportunity_map = {}
        link_opportunity = self.link_opportunity
        links = list(self.links.keys())
        for t in pending:
            total = sum(link_opportunity(t, link) for link in links)
            opportunity_map[t.task_id] = total
        
        pending.sort(key=lambda t: -opportunity_map[t.task_id])
        cnt = 0
        place_func = self.place_task_multicand
        for t in pending:
            a = place_func(t, n_cand=3)
            if a:
                self.assignments.append(a)
                self.unscheduled.discard(t.task_id)
                cnt += 1
                if cnt >= max_trials:
                    break
        return cnt

    def repair_random(self, max_trials: int = 256) -> int:
        """随机修复（星间互联版本）"""
        tabu_set = set(self.tabu)
        pending = [
            t for t in self.tasks 
            if t.task_id in self.unscheduled and t.task_id not in tabu_set
        ]
        random.shuffle(pending)
        cnt = 0
        place_func = self.place_task_multicand
        for t in pending:
            a = place_func(t, n_cand=2)
            if a:
                self.assignments.append(a)
                self.unscheduled.discard(t.task_id)
                cnt += 1
                if cnt >= max_trials:
                    break
        return cnt

    def run_alns_ts(self, print_every: int = 20) -> Tuple[float, List[Assignment], List[float]]:
        """运行ISL-ALNS+TS算法（星间互联版本）- 优化：减少循环内的重复计算"""
        self.build_initial()
        temp_start = 10.0
        temp_end = 0.05
        k_max_ratio = (0.30, 0.10)
        temps = np.linspace(temp_start, temp_end, self.config.ITERATIONS)
        trace = [self.best_value]

        destroy_ops = [self.remove_cluster, self.remove_worst_rf, self.remove_oldest]
        repair_ops = [self.repair_profit, self.repair_opportunity, self.repair_random]
        repair_weights = [2.0, 1.5, 1.0]  # 优化：缓存权重

        # 优化：预计算常用变量
        iterations = self.config.ITERATIONS
        reheat_period = self.config.REHEAT_PERIOD
        math_exp = math.exp
        random_choice = random.choice
        random_choices = random.choices
        random_random = random.random

        print(f"[ISL-ALNS] Start iterations={iterations}  init_cur={self.cur_value:.3f}  "
              f"init_best={self.best_value:.3f}", flush=True)

        for it in range(iterations):
            # 备份当前状态
            cur_backup_assign = list(self.assignments)
            cur_backup_slots = {ln: list(v) for ln, v in self.slots_by_link.items()}
            cur_backup_uns = set(self.unscheduled)

            # 选择算子
            rem = random_choice(destroy_ops)
            rep = random_choices(repair_ops, weights=repair_weights)[0]
            ratio = k_max_ratio[0] - (k_max_ratio[0] - k_max_ratio[1]) * (it / max(1, iterations - 1))
            k = max(5, int(ratio * max(1, len(self.assignments))))

            # 执行销毁和修复
            removed = rem(k)
            rep(max_trials=256)

            # 计算新解
            new_val = self.objective(self.assignments)
            delta = new_val - self.cur_value
            T = temps[it]
            accept = (delta >= 0) or (random_random() < math_exp(delta / max(T, 1e-9)))

            # 接受/拒绝新解
            if accept:
                self.cur_value = new_val
                if self.cur_value > self.best_value:
                    self.best_value = self.cur_value
                    self.best_assignments = list(self.assignments)
                for a in removed:
                    if random_random() < 0.3:
                        self.tabu.append(a.task_id)
            else:
                self.assignments = cur_backup_assign
                self.slots_by_link = defaultdict(list, cur_backup_slots)
                self.unscheduled = cur_backup_uns

            # 回温
            if reheat_period and (it + 1) % reheat_period == 0:
                temps = temps * 1.3
                temps = np.clip(temps, temp_end, temp_start)

            trace.append(self.best_value)

            # 打印进度
            if (it + 1) % print_every == 0 or it == 0 or (it + 1) == iterations:
                print(f"[Iter {it + 1}/{iterations}] cur={self.cur_value:.3f}  best={self.best_value:.3f}  "
                      f"d={rem.__name__}  r={rep.__name__}", flush=True)

        # 重建最优解的资源预留
        self.assignments = list(self.best_assignments)
        self.slots_by_link = defaultdict(list)
        for a in self.assignments:
            c = a.chunk
            self.reserve(c.link, int(c.start), math.ceil(c.finish), a.task_id)

        print(f"[ISL-ALNS] Done  best={self.best_value:.3f}  scheduled={len(self.best_assignments)}", flush=True)
        return self.best_value, self.best_assignments, trace