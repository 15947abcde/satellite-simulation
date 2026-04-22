#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
遗传算法（GA）调度器（单链路+无分片）
核心逻辑：基于遗传算法的任务调度优化
约束：
- 每个任务仅调度在归属链路（home_link）
- 每个任务为连续的单个Chunk
加速优化：
1) 预计算每个任务在归属链路上的候选段
2) 插入时保持链路时隙有序，避免重复排序
"""

import math
import random
import re
import glob
import os
from bisect import bisect_left
from collections import defaultdict, deque, namedtuple
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ================= 配置类（统一管理参数）=================
class SchedulerConfigGA:
    def __init__(self):
        # GA算法参数
        self.ITERATIONS = 500
        self.SEED = 42
        self.GB_IS_GIB = False
        self.RF_PENALTY_PER_GBIT = 0.0
        self.SERVICE_WINDOW_SEC = 36000
        # 惩罚项参数
        self.EARLY_WINDOW_SEC = 0
        self.EARLY_PENALTY_COEF = 1e-11
        self.RANDOM_RATE_JITTER = 0.02
        self.REHEAT_PERIOD = 100
        # 速率映射
        self.RATE_MAP_GBPS = {1: 1.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0}
        self.RATE_MAP = {k: v * 1e9 for k, v in self.RATE_MAP_GBPS.items()}
        self.LINK_TYPE = {1: "RF", 2: "OPT", 3: "OPT", 4: "OPT", 5: "OPT"}
        # 切换时间
        self.MOD_SWITCH_TIME = 0.5
        self.LINK_SWITCH_TIME = 1.0
        # GA运行参数
        self.POP_SIZE = 30
        self.GENERATIONS = 120
        self.CX_PROB = 0.9
        self.MUT_PROB = 0.2
        # 算法名称
        self.ALGO_NAME = "GA"


# ================= 辅助函数 =================
def mean_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """返回: (mean, lower, upper)，t 分布 95% CI"""
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
    """自动选择计算设备（CUDA/CPU）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_bits(size_gb: float, gb_is_gib: bool) -> float:
    """比特转换逻辑"""
    if isinstance(size_gb, str):
        size_gb = float(size_gb)
    if gb_is_gib:
        return size_gb * (1024 ** 3) * 8.0
    return size_gb * (1000 ** 3) * 8.0


# ================= 数据结构 =================
Segment = namedtuple("Segment", ["start", "end", "rate", "mod_level", "link_type"])


class LinkFTW:
    """链路可见时间窗口（GA版本）"""
    def __init__(self, name: str):
        self.name = name
        self.segments: List[Segment] = []

    def add_segment(self, start: int, end: int, mod: int, config: SchedulerConfigGA):
        """添加链路段（适配配置类）"""
        if end > start:
            self.segments.append(
                Segment(start, end, config.RATE_MAP[int(mod)], int(mod), config.LINK_TYPE[int(mod)])
            )

    def finalize(self):
        """合并连续相同调制的段"""
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
        """遍历指定时间窗口内的段"""
        for s in self.segments:
            if s.end <= t0:
                continue
            if s.start >= t1:
                break
            yield Segment(max(s.start, t0), min(s.end, t1), s.rate, s.mod_level, s.link_type)


class Task:
    """任务数据结构（GA版本，单链路归属）"""
    __slots__ = ("task_id", "arrival", "deadline", "size_bits", "priority", "value", "home_link", "gen_time")

    def __init__(self, task_id, arrival, deadline, size_bits, priority, value, home_link, gen_time):
        self.task_id = str(task_id)
        self.arrival = int(arrival)
        self.deadline = int(deadline)
        self.size_bits = float(size_bits)
        self.priority = int(priority)
        self.value = float(value)
        self.home_link = str(home_link)  # 单链路归属约束
        self.gen_time = int(gen_time)


# 无分片Chunk（GA版本）
Chunk = namedtuple(
    "Chunk",
    ["link", "start", "finish", "bits", "rf_bits", "mod_switches", "link_switches", "mod_level", "link_type"]
)
Assignment = namedtuple("Assignment", ["task_id", "chunks", "total_bits", "value"])


# ================= 数据加载函数（GA版本）=================
def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """解析文件名提取链路名和时间（GA版本增强解析）"""
    name = Path(filename).stem.strip()
    if '-' in name:
        prefix, tail = name.rsplit('-', 1)
        tail = tail.strip()
        if tail.isdigit():
            return prefix, int(tail)
    if '_' in name:
        prefix, tail = name.rsplit('_', 1)
        tail = tail.strip()
        if tail.isdigit():
            return prefix, int(tail)
    m = re.search(r'(\d+)\s*$', name)
    if m:
        return name[:m.start()].rstrip('-_ '), int(m.group(1))
    return None, None


def load_tasks_from_dir(tasks_dir: str, config: SchedulerConfigGA) -> List[Task]:
    """加载任务数据（GA版本，增强优先级分配）"""
    tasks = []
    files = []
    for ext in ("*.csv", "*.CSV"):
        files += glob.glob(os.path.join(tasks_dir, ext))
    if not files:
        raise FileNotFoundError(f"No task CSV files in {tasks_dir}")

    for f in files:
        fname = Path(f).name
        link, arr = parse_filename(fname)
        if link is None or arr is None:
            print(f"[SKIP:name] {fname}")
            continue

        # 鲁棒读取CSV（自动识别分隔符，处理表头）
        df = pd.read_csv(
            f,
            header=None,
            names=["value", "size_gb"],
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        # 过滤非数值表头
        def _num(x):
            try:
                float(x)
                return True
            except:
                return False

        if len(df) > 0 and (not _num(df.iloc[0, 0]) or not _num(df.iloc[0, 1])):
            df = df.iloc[1:].copy()

        # 清洗脏数据
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["size_gb"] = pd.to_numeric(df["size_gb"], errors="coerce")
        df = df.dropna(subset=["value", "size_gb"]).sort_values("value", ascending=False).reset_index(drop=True)

        if df.empty:
            print(f"[SKIP:empty] {fname}")
            continue

        # GA版本：按价值分配优先级
        n = len(df)
        hi = int(n * 0.2)
        lo = int(n * 0.8)
        for i, row in df.iterrows():
            pr = 10 if i < hi else (5 if i < lo else 1)
            size_bits = to_bits(row["size_gb"], config.GB_IS_GIB)
            deadline = arr + config.SERVICE_WINDOW_SEC
            tid = f"{fname}_task{i}"
            tasks.append(Task(
                tid, arr, deadline, size_bits, pr,
                float(row["value"]), link, gen_time=arr
            ))

    if not tasks:
        raise ValueError("No valid tasks found (after parsing/cleaning).")
    return tasks


def load_link_ftw_csv(path: str, link_name: str, config: SchedulerConfigGA) -> LinkFTW:
    """加载链路可见性数据（GA版本，增强格式兼容）"""
    try:
        df = pd.read_csv(path)
        if not {"valid_time_seconds", "modulation_level"}.issubset(df.columns):
            df = pd.read_csv(path, header=None, names=["valid_time_seconds", "modulation_level"])
    except:
        df = pd.read_csv(path, header=None, names=["valid_time_seconds", "modulation_level"])

    df = df.sort_values("valid_time_seconds")
    ftw = LinkFTW(link_name)
    prev_m = None
    run_s = None
    last = None

    for _, r in df.iterrows():
        try:
            t = int(r["valid_time_seconds"])
            m = int(r["modulation_level"])
        except (ValueError, TypeError):
            continue  # 跳过无效行

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


def load_all_links(visibility_dir: str, config: SchedulerConfigGA) -> Dict[str, LinkFTW]:
    """加载所有链路数据（GA版本，单链路归属）"""
    link_map = {}
    csvs = []
    for ext in ("*.csv", "*.CSV"):
        csvs += glob.glob(os.path.join(visibility_dir, ext))
    if not csvs:
        raise FileNotFoundError(f"No visibility CSV files in {visibility_dir}")

    for p in csvs:
        link_name = Path(p).stem
        link_map[link_name] = load_link_ftw_csv(p, link_name, config)
    return link_map


# ================= 核心调度器类（遗传算法）=================
class SchedulerGA:
    def __init__(self, tasks: List[Task], links: Dict[str, LinkFTW], device, config: SchedulerConfigGA, seed=42):
        """初始化（GA版本，单链路归属约束）"""
        self.tasks = tasks
        self.links = links
        self.device = device
        self.config = config
        self.rf_penalty = config.RF_PENALTY_PER_GBIT

        # 调度状态
        self.assignments: List[Assignment] = []
        self.unscheduled = set(t.task_id for t in tasks)
        self.slots_by_link: Dict[str, List[Tuple[int, int, str]]] = defaultdict(list)

        # 最优解跟踪
        self.best_assignments = None
        self.best_value = -1e18
        self.cur_assignments = None
        self.cur_value = -1e18

        # 禁忌表（GA预留扩展）
        self.tabu = deque(maxlen=64)

        # 加速优化：预计算每个任务的候选段
        self._task_seg_cache: Dict[str, List[Tuple[int, int, float, int, str]]] = {}
        self._build_task_seg_cache()

        # 初始化随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _build_task_seg_cache(self):
        """预计算每个任务在归属链路上的候选段（加速优化）"""
        cache = {}
        for t in self.tasks:
            ln = t.home_link
            if ln not in self.links:
                cache[t.task_id] = []
                continue
            segs = []
            for seg in self.links[ln].iter_segments_in_window(t.arrival, t.deadline):
                segs.append((int(seg.start), int(seg.end), float(seg.rate), int(seg.mod_level), str(seg.link_type)))
            cache[t.task_id] = segs
        self._task_seg_cache = cache

    def get_free_gaps(self, link: str, a: int, b: int) -> List[Tuple[int, int]]:
        """获取链路空闲时隙（GA版本，时隙已排序）"""
        used = self.slots_by_link[link]  # 已按start排序
        gaps = []
        cur = a
        for s, e, _ in used:
            if e <= cur:
                continue
            if s > cur:
                gaps.append((cur, min(s, b)))
                cur = min(s, b)
            cur = max(cur, e)
            if cur >= b:
                break
        if cur < b:
            gaps.append((cur, b))
        return gaps

    def reserve(self, link: str, start: int, finish: int, task_id: str):
        """预留链路时隙（GA版本，插入时保持有序）"""
        lst = self.slots_by_link[link]
        pos = bisect_left([x[0] for x in lst], start)
        lst.insert(pos, (start, finish, task_id))

    def candidate_heuristic_cost(self, t: Task, chunks: List[Chunk]) -> float:
        """候选方案的启发式成本计算（GA版本）"""
        early_end = t.arrival + self.config.EARLY_WINDOW_SEC
        early_bits = 0.0
        for c in chunks:
            dur = max(1e-9, c.finish - c.start)
            over = max(0.0, min(c.finish, early_end) - max(c.start, t.arrival))
            if over > 0:
                early_bits += (over / dur) * c.bits
        return -t.value + self.config.EARLY_PENALTY_COEF * early_bits

    def try_pack_task_once(self, t: Task) -> Optional[List[Chunk]]:
        """尝试打包单个任务（单链路+无分片，GA核心）"""
        ln = t.home_link
        if ln not in self.links:
            return None

        need_bits = t.size_bits
        base_segs = self._task_seg_cache.get(t.task_id, [])
        if not base_segs:
            return None

        # 速率抖动（GA版本优化）
        seg_list = []
        for (ss, ee, rate, mod_level, link_type) in base_segs:
            jitter = 1.0 + random.uniform(-self.config.RANDOM_RATE_JITTER, self.config.RANDOM_RATE_JITTER)
            seg_list.append((ss, ee, rate, mod_level, link_type, rate * jitter))

        # 按起始时间、速率、链路类型排序
        seg_list.sort(key=lambda x: (x[0], -x[5], x[4] != "OPT"))
        snapshot = list(self.slots_by_link[ln])

        for ss, ee, rate, mod_level, link_type, _ in seg_list:
            gaps = self.get_free_gaps(ln, ss, ee)
            for (gs, ge) in gaps:
                dur = need_bits / rate
                if gs + dur <= ge:
                    # 找到可用时隙，创建Chunk
                    start = float(gs)
                    finish = float(gs + dur)
                    rf_bits = need_bits if link_type == "RF" else 0.0
                    self.reserve(ln, int(start), int(math.ceil(finish)), t.task_id)
                    chunk = Chunk(
                        ln, start, finish, need_bits, rf_bits,
                        0, 0, int(mod_level), str(link_type)
                    )
                    return [chunk]

        # 恢复快照
        self.slots_by_link[ln] = snapshot
        return None

    def place_task_multicand(self, t: Task, n_cand: int = 3) -> Optional[Assignment]:
        """多候选方案放置任务（GA版本）"""
        best = None
        best_cost = None
        best_reserve = None
        snapshot = {ln: list(v) for ln, v in self.slots_by_link.items()}

        for _ in range(n_cand):
            self.slots_by_link = defaultdict(list, {ln: list(v) for ln, v in snapshot.items()})
            chunks = self.try_pack_task_once(t)
            if chunks is None:
                continue
            cost = self.candidate_heuristic_cost(t, chunks)
            if (best is None) or (cost < best_cost):
                best = chunks
                best_cost = cost
                best_reserve = {ln: list(v) for ln, v in self.slots_by_link.items()}

        if best is None:
            self.slots_by_link = defaultdict(list, snapshot)
            return None

        self.slots_by_link = defaultdict(list, best_reserve)
        return Assignment(t.task_id, best, sum(c.bits for c in best), t.value)

    def objective(self, assigns: List[Assignment]) -> float:
        """目标函数计算（GA版本，含RF惩罚）"""
        v = 0.0
        for a in assigns:
            rf_bits = sum(c.rf_bits for c in a.chunks)
            v += a.value - self.rf_penalty * (rf_bits / 1e9)
        return v

    def _reset_state(self):
        """重置调度状态"""
        self.assignments = []
        self.unscheduled = set(t.task_id for t in self.tasks)
        self.slots_by_link = defaultdict(list)
        self.best_assignments = None
        self.best_value = -1e18
        self.cur_assignments = None
        self.cur_value = -1e18
        self.tabu = deque(maxlen=64)

    def _finalize_best(self, assigns: List[Assignment]):
        """更新最优解"""
        self.best_assignments = list(assigns)
        self.best_value = self.objective(assigns)

    def run_ga(self) -> Tuple[float, List[Assignment], List[float]]:
        """运行遗传算法（核心入口）"""
        print(f"[GA] Start scheduling {len(self.tasks)} tasks...", flush=True)
        print(f"[GA] Pop size: {self.config.POP_SIZE} | Generations: {self.config.GENERATIONS}", flush=True)
        
        # 执行GA算法
        best_assignments, trace = self.solve_ga(
            pop_size=self.config.POP_SIZE,
            generations=self.config.GENERATIONS,
            cx_prob=self.config.CX_PROB,
            mut_prob=self.config.MUT_PROB
        )
        
        # 获取最优值
        best_value = self.best_value
        print(f"[GA] Done | Best Value = {best_value:.3f} | Scheduled Tasks = {len(best_assignments)}", flush=True)
        return best_value, best_assignments, trace

    def solve_ga(self, pop_size: int = 30, generations: int = 120, cx_prob: float = 0.9, mut_prob: float = 0.2) -> Tuple[List[Assignment], List[float]]:
        """遗传算法核心实现（保持原逻辑不变）"""
        self._reset_state()

        tasks = self.tasks
        n = len(tasks)
        idxs = list(range(n))

        place = self.place_task_multicand
        objective = self.objective

        def decode(order_idx) -> Tuple[float, List[Assignment]]:
            """解码：将任务顺序转换为调度结果"""
            backup_assign = self.assignments
            backup_slots = self.slots_by_link
            backup_uns = self.unscheduled

            self.assignments = []
            self.slots_by_link = defaultdict(list)
            self.unscheduled = set(t.task_id for t in tasks)

            ok = []
            uns = self.unscheduled
            for i in order_idx:
                t = tasks[i]
                tid = t.task_id
                if tid in uns:
                    a = place(t, n_cand=2)
                    if a:
                        ok.append(a)
                        uns.remove(tid)

            val = objective(ok)

            # 恢复状态
            self.assignments = backup_assign
            self.slots_by_link = backup_slots
            self.unscheduled = backup_uns
            return val, ok

        def ox(p1, p2):
            """顺序交叉（OX）"""
            a, b = sorted(random.sample(range(n), 2))
            child = [-1] * n
            child[a:b] = p1[a:b]
            fill = [x for x in p2 if x not in child[a:b]]
            ptr = 0
            for i in range(n):
                if child[i] == -1:
                    child[i] = fill[ptr]
                    ptr += 1
            return child

        def mutate_swap(p):
            """交换变异"""
            i, j = random.sample(range(n), 2)
            p[i], p[j] = p[j], p[i]

        # 初始化种群
        pop = []
        for _ in range(pop_size):
            p = idxs[:]
            random.shuffle(p)
            pop.append(p)

        best_val = -1e18
        best_assign = None
        trace = []

        # 迭代进化
        for _g in range(generations):
            # 评估种群
            scored = []
            for p in pop:
                v, assigns = decode(p)
                scored.append((v, p, assigns))
            scored.sort(key=lambda x: x[0], reverse=True)

            # 更新最优解
            if scored[0][0] > best_val:
                best_val = scored[0][0]
                best_assign = scored[0][2]

            trace.append(best_val)

            # 精英保留
            elites = [scored[i][1] for i in range(max(2, pop_size // 6))]
            new_pop = elites[:]

            # 锦标赛选择
            def tournament():
                cand = random.sample(scored, k=min(4, len(scored)))
                cand.sort(key=lambda x: x[0], reverse=True)
                return cand[0][1][:]

            # 生成新种群
            while len(new_pop) < pop_size:
                p1 = tournament()
                p2 = tournament()
                if random.random() < cx_prob:
                    c = ox(p1, p2)
                else:
                    c = p1[:]
                if random.random() < mut_prob:
                    mutate_swap(c)
                new_pop.append(c)
            pop = new_pop

        # 提交最优解
        self._reset_state()
        for a in best_assign:
            self.assignments.append(a)
            self.unscheduled.discard(a.task_id)
            for c in a.chunks:
                self.reserve(c.link, int(c.start), int(math.ceil(c.finish)), a.task_id)

        self._finalize_best(self.assignments)
        return self.best_assignments, trace