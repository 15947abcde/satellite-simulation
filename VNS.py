#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VNS (变邻域搜索) Scheduler（单链路+无分片）
Neighborhoods:
- level 0: remove_cluster + repair_profit (multi-cand)
- level 1: remove_worst_rf + repair_opportunity (multi-cand)
- level 2: small random remove + strong reinsert (higher n_cand)
"""

import math
import random
import re
import glob
import os
from collections import defaultdict, deque, namedtuple
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ================= 配置类（统一管理参数）=================
class SchedulerConfigVNS:
    def __init__(self):
        # 核心参数
        self.SEED = 42
        self.GB_IS_GIB = False
        self.RF_PENALTY_PER_GBIT = 0.0
        self.SERVICE_WINDOW_SEC = 36000
        # VNS算法参数
        self.VNS_ITERS = 20               # VNS总迭代次数
        self.NO_IMPROVE_TO_SWITCH = 10     # 无改进切换邻域的阈值
        self.RANDOM_RATE_JITTER = 0.02     # 速率抖动系数
        self.TABU_LEN = 64                 # 禁忌表长度
        # 邻域操作参数
        self.CLUSTER_REMOVE_RATE = 0.20    # 簇移除比例
        self.RF_REMOVE_RATE = 0.12         # RF任务移除比例
        self.MIN_REMOVE = 5                # 最小移除数量
        self.MAX_REPAIR_TRIALS = 256       # 修复最大尝试次数
        # 速率映射
        self.RATE_MAP_GBPS = {1: 1.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0}
        self.RATE_MAP = {k: v * 1e9 for k, v in self.RATE_MAP_GBPS.items()}
        self.LINK_TYPE = {1: "RF", 2: "OPT", 3: "OPT", 4: "OPT", 5: "OPT"}
        # 算法名称
        self.ALGO_NAME = "VNS"


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
    """链路可见时间窗口（VNS版本）"""
    def __init__(self, name: str):
        self.name = name
        self.segments: List[Segment] = []

    def add_segment(self, start: int, end: int, mod: int, config: SchedulerConfigVNS):
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
    """任务数据结构（VNS版本，单链路归属）"""
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


# 无分片Chunk（VNS版本）
Chunk = namedtuple(
    "Chunk",
    ["link", "start", "finish", "bits", "rf_bits", "mod_switches", "link_switches", "mod_level", "link_type"]
)
Assignment = namedtuple("Assignment", ["task_id", "chunks", "total_bits", "value"])


# ================= 数据加载函数（VNS版本）=================
def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """解析文件名提取链路名和时间"""
    name = Path(filename).stem.strip()
    m = re.search(r'(.+?)[-_]?(\d+)$', name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def load_tasks_from_dir(tasks_dir: str, config: SchedulerConfigVNS) -> List[Task]:
    """加载任务数据（VNS版本，鲁棒解析）"""
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
        df = df.dropna(subset=["value", "size_gb"]).reset_index(drop=True)

        if df.empty:
            print(f"[SKIP:empty] {fname}")
            continue

        for i, row in df.iterrows():
            size_bits = to_bits(row["size_gb"], config.GB_IS_GIB)
            deadline = arr + config.SERVICE_WINDOW_SEC
            tid = f"{fname}_task{i}"
            tasks.append(Task(
                tid, arr, deadline, size_bits, 1,
                float(row["value"]), link, gen_time=arr
            ))

    if not tasks:
        raise ValueError("No valid tasks found (after parsing/cleaning).")
    return tasks


def load_link_ftw_csv(path: str, link_name: str, config: SchedulerConfigVNS) -> LinkFTW:
    """加载链路可见性数据（VNS版本，鲁棒解析）"""
    # 先按“可能有表头”读
    try:
        df = pd.read_csv(path)
        if not {"valid_time_seconds", "modulation_level"}.issubset(df.columns):
            # 不是标准列名 -> 当成无表头两列
            df = pd.read_csv(path, header=None, names=["valid_time_seconds", "modulation_level"])
    except Exception:
        df = pd.read_csv(path, header=None, names=["valid_time_seconds", "modulation_level"])

    # 清洗数据
    df["valid_time_seconds"] = pd.to_numeric(df["valid_time_seconds"], errors="coerce")
    df["modulation_level"] = pd.to_numeric(df["modulation_level"], errors="coerce")
    df = df.dropna(subset=["valid_time_seconds", "modulation_level"])

    if df.empty:
        ftw = LinkFTW(link_name)
        ftw.finalize()
        return ftw

    df = df.sort_values("valid_time_seconds")

    ftw = LinkFTW(link_name)
    prev_m = None
    run_s = None
    last = None

    for _, r in df.iterrows():
        t = int(r["valid_time_seconds"])
        m = int(r["modulation_level"])

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


def load_all_links(visibility_dir: str, config: SchedulerConfigVNS) -> Dict[str, LinkFTW]:
    """加载所有链路数据（VNS版本，单链路归属）"""
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


# ================= 核心调度器类（VNS）=================
class SchedulerVNS:
    def __init__(self, tasks: List[Task], links: Dict[str, LinkFTW], device, config: SchedulerConfigVNS, seed=42):
        """初始化（VNS版本，单链路归属约束）"""
        self.tasks = tasks
        self.links = links
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
        self.slots_by_link = defaultdict(list)

        # 最优解跟踪
        self.best_assignments = None
        self.best_value = -1e18

        # 禁忌表
        self.tabu = deque(maxlen=config.TABU_LEN)

    def get_free_gaps(self, link: str, a: int, b: int) -> List[Tuple[int, int]]:
        """获取链路空闲时隙（VNS版本）"""
        used = sorted([(s, e) for (s, e, _) in self.slots_by_link[link]], key=lambda x: x[0])
        gaps = []
        cur = a
        for s, e in used:
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

    def reserve(self, link: str, start: int, finish: int, tid: str):
        """预留链路时隙（VNS版本）"""
        self.slots_by_link[link].append((start, finish, tid))

    def _reset_state(self):
        """重置调度状态"""
        self.assignments = []
        self.unscheduled = set(t.task_id for t in self.tasks)
        self.slots_by_link = defaultdict(list)
        self.best_assignments = None
        self.best_value = -1e18
        self.tabu = deque(maxlen=self.config.TABU_LEN)

    def objective(self, assigns: List[Assignment]) -> float:
        """目标函数计算（含RF惩罚）"""
        v = 0.0
        for a in assigns:
            rf_bits = sum(c.rf_bits for c in a.chunks)
            v += a.value - self.rf_penalty * (rf_bits / 1e9)
        return v

    def try_pack_task_once(self, t: Task) -> Optional[List[Chunk]]:
        """单次打包任务（无分片，VNS基础版）"""
        ln = t.home_link
        if ln not in self.links:
            return None

        need_bits = t.size_bits
        snapshot = list(self.slots_by_link[ln])

        for seg in self.links[ln].iter_segments_in_window(t.arrival, t.deadline):
            gaps = self.get_free_gaps(ln, seg.start, seg.end)
            for gs, ge in gaps:
                dur = need_bits / seg.rate
                if gs + dur <= ge:
                    start = float(gs)
                    finish = float(gs + dur)
                    rf_bits = need_bits if seg.link_type == "RF" else 0.0
                    self.reserve(ln, int(start), int(math.ceil(finish)), t.task_id)
                    return [Chunk(ln, start, finish, need_bits, rf_bits, 0, 0, seg.mod_level, seg.link_type)]

        self.slots_by_link[ln] = snapshot
        return None

    def try_pack_task_multicand(self, t: Task, n_cand: int = 6) -> Optional[List[Chunk]]:
        """多候选打包任务（VNS优化版）"""
        ln = t.home_link
        if ln not in self.links:
            return None

        need_bits = t.size_bits
        snapshot = list(self.slots_by_link[ln])

        segs = []
        for seg in self.links[ln].iter_segments_in_window(t.arrival, t.deadline):
            jitter = 1.0 + random.uniform(-self.config.RANDOM_RATE_JITTER, self.config.RANDOM_RATE_JITTER)
            segs.append((seg, seg.rate * jitter))

        if not segs:
            self.slots_by_link[ln] = snapshot
            return None

        segs.sort(key=lambda x: (x[0].start, -x[1], x[0].link_type != "OPT"))

        best = None  # (finish_time, start, seg)
        best_state = None

        for seg, _ in segs[:min(n_cand, len(segs))]:
            self.slots_by_link[ln] = list(snapshot)
            gaps = self.get_free_gaps(ln, seg.start, seg.end)
            for gs, ge in gaps:
                dur = need_bits / seg.rate
                if gs + dur <= ge:
                    start = float(gs)
                    finish = float(gs + dur)
                    score = finish
                    if (best is None) or (score < best[0]):
                        rf_bits = need_bits if seg.link_type == "RF" else 0.0
                        _ = [Chunk(ln, start, finish, need_bits, rf_bits, 0, 0, seg.mod_level, seg.link_type)]
                        self.reserve(ln, int(start), int(math.ceil(finish)), t.task_id)
                        best = (score, start, seg)
                        best_state = list(self.slots_by_link[ln])
                    break

        if best is None:
            self.slots_by_link[ln] = snapshot
            return None

        self.slots_by_link[ln] = best_state
        seg = best[2]
        start = float(best[1])
        finish = float(best[0])
        rf_bits = need_bits if seg.link_type == "RF" else 0.0
        return [Chunk(ln, start, finish, need_bits, rf_bits, 0, 0, seg.mod_level, seg.link_type)]

    def place_task(self, t: Task, n_cand: int = 1) -> Optional[Assignment]:
        """放置任务（适配单/多候选）"""
        chunks = self.try_pack_task_once(t) if n_cand <= 1 else self.try_pack_task_multicand(t, n_cand=n_cand)
        if chunks is None:
            return None
        return Assignment(t.task_id, chunks, sum(c.bits for c in chunks), t.value)

    def _remove_assignments(self, removed: List[Assignment]):
        """移除任务分配（VNS邻域操作基础）"""
        removed_ids = {a.task_id for a in removed}
        for a in removed:
            self.unscheduled.add(a.task_id)
            for ln in list(self.slots_by_link.keys()):
                self.slots_by_link[ln] = [(s, e, tid) for (s, e, tid) in self.slots_by_link[ln] if tid != a.task_id]
        self.assignments = [x for x in self.assignments if x.task_id not in removed_ids]

    def remove_cluster(self):
        """邻域0：簇移除操作"""
        if not self.assignments:
            return []
        spans = []
        for a in self.assignments:
            st = min(c.start for c in a.chunks)
            ft = max(c.finish for c in a.chunks)
            spans.append((st, ft, a))
        spans.sort(key=lambda x: x[0])
        center = random.choice(spans)[0]
        spans.sort(key=lambda x: abs(x[0] - center))
        k = max(self.config.MIN_REMOVE, int(self.config.CLUSTER_REMOVE_RATE * max(1, len(self.assignments))))
        removed = []
        for _, _, a in spans:
            removed.append(a)
            if len(removed) >= min(k, len(self.assignments)):
                break
        self._remove_assignments(removed)
        return removed

    def remove_worst_rf(self):
        """邻域1：RF惩罚最大任务移除"""
        if not self.assignments:
            return []
        order = sorted(self.assignments, key=lambda a: sum(c.rf_bits for c in a.chunks), reverse=True)
        k = max(self.config.MIN_REMOVE, int(self.config.RF_REMOVE_RATE * max(1, len(self.assignments))))
        removed = order[:min(k, len(order))]
        self._remove_assignments(removed)
        return removed

    def repair_profit(self, n_cand: int = 4):
        """利润导向修复（邻域0/2）"""
        pending = [t for t in self.tasks if t.task_id in self.unscheduled and t.task_id not in self.tabu]
        pending.sort(key=lambda t: t.value, reverse=True)
        cnt = 0
        for t in pending:
            a = self.place_task(t, n_cand=n_cand)
            if a:
                self.assignments.append(a)
                self.unscheduled.discard(t.task_id)
                cnt += 1
                if cnt >= self.config.MAX_REPAIR_TRIALS:
                    break
        return cnt

    def repair_opportunity(self, n_cand: int = 4):
        """机会导向修复（邻域1/2）"""
        pending = [t for t in self.tasks if t.task_id in self.unscheduled and t.task_id not in self.tabu]

        def opp(t: Task):
            if t.home_link not in self.links:
                return 0.0
            s = 0.0
            for seg in self.links[t.home_link].iter_segments_in_window(t.arrival, t.deadline):
                w = 1.0 if seg.link_type == "OPT" else 0.8
                s += w * seg.rate * (seg.end - seg.start)
            return s

        pending.sort(key=lambda t: -opp(t))
        cnt = 0
        for t in pending:
            a = self.place_task(t, n_cand=n_cand)
            if a:
                self.assignments.append(a)
                self.unscheduled.discard(t.task_id)
                cnt += 1
                if cnt >= self.config.MAX_REPAIR_TRIALS:
                    break
        return cnt

    def solve_greedy_density(self) -> Tuple[List[Assignment], List[float]]:
        """贪心初始解（密度优先）"""
        self._reset_state()
        order = sorted(self.tasks, key=lambda t: (t.value / max(1.0, t.size_bits)), reverse=True)
        for t in order:
            if t.task_id in self.unscheduled:
                a = self.place_task(t, n_cand=2)
                if a:
                    self.assignments.append(a)
                    self.unscheduled.discard(t.task_id)
        self.best_assignments = list(self.assignments)
        self.best_value = self.objective(self.assignments)
        return self.best_assignments, [self.best_value]

    def run_vns(self) -> Tuple[float, List[Assignment], List[float]]:
        """运行VNS算法（核心入口）"""
        print(f"[VNS] Start Iterations={self.config.VNS_ITERS} | NoImproveSwitch={self.config.NO_IMPROVE_TO_SWITCH}", flush=True)
        
        # 贪心初始解
        self.solve_greedy_density()
        best = list(self.best_assignments)
        best_val = self.objective(best)
        trace = [best_val]

        level = 0
        no_imp = 0

        # VNS主循环
        for vns_iter in range(self.config.VNS_ITERS):
            # 备份当前状态
            backup_assign = list(self.assignments)
            backup_slots = {ln: list(v) for ln, v in self.slots_by_link.items()}
            backup_uns = set(self.unscheduled)

            # 邻域操作
            if level == 0:
                self.remove_cluster()
                self.repair_profit(n_cand=6)
            elif level == 1:
                self.remove_worst_rf()
                self.repair_opportunity(n_cand=6)
            else:
                # 邻域2：随机小移除 + 强重插入
                if self.assignments:
                    k = min(6, len(self.assignments))
                    pick = random.sample(self.assignments, k=k)
                    self._remove_assignments(pick)

                if random.random() < 0.5:
                    self.repair_profit(n_cand=10)
                else:
                    self.repair_opportunity(n_cand=10)

            # 评估新解
            val = self.objective(self.assignments)
            if val > best_val:
                best_val = val
                best = list(self.assignments)
                no_imp = 0
                level = min(2, level + 1)
            else:
                # 回滚到备份
                self.assignments = backup_assign
                self.slots_by_link = defaultdict(list, backup_slots)
                self.unscheduled = backup_uns
                no_imp += 1
                if no_imp >= self.config.NO_IMPROVE_TO_SWITCH:
                    no_imp = 0
                    level = max(0, level - 1)

            trace.append(best_val)

            # 打印VNS进度
            if (vns_iter + 1) % 20 == 0 or vns_iter == 0 or (vns_iter + 1) == self.config.VNS_ITERS:
                print(f"[VNS Iter {vns_iter+1}/{self.config.VNS_ITERS}] Level={level} | Best Value = {best_val:.3f}", flush=True)

        # 恢复最优解状态
        self._reset_state()
        for a in best:
            self.assignments.append(a)
            self.unscheduled.discard(a.task_id)
            for c in a.chunks:
                self.reserve(c.link, int(c.start), int(math.ceil(c.finish)), a.task_id)

        self.best_assignments = list(self.assignments)
        self.best_value = self.objective(self.assignments)

        print(f"[VNS] Done | Best Value = {self.best_value:.3f} | Scheduled = {len(self.best_assignments)}", flush=True)
        return self.best_value, self.best_assignments, trace