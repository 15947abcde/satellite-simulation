#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Greedy-Density 调度器（单链路+无分片）
核心逻辑：按价值密度（value/size_bits）贪心调度
约束：
- 每个任务仅调度在归属链路（home_link）
- 每个任务为连续的单个Chunk
"""

import math
import random
import re
import glob
import os
from collections import defaultdict, namedtuple
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


# ================= 配置类（统一管理参数）=================
class SchedulerConfigGreedy:
    def __init__(self):
        # 核心参数
        self.SEED = 42
        self.GB_IS_GIB = False
        self.RF_PENALTY_PER_GBIT = 0.0
        self.SERVICE_WINDOW_SEC = 36000
        # 速率映射
        self.RATE_MAP_GBPS = {1: 1.0, 2: 10.0, 3: 10.0, 4: 10.0, 5: 10.0}
        self.RATE_MAP = {k: v * 1e9 for k, v in self.RATE_MAP_GBPS.items()}
        self.LINK_TYPE = {1: "RF", 2: "OPT", 3: "OPT", 4: "OPT", 5: "OPT"}
        # 算法名称
        self.ALGO_NAME = "GreedyDensity"


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
    """链路可见时间窗口（贪心密度版本）"""
    def __init__(self, name: str):
        self.name = name
        self.segments: List[Segment] = []

    def add_segment(self, start: int, end: int, mod: int, config: SchedulerConfigGreedy):
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
    """任务数据结构（贪心密度版本，单链路归属）"""
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


# 无分片Chunk（贪心密度版本）
Chunk = namedtuple(
    "Chunk",
    ["link", "start", "finish", "bits", "rf_bits", "mod_switches", "link_switches", "mod_level", "link_type"]
)
Assignment = namedtuple("Assignment", ["task_id", "chunks", "total_bits", "value"])


# ================= 数据加载函数（贪心密度版本）=================
def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """解析文件名提取链路名和时间"""
    name = Path(filename).stem.strip()
    m = re.search(r'(.+?)[-_]?(\d+)$', name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def load_tasks_from_dir(tasks_dir: str, config: SchedulerConfigGreedy) -> List[Task]:
    """加载任务数据（贪心密度版本，鲁棒解析中文表头）"""
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

        # 鲁棒读取CSV（自动识别分隔符，处理中文表头）
        df = pd.read_csv(
            f,
            header=None,
            names=["value", "size_gb"],
            sep=None,
            engine="python",
            encoding="utf-8-sig"
        )

        # 清洗脏数据（非数值转NaN后删除）
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


def load_link_ftw_csv(path: str, link_name: str, config: SchedulerConfigGreedy) -> LinkFTW:
    """加载链路可见性数据（贪心密度版本）"""
    # 读取链路数据（无表头，两列：时间+调制方式）
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


def load_all_links(visibility_dir: str, config: SchedulerConfigGreedy) -> Dict[str, LinkFTW]:
    """加载所有链路数据（贪心密度版本，单链路归属）"""
    links = {}
    csv_paths = glob.glob(os.path.join(visibility_dir, "*.csv")) + glob.glob(os.path.join(visibility_dir, "*.CSV"))
    for p in csv_paths:
        ln = Path(p).stem
        links[ln] = load_link_ftw_csv(p, ln, config)
    return links


# ================= 核心调度器类（贪心密度）=================
class SchedulerGreedy:
    def __init__(self, tasks: List[Task], links: Dict[str, LinkFTW], device, config: SchedulerConfigGreedy, seed=42):
        """初始化（贪心密度版本，单链路归属约束）"""
        self.tasks = tasks
        self.links = links
        self.device = device
        self.config = config

        # 调度状态
        self.assignments: List[Assignment] = []
        self.unscheduled = set(t.task_id for t in tasks)
        self.slots_by_link = defaultdict(list)

        # 最优解跟踪
        self.best_assignments = []
        self.best_value = 0.0

        # 初始化随机种子（预留扩展）
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def get_free_gaps(self, link: str, a: int, b: int) -> List[Tuple[int, int]]:
        """获取链路空闲时隙（贪心密度版本）"""
        used = sorted(self.slots_by_link[link], key=lambda x: x[0])
        gaps = []
        cur = a
        for s, e, _ in used:
            if e <= cur:
                continue
            if s > cur:
                gaps.append((cur, min(s, b)))
            cur = max(cur, e)
            if cur >= b:
                break
        if cur < b:
            gaps.append((cur, b))
        return gaps

    def reserve(self, link: str, start: int, finish: int, tid: str):
        """预留链路时隙（贪心密度版本）"""
        self.slots_by_link[link].append((start, finish, tid))

    def try_pack_task(self, t: Task) -> Optional[Assignment]:
        """尝试打包单个任务（单链路+无分片，贪心核心）"""
        ln = t.home_link
        if ln not in self.links:
            return None

        # 遍历归属链路的可用段
        for seg in self.links[ln].iter_segments_in_window(t.arrival, t.deadline):
            gaps = self.get_free_gaps(ln, seg.start, seg.end)
            for gs, ge in gaps:
                # 计算任务所需时间
                dur = t.size_bits / seg.rate
                if gs + dur <= ge:
                    # 找到可用时隙，创建Chunk
                    start = float(gs)
                    finish = float(gs + dur)
                    rf_bits = t.size_bits if seg.link_type == "RF" else 0.0
                    self.reserve(ln, int(start), int(math.ceil(finish)), t.task_id)
                    return Assignment(
                        t.task_id,
                        [Chunk(ln, start, finish, t.size_bits, rf_bits, 0, 0, seg.mod_level, seg.link_type)],
                        t.size_bits,
                        t.value
                    )
        return None

    def run_greedy_density(self) -> Tuple[float, List[Assignment], List[float]]:
        """运行贪心密度调度（核心入口）"""
        print(f"[Greedy-Density] Start scheduling {len(self.tasks)} tasks...", flush=True)
        
        # 按价值密度降序排序任务
        task_order = sorted(
            self.tasks,
            key=lambda t: t.value / max(1.0, t.size_bits),  # 避免除零
            reverse=True
        )

        # 贪心调度
        for t in task_order:
            if t.task_id not in self.unscheduled:
                continue
            assignment = self.try_pack_task(t)
            if assignment:
                self.assignments.append(assignment)
                self.unscheduled.remove(t.task_id)

        # 更新最优解
        self.best_assignments = list(self.assignments)
        self.best_value = sum(a.value for a in self.assignments)

        print(f"[Greedy-Density] Done | Scheduled {len(self.best_assignments)} tasks | Total Value = {self.best_value:.3f}", flush=True)
        return self.best_value, self.best_assignments, [self.best_value]  # trace仅含最终值