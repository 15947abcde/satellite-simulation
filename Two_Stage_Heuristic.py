#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-Stage Heuristic Scheduler（单链路+无分片）
Stage 1: 贪心构造（小任务优先+早截止期）
Stage 2: 有限替换（移除低密度任务，插入高价值任务）
"""

import math
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
class SchedulerConfigTwoStage:
    def __init__(self):
        # 核心参数
        self.SEED = 42
        self.GB_IS_GIB = False
        self.RF_PENALTY_PER_GBIT = 0.0
        self.SERVICE_WINDOW_SEC = 36000
        # 两阶段算法参数
        self.STAGE2_ITERS = 120  # 第二阶段迭代次数
        self.DESTROY_RATE = 0.05  # 销毁率
        self.MIN_DESTROY = 3      # 最小销毁数量
        # 速率映射
        self.RATE_MAP_GBPS = {1: 1.0, 2: 10.0, 3: 20.0, 4: 30.0, 5: 40.0}
        self.RATE_MAP = {k: v * 1e9 for k, v in self.RATE_MAP_GBPS.items()}
        self.LINK_TYPE = {1: "RF", 2: "OPT", 3: "OPT", 4: "OPT", 5: "OPT"}
        # 算法名称
        self.ALGO_NAME = "TwoStage"


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
    """链路可见时间窗口（两阶段启发式版本）"""
    def __init__(self, name: str):
        self.name = name
        self.segments: List[Segment] = []

    def add_segment(self, start: int, end: int, mod: int, config: SchedulerConfigTwoStage):
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
    """任务数据结构（两阶段启发式版本，单链路归属）"""
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


# 无分片Chunk（两阶段启发式版本）
Chunk = namedtuple(
    "Chunk",
    ["link", "start", "finish", "bits", "rf_bits", "mod_switches", "link_switches", "mod_level", "link_type"]
)
Assignment = namedtuple("Assignment", ["task_id", "chunks", "total_bits", "value"])


# ================= 数据加载函数（两阶段启发式版本）=================
def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """解析文件名提取链路名和时间"""
    name = Path(filename).stem.strip()
    m = re.search(r'(.+?)[-_]?(\d+)$', name)
    if m:
        return m.group(1), int(m.group(2))
    return None, None


def load_tasks_from_dir(tasks_dir: str, config: SchedulerConfigTwoStage) -> List[Task]:
    """加载任务数据（两阶段启发式版本，鲁棒解析）"""
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


def load_link_ftw_csv(path: str, link_name: str, config: SchedulerConfigTwoStage) -> LinkFTW:
    """加载链路可见性数据（两阶段启发式版本，鲁棒解析）"""
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


def load_all_links(visibility_dir: str, config: SchedulerConfigTwoStage) -> Dict[str, LinkFTW]:
    """加载所有链路数据（两阶段启发式版本，单链路归属）"""
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


# ================= 核心调度器类（两阶段启发式）=================
class SchedulerTwoStage:
    def __init__(self, tasks: List[Task], links: Dict[str, LinkFTW], device, config: SchedulerConfigTwoStage, seed=42):
        """初始化（两阶段启发式版本，单链路归属约束）"""
        self.tasks = tasks
        self.links = links
        self.device = device
        self.config = config

        # 初始化随机种子
        import random
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
        self.best_assignments = []
        self.best_value = 0.0

    def get_free_gaps(self, link: str, a: int, b: int) -> List[Tuple[int, int]]:
        """获取链路空闲时隙（两阶段启发式版本）"""
        used = sorted(self.slots_by_link[link])
        gaps = []
        cur = a
        for s, e, _ in used:
            if s > cur:
                gaps.append((cur, min(s, b)))
            cur = max(cur, e)
            if cur >= b:
                break
        if cur < b:
            gaps.append((cur, b))
        return gaps

    def reserve(self, link: str, start: int, finish: int, tid: str):
        """预留链路时隙（两阶段启发式版本）"""
        self.slots_by_link[link].append((start, finish, tid))

    def try_pack(self, t: Task) -> Optional[Assignment]:
        """单链路打包任务（无分片，两阶段启发式核心）"""
        ln = t.home_link
        if ln not in self.links:
            return None

        for seg in self.links[ln].iter_segments_in_window(t.arrival, t.deadline):
            for gs, ge in self.get_free_gaps(ln, seg.start, seg.end):
                dur = t.size_bits / seg.rate
                if gs + dur <= ge:
                    start = float(gs)
                    finish = float(gs + dur)
                    rf_bits = t.size_bits if seg.link_type == "RF" else 0.0
                    self.reserve(ln, int(start), int(math.ceil(finish)), t.task_id)
                    return Assignment(
                        task_id=t.task_id,
                        chunks=[Chunk(
                            ln, start, finish, t.size_bits, rf_bits, 0, 0, seg.mod_level, seg.link_type
                        )],
                        total_bits=t.size_bits,
                        value=t.value
                    )
        return None

    def run_two_stage(self) -> Tuple[float, List[Assignment], List[float]]:
        """运行两阶段启发式算法（核心入口）"""
        print(f"[Two-Stage] Start Stage1 (Greedy) + Stage2 (Replace, Iters={self.config.STAGE2_ITERS})", flush=True)

        # ---------- Stage 1: 贪心构造（小任务优先+早截止期）----------
        order = sorted(self.tasks, key=lambda t: (t.size_bits, t.deadline, -t.value))
        for t in order:
            if t.task_id in self.unscheduled:
                a = self.try_pack(t)
                if a:
                    self.assignments.append(a)
                    self.unscheduled.remove(t.task_id)

        # 初始化最优解
        best = list(self.assignments)
        best_val = sum(a.value for a in best)
        trace = [best_val]

        # ---------- Stage 2: 有限替换（移除低密度，插入高价值）----------
        id2t = {t.task_id: t for t in self.tasks}
        for stage2_iter in range(self.config.STAGE2_ITERS):
            # 备份当前状态
            backup = list(self.assignments)
            backup_slots = {ln: list(v) for ln, v in self.slots_by_link.items()}
            backup_uns = set(self.unscheduled)

            # 销毁：移除低密度任务
            if self.assignments:
                scored = []
                for a in self.assignments:
                    t = id2t[a.task_id]
                    dens = t.value / max(1.0, t.size_bits)
                    scored.append((dens, a))
                scored.sort(key=lambda x: x[0])
                # 计算销毁数量
                k = max(self.config.MIN_DESTROY, int(self.config.DESTROY_RATE * len(scored)))
                # 移除低密度任务
                for _, a in scored[:k]:
                    self.assignments.remove(a)
                    self.unscheduled.add(a.task_id)
                    # 清理时隙
                    self.slots_by_link[a.chunks[0].link] = [
                        x for x in self.slots_by_link[a.chunks[0].link] if x[2] != a.task_id
                    ]

            # 修复：插入高价值任务
            uns = sorted([id2t[i] for i in self.unscheduled], key=lambda t: -t.value)
            for t in uns:
                a = self.try_pack(t)
                if a:
                    self.assignments.append(a)
                    self.unscheduled.remove(t.task_id)

            # 评估新解
            cur_val = sum(a.value for a in self.assignments)
            if cur_val > best_val:
                best_val = cur_val
                best = list(self.assignments)
            else:
                # 回滚到备份
                self.assignments = backup
                self.slots_by_link = defaultdict(list, backup_slots)
                self.unscheduled = backup_uns

            trace.append(best_val)

            # 打印Stage2进度
            if (stage2_iter + 1) % 20 == 0 or stage2_iter == 0 or (stage2_iter + 1) == self.config.STAGE2_ITERS:
                print(f"[Stage2 Iter {stage2_iter+1}/{self.config.STAGE2_ITERS}] Best Value = {best_val:.3f}", flush=True)

        # 提交最优解
        self.best_assignments = best
        self.best_value = best_val

        print(f"[Two-Stage] Done | Best Value = {self.best_value:.3f} | Scheduled = {len(self.best_assignments)}", flush=True)
        return self.best_value, self.best_assignments, trace