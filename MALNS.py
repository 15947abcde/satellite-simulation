#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALNS+TS scheduler (piecewise-rate) with cross-window & cross-link support.
核心算法模块 - 无输入输出逻辑，仅包含调度算法相关实现
"""

import math
import random
from collections import defaultdict, deque, namedtuple
from typing import List, Dict, Optional, Tuple

import numpy as np
import torch


# ================= 统计辅助函数 =================
def mean_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    返回: (mean, lower, upper)，t 分布 95% CI
    """
    n = len(data)
    if n == 0:
        return (0.0, 0.0, 0.0)
    mean = sum(data) / n
    if n == 1:
        return (mean, mean, mean)

    var = sum((x - mean) ** 2 for x in data) / (n - 1)
    std = math.sqrt(var)

    # n=10 时 t(0.975,9) ≈ 2.262
    t_table = {
        9: 2.262, 10: 2.228, 11: 2.201
    }
    t_val = t_table.get(n - 1, 1.96)

    margin = t_val * std / math.sqrt(n)
    return (mean, mean - margin, mean + margin)


# ================= 算法参数配置（可通过外部修改） =================
class SchedulerConfig:
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
        self.RATE_MAP_GBPS = {1: 1.0, 2: 10.0, 3: 20.0, 4: 30.0, 5: 40.0}
        self.RATE_MAP = {k: v * 1e9 for k, v in self.RATE_MAP_GBPS.items()}
        self.LINK_TYPE = {1: "RF", 2: "BPSK", 3: "QPSK", 4: "8QAM", 5: "16QAM"}
        self.MOD_SWITCH_TIME = 0.5
        self.LINK_SWITCH_TIME = 1.0


# ================= 工具函数 =================
def device_autoselect() -> torch.device:
    """自动选择计算设备（CUDA/CPU）"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_bits(size_gb: float, gb_is_gib: bool) -> float:
    """将GB转换为比特数"""
    if isinstance(size_gb, str):
        size_gb = float(size_gb)
    if gb_is_gib:
        return size_gb * (1024 ** 3) * 8.0
    return size_gb * (1000 ** 3) * 8.0


# ================= 数据结构 =================
Segment = namedtuple("Segment", ["start", "end", "rate", "mod_level", "link_type"])


class LinkFTW:
    def __init__(self, name: str):
        self.name = name
        self.segments: List[Segment] = []

    def add_segment(self, start: int, end: int, mod: int):
        if end > start:
            self.segments.append(Segment(start, end, SchedulerConfig().RATE_MAP[int(mod)], int(mod),
                                         SchedulerConfig().LINK_TYPE[int(mod)]))

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


class Task:
    __slots__ = ("task_id", "arrival", "deadline", "size_bits", "priority", "value", "home_link", "gen_time")

    def __init__(self, task_id, arrival, deadline, size_bits, priority, value, home_link, gen_time):
        self.task_id = str(task_id)
        self.arrival = int(arrival)
        self.deadline = int(deadline)
        self.size_bits = float(size_bits)
        self.priority = int(priority)
        self.value = float(value)
        self.home_link = str(home_link)
        self.gen_time = int(gen_time)


# 去分片：Chunk 保留字段（mod/link switch计数保留但恒为0）
Chunk = namedtuple(
    "Chunk",
    ["link", "start", "finish", "bits", "rf_bits", "mod_switches", "link_switches", "mod_level", "link_type"]
)

# 去分片：Assignment 只包含一个 chunk
Assignment = namedtuple("Assignment", ["task_id", "chunk", "total_bits", "value"])


# ================= 核心调度器类 =================
class Scheduler:
    def __init__(self, tasks: List[Task], links: Dict[str, LinkFTW], device, config: SchedulerConfig, seed: int = 42):
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

        # 调度结果
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
        """获取链路在指定时间段内的空闲时隙"""
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

    def reserve(self, link: str, start: int, finish: int, task_id: str):
        """预留链路时隙"""
        self.slots_by_link[link].append((start, finish, task_id))

    def link_opportunity(self, t: Task, link: str) -> float:
        """计算任务在指定链路上的传输机会分数"""
        score = 0.0
        for seg in self.links[link].iter_segments_in_window(t.arrival, t.deadline):
            w = 1.0 if seg.link_type == "OPT" else 0.8
            score += w * seg.rate * (seg.end - seg.start)
        return score

    def candidate_heuristic_cost(self, t: Task, chunk: Chunk) -> float:
        """计算候选chunk的启发式成本"""
        early_end = t.arrival + self.config.EARLY_WINDOW_SEC
        dur = max(1e-9, chunk.finish - chunk.start)
        over = max(0.0, min(chunk.finish, early_end) - max(chunk.start, t.arrival))
        early_bits = (over / dur) * chunk.bits if over > 0 else 0.0
        return -t.value + self.config.EARLY_PENALTY_COEF * early_bits

    def try_pack_task_once(self, t: Task) -> Optional[Chunk]:
        """尝试将任务一次性打包到单个连续时隙"""
        need_bits = t.size_bits
        cand_links = [t.home_link] + [ln for ln in self.links.keys() if ln != t.home_link]
        best_chunk = None
        best_cost = float("inf")

        for ln in cand_links:
            for seg in self.links[ln].iter_segments_in_window(t.arrival, t.deadline):
                # 速率抖动
                jitter = 1.0 + random.uniform(-self.config.RANDOM_RATE_JITTER, self.config.RANDOM_RATE_JITTER)
                rate = seg.rate * jitter

                # 该seg内的空闲gap
                gaps = self.get_free_gaps(ln, seg.start, seg.end)
                for (gs, ge) in gaps:
                    if ge <= gs:
                        continue
                    # 连续窗口容量
                    cap_bits = (ge - gs) * rate
                    if cap_bits < need_bits:
                        continue

                    # 任务所需时长
                    dur = need_bits / rate
                    start = gs
                    finish = start + dur
                    if finish > ge:
                        continue

                    rf_bits = need_bits if seg.link_type == "RF" else 0.0
                    chunk = Chunk(
                        ln, start, finish, need_bits, rf_bits,
                        0, 0,  # 去分片后不再发生边界切换
                        seg.mod_level,
                        seg.link_type
                    )
                    cost = self.candidate_heuristic_cost(t, chunk)
                    if cost < best_cost:
                        best_cost = cost
                        best_chunk = chunk

        if best_chunk is None:
            return None

        # 预留资源
        self.reserve(best_chunk.link, int(best_chunk.start), int(math.ceil(best_chunk.finish)), t.task_id)
        return best_chunk

    def place_task_multicand(self, t: Task, n_cand: int = 3) -> Optional[Assignment]:
        """多候选位置尝试放置任务"""
        best = None
        best_cost = None
        best_reserve = None
        snapshot = {ln: list(v) for ln, v in self.slots_by_link.items()}

        for _ in range(n_cand):
            self.slots_by_link = defaultdict(list, {ln: list(v) for ln, v in snapshot.items()})
            chunk = self.try_pack_task_once(t)
            if chunk is None:
                continue
            cost = self.candidate_heuristic_cost(t, chunk)
            if (best is None) or (cost < best_cost):
                best = chunk
                best_cost = cost
                best_reserve = {ln: list(v) for ln, v in self.slots_by_link.items()}

        if best is None:
            self.slots_by_link = defaultdict(list, snapshot)
            return None

        self.slots_by_link = defaultdict(list, best_reserve)
        return Assignment(t.task_id, best, float(best.bits), t.value)

    def objective(self, assigns):
        if not assigns:
            return -1e18

        total_value = 0.0
        total_time = 0.0

        for a in assigns:
            rf_bits = a.chunk.rf_bits
            total_value += a.value - self.rf_penalty * (rf_bits / 1e9)

            # 单 chunk 情况
            total_time += (a.chunk.finish - a.chunk.start)

        avg_time = total_time / len(assigns)

        λ = 1.0   # 你可以调节这个参数
        return total_value - λ * avg_time


    def build_initial(self):
        proxy = 20e9
        order = sorted(self.tasks, key=lambda t: (t.value / max(1.0, t.size_bits / proxy), -t.arrival))
        for t in order:
            if t.task_id not in self.unscheduled:
                continue
            a = self.place_task_multicand(t, n_cand=3)
            if a:
                self.assignments.append(a)
                self.unscheduled.discard(t.task_id)  # ✅ discard
        self.cur_assignments = list(self.assignments)
        self.cur_value = self.objective(self.assignments)
        self.best_assignments = list(self.assignments)
        self.best_value = self.cur_value


    def _remove_assignments(self, removed: List[Assignment]):
        """移除指定的调度分配"""
        for a in removed:
            self.unscheduled.add(a.task_id)
            for ln in list(self.slots_by_link.keys()):
                self.slots_by_link[ln] = [(s, e, tid) for (s, e, tid) in self.slots_by_link[ln] if tid != a.task_id]
        self.assignments = [x for x in self.assignments if x.task_id not in {a.task_id for a in removed}]

    def remove_cluster(self, k: int) -> List[Assignment]:
        """聚类移除策略"""
        if not self.assignments:
            return []
        spans = []
        for a in self.assignments:
            spans.append((a.chunk.start, a.chunk.finish, a))
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
        """移除RF传输最多的任务"""
        if not self.assignments:
            return []
        order = sorted(self.assignments, key=lambda a: a.chunk.rf_bits, reverse=True)
        removed = order[:min(k, len(order))]
        self._remove_assignments(removed)
        return removed

    def remove_oldest(self, k: int) -> List[Assignment]:
        """移除最早调度的任务"""
        if not self.assignments:
            return []
        order = sorted(self.assignments, key=lambda a: a.chunk.start)
        removed = order[:min(k, len(order))]
        self._remove_assignments(removed)
        return removed

    def repair_profit(self, max_trials: int = 256) -> int:
        pending = [t for t in self.tasks if t.task_id in self.unscheduled and t.task_id not in self.tabu]
        pending.sort(key=lambda t: t.value, reverse=True)
        cnt = 0
        for t in pending:
            if t.task_id not in self.unscheduled:   # ✅ 二次检查
                continue
            a = self.place_task_multicand(t, n_cand=4)
            if a:
                self.assignments.append(a)
                self.unscheduled.discard(t.task_id)  # ✅ discard
                cnt += 1
                if cnt >= max_trials:
                    break
        return cnt

    def repair_opportunity(self, max_trials: int = 256) -> int:
        pending = [t for t in self.tasks if t.task_id in self.unscheduled and t.task_id not in self.tabu]
        pending.sort(key=lambda t: -max(self.link_opportunity(t, ln) for ln in self.links.keys()))
        cnt = 0
        for t in pending:
            if t.task_id not in self.unscheduled:   # ✅ 二次检查
                continue
            a = self.place_task_multicand(t, n_cand=3)
            if a:
                self.assignments.append(a)
                self.unscheduled.discard(t.task_id)  # ✅ discard
                cnt += 1
                if cnt >= max_trials:
                    break
        return cnt


    def repair_random(self, max_trials: int = 256) -> int:
        pending = [t for t in self.tasks if t.task_id in self.unscheduled and t.task_id not in self.tabu]
        random.shuffle(pending)
        cnt = 0
        for t in pending:
            if t.task_id not in self.unscheduled:   # ✅ 二次检查
                continue
            a = self.place_task_multicand(t, n_cand=2)
            if a:
                self.assignments.append(a)
                self.unscheduled.discard(t.task_id)  # ✅ discard
                cnt += 1
                if cnt >= max_trials:
                    break
        return cnt


    def run_alns_ts(self, print_every: int = 20) -> Tuple[float, List[Assignment], List[float]]:
        """
        运行ALNS+TS算法
        返回: (最优目标值, 最优分配列表, 迭代轨迹)
        """
        self.build_initial()
        temp_start = 10.0
        temp_end = 0.05
        k_max_ratio = (0.30, 0.10)
        temps = np.linspace(temp_start, temp_end, self.config.ITERATIONS)
        trace = [self.best_value]

        destroy_ops = [self.remove_cluster, self.remove_worst_rf, self.remove_oldest]
        repair_ops = [self.repair_profit, self.repair_opportunity, self.repair_random]

        print(f"[ALNS] Start iterations={self.config.ITERATIONS}  init_cur={self.cur_value:.3f}  "
              f"init_best={self.best_value:.3f}", flush=True)

        for it in range(self.config.ITERATIONS):
            # 备份当前状态
            cur_backup_assign = list(self.assignments)
            cur_backup_slots = {ln: list(v) for ln, v in self.slots_by_link.items()}
            cur_backup_uns = set(self.unscheduled)

            # 选择销毁和修复策略
            rem = random.choice(destroy_ops)
            rep = random.choices(repair_ops, weights=[2.0, 1.5, 1.0])[0]
            ratio = k_max_ratio[0] - (k_max_ratio[0] - k_max_ratio[1]) * (it / max(1, self.config.ITERATIONS - 1))
            k = max(5, int(ratio * max(1, len(self.assignments))))

            # 执行销毁和修复
            removed = rem(k)
            rep(max_trials=256)

            # 计算新解的目标值
            new_val = self.objective(self.assignments)
            delta = new_val - self.cur_value
            T = temps[it]
            accept = (delta >= 0) or (random.random() < math.exp(delta / max(T, 1e-9)))

            # 接受/拒绝新解
            if accept:
                self.cur_value = new_val
                if self.cur_value > self.best_value:
                    self.best_value = self.cur_value
                    self.best_assignments = list(self.assignments)
                for a in removed:
                    if random.random() < 0.3:
                        self.tabu.append(a.task_id)
            else:
                self.assignments = cur_backup_assign
                self.slots_by_link = defaultdict(list, cur_backup_slots)
                self.unscheduled = cur_backup_uns

            # 温度重加热
            if self.config.REHEAT_PERIOD and (it + 1) % self.config.REHEAT_PERIOD == 0:
                temps = temps * 1.3
                temps = np.clip(temps, temp_end, temp_start)

            trace.append(self.best_value)

            # 打印进度
            if (it + 1) % print_every == 0 or it == 0 or (it + 1) == self.config.ITERATIONS:
                print(f"[Iter {it + 1}/{self.config.ITERATIONS}] cur={self.cur_value:.3f}  best={self.best_value:.3f}  "
                      f"d={rem.__name__}  r={rep.__name__}", flush=True)

        # 从最优解重建资源预留
        self.assignments = list(self.best_assignments)
        self.slots_by_link = defaultdict(list)
        for a in self.assignments:
            c = a.chunk
            self.reserve(c.link, int(c.start), int(math.ceil(c.finish)), a.task_id)

        print(f"[ALNS] Done  best={self.best_value:.3f}  scheduled={len(self.best_assignments)}", flush=True)
        return self.best_value, self.best_assignments, trace


# ================= 数据加载辅助函数 =================
def parse_filename(filename: str) -> Tuple[Optional[str], Optional[int]]:
    """解析文件名提取链路名和时间"""
    from pathlib import Path
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
    import re
    m = re.search(r'(\d+)\s*$', name)
    if m:
        return name[:m.start()].rstrip('-_ '), int(m.group(1))
    return None, None


def load_tasks_from_dir(tasks_dir: str, gb_is_gib: bool) -> List[Task]:
    """从目录加载任务数据"""
    import glob
    import os
    import pandas as pd
    tasks = []
    files = []
    for ext in ("*.csv", "*.CSV"):
        files += glob.glob(os.path.join(tasks_dir, ext))
    if not files:
        raise FileNotFoundError(f"No task CSV files in {tasks_dir}")

    for f in files:
        fname = os.path.basename(f)
        link, arr = parse_filename(fname)
        if link is None or arr is None:
            print(f"[SKIP:name] {fname}")
            continue

        df = pd.read_csv(f, header=None, names=["value", "size_gb"], sep=None, engine="python", encoding="utf-8-sig")

        def _num(x):
            try:
                float(x)
                return True
            except:
                return False

        if len(df) > 0 and (not _num(df.iloc[0, 0]) or not _num(df.iloc[0, 1])):
            df = df.iloc[1:].copy()

        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["size_gb"] = pd.to_numeric(df["size_gb"], errors="coerce")
        df = df.dropna(subset=["value", "size_gb"])
        if df.empty:
            print(f"[SKIP:empty] {fname}")
            continue

        df = df.sort_values("value", ascending=False).reset_index(drop=True)
        n = len(df)
        hi = int(n * 0.2)
        lo = int(n * 0.8)

        for i, row in df.iterrows():
            pr = 10 if i < hi else (5 if i < lo else 1)
            size_bits = to_bits(row["size_gb"], gb_is_gib)
            deadline = arr + SchedulerConfig().SERVICE_WINDOW_SEC
            tid = f"{fname}_task{i}"
            tasks.append(Task(tid, arr, deadline, size_bits, pr, float(row["value"]), link, gen_time=arr))

    if not tasks:
        raise ValueError("No valid tasks found (after parsing/cleaning).")
    return tasks


def load_link_ftw_csv(path: str, link_name: str) -> LinkFTW:
    """从CSV加载链路可见性数据"""
    import pandas as pd
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
        except:
            continue
        if prev_m is None:
            prev_m = m
            run_s = t
            last = t
        else:
            if m == prev_m and t == last + 1:
                last = t
            else:
                ftw.add_segment(run_s, last + 1, prev_m)
                prev_m = m
                run_s = t
                last = t

    if prev_m is not None:
        ftw.add_segment(run_s, last + 1, prev_m)

    ftw.finalize()
    return ftw


def load_all_links(visibility_dir: str) -> Dict[str, LinkFTW]:
    """加载所有链路的可见性数据"""
    import glob
    import os
    from pathlib import Path
    link_map = {}
    csvs = []
    for ext in ("*.csv", "*.CSV"):
        csvs += glob.glob(os.path.join(visibility_dir, ext))
    if not csvs:
        raise FileNotFoundError(f"No visibility CSV files in {visibility_dir}")

    for p in csvs:
        link_name = Path(p).stem
        link_map[link_name] = load_link_ftw_csv(p, link_name)
    return link_map