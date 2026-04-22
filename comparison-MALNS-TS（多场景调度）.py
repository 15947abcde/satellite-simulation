#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALNS+TS scheduler (piecewise-rate) with cross-window & cross-link (ISL/virtual FTW) support.
- Arrival time = ready time (NOT forced start).
- Each task has 4h service window (deadline = arrival + 4h).
- Tasks may be split across multiple visibility windows and multiple links.
- Per-link single-channel constraint (no overlap).
- Switching penalties at modulation/link change boundaries.
- Stronger destroy/repair; SA w.r.t current + reheating; randomized multi-candidate repair.
"""

import os, sys, math, random, time, glob, re
from collections import defaultdict, deque, namedtuple
from pathlib import Path
from typing import List, Dict, Optional, Tuple
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================= USER CONFIG =================
# ================= MULTI-SCENARIO CONFIG =================
TASKS_DIR_GLOB = os.path.join(BASE_DIR, "业务部署-30卫星-*业务")
VIS_DIR_PREFIX = os.path.join(BASE_DIR, "卫星部署-30卫星-")

OUTPUT_PREFIX = "M-fangzhen-30卫星-"
# ========================================================

ITERATIONS  = 10            # ALNS迭代次数
SEED        = 42
GB_IS_GIB   = False           # 任务表中 size_gb 是否为GiB
RF_PENALTY_PER_GBIT = 0.0     # RF使用惩罚（每Gbit）
SERVICE_WINDOW_SEC  = 14400  # 每个业务可持续时间=1小时
# 修复启发式：早窗轻惩罚（鼓励把低价值任务后移），取值很小即可
EARLY_WINDOW_SEC    = 0    # “早窗”定义为到达后30分钟
EARLY_PENALTY_COEF  = 1e-11   # 每比特·早窗占用的启发式扣分（仅用于修复候选评分）
RANDOM_RATE_JITTER  = 0.02    # 段排序扰动：±2% 速率抖动
REHEAT_PERIOD       = 100     # 每多少步回温一次
# =================================================

RATE_MAP_GBPS = {1:1.0, 2:10.0, 3:20.0, 4:30.0, 5:40.0}
RATE_MAP = {k: v*1e9 for k,v in RATE_MAP_GBPS.items()}
LINK_TYPE = {1:"RF", 2:"OPT", 3:"OPT", 4:"OPT", 5:"OPT"}

MOD_SWITCH_TIME = 0.5     # s
LINK_SWITCH_TIME = 1.0    # s

def device_autoselect():
    # torch removed; scheduler does not actually use device for computation
    return "cpu"

def to_bits(size_gb: float, gb_is_gib: bool) -> float:
    if isinstance(size_gb, str):
        size_gb = float(size_gb)
    if gb_is_gib:
        return size_gb*(1024**3)*8.0
    return size_gb*(1000**3)*8.0

# ---------------- Data structures ----------------
Segment = namedtuple("Segment", ["start","end","rate","mod_level","link_type"])

class LinkFTW:
    def __init__(self, name:str):
        self.name = name
        self.segments: List[Segment] = []

    def add_segment(self, start:int, end:int, mod:int):
        if end>start:
            self.segments.append(Segment(start, end, RATE_MAP[int(mod)], int(mod), LINK_TYPE[int(mod)]))

    def finalize(self):
        if not self.segments: return
        self.segments.sort(key=lambda s:(s.start, s.mod_level))
        merged=[]
        cur=self.segments[0]
        for seg in self.segments[1:]:
            if seg.mod_level==cur.mod_level and seg.start<=cur.end:
                cur = Segment(cur.start, max(cur.end, seg.end), cur.rate, cur.mod_level, cur.link_type)
            else:
                merged.append(cur); cur=seg
        merged.append(cur)
        self.segments=merged

    def iter_segments_in_window(self, t0:int, t1:int):
        for s in self.segments:
            if s.end<=t0: continue
            if s.start>=t1: break
            yield Segment(max(s.start,t0), min(s.end,t1), s.rate, s.mod_level, s.link_type)

class Task:
    # ✅ 新增 gen_time（生成时间：来自任务表文件名最后一段）
    __slots__=("task_id","arrival","deadline","size_bits","priority","value","home_link","gen_time")
    def __init__(self, task_id, arrival, deadline, size_bits, priority, value, home_link, gen_time):
        self.task_id=str(task_id)
        self.arrival=int(arrival)
        self.deadline=int(deadline)
        self.size_bits=float(size_bits)
        self.priority=int(priority)
        self.value=float(value)
        self.home_link=str(home_link)
        self.gen_time=int(gen_time)

# 每个任务可被分拆为多段（多链路/多窗口）
# ✅ 新增：mod_level, link_type 以便导出“调制格式”和“RF/FSO”
Chunk = namedtuple(
    "Chunk",
    ["link","start","finish","bits","rf_bits","mod_switches","link_switches","mod_level","link_type"]
)
Assignment = namedtuple("Assignment", ["task_id","chunks","total_bits","value"])

# ---------------- Loading ----------------
def parse_filename(filename:str)->Tuple[Optional[str],Optional[int]]:
    name = Path(filename).stem.strip()
    if '-' in name:
        prefix, tail = name.rsplit('-',1)
        tail=tail.strip()
        if tail.isdigit():
            return prefix, int(tail)
    if '_' in name:
        prefix, tail = name.rsplit('_',1)
        tail=tail.strip()
        if tail.isdigit():
            return prefix, int(tail)
    m=re.search(r'(\d+)\s*$', name)
    if m:
        return name[:m.start()].rstrip('-_ '), int(m.group(1))
    return None, None

def load_tasks_from_dir(tasks_dir:str, gb_is_gib:bool)->List[Task]:
    tasks=[]
    files=[]
    for ext in ("*.csv","*.CSV"):
        files += glob.glob(os.path.join(tasks_dir, ext))
    files = sorted(set(files))
    if not files:
        raise FileNotFoundError(f"No task CSV files in {tasks_dir}")
    for f in files:
        fname=Path(f).name
        link, arr = parse_filename(fname)
        if link is None or arr is None:
            print(f"[SKIP:name] {fname}")
            continue

        df = pd.read_csv(f, header=None, names=["value","size_gb"], sep=None, engine="python", encoding="utf-8-sig")
        def _num(x):
            try: float(x); return True
            except: return False
        if len(df)>0 and (not _num(df.iloc[0,0]) or not _num(df.iloc[0,1])):
            df=df.iloc[1:].copy()
        df["value"]=pd.to_numeric(df["value"], errors="coerce")
        df["size_gb"]=pd.to_numeric(df["size_gb"], errors="coerce")
        df=df.dropna(subset=["value","size_gb"])
        if df.empty:
            print(f"[SKIP:empty] {fname}")
            continue

        df=df.sort_values("value", ascending=False).reset_index(drop=True)
        n=len(df); hi=int(n*0.2); lo=int(n*0.8)
        for i,row in df.iterrows():
            pr = 10 if i<hi else (5 if i<lo else 1)
            size_bits = to_bits(row["size_gb"], gb_is_gib)
            deadline = arr + SERVICE_WINDOW_SEC
            tid = f"{fname}_task{i}"
            # ✅ gen_time = arr（来自任务文件名最后一段），arrival 初始也等于 arr
            tasks.append(Task(tid, arr, deadline, size_bits, pr, float(row["value"]), link, gen_time=arr))

    if not tasks:
        raise ValueError("No valid tasks found (after parsing/cleaning).")
    return tasks

def load_link_ftw_csv(path:str, link_name:str)->LinkFTW:
    try:
        df=pd.read_csv(path)
        if not {"valid_time_seconds","modulation_level"}.issubset(df.columns):
            df=pd.read_csv(path, header=None, names=["valid_time_seconds","modulation_level"])
    except:
        df=pd.read_csv(path, header=None, names=["valid_time_seconds","modulation_level"])
    df=df.sort_values("valid_time_seconds")
    ftw=LinkFTW(link_name)
    prev_m=None; run_s=None; last=None
    for _,r in df.iterrows():
        try:
            t=int(r["valid_time_seconds"]); m=int(r["modulation_level"])
        except: continue
        if prev_m is None:
            prev_m=m; run_s=t; last=t
        else:
            if m==prev_m and t==last+1:
                last=t
            else:
                ftw.add_segment(run_s, last+1, prev_m)
                prev_m=m; run_s=t; last=t
    if prev_m is not None:
        ftw.add_segment(run_s, last+1, prev_m)
    ftw.finalize()
    return ftw

def load_all_links(visibility_dir:str)->Dict[str,LinkFTW]:
    link_map={}
    csvs=[]
    for ext in ("*.csv","*.CSV"):
        csvs += glob.glob(os.path.join(visibility_dir, ext))
    csvs = sorted(set(csvs))
    if not csvs:
        raise FileNotFoundError(f"No visibility CSV files in {visibility_dir}")
    for p in csvs:
        link_name = Path(p).stem
        link_map[link_name]=load_link_ftw_csv(p, link_name)
    return link_map

# ---------------- Scheduler ----------------
class Scheduler:
    def __init__(self, tasks:List[Task], links:Dict[str,LinkFTW], device, rf_penalty=0.0, seed=42):
        self.tasks=tasks
        self.links=links                    # 可使用“可见性目录中所有链路”
        self.device=device                  # kept for compatibility; not used in computations
        self.rf_penalty=rf_penalty
        random.seed(seed); np.random.seed(seed)
        # state
        self.assignments: List[Assignment] = []
        self.unscheduled = set(t.task_id for t in tasks)
        self.slots_by_link: Dict[str, List[Tuple[int,int,str]]] = defaultdict(list)  # link -> [(start,end,task_id)]
        # best/current
        self.best_assignments=None; self.best_value=-1e18
        self.cur_assignments=None;  self.cur_value=-1e18
        # tabu (用于避免立即回插同一任务，轻度)
        self.tabu=deque(maxlen=64)

    # ----- per-link gap utilities -----
    def get_free_gaps(self, link:str, a:int, b:int)->List[Tuple[int,int]]:
        """Return non-overlap gaps within [a,b) on a link."""
        used=sorted([(s,e) for (s,e,_) in self.slots_by_link[link]], key=lambda x:x[0])
        gaps=[]; cur=a
        for s,e in used:
            if e<=cur: continue
            if s>cur:
                gaps.append((cur, min(s,b)))
                cur = min(s,b)
            cur = max(cur, e)
            if cur>=b: break
        if cur<b: gaps.append((cur,b))
        return gaps

    def reserve(self, link:str, start:int, finish:int, task_id:str):
        self.slots_by_link[link].append((start, finish, task_id))

    # ---------- scoring helpers ----------
    def link_opportunity(self, t:Task, link:str)->float:
        score=0.0
        for seg in self.links[link].iter_segments_in_window(t.arrival, t.deadline):
            w = 1.0 if seg.link_type=="OPT" else 0.8
            score += w*seg.rate*(seg.end-seg.start)
        return score

    def candidate_heuristic_cost(self, t:Task, chunks:List[Chunk])->float:
        """用于修复阶段选择候选：价值越大越好（负号）；早窗占用越多越差（加罚）"""
        # RF 真实惩罚进入目标函数，这里只加早窗启发式
        early_end = t.arrival + EARLY_WINDOW_SEC
        early_bits = 0.0
        for c in chunks:
            dur = max(1e-9, c.finish - c.start)
            # 该块前 EARLY 窗口占用
            over = max(0.0, min(c.finish, early_end) - max(c.start, t.arrival))
            if over>0:
                early_bits += (over/dur) * c.bits
        # 越小越好：-value + alpha*early_bits
        return -t.value + EARLY_PENALTY_COEF*early_bits

    # ---------- packing core (randomized multi-candidate) ----------
    def try_pack_task_once(self, t:Task, rate_jitter:float=0.0)->Optional[List[Chunk]]:
        remaining = t.size_bits
        last_mod=None; last_link=None
        chunks=[]
        # 候选链路（所有链路，home_link 最优先）
        cand_links = [t.home_link] + [ln for ln in self.links.keys() if ln!=t.home_link]
        # 收集(ln,seg)并引入速率微扰，排序：早+高速(扰动后)+光优先
        seg_list=[]
        for ln in cand_links:
            for seg in self.links[ln].iter_segments_in_window(t.arrival, t.deadline):
                jitter = 1.0 + random.uniform(-rate_jitter, rate_jitter)
                seg_list.append((ln, seg, seg.rate*jitter))
        seg_list.sort(key=lambda x: (x[1].start, -x[2], x[1].link_type!="OPT"))

        for ln, seg, _ in seg_list:
            if remaining<=0: break
            gaps = self.get_free_gaps(ln, seg.start, seg.end)
            for (gs, ge) in gaps:
                if remaining<=0: break
                cur_start = gs
                penalty = 0.0
                if last_mod is not None and seg.mod_level!=last_mod: penalty += MOD_SWITCH_TIME
                if last_link is not None and ln!=last_link:       penalty += LINK_SWITCH_TIME
                cur_start = min(max(cur_start + penalty, gs), ge)
                if cur_start>=ge: continue
                avail_t = ge - cur_start
                capacity = avail_t * seg.rate
                if capacity<=0: continue
                use_bits = min(remaining, capacity)
                dur = use_bits / seg.rate
                cur_finish = cur_start + dur
                rf_bits = use_bits if seg.link_type=="RF" else 0.0

                # ✅ 这里把 mod_level / link_type 写入 Chunk，供最终导出调制格式与RF/FSO
                chunks.append(Chunk(
                    ln, cur_start, cur_finish, use_bits, rf_bits,
                    1 if (last_mod is not None and seg.mod_level!=last_mod) else 0,
                    1 if (last_link is not None and ln!=last_link) else 0,
                    seg.mod_level,
                    seg.link_type
                ))
                # 预留
                self.reserve(ln, int(cur_start), int(math.ceil(cur_finish)), t.task_id)
                remaining -= use_bits
                last_mod = seg.mod_level
                last_link = ln
                if remaining<=0: break

        if remaining>0:
            # 回滚预留
            for ln in list(self.slots_by_link.keys()):
                self.slots_by_link[ln] = [(s,e,tid) for (s,e,tid) in self.slots_by_link[ln] if tid!=t.task_id]
            return None
        return chunks

    def place_task_multicand(self, t:Task, n_cand:int=3)->Optional[Assignment]:
        """生成多候选（随机扰动排序），取启发式成本最小者。"""
        best=None; best_cost=None; best_reserve=None
        snapshot = {ln:list(v) for ln,v in self.slots_by_link.items()}

        for _ in range(n_cand):
            self.slots_by_link = defaultdict(list, {ln:list(v) for ln,v in snapshot.items()})
            chunks = self.try_pack_task_once(t, rate_jitter=RANDOM_RATE_JITTER)
            if chunks is None:
                continue
            cost = self.candidate_heuristic_cost(t, chunks)
            if (best is None) or (cost<best_cost):
                best=chunks; best_cost=cost
                best_reserve = {ln:list(v) for ln,v in self.slots_by_link.items()}

        if best is None:
            self.slots_by_link = defaultdict(list, snapshot)
            return None

        self.slots_by_link = defaultdict(list, best_reserve)
        return Assignment(t.task_id, best, sum(c.bits for c in best), t.value)

    # ---------- objective ----------
    def objective(self, assigns:List[Assignment])->float:
        v=0.0
        for a in assigns:
            rf_bits = sum(c.rf_bits for c in a.chunks)
            v += a.value - self.rf_penalty*(rf_bits/1e9)  # RF惩罚单位：每Gbit
        return v

    # ---------- build initial ----------
    def build_initial(self):
        proxy=20e9
        order = sorted(self.tasks, key=lambda t:(t.value/max(1.0, t.size_bits/proxy), -t.arrival))
        for t in order:
            if t.task_id in self.unscheduled:
                a = self.place_task_multicand(t, n_cand=3)
                if a:
                    self.assignments.append(a)
                    self.unscheduled.remove(t.task_id)
        self.cur_assignments = list(self.assignments)
        self.cur_value = self.objective(self.assignments)
        self.best_assignments = list(self.assignments)
        self.best_value = self.cur_value

    # ---------- destroy operators ----------
    def _remove_assignments(self, removed:List[Assignment]):
        for a in removed:
            self.unscheduled.add(a.task_id)
            for ln in list(self.slots_by_link.keys()):
                self.slots_by_link[ln]=[(s,e,tid) for (s,e,tid) in self.slots_by_link[ln] if tid!=a.task_id]
        self.assignments=[x for x in self.assignments if x.task_id not in {a.task_id for a in removed}]

    def remove_cluster(self, k:int):
        if not self.assignments: return []
        spans=[]
        for a in self.assignments:
            st=min(c.start for c in a.chunks); ft=max(c.finish for c in a.chunks)
            spans.append((st,ft,a))
        spans.sort(key=lambda x:x[0])
        center=random.choice(spans)[0]
        spans.sort(key=lambda x: abs(x[0]-center))
        removed=[]
        for _,_,a in spans:
            removed.append(a)
            if len(removed)>=min(k,len(self.assignments)): break
        self._remove_assignments(removed)
        return removed

    def remove_worst_rf(self, k:int):
        if not self.assignments: return []
        order=sorted(self.assignments, key=lambda a: sum(c.rf_bits for c in a.chunks), reverse=True)
        removed=order[:min(k,len(order))]
        self._remove_assignments(removed)
        return removed

    def remove_oldest(self, k:int):
        if not self.assignments: return []
        order=sorted(self.assignments, key=lambda a: min(c.start for c in a.chunks))
        removed=order[:min(k,len(order))]
        self._remove_assignments(removed)
        return removed

    # ---------- repair operators ----------
    def repair_profit(self, max_trials:int=256):
        pending=[t for t in self.tasks if t.task_id in self.unscheduled and t.task_id not in self.tabu]
        pending.sort(key=lambda t:t.value, reverse=True)
        cnt=0
        for t in pending:
            a=self.place_task_multicand(t, n_cand=4)
            if a:
                self.assignments.append(a); self.unscheduled.remove(t.task_id); cnt+=1
                if cnt>=max_trials: break
        return cnt

    def repair_opportunity(self, max_trials:int=256):
        pending=[t for t in self.tasks if t.task_id in self.unscheduled and t.task_id not in self.tabu]
        pending.sort(key=lambda t: -max(self.link_opportunity(t, ln) for ln in self.links.keys()))
        cnt=0
        for t in pending:
            a=self.place_task_multicand(t, n_cand=3)
            if a:
                self.assignments.append(a); self.unscheduled.remove(t.task_id); cnt+=1
                if cnt>=max_trials: break
        return cnt

    def repair_random(self, max_trials:int=256):
        pending=[t for t in self.tasks if t.task_id in self.unscheduled and t.task_id not in self.tabu]
        random.shuffle(pending)
        cnt=0
        for t in pending:
            a=self.place_task_multicand(t, n_cand=2)
            if a:
                self.assignments.append(a); self.unscheduled.remove(t.task_id); cnt+=1
                if cnt>=max_trials: break
        return cnt

    # ---------- ALNS main (relative-to-current SA + reheating) ----------
    def alns_ts(self, iterations:int=200, temp_start:float=10.0, temp_end:float=0.05, k_max_ratio=(0.30,0.10),
                print_every:int=20):
        """带迭代进度打印的 ALNS 主循环。仅输出，不改变任何算法逻辑。"""
        self.build_initial()
        temps = np.linspace(temp_start, temp_end, iterations)
        trace = [self.best_value]

        destroy_ops = [self.remove_cluster, self.remove_worst_rf, self.remove_oldest]
        repair_ops  = [self.repair_profit, self.repair_opportunity, self.repair_random]

        print(f"[ALNS] Start iterations={iterations}  init_cur={self.cur_value:.3f}  init_best={self.best_value:.3f}",
              flush=True)

        for it in range(iterations):
            cur_backup_assign = list(self.assignments)
            cur_backup_slots  = {ln:list(v) for ln,v in self.slots_by_link.items()}
            cur_backup_uns    = set(self.unscheduled)

            rem = random.choice(destroy_ops)
            rep = random.choices(repair_ops, weights=[2.0, 1.5, 1.0])[0]
            ratio = k_max_ratio[0] - (k_max_ratio[0]-k_max_ratio[1])*(it/max(1,iterations-1))
            k = max(5, int(ratio*max(1, len(self.assignments))))

            removed = rem(k)
            rep(max_trials=256)

            new_val = self.objective(self.assignments)
            delta   = new_val - self.cur_value
            T       = temps[it]
            accept  = (delta >= 0) or (random.random() < math.exp(delta / max(T,1e-9)))

            if accept:
                self.cur_value = new_val
                if self.cur_value > self.best_value:
                    self.best_value = self.cur_value
                    self.best_assignments = list(self.assignments)
                for a in removed:
                    if random.random() < 0.3:
                        self.tabu.append(a.task_id)
            else:
                self.assignments   = cur_backup_assign
                self.slots_by_link = defaultdict(list, cur_backup_slots)
                self.unscheduled   = cur_backup_uns

            if REHEAT_PERIOD and (it + 1) % REHEAT_PERIOD == 0:
                temps = temps * 1.3
                temps = np.clip(temps, temp_end, temp_start)

            trace.append(self.best_value)

            if (it + 1) % print_every == 0 or it == 0 or (it + 1) == iterations:
                print(f"[Iter {it+1}/{iterations}] cur={self.cur_value:.3f}  best={self.best_value:.3f}  "
                      f"d={rem.__name__}  r={rep.__name__}",
                      flush=True)

        self.assignments = list(self.best_assignments)
        self.slots_by_link = defaultdict(list)
        for a in self.assignments:
            for c in a.chunks:
                self.reserve(c.link, int(c.start), int(c.finish + 0.999999), a.task_id)

        print(f"[ALNS] Done  best={self.best_value:.3f}  scheduled={len(self.best_assignments)}",
              flush=True)
        return trace

def main():
    dev = device_autoselect()
    print(f"[INFO] Device: CPU")

    # ✅ 1) 场景目录按“业务规模数字”排序（避免字符串排序错乱）
    task_dirs = sorted(
        glob.glob(TASKS_DIR_GLOB),
        key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1))
    )
    if not task_dirs:
        raise FileNotFoundError(f"No task directories found: {TASKS_DIR_GLOB}")

    print("[INFO] Detected scenarios:")
    for d in task_dirs:
        print("  -", d)

    business_summary = []

    for TASKS_DIR in task_dirs:
        tag = os.path.basename(TASKS_DIR).replace("业务部署-30卫星-", "")
        VISIBILITY_DIR = f"{VIS_DIR_PREFIX}{tag}"

        # ✅ 2) summary 的 Scenario 用纯数字，避免出现中文
        scale = int(re.search(r"(\d+)", str(tag)).group(1))

        OUTPUT_PLOT = f"{OUTPUT_PREFIX}{tag}.png"
        OUTPUT_CSV  = f"{OUTPUT_PREFIX}{tag}.csv"

        if not os.path.isdir(VISIBILITY_DIR):
            print(f"[SKIP] Visibility dir not found: {VISIBILITY_DIR}")
            continue

        print("\n" + "="*60)
        print(f"[SCENARIO] {tag}")
        print(f"  Tasks: {TASKS_DIR}")
        print(f"  Visibility: {VISIBILITY_DIR}")
        print("="*60)

        # ---------- 原 main 内逻辑，一行不改 ----------
        tasks = load_tasks_from_dir(TASKS_DIR, GB_IS_GIB)

        TX_START = 36000
        TX_END   = 50400
        for t in tasks:
            t.arrival = max(t.arrival, TX_START)
            t.deadline = min(t.deadline, TX_END)

        links = load_all_links(VISIBILITY_DIR)
        print(f"[INFO] Tasks={len(tasks)} Links={len(links)}")

        sched = Scheduler(
            tasks, links, dev,
            rf_penalty=RF_PENALTY_PER_GBIT,
            seed=SEED
        )

        print(f"[INFO] ALNS iters={ITERATIONS}")
        t0 = time.time()
        trace = sched.alns_ts(iterations=ITERATIONS)
        t1 = time.time()

        print(f"[INFO] Done in {t1-t0:.2f}s  Best={sched.best_value:.3f}")
        print(f"[INFO] Scheduled={len(sched.best_assignments)}/{len(tasks)}")

        # ---------- plot ----------
        plt.figure(figsize=(8, 5))
        plt.plot(trace, label=tag, linewidth=2)
        plt.xlabel("Number of iterations")
        plt.ylabel("Profits")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUTPUT_PLOT, dpi=150)
        plt.close()
        print(f"[INFO] Saved plot -> {OUTPUT_PLOT}")

        # ---------- CSV export（你原来的那一段，完全不动） ----------
        task_by_id = {t.task_id: t for t in tasks}
        out_rows = []

        for a in sched.best_assignments:
            task = task_by_id.get(a.task_id, None)
            gen_time = task.gen_time if task else None
            chunks_sorted = sorted(a.chunks, key=lambda c: (c.start, c.finish, c.link))

            for idx, c in enumerate(chunks_sorted, 1):
                out_rows.append({
                    "task_id": f"{a.task_id}_part{idx}",
                    "value": float(a.value),
                    "gen_time": gen_time,
                    "links_used": c.link,
                    "link_type": "RF" if c.link_type == "RF" else "FSO",
                    "modulation_level": int(c.mod_level),
                    "start": float(c.start),
                    "finish": float(c.finish),
                    "duration": float(c.finish - c.start),
                    "FSO_bits_total": float(c.bits - c.rf_bits),
                    "RF_bits_total": float(c.rf_bits),
                })

        col_order = [
            "task_id","value","gen_time","links_used","link_type","modulation_level",
            "start","finish","duration","FSO_bits_total","RF_bits_total"
        ]
        pd.DataFrame(out_rows, columns=col_order).to_csv(
            OUTPUT_CSV, index=False, encoding="utf-8-sig"
        )
        print(f"[INFO] Saved schedule -> {OUTPUT_CSV}")

        # ✅ summary：Scenario 用 scale(纯数字)，并确保 Final_Value 是 float
        business_summary.append({
            "Scenario": scale,
            "Number_of_Tasks": len(tasks),
            "Scheduled_Tasks": len(sched.best_assignments),
            "Final_Value": float(sched.best_value)
        })

    # ✅ summary 统一按业务规模升序，且用 utf-8-sig
    summary_df = pd.DataFrame(business_summary)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["Scenario"]).reset_index(drop=True)

    summary_output = "value-scheduling_task-number_task-MALNS.csv"
    summary_df.to_csv(summary_output, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved business summary -> {summary_output}")

    print("\n🎉 All scenarios finished.")

if __name__=="__main__":
    main()
TASKS_PER_SAT = 50
