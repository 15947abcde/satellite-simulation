#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import csv
import shutil
import re
import math
from collections import defaultdict
from typing import List, Tuple, Dict
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ======================================================
# 实验配置
# ======================================================

# ✅ 星座规模（总卫星数）
CONSTELLATION_SIZES = [72, 96, 120, 144, 168, 192,216,240,264,288,320]

# ✅ 多随机种子：每个seed生成一套数据（建议 5~20 个）
SEEDS = [41, 42, 43]


# ✅ 观测卫星占比（观测卫星数 = round(NUM_SATS * OBS_RATIO)）
OBS_RATIO = 1 / 4

# ✅ 全网业务总数（固定）
TOTAL_TASKS = 2400

# ✅ 总业务量上限（GB）
TOTAL_VOLUME_CAP_GB = 20000.0

# ✅ 建议生成目标总量（GB），留余量避免触顶
TARGET_TOTAL_VOLUME_GB = 19000.0

# ✅ 单业务大小范围（GB）
MIN_TASK_GB = 0.1
MAX_TASK_GB = 20.0

# ✅ 到达时间范围
MIN_ARRIVAL_TIME = 32400
MAX_ARRIVAL_TIME = 36000

# ✅ 可见性源目录
BASE_DIR = os.path.dirname(__file__)
VISIBILITY_DIR = os.path.join(BASE_DIR, "星地可见性关系-调制格式分级")

# ✅ 分配“不要太随机”的控制：每颗观测卫星业务数在平均值±比例范围内
TASK_COUNT_DEV_RATIO = 0.15  # ±15%

# ✅ 非观测卫星：仅一条占位业务（避免调度器空序列问题）
DUMMY_VALUE = 1
DUMMY_SIZE_GB = 1.0

# ======================================================
# 混合分布参数（你可以按需要调）
# ======================================================
# 小任务占比：比如 85% 小任务，15% 大任务
MIX_SMALL_RATIO = 0.85

# 小任务 lognormal 参数（会截断到 [MIN_TASK_GB, MAX_TASK_GB]）
# 目标：多数在 0.2~3GB 左右
SMALL_SIGMA = 0.65

# 大任务 lognormal 参数（截断到 [MIN_TASK_GB, MAX_TASK_GB]）
# 目标：少量在 8~20GB
LARGE_SIGMA = 0.45

# ======================================================
# 卫星全集（保持你原逻辑：16*20=320颗）
# ======================================================

def create_all_satellites():
    return [f"{o:02d}{s:02d}" for o in range(1, 17) for s in range(1, 21)]

# ======================================================
# 可见性索引
# ======================================================

VIS_PATTERN = re.compile(r"^Facility(\d{1,2})[-_]?Satellite(\d{4})\.(csv|CSV)$")

def index_visibility_files(root: str):
    vis_map = defaultdict(list)
    vis_path = {}
    for cur, _, files in os.walk(root):
        for fn in files:
            m = VIS_PATTERN.match(fn)
            if not m:
                continue
            f_id = m.group(1).zfill(2)
            s_id = m.group(2)
            p = os.path.join(cur, fn)
            vis_map[s_id].append((f_id, p))
            vis_path[(f_id, s_id)] = p
    return vis_map, vis_path

# ======================================================
# 选择 NUM_SATS 颗卫星，并为每颗卫星选一个 facility（唯一配对）
# ======================================================

def select_satellites_and_facilities(
    num_sats: int,
    all_sats: List[str],
    vis_map: Dict[str, List[Tuple[str, str]]],
    require_unique_pair: bool = True
) -> List[Tuple[str, str]]:
    sats_with_vis = [s for s in all_sats if s in vis_map and len(vis_map[s]) > 0]
    if len(sats_with_vis) < num_sats:
        raise RuntimeError(f"可见性库中可用卫星不足：需要 {num_sats}，但只有 {len(sats_with_vis)}")

    selected_sats = random.sample(sats_with_vis, num_sats)

    used_pairs = set()
    pairs = []
    for sat in selected_sats:
        facs = vis_map[sat][:]
        random.shuffle(facs)
        chosen = None
        for fac_id, _path in facs:
            if (fac_id, sat) not in used_pairs:
                chosen = fac_id
                if require_unique_pair:
                    used_pairs.add((fac_id, sat))
                break
        if chosen is None:
            chosen = facs[0][0]
            if require_unique_pair:
                used_pairs.add((chosen, sat))
        pairs.append((sat, chosen))
    return pairs

# ======================================================
# 观测卫星集合
# ======================================================

def pick_observation_sats(selected_pairs: List[Tuple[str, str]], obs_ratio: float) -> List[Tuple[str, str]]:
    num_sats = len(selected_pairs)
    n_obs = int(round(num_sats * obs_ratio))
    n_obs = max(1, n_obs)
    return random.sample(selected_pairs, n_obs)

# ======================================================
# 业务数分配：尽量均匀，差异不大（受限于 ±dev_ratio）
# ======================================================

def allocate_task_counts(n_bins: int, total: int, dev_ratio: float) -> List[int]:
    if n_bins <= 0:
        return []
    avg = total / n_bins
    lo = max(0, int(math.floor(avg * (1.0 - dev_ratio))))
    hi = max(lo + 1, int(math.ceil(avg * (1.0 + dev_ratio))))

    base = [int(round(avg))] * n_bins
    s = sum(base)

    i = 0
    while s < total:
        if base[i] < hi:
            base[i] += 1
            s += 1
        i = (i + 1) % n_bins
        if i == 0 and all(x >= hi for x in base):
            hi += 1

    i = 0
    while s > total:
        if base[i] > lo:
            base[i] -= 1
            s -= 1
        i = (i + 1) % n_bins
        if i == 0 and all(x <= lo for x in base):
            lo = max(0, lo - 1)

    random.shuffle(base)
    return base

# ======================================================
# ✅ 混合分布：大量小任务 + 少量大任务
# ======================================================

def sample_truncated_lognormal(n: int, low: float, high: float, mu: float, sigma: float) -> List[float]:
    out = []
    # 简单拒绝采样：n=2400规模很快
    while len(out) < n:
        x = random.lognormvariate(mu, sigma)
        if low <= x <= high:
            out.append(x)
    return out

def generate_task_sizes_gb_mixture(total_tasks: int, target_total_gb: float) -> List[float]:
    """
    生成 total_tasks 个任务大小(GB)：
    - small: 占比 MIX_SMALL_RATIO
    - large: 占比 1 - MIX_SMALL_RATIO
    - 都截断到 [MIN_TASK_GB, MAX_TASK_GB]
    - 最后整体缩放到 target_total_gb（并确保不超过）
    """
    n_small = int(round(total_tasks * MIX_SMALL_RATIO))
    n_large = total_tasks - n_small

    # small: 目标均值 ~ 1GB 左右
    small_mean = 1.2
    mu_small = math.log(small_mean) - 0.5 * (SMALL_SIGMA ** 2)

    # large: 目标均值 ~ 12GB 左右（但会被截断到20）
    large_mean = 12.0
    mu_large = math.log(large_mean) - 0.5 * (LARGE_SIGMA ** 2)

    small = sample_truncated_lognormal(n_small, MIN_TASK_GB, MAX_TASK_GB, mu_small, SMALL_SIGMA)
    large = sample_truncated_lognormal(n_large, MIN_TASK_GB, MAX_TASK_GB, mu_large, LARGE_SIGMA)

    sizes = small + large
    random.shuffle(sizes)

    # 缩放到 target_total_gb
    s = sum(sizes)
    if s <= 0:
        sizes = [MIN_TASK_GB] * total_tasks
        s = sum(sizes)

    scale = target_total_gb / s
    sizes = [x * scale for x in sizes]

    # clip，避免缩放后越界
    sizes = [min(MAX_TASK_GB, max(MIN_TASK_GB, x)) for x in sizes]

    # 若 clip 导致总量略超 target，再缩一次
    s2 = sum(sizes)
    if s2 > target_total_gb:
        scale2 = target_total_gb / s2
        sizes = [min(MAX_TASK_GB, max(MIN_TASK_GB, x * scale2)) for x in sizes]

    # 两位小数
    sizes = [round(x, 2) for x in sizes]

    # 最后确保不超过 target（做一点“削峰”）
    total = round(sum(sizes), 2)
    if total > target_total_gb:
        idx = sorted(range(len(sizes)), key=lambda i: sizes[i], reverse=True)
        over = round(total - target_total_gb, 2)
        for i in idx:
            if over <= 0:
                break
            dec = min(over, round(sizes[i] - MIN_TASK_GB, 2))
            if dec > 0:
                sizes[i] = round(sizes[i] - dec, 2)
                over = round(over - dec, 2)

    return sizes

# ======================================================
# 生成业务行：value + size_gb（带表头）
# ======================================================

def build_task_rows(values: List[int], sizes_gb: List[float]) -> List[List]:
    rows = [[int(v), float(sz)] for v, sz in zip(values, sizes_gb)]
    rows.sort(key=lambda x: (-x[0], -x[1]))
    rows.insert(0, ["价值量", "业务量(GB)"])
    return rows

def build_dummy_task_rows() -> List[List]:
    return [
        ["价值量", "业务量(GB)"],
        [int(DUMMY_VALUE), float(DUMMY_SIZE_GB)],
    ]

# ======================================================
# 生成一个场景（给定 NUM_SATS, seeds）
# ======================================================

def generate_scenario(num_sats: int, vis_map, vis_path, seed: int, out_root: str):
    business_out_dir = os.path.join(out_root, f"业务部署-{num_sats}卫星-{TOTAL_TASKS}业务")
    sat_deploy_dir   = os.path.join(out_root, f"卫星部署-{num_sats}卫星-{TOTAL_TASKS}业务")

    if os.path.isdir(business_out_dir):
        shutil.rmtree(business_out_dir)
    if os.path.isdir(sat_deploy_dir):
        shutil.rmtree(sat_deploy_dir)

    os.makedirs(business_out_dir, exist_ok=True)
    os.makedirs(sat_deploy_dir, exist_ok=True)

    all_sats = create_all_satellites()

    # 1) 选 NUM_SATS 颗卫星，并为每颗绑定一个 facility
    selected_pairs = select_satellites_and_facilities(
        num_sats=num_sats,
        all_sats=all_sats,
        vis_map=vis_map,
        require_unique_pair=True
    )

    # 2) 选观测卫星（有真实业务）
    obs_pairs = pick_observation_sats(selected_pairs, OBS_RATIO)
    n_obs = len(obs_pairs)

    # 3) 生成全网真实业务（混合分布）
    target_total = min(TARGET_TOTAL_VOLUME_GB, TOTAL_VOLUME_CAP_GB)
    sizes = generate_task_sizes_gb_mixture(TOTAL_TASKS, target_total)

    # 价值量：与你之前一致，按 size 分位映射到 1..10 + 少量噪声
    idx_sorted = sorted(range(TOTAL_TASKS), key=lambda i: sizes[i])
    values = [0] * TOTAL_TASKS
    for rank, i in enumerate(idx_sorted):
        q = rank / max(1, TOTAL_TASKS - 1)
        base = 1 + int(q * 9)  # 1..10
        jitter = random.choice([0, 0, 0, 1, -1])
        values[i] = min(10, max(1, base + jitter))

    # 4) 将 2400 个业务分配到观测卫星（差异不大）
    counts = allocate_task_counts(n_obs, TOTAL_TASKS, TASK_COUNT_DEV_RATIO)
    perm = list(range(TOTAL_TASKS))
    random.shuffle(perm)

    cursor = 0
    obs_task_map = {}
    for (sat, fac), cnt in zip(obs_pairs, counts):
        take = perm[cursor:cursor + cnt]
        cursor += cnt

        sat_sizes = [sizes[i] for i in take]
        sat_vals  = [values[i] for i in take]
        arrival = random.randint(MIN_ARRIVAL_TIME, MAX_ARRIVAL_TIME)

        rows = build_task_rows(sat_vals, sat_sizes)
        obs_task_map[(sat, fac)] = (arrival, rows)

    # 5) 写业务文件：所有 selected_pairs 都生成 CSV
    #    - 观测卫星：真实业务
    #    - 非观测：dummy 1条
    for i, (sat, fac) in enumerate(selected_pairs, 1):
        if (sat, fac) in obs_task_map:
            arrival, rows = obs_task_map[(sat, fac)]
            flag = "OBS "
        else:
            arrival = random.randint(MIN_ARRIVAL_TIME, MAX_ARRIVAL_TIME)
            rows = build_dummy_task_rows()
            flag = "DUM "

        fn = f"Facility{fac}-Satellite{sat}-{arrival}.csv"
        out_path = os.path.join(business_out_dir, fn)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(rows)

        print(f"[seed={seed}] [TASK {i}/{len(selected_pairs)}] {flag}-> {fn} ✅")

    # 6) 拷贝可见性文件（给全部 NUM_SATS 卫星）
    for (sat, fac) in selected_pairs:
        src = vis_path.get((fac, sat), None)
        if src is None:
            print(f"[WARN] missing visibility for Facility{fac}-Satellite{sat}, skipped.")
            continue
        dst = os.path.join(sat_deploy_dir, f"Facility{fac}-Satellite{sat}.csv")
        shutil.copy2(src, dst)

    total_vol = round(sum(sizes), 2)
    print("\n" + "-" * 60)
    print(f"[SCENARIO DONE] seed={seed}  NUM_SATS={num_sats}  OBS_SATS={n_obs}  DUMMY_SATS={num_sats - n_obs}")
    print(f"  Root        : {out_root}")
    print(f"  Business Dir: {business_out_dir}")
    print(f"  Vis Dir     : {sat_deploy_dir}")
    print(f"  Real Tasks  : {TOTAL_TASKS} (only on OBS sats)")
    print(f"  Total Volume: {total_vol} GB  (cap={TOTAL_VOLUME_CAP_GB} GB, target={target_total} GB)")
    print("-" * 60 + "\n")

# ======================================================
# main
# ======================================================

if __name__ == "__main__":
    print("VISIBILITY_DIR =", VISIBILITY_DIR)
    print("TOTAL_TASKS =", TOTAL_TASKS)
    print("OBS_RATIO =", OBS_RATIO)
    print("TARGET_TOTAL_VOLUME_GB =", TARGET_TOTAL_VOLUME_GB)
    print("MIX_SMALL_RATIO =", MIX_SMALL_RATIO)
    print("SEEDS =", SEEDS)

    vis_map, vis_path = index_visibility_files(VISIBILITY_DIR)
    if not vis_map:
        raise RuntimeError(f"未在目录 {VISIBILITY_DIR} 下索引到任何可见性文件，请检查路径/命名。")

    for seed in SEEDS:
        random.seed(seed)

        # ✅ 每个 seeds 输出到独立目录，避免覆盖
        out_root = os.path.join(BASE_DIR, f"seed{seed}")
        os.makedirs(out_root, exist_ok=True)

        for num_sats in CONSTELLATION_SIZES:
            generate_scenario(num_sats, vis_map, vis_path, seed, out_root)

    print("\n✅ 全部场景（多seed）生成完成")
