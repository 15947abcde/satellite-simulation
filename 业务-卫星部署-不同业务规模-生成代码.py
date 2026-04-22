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
# 配置参数
# ======================================================

# ✅ 任务数场景列表：会依次输出 50任务、60任务、70任务...
TASKS_PER_SAT_LIST = [50, 60, 70, 80, 90, 100, 110, 120, 130, 140]

# ✅ 多随机种子：每个seed生成一套数据（建议 5~20 个）
SEEDS = [41, 42, 43]

# ✅ 输出根目录：参考你给的网络规模脚本（seeds/seed{seed}/...）
OUTPUT_ROOT = os.path.join(BASE_DIR, "seeds-业务")

# 可见性文件目录（可能有多级子目录）
VISIBILITY_DIR = os.path.join(BASE_DIR, "星地可见性关系-调制格式分级")

TOTAL_SELECTED_SATS = 30  # 卫星数量（固定 30）

# 到达时间范围
MIN_ARRIVAL_TIME = 0
MAX_ARRIVAL_TIME = 7200

MAX_TRAFFIC_PER_ROW = 20.0  # 每条业务最大 20GB

# ✅ 是否要求 Facility–Satellite 组合唯一
REQUIRE_UNIQUE_VISIBILITY_PAIR = True

# ✅ 如果选到某颗卫星在目录里没有任何可见 facility，是否允许换卫星
ALLOW_REPLACE_SAT_IF_NO_FACILITY = True

# ======================================================
# ✅ BUSINESS_TRAFFIC 随 TASKS_PER_SAT 动态递增（不改规则）
# ======================================================

BASE_TASKS = 50
BASE_BUSINESS_TRAFFIC = {400: 15, 600: 10, 800: 5}
INCREASE_GB_PER_10_TASKS = 50


def build_dynamic_business_traffic(tasks_per_sat: int) -> dict:
    if tasks_per_sat < BASE_TASKS:
        raise ValueError(f"TASKS_PER_SAT={tasks_per_sat} 小于基准 {BASE_TASKS}，不符合递增规则")
    diff = tasks_per_sat - BASE_TASKS
    if diff % 10 != 0:
        raise ValueError(
            f"TASKS_PER_SAT={tasks_per_sat} 需要是 50 的基础上按 10 递增（50/60/70/...），当前不满足"
        )
    step = diff // 10
    delta = step * INCREASE_GB_PER_10_TASKS
    return {k + delta: v for k, v in BASE_BUSINESS_TRAFFIC.items()}


# ======================================================
# 生成卫星全集
# ======================================================

def create_all_satellites():
    return [f"{orbit:02d}{sat:02d}" for orbit in range(1, 17) for sat in range(1, 21)]


# ======================================================
# 业务总量分配（30份，恰好对应30颗卫星）
# ======================================================

def build_total_traffic_list(business_traffic: dict):
    lst = []
    for traffic, count in business_traffic.items():
        lst.extend([traffic] * count)
    random.shuffle(lst)
    return lst


# ======================================================
# ✅ 生成：每颗卫星固定 n_tasks 条业务，总量精确守恒
# ======================================================

def generate_fixed_count_tasks(total_traffic_gb: float, n_tasks: int):
    total_traffic_gb = round(float(total_traffic_gb), 2)

    if total_traffic_gb > n_tasks * MAX_TRAFFIC_PER_ROW:
        raise ValueError(
            f"不可行：总量 {total_traffic_gb}GB 超过 n_tasks*MAX_TRAFFIC_PER_ROW = "
            f"{n_tasks}*{MAX_TRAFFIC_PER_ROW}={n_tasks * MAX_TRAFFIC_PER_ROW}GB"
        )
    if total_traffic_gb < n_tasks * 0.01:
        raise ValueError(
            f"不可行：总量 {total_traffic_gb}GB 小于 n_tasks*0.01 = {n_tasks * 0.01}GB"
        )

    rows = []
    remaining = total_traffic_gb

    for i in range(n_tasks - 1):
        remaining_slots = n_tasks - i
        min_remain_after = (remaining_slots - 1) * 0.01
        max_remain_after = (remaining_slots - 1) * MAX_TRAFFIC_PER_ROW

        min_needed = max(0.01, remaining - max_remain_after)
        max_allowed = min(MAX_TRAFFIC_PER_ROW, remaining - min_remain_after)

        min_needed = round(min_needed, 2)
        max_allowed = round(max_allowed, 2)
        if min_needed > max_allowed:
            traffic = max_allowed
        else:
            traffic = round(random.uniform(min_needed, max_allowed), 2)

        rows.append([random.randint(1, 10), traffic])
        remaining = round(remaining - traffic, 2)

    last = round(remaining, 2)
    if last < 0.01:
        last = 0.01
    if last > MAX_TRAFFIC_PER_ROW:
        last = MAX_TRAFFIC_PER_ROW
    rows.append([random.randint(1, 10), last])

    s = round(sum(r[1] for r in rows), 2)
    diff = round(total_traffic_gb - s, 2)
    if abs(diff) >= 0.01:
        new_last = round(rows[-1][1] + diff, 2)
        if 0.01 <= new_last <= MAX_TRAFFIC_PER_ROW:
            rows[-1][1] = new_last
        else:
            for k in range(len(rows)):
                newv = round(rows[k][1] + diff, 2)
                if 0.01 <= newv <= MAX_TRAFFIC_PER_ROW:
                    rows[k][1] = newv
                    break

    rows_sorted = sorted(rows, key=lambda x: (-x[0], -x[1]))
    rows_sorted.insert(0, ["价值量", "业务量(GB)"])
    return rows_sorted


# ======================================================
# 可见性索引（递归扫描）
# ======================================================

VIS_PATTERN = re.compile(r"^Facility(\d{2})-Satellite(\d{4})\.csv$")


def index_visibility_files(root_dir):
    vis_map = defaultdict(list)
    vis_path_map = {}

    for cur_dir, _, files in os.walk(root_dir):
        for fn in files:
            m = VIS_PATTERN.match(fn)
            if not m:
                continue
            facility_id = m.group(1)
            sat_id = m.group(2)
            full_path = os.path.join(cur_dir, fn)

            vis_map[sat_id].append((facility_id, full_path))
            vis_path_map[(facility_id, sat_id)] = full_path

    return vis_map, vis_path_map


# ======================================================
# ✅ 选择 30 颗卫星并为每颗卫星找到一个真实存在的 facility
# ======================================================

def select_30_sat_facility_pairs(all_sats, total_traffic_list, vis_map):
    need = len(total_traffic_list)

    sats_with_vis = [s for s in all_sats if s in vis_map and len(vis_map[s]) > 0]
    if len(sats_with_vis) < need:
        raise RuntimeError(f"可见性文件覆盖的卫星数量不足：只有 {len(sats_with_vis)} 颗可用，需求 {need} 颗")

    base_sats = random.sample(sats_with_vis, need)

    used_pairs = set()
    records = []

    for i in range(need):
        sat_id = base_sats[i]
        total_traffic = total_traffic_list[i]

        choices = vis_map[sat_id][:]
        random.shuffle(choices)

        picked_facility = None
        for facility_id, _path in choices:
            key = (facility_id, sat_id)
            if REQUIRE_UNIQUE_VISIBILITY_PAIR and key in used_pairs:
                continue
            picked_facility = facility_id
            used_pairs.add(key)
            break

        if picked_facility is None:
            if not ALLOW_REPLACE_SAT_IF_NO_FACILITY:
                raise RuntimeError(f"卫星 {sat_id} 找不到可用 facility（唯一性约束下）")
            while True:
                sat_id = random.choice(sats_with_vis)
                choices = vis_map[sat_id][:]
                random.shuffle(choices)
                picked_facility = None
                for facility_id, _path in choices:
                    key = (facility_id, sat_id)
                    if REQUIRE_UNIQUE_VISIBILITY_PAIR and key in used_pairs:
                        continue
                    picked_facility = facility_id
                    used_pairs.add(key)
                    break
                if picked_facility is not None:
                    break

        arrival_time = random.randint(MIN_ARRIVAL_TIME, MAX_ARRIVAL_TIME)
        records.append((sat_id, total_traffic, picked_facility, arrival_time))

    return records


# ======================================================
# ✅ 生成业务文件 + 复制可见性文件
# ======================================================

def generate_and_copy(records, vis_path_map, tasks_per_sat: int, business_out_dir: str, sat_deploy_dir: str):
    # ✅ 与参考脚本一致：若已存在则清理，保证每次生成干净
    if os.path.isdir(business_out_dir):
        shutil.rmtree(business_out_dir)
    if os.path.isdir(sat_deploy_dir):
        shutil.rmtree(sat_deploy_dir)

    os.makedirs(business_out_dir, exist_ok=True)
    os.makedirs(sat_deploy_dir, exist_ok=True)

    for idx, (sat_id, total_traffic, facility_id, arrival_time) in enumerate(records, 1):
        orbit_xx = sat_id[:2]
        sat_yy = sat_id[2:]

        business_file_name = f"Facility{facility_id}-Satellite{orbit_xx}{sat_yy}-{arrival_time}.csv"
        business_file_path = os.path.join(business_out_dir, business_file_name)

        csv_data = generate_fixed_count_tasks(total_traffic, tasks_per_sat)
        with open(business_file_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(csv_data)

        visibility_name = f"Facility{facility_id}-Satellite{orbit_xx}{sat_yy}.csv"
        src_vis_path = vis_path_map.get((facility_id, sat_id))
        if not src_vis_path or not os.path.exists(src_vis_path):
            raise RuntimeError(f"内部错误：预期存在可见性文件但未找到：{visibility_name}")

        dst_vis_path = os.path.join(sat_deploy_dir, visibility_name)
        shutil.copy2(src_vis_path, dst_vis_path)

        print(
            f"[{idx}/{len(records)}] 业务: {business_file_name} "
            f"(总量{total_traffic}GB, 条数{tasks_per_sat}) | 可见性: {visibility_name} ✅"
        )

    print("\n========== 场景完成 ==========")
    print(f"  Business Dir: {os.path.abspath(business_out_dir)}")
    print(f"  Vis Dir     : {os.path.abspath(sat_deploy_dir)}")
    print(f"  每颗卫星业务条数：{tasks_per_sat}（严格固定）")
    print("==============================\n")


# ======================================================
# main（✅核心修改：同一seed下仅选一次卫星，所有任务数场景复用）
# ======================================================

if __name__ == "__main__":
    print("VISIBILITY_DIR =", VISIBILITY_DIR)
    print("TASKS_PER_SAT_LIST =", TASKS_PER_SAT_LIST)
    print("SEEDS =", SEEDS)

    vis_map, vis_path_map = index_visibility_files(VISIBILITY_DIR)
    if not vis_map:
        raise RuntimeError(f"未在目录 {VISIBILITY_DIR} 下索引到任何可见性文件，请检查路径/命名。")

    all_sats = create_all_satellites()
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    for seed in SEEDS:
        random.seed(seed)

        # ✅ 参考脚本：每个 seed 输出到独立目录
        out_root = os.path.join(OUTPUT_ROOT, f"seed{seed}")
        os.makedirs(out_root, exist_ok=True)

        # ===================== 核心修改开始 =====================
        # 1. 基于第一个任务数场景，生成固定的卫星+Facility+到达时间（仅选一次）
        first_tasks_per_sat = TASKS_PER_SAT_LIST[0]
        first_business_traffic = build_dynamic_business_traffic(first_tasks_per_sat)
        first_total_traffic_list = build_total_traffic_list(first_business_traffic)
        first_records = select_30_sat_facility_pairs(all_sats, first_total_traffic_list, vis_map)

        # 2. 提取固定的卫星信息（SatID + FacilityID + 到达时间），仅业务量后续动态变化
        fixed_sat_base = [(r[0], r[2], r[3]) for r in first_records]
        # ===================== 核心修改结束 =====================

        for tasks_per_sat in TASKS_PER_SAT_LIST:
            # 生成当前任务数场景的业务总量列表
            business_traffic = build_dynamic_business_traffic(tasks_per_sat)
            total_traffic_list = build_total_traffic_list(business_traffic)

            # 复用固定卫星，仅替换业务总量
            current_records = []
            for i in range(len(fixed_sat_base)):
                sat_id, facility_id, arrival_time = fixed_sat_base[i]
                total_traffic = total_traffic_list[i]
                current_records.append((sat_id, total_traffic, facility_id, arrival_time))

            business_out_dir = os.path.join(out_root, f"业务部署-30卫星-{tasks_per_sat}业务")
            sat_deploy_dir = os.path.join(out_root, f"卫星部署-30卫星-{tasks_per_sat}业务")

            print(f"\n==================== seed={seed} | tasks_per_sat={tasks_per_sat} ====================")
            generate_and_copy(current_records, vis_path_map, tasks_per_sat, business_out_dir, sat_deploy_dir)

    print("\n✅ 全部任务数场景 + 全部 seed 生成完成")