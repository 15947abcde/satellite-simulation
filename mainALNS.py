#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ALNS+TS调度器 - 统一主程序模块
支持切换四种算法：
1. ALNS（无星间转发 + 无分片）- ALNS.py
2. MALNS（带星间转发 + 分片）- MALNS.py
3. DRA-ALNS（动态速率匹配 + 无星间转发）- DRA_ALNS.py
4. ISL-ALNS（星间互联 + 无速率匹配）- ISL_ALNS.py
"""

import os
import re
import glob
import time
import math
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import numpy as np
BASE_DIR = Path(__file__).resolve().parent
# ================= 核心工具函数 =================
def discover_seeds(data_root: str) -> List[int]:
    """发现seed目录（主程序专属工具函数）"""
    seed_dirs = glob.glob(os.path.join(data_root, "seed*"))
    seeds = []
    for p in seed_dirs:
        name = os.path.basename(p)
        m = re.fullmatch(r"seed(\d+)", name)
        if m and os.path.isdir(p):
            seeds.append(int(m.group(1)))
    seeds = sorted(seeds)
    if not seeds:
        raise FileNotFoundError(f"No seedXX directories found under: {data_root}")
    return seeds

def mean_ci(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """统一的置信区间计算函数"""
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

# ================= 交互式算法选择（新增ISL-ALNS）=================
def select_algorithm_interactive():
    """交互式选择算法（终端提问，输入1/2/3/4选择）"""
    print("\n===== 选择要运行的算法 =====")
    print("1 - ALNS（无星间转发 + 无分片）")
    print("2 - MALNS（带星间转发 + 分片）")
    print("3 - DRA-ALNS（动态速率匹配 + 无星间转发）")
    print("4 - ISL-ALNS（星间互联 + 无速率匹配）")
    while True:
        choice = input("请输入 1/2/3/4 选择算法：").strip()
        if choice == "1":
            return "ALNS"
        elif choice == "2":
            return "MALNS"
        elif choice == "3":
            return "DRA-ALNS"
        elif choice == "4":
            return "ISL-ALNS"
        else:
            print("❌ 输入无效，请输入 1、2、3 或 4！")

# 获取算法类型（交互式选择）
ALGORITHM_TYPE = select_algorithm_interactive()

# ================= 精准导入配置（新增ISL-ALNS导入）=================
# 仅导入算法核心类/函数，工具函数已在main.py定义
if ALGORITHM_TYPE == "ALNS":
    # 从ALNS.py导入（无星间转发+无分片）
    from ALNS import (
        SchedulerConfigNoForward as SchedulerConfig,
        SchedulerNoForward as Scheduler,
        device_autoselect,
        load_tasks_from_dir,
        load_all_links
    )
    ALGO_NAME = "ALNS"
elif ALGORITHM_TYPE == "MALNS":
    # 从MALNS.py导入（带星间转发+分片）
    from MALNS import (
        SchedulerConfig,
        Scheduler,
        device_autoselect,
        load_tasks_from_dir,
        load_all_links
    )
    ALGO_NAME = "MALNS"
elif ALGORITHM_TYPE == "DRA-ALNS":
    # 从DRA_ALNS.py导入（动态速率匹配）
    from DRA_ALNS import (
        SchedulerConfigDRA as SchedulerConfig,
        SchedulerDRA as Scheduler,
        device_autoselect,
        load_tasks_from_dir,
        load_all_links
    )
    ALGO_NAME = "DRA-ALNS"
elif ALGORITHM_TYPE == "ISL-ALNS":
    # 从ISL_ALNS.py导入（星间互联）
    from ISL_ALNS import (
        SchedulerConfigISL as SchedulerConfig,
        SchedulerISL as Scheduler,
        device_autoselect,
        load_tasks_from_dir,
        load_all_links
    )
    ALGO_NAME = "ISL-ALNS"
else:
    raise ValueError(f"无效的算法类型：{ALGORITHM_TYPE}")

# ================= 辅助函数（保持不变）=================
def select_simulation_mode() -> int:
    """选择仿真模式"""
    print("\n===== 选择仿真模式 =====")
    print("1 - 不同业务规模（固定卫星数）")
    print("2 - 不同网络规模（固定业务数）")
    while True:
        choice = input("请输入 1 或 2：").strip()
        if choice in ("1", "2"):
            return int(choice)
        print("❌ 输入无效，请重新输入。")

# ================= 主流程（适配ISL-ALNS）=================
def main():
    # 初始化配置
    config = SchedulerConfig()
    dev = device_autoselect()
    TX_START, TX_END = 3600, 43200
    SCHED_PERIOD_SEC = TX_END - TX_START

    # 打印配置信息
    print(f"\n[INFO] 已选择算法: {ALGORITHM_TYPE}")
    print(f"[INFO] 算法名称: {ALGO_NAME}")
    print(f"[INFO] 计算设备: {dev} (CUDA={dev.type == 'cuda'})")
    print(f"[INFO] 迭代次数: {config.ITERATIONS}")
    if ALGORITHM_TYPE == "ALNS":
        print(f"[INFO] 核心约束: NO Task Split | NO Multi-Sat Forward | Only Home Link")
    elif ALGORITHM_TYPE == "MALNS":
        print(f"[INFO] 核心约束: Support Task Split | Multi-Sat Forward | Multi-Link")
    elif ALGORITHM_TYPE == "DRA-ALNS":
        print(f"[INFO] 核心约束: Dynamic Rate Matching | NO Multi-Sat Forward | Only Home Link")
    elif ALGORITHM_TYPE == "ISL-ALNS":
        print(f"[INFO] 核心约束: NO Task Split | Multi-Sat Forward | Any Link Allowed (ISL)")

    # 选择仿真模式
    mode = select_simulation_mode()

    # ================= 模式配置 =================
    if mode == 1:
        # 业务规模模式
        MODE_SUFFIX = "业务规模"
        SUMMARY_COL = "Business_Scale"
        DATA_ROOT = BASE_DIR / "seeds-业务"
        TASKS_DIR_PATTERN = "业务部署-30卫星-*业务"
        SCALE_PATTERN = r"(\d+)业务"
        if ALGORITHM_TYPE == "ALNS":
            MODE_SUFFIX += "-无分片"
        elif ALGORITHM_TYPE == "DRA-ALNS":
            MODE_SUFFIX += "-动态速率匹配"
        elif ALGORITHM_TYPE == "ISL-ALNS":
            MODE_SUFFIX += "-星间互联"
    else:
        # 网络规模模式
        MODE_SUFFIX = "网络规模"
        SUMMARY_COL = "Network_Scale"
        DATA_ROOT = BASE_DIR / "seeds-网络"
        TASKS_DIR_PATTERN = "业务部署-*卫星-2400业务"
        SCALE_PATTERN = r"(\d+)卫星"
        if ALGORITHM_TYPE == "ALNS":
            MODE_SUFFIX += "-无分片"
        elif ALGORITHM_TYPE == "DRA-ALNS":
            MODE_SUFFIX += "-动态速率匹配"
        elif ALGORITHM_TYPE == "ISL-ALNS":
            MODE_SUFFIX += "-星间互联"

    # 检查数据根目录是否存在
    if not DATA_ROOT.exists():
        print(f"❌ 数据根目录不存在：{DATA_ROOT}")
        return

    # 发现seed目录
    try:
        SEEDS = discover_seeds(str(DATA_ROOT))
        scan_seed = SEEDS[0]
        print(f"\n[INFO] 发现Seed目录: {SEEDS}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return

    # 发现所有场景目录
    task_dirs = sorted(
        glob.glob(os.path.join(DATA_ROOT, f"seed{scan_seed}", TASKS_DIR_PATTERN)),
        key=lambda p: int(re.search(SCALE_PATTERN, os.path.basename(p)).group(1))
    )

    if not task_dirs:
        print(f"❌ 未找到{MODE_SUFFIX}场景目录")
        return

    summary_records = []

    # ================= 遍历每个场景 =================
    for idx, base_dir in enumerate(task_dirs, 1):
        # 提取规模值（业务数/卫星数）
        scale = int(re.search(SCALE_PATTERN, os.path.basename(base_dir)).group(1))
        print(f"\n📦 [{idx}/{len(task_dirs)}] {MODE_SUFFIX} = {scale}")

        # 统计变量
        final_values = []
        scheduled_nums = []
        avg_rates_gbps = []
        total_rates_gbps = []
        avg_transfer_times = []
        total_tasks = 0

        # 遍历每个seed
        for si, seed in enumerate(SEEDS, 1):
            print(f"  🌱 Seed {seed} ({si}/{len(SEEDS)}) ... ", end="", flush=True)

            # 构建路径
            if mode == 1:
                # 业务规模路径
                TASKS_DIR = os.path.join(DATA_ROOT, f"seed{seed}", f"业务部署-30卫星-{scale}业务")
                VIS_DIR = os.path.join(DATA_ROOT, f"seed{seed}", f"卫星部署-30卫星-{scale}业务")
            else:
                # 网络规模路径
                TASKS_DIR = os.path.join(DATA_ROOT, f"seed{seed}", f"业务部署-{scale}卫星-2400业务")
                VIS_DIR = os.path.join(DATA_ROOT, f"seed{seed}", f"卫星部署-{scale}卫星-2400业务")

            # 检查路径有效性
            if not os.path.isdir(TASKS_DIR) or not os.path.isdir(VIS_DIR):
                print("skip ❌ (路径不存在)")
                continue

            # 加载数据（适配不同算法的参数差异）
            try:
                if ALGORITHM_TYPE in ("ALNS", "DRA-ALNS", "ISL-ALNS"):
                    # ALNS/DRA-ALNS/ISL-ALNS的load_tasks_from_dir需要config参数
                    tasks = load_tasks_from_dir(TASKS_DIR, config)
                else:
                    # MALNS的load_tasks_from_dir需要gb_is_gib参数
                    tasks = load_tasks_from_dir(TASKS_DIR, config.GB_IS_GIB)
                total_tasks = len(tasks)
            except Exception as e:
                print(f"skip ❌ (加载任务失败: {e})")
                continue

            # 调整任务时间范围
            for t in tasks:
                t.arrival = max(t.arrival, TX_START)
                t.deadline = min(t.deadline, TX_END)

            # 加载链路数据（适配不同算法的参数差异）
            try:
                if ALGORITHM_TYPE in ("ALNS", "DRA-ALNS", "ISL-ALNS"):
                    # ALNS/DRA-ALNS/ISL-ALNS的load_all_links需要config参数
                    links = load_all_links(VIS_DIR, config)
                else:
                    # MALNS的load_all_links不需要额外参数
                    links = load_all_links(VIS_DIR)
            except Exception as e:
                print(f"skip ❌ (加载链路失败: {e})")
                continue

            # 初始化调度器
            sched = Scheduler(
                tasks=tasks,
                links=links,
                device=dev,
                config=config,
                seed=seed
            )

            # 运行调度算法
            t0 = time.time()
            try:
                best_value, best_assignments, _ = sched.run_alns_ts(print_every=20)
                t1 = time.time()
                print(f"done ⏱ {t1 - t0:.1f}s")
            except Exception as e:
                print(f"fail ❌ (运行算法失败: {e})")
                continue

            # 收集结果
            final_values.append(float(best_value))
            scheduled_nums.append(len(best_assignments))

            if best_assignments:
                # 计算平均速率
                bits = sum(a.chunk.bits for a in best_assignments)
                t0_task = min(a.chunk.start for a in best_assignments)
                t1_task = max(a.chunk.finish for a in best_assignments)
                avg_rates_gbps.append((bits / max(1.0, t1_task - t0_task)) / 1e9)

                # 固定周期速率
                total_rates_gbps.append((bits / SCHED_PERIOD_SEC) / 1e9)

                # 平均传输时间
                avg_transfer_times.append(
                    np.mean([a.chunk.finish - a.chunk.start for a in best_assignments])
                )
            else:
                avg_rates_gbps.append(0.0)
                total_rates_gbps.append(0.0)
                avg_transfer_times.append(0.0)

        # ================= 统计结果 =================
        if final_values:
            mv, vl, vh = mean_ci(final_values)
            ms, sl, sh = mean_ci(scheduled_nums)
            mrate, rl, rh = mean_ci(avg_rates_gbps)
            mtr, trl, trh = mean_ci(total_rates_gbps)
            matt, atl, ath = mean_ci(avg_transfer_times)


            record = {
                SUMMARY_COL: scale,
                "Number_of_Tasks": total_tasks,
                "Mean_Final_Value": round(mv, 3),
                "Final_Value_CI_Low": round(vl, 3),
                "Final_Value_CI_High": round(vh, 3),
                "Mean_Scheduled_Tasks": round(ms, 2),
                "Scheduled_Tasks_CI_Low": round(sl, 2),
                "Scheduled_Tasks_CI_High": round(sh, 2),
                "Mean_Total_Avg_Rate_Gbps": round(mrate, 6),
                "Total_Avg_Rate_CI_Low_Gbps": round(rl, 6),
                "Total_Avg_Rate_CI_High_Gbps": round(rh, 6),
                "Mean_Total_Transfer_Rate_Gbps": round(mtr, 6),
                "Total_Transfer_Rate_CI_Low_Gbps": round(trl, 6),
                "Total_Transfer_Rate_CI_High_Gbps": round(trh, 6),
                "Mean_Avg_Transfer_Time_sec": round(matt, 3),
                "Avg_Transfer_Time_CI_Low_sec": round(atl, 3),
                "Avg_Transfer_Time_CI_High_sec": round(ath, 3),
                "Seeds_Count": len(SEEDS),
            }
            # 网络规模模式移除Number_of_Tasks
            if mode == 2:
                del record["Number_of_Tasks"]
            summary_records.append(record)

    # ================= 保存结果 =================
    if summary_records:
        df = pd.DataFrame(summary_records).sort_values(SUMMARY_COL)
        if mode == 1:
            out_file = f"value-scheduling_task-number_task-{ALGO_NAME}-{MODE_SUFFIX}.csv"
        else:
            out_file = f"value-scheduling_task-mean-ci-{ALGO_NAME}-{MODE_SUFFIX}.csv"
        
        df.to_csv(out_file, index=False, encoding="utf-8-sig")
        print(f"\n📊 汇总完成，已保存：{out_file}")
    else:
        print("\n❌ 无有效结果可保存")

    print(f"🎉 全部仿真完成！算法：{ALGO_NAME} | 模式：{MODE_SUFFIX}")

if __name__ == "__main__":
    main()