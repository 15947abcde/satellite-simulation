#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一调度主程序
支持切换五种算法：
5 - GRASP-ILS（单链路 + 无分片）
6 - 两阶段启发式（单链路 + 无分片）
7 - VNS（单链路 + 无分片）
8 - GreedyDensity（单链路 + 无分片）
9 - GA（单链路 + 无分片）
"""

import os
import re
import glob
import math
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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


def select_simulation_mode():
    """选择仿真模式"""
    print("\n===== 选择仿真模式 =====")
    print("1 - 不同业务规模（固定卫星数）")
    print("2 - 不同网络规模（固定业务数）")
    while True:
        c = input("请输入 1 或 2：").strip()
        if c in ("1", "2"):
            return int(c)
        print("❌ 输入无效，请重新输入。")


def discover_seeds(data_root: str) -> List[int]:
    """发现所有seed目录"""
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


def select_algorithm_interactive():
    """交互式选择算法（剔除前四个算法）"""
    print("\n===== 选择要运行的算法 =====")
    print("5 - GRASP-ILS（单链路 + 无分片）")
    print("6 - 两阶段启发式（单链路 + 无分片）")
    print("7 - VNS（单链路 + 无分片）")
    print("8 - GreedyDensity（单链路 + 无分片）")
    print("9 - GA（单链路 + 无分片）")
    while True:
        choice = input("请输入 5/6/7/8/9 选择算法：").strip()
        if choice == "5":
            return "GRASP-ILS"
        elif choice == "6":
            return "TwoStage"
        elif choice == "7":
            return "VNS"
        elif choice == "8":
            return "GreedyDensity"
        elif choice == "9":
            return "GA"
        else:
            print("❌ 输入无效，请输入 5/6/7/8/9 之间的数字！")


# ================= 算法选择与导入（剔除前四个算法的导入分支） =================
ALGORITHM_TYPE = select_algorithm_interactive()
print(f"\n[INFO] 已选择算法: {ALGORITHM_TYPE}")

# 精准导入配置（仅保留后五个算法）
if ALGORITHM_TYPE == "GRASP-ILS":
    from GRASP_ILS import (
        SchedulerConfigGRASP as SchedulerConfig,
        SchedulerGRASP as Scheduler,
        device_autoselect,
        load_tasks_from_dir,
        load_all_links
    )
    ALGO_NAME = "GRASP_ILS"
elif ALGORITHM_TYPE == "TwoStage":
    from Two_Stage_Heuristic import (
        SchedulerConfigTwoStage as SchedulerConfig,
        SchedulerTwoStage as Scheduler,
        device_autoselect,
        load_tasks_from_dir,
        load_all_links
    )
    ALGO_NAME = "TwoStage"
elif ALGORITHM_TYPE == "VNS":
    from VNS import (
        SchedulerConfigVNS as SchedulerConfig,
        SchedulerVNS as Scheduler,
        device_autoselect,
        load_tasks_from_dir,
        load_all_links
    )
    ALGO_NAME = "VNS"
elif ALGORITHM_TYPE == "GreedyDensity":
    from GreedyDensity import (
        SchedulerConfigGreedy as SchedulerConfig,
        SchedulerGreedy as Scheduler,
        device_autoselect,
        load_tasks_from_dir,
        load_all_links
    )
    ALGO_NAME = "GreedyDensity"
elif ALGORITHM_TYPE == "GA":
    from GA import (
        SchedulerConfigGA as SchedulerConfig,
        SchedulerGA as Scheduler,
        device_autoselect,
        load_tasks_from_dir,
        load_all_links
    )
    ALGO_NAME = "GA"
else:
    raise ValueError(f"无效的算法类型：{ALGORITHM_TYPE}")

# ================= 全局配置 =================
config = SchedulerConfig()
dev = device_autoselect()
print(f"[INFO] 算法名称: {ALGO_NAME}")
print(f"[INFO] 计算设备: {dev} (CUDA={torch.cuda.is_available()})")

# 适配算法参数打印
if ALGO_NAME == "GRASP_ILS":
    print(f"[INFO] GRASP迭代次数: {config.GRASP_ITERS}")
    print(f"[INFO] RCL列表大小: {config.RCL_K}")
    print(f"[INFO] ILS改进步数: {config.ILS_STEPS}")
elif ALGO_NAME == "TwoStage":
    print(f"[INFO] Stage2迭代次数: {config.STAGE2_ITERS}")
    print(f"[INFO] 销毁率: {config.DESTROY_RATE}")
    print(f"[INFO] 最小销毁数量: {config.MIN_DESTROY}")
elif ALGO_NAME == "VNS":
    print(f"[INFO] VNS迭代次数: {config.VNS_ITERS}")
    print(f"[INFO] 无改进切换阈值: {config.NO_IMPROVE_TO_SWITCH}")
    print(f"[INFO] 速率抖动系数: {config.RANDOM_RATE_JITTER}")
elif ALGO_NAME == "GreedyDensity":
    print(f"[INFO] 价值密度排序: 降序（value/size_bits）")
    print(f"[INFO] 单链路约束: 任务仅调度在归属链路")
elif ALGO_NAME == "GA":
    print(f"[INFO] GA种群大小: {config.POP_SIZE}")
    print(f"[INFO] GA进化代数: {config.GENERATIONS}")
    print(f"[INFO] 交叉概率: {config.CX_PROB} | 变异概率: {config.MUT_PROB}")
    print(f"[INFO] 速率抖动系数: {config.RANDOM_RATE_JITTER}")

# 时间窗口配置
TX_START = 3600
TX_END = 43200
SCHED_PERIOD_SEC = TX_END - TX_START

# ================= 主仿真流程 =================
def main():
    mode = select_simulation_mode()
    
    business_summary = []
    network_summary = []
    
    DATA_ROOT_MODE1 = os.path.join(BASE_DIR, "seeds-业务")
    DATA_ROOT_MODE2 = os.path.join(BASE_DIR, "seeds-网络")

    if mode == 1:
        # 业务规模模式
        SEEDS = discover_seeds(DATA_ROOT_MODE1)
        print(f"\n[INFO] 发现Seed目录: {SEEDS}")
        scan_seed = SEEDS[0]
        TASKS_DIR_GLOB = os.path.join(DATA_ROOT_MODE1, f"seed{scan_seed}", "业务部署-30卫星-*业务")
        task_dirs = sorted(
            glob.glob(TASKS_DIR_GLOB),
            key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1))
        )
        scale_list = [int(re.search(r"(\d+)业务", os.path.basename(p)).group(1)) for p in task_dirs]
        business_summary = []
        boxplot_data = []  # 保留原代码的箱线图数据

        for i, scale in enumerate(scale_list, 1):
            print(f"\n📦 [{i}/{len(scale_list)}] 业务规模 = {scale}")
            values = []
            sched_nums = []
            total_tasks_list = []
            total_transfer_rates = []
            avg_transfer_times = []

            for si, seed in enumerate(SEEDS, 1):
                print(f"  🌱 Seed {seed} ({si}/{len(SEEDS)}) ...", end=" ")
                # 构造路径
                seed_TASKS_DIR = os.path.join(DATA_ROOT_MODE1, f"seed{seed}", f"业务部署-30卫星-{scale}业务")
                seed_VIS_DIR = os.path.join(DATA_ROOT_MODE1, f"seed{seed}", f"卫星部署-30卫星-{scale}业务")

                # 检查路径
                if not os.path.isdir(seed_TASKS_DIR) or not os.path.isdir(seed_VIS_DIR):
                    print("skip ❌ (路径不存在)")
                    continue

                # 加载任务（适配不同算法）
                try:
                    if ALGO_NAME in ["GRASP_ILS", "TwoStage", "VNS", "GreedyDensity", "GA"]:
                        tasks = load_tasks_from_dir(seed_TASKS_DIR, config)
                    else:
                        tasks = load_tasks_from_dir(seed_TASKS_DIR, config.GB_IS_GIB)
                    tasks = [t for t in tasks if t.deadline > t.arrival and t.size_bits > 0]
                    if not tasks:
                        print("skip ❌ (无有效任务)")
                        continue
                except Exception as e:
                    print(f"skip ❌ (加载任务失败: {str(e)})")
                    continue

                # 加载链路（适配不同算法）
                try:
                    if ALGO_NAME in ["GRASP_ILS", "TwoStage", "VNS", "GreedyDensity", "GA"]:
                        links = load_all_links(seed_VIS_DIR, config)
                    else:
                        links = load_all_links(seed_VIS_DIR, config.RATE_MAP, config.LINK_TYPE)
                    if not links:
                        print("skip ❌ (无有效链路)")
                        continue
                except Exception as e:
                    print(f"skip ❌ (加载链路失败: {str(e)})")
                    continue

                # 调整任务时间
                for t in tasks:
                    t.arrival = max(t.arrival, TX_START)
                    t.deadline = min(t.deadline, TX_END)

                # 运行算法（适配不同算法）
                try:
                    sched = Scheduler(tasks, links, dev, config, seed=seed)
                    if ALGO_NAME == "GRASP_ILS":
                        best_value, best_assignments, trace = sched.run_grasp_ils()
                    elif ALGO_NAME == "TwoStage":
                        best_value, best_assignments, trace = sched.run_two_stage()
                    elif ALGO_NAME == "VNS":
                        best_value, best_assignments, trace = sched.run_vns()
                    elif ALGO_NAME == "GreedyDensity":
                        best_value, best_assignments, trace = sched.run_greedy_density()
                    elif ALGO_NAME == "GA":
                        best_value, best_assignments, trace = sched.run_ga()
                    else:
                        best_value, best_assignments, trace = sched.run_alns()

                    # 计算统计指标
                    total_tasks_list.append(len(tasks))
                    values.append(float(best_value))
                    sched_nums.append(len(best_assignments))

                    # 传输速率和时间
                    if best_assignments:
                        total_bits = sum(c.bits for a in best_assignments for c in a.chunks)
                        durations = [c.finish - c.start for a in best_assignments for c in a.chunks]
                        total_transfer_rates.append((total_bits / SCHED_PERIOD_SEC) / 1e9)
                        avg_transfer_times.append(float(np.mean(durations)) if durations else 0.0)
                    else:
                        total_transfer_rates.append(0.0)
                        avg_transfer_times.append(0.0)

                    print(f"done ✅ (调度任务数: {len(best_assignments)})")
                except Exception as e:
                    print(f"skip ❌ (运行算法失败: {str(e)})")
                    continue

            # 统计结果
            if values:
                mv, val_ci_low, val_ci_high = mean_ci(values)
                ms, sched_ci_low, sched_ci_high = mean_ci(sched_nums)

                # ===== 关键改动：传输速率 mean + CI =====
                if total_transfer_rates:
                    mt, rate_ci_low, rate_ci_high = mean_ci(total_transfer_rates)
                else:
                    mt, rate_ci_low, rate_ci_high = 0.0, 0.0, 0.0

                mat = mean_ci(avg_transfer_times)[0] if avg_transfer_times else 0.0
                avg_total_tasks = np.mean(total_tasks_list) if total_tasks_list else 0

                business_summary.append({
                    "Business_Scale": scale,
                    "Number_of_Tasks": int(avg_total_tasks),
                    "Mean_Final_Value": round(mv, 3),
                    "Final_Value_CI_Low": round(val_ci_low, 3),
                    "Final_Value_CI_High": round(val_ci_high, 3),
                    "Mean_Scheduled_Tasks": round(ms, 2),
                    "Scheduled_Tasks_CI_Low": round(sched_ci_low, 2),
                    "Scheduled_Tasks_CI_High": round(sched_ci_high, 2),

                    # ===== 新增三列 =====
                    "Mean_Total_Transfer_Rate_Gbps": round(float(mt), 6),
                    "Total_Transfer_Rate_CI_Low": round(float(rate_ci_low), 6),
                    "Total_Transfer_Rate_CI_High": round(float(rate_ci_high), 6),

                    "Mean_Avg_Transfer_Time_sec": round(float(mat), 3),
                    "Seeds_Count": len(SEEDS)
                })

                # 箱线图数据（保留原逻辑）
                boxplot_data.extend([{"Scale": scale, "Value": v} for v in values])

        # 保存结果
        if business_summary:
            df = pd.DataFrame(business_summary).sort_values("Business_Scale")
            df.to_csv(
                f"value-scheduling_task-number_task-{ALGO_NAME}-业务规模.csv",
                index=False,
                encoding="utf-8-sig"
            )
            print(f"\n✅ 结果已保存至: value-scheduling_task-number_task-{ALGO_NAME}-业务规模.csv")
        if boxplot_data:
            pd.DataFrame(boxplot_data).to_csv(
                f"boxplot_data-{ALGO_NAME}-业务规模.csv",
                index=False,
                encoding="utf-8-sig"
            )
            print(f"✅ 箱线图数据已保存至: boxplot_data-{ALGO_NAME}-业务规模.csv")
        else:
            print("\n❌ 无有效结果可保存")

    else:
        # 网络规模模式
        SEEDS = discover_seeds(DATA_ROOT_MODE2)
        print(f"\n[INFO] 发现Seed目录: {SEEDS}")
        scan_seed = SEEDS[0]
        TASKS_DIR_GLOB = os.path.join(DATA_ROOT_MODE2, f"seed{scan_seed}", "业务部署-*卫星-2400业务")
        task_dirs = sorted(
            glob.glob(TASKS_DIR_GLOB),
            key=lambda p: int(re.search(r"(\d+)", os.path.basename(p)).group(1))
        )
        scale_list = [int(re.search(r"(\d+)卫星", os.path.basename(p)).group(1)) for p in task_dirs]
        network_summary = []
        boxplot_data = []  # 保留原代码的箱线图数据

        for i, scale in enumerate(scale_list, 1):
            print(f"\n📦 [{i}/{len(scale_list)}] 网络规模 = {scale}")
            values = []
            sched_nums = []
            total_tasks_list = []
            total_transfer_rates = []
            avg_transfer_times = []

            for si, seed in enumerate(SEEDS, 1):
                print(f"  🌱 Seed {seed} ({si}/{len(SEEDS)}) ...", end=" ")
                # 构造路径
                seed_TASKS_DIR = os.path.join(DATA_ROOT_MODE2, f"seed{seed}", f"业务部署-{scale}卫星-2400业务")
                seed_VIS_DIR = os.path.join(DATA_ROOT_MODE2, f"seed{seed}", f"卫星部署-{scale}卫星-2400业务")

                # 检查路径
                if not os.path.isdir(seed_TASKS_DIR) or not os.path.isdir(seed_VIS_DIR):
                    print("skip ❌ (路径不存在)")
                    continue

                # 加载任务（适配不同算法）
                try:
                    if ALGO_NAME in ["GRASP_ILS", "TwoStage", "VNS", "GreedyDensity", "GA"]:
                        tasks = load_tasks_from_dir(seed_TASKS_DIR, config)
                    else:
                        tasks = load_tasks_from_dir(seed_TASKS_DIR, config.GB_IS_GIB)
                    tasks = [t for t in tasks if t.deadline > t.arrival and t.size_bits > 0]
                    if not tasks:
                        print("skip ❌ (无有效任务)")
                        continue
                except Exception as e:
                    print(f"skip ❌ (加载任务失败: {str(e)})")
                    continue

                # 加载链路（适配不同算法）
                try:
                    if ALGO_NAME in ["GRASP_ILS", "TwoStage", "VNS", "GreedyDensity", "GA"]:
                        links = load_all_links(seed_VIS_DIR, config)
                    else:
                        links = load_all_links(seed_VIS_DIR, config.RATE_MAP, config.LINK_TYPE)
                    if not links:
                        print("skip ❌ (无有效链路)")
                        continue
                except Exception as e:
                    print(f"skip ❌ (加载链路失败: {str(e)})")
                    continue

                # 调整任务时间
                for t in tasks:
                    t.arrival = max(t.arrival, TX_START)
                    t.deadline = min(t.deadline, TX_END)

                # 运行算法（适配不同算法）
                try:
                    sched = Scheduler(tasks, links, dev, config, seed=seed)
                    if ALGO_NAME == "GRASP_ILS":
                        best_value, best_assignments, trace = sched.run_grasp_ils()
                    elif ALGO_NAME == "TwoStage":
                        best_value, best_assignments, trace = sched.run_two_stage()
                    elif ALGO_NAME == "VNS":
                        best_value, best_assignments, trace = sched.run_vns()
                    elif ALGO_NAME == "GreedyDensity":
                        best_value, best_assignments, trace = sched.run_greedy_density()
                    elif ALGO_NAME == "GA":
                        best_value, best_assignments, trace = sched.run_ga()
                    else:
                        best_value, best_assignments, trace = sched.run_alns()

                    # 计算统计指标
                    total_tasks_list.append(len(tasks))
                    values.append(float(best_value))
                    sched_nums.append(len(best_assignments))

                    # 传输速率和时间
                    if best_assignments:
                        total_bits = sum(c.bits for a in best_assignments for c in a.chunks)
                        durations = [c.finish - c.start for a in best_assignments for c in a.chunks]
                        total_transfer_rates.append((total_bits / SCHED_PERIOD_SEC) / 1e9)
                        avg_transfer_times.append(float(np.mean(durations)) if durations else 0.0)
                    else:
                        total_transfer_rates.append(0.0)
                        avg_transfer_times.append(0.0)

                    print(f"done ✅ (调度任务数: {len(best_assignments)})")
                except Exception as e:
                    print(f"skip ❌ (运行算法失败: {str(e)})")
                    continue

            # 统计结果
            if values:
                mv, val_ci_low, val_ci_high = mean_ci(values)
                ms, sched_ci_low, sched_ci_high = mean_ci(sched_nums)

                # ===== 关键改动：传输速率 mean + CI =====
                if total_transfer_rates:
                    mt, rate_ci_low, rate_ci_high = mean_ci(total_transfer_rates)
                else:
                    mt, rate_ci_low, rate_ci_high = 0.0, 0.0, 0.0

                mat = mean_ci(avg_transfer_times)[0] if avg_transfer_times else 0.0
                avg_total_tasks = np.mean(total_tasks_list) if total_tasks_list else 0

                network_summary.append({
                    "Network_Scale": scale,
                    "Number_of_Tasks": int(avg_total_tasks),
                    "Mean_Final_Value": round(mv, 3),
                    "Final_Value_CI_Low": round(val_ci_low, 3),
                    "Final_Value_CI_High": round(val_ci_high, 3),
                    "Mean_Scheduled_Tasks": round(ms, 2),
                    "Scheduled_Tasks_CI_Low": round(sched_ci_low, 2),
                    "Scheduled_Tasks_CI_High": round(sched_ci_high, 2),

                    # ===== 新增三列 =====
                    "Mean_Total_Transfer_Rate_Gbps": round(float(mt), 6),
                    "Total_Transfer_Rate_CI_Low": round(float(rate_ci_low), 6),
                    "Total_Transfer_Rate_CI_High": round(float(rate_ci_high), 6),

                    "Mean_Avg_Transfer_Time_sec": round(float(mat), 3),
                    "Seeds_Count": len(SEEDS)
                })
                # 箱线图数据（保留原逻辑）
                boxplot_data.extend([{"Scale": scale, "Value": v} for v in values])


        # 保存结果
        if network_summary:
            df = pd.DataFrame(network_summary).sort_values("Network_Scale")
            df.to_csv(
                f"value-scheduling_task-mean-ci-{ALGO_NAME}-网络规模.csv",
                index=False,
                encoding="utf-8-sig"
            )
            print(f"\n✅ 结果已保存至: value-scheduling_task-mean-ci-{ALGO_NAME}-网络规模.csv")
        if boxplot_data:
            pd.DataFrame(boxplot_data).to_csv(
                f"boxplot_data-{ALGO_NAME}-网络规模.csv",
                index=False,
                encoding="utf-8-sig"
            )
            print(f"✅ 箱线图数据已保存至: boxplot_data-{ALGO_NAME}-网络规模.csv")
        else:
            print("\n❌ 无有效结果可保存")

    print(f"\n🎉 全部仿真完成！算法：{ALGO_NAME} | 模式：{'业务规模' if mode == 1 else '网络规模'}")


if __name__ == "__main__":
    main()