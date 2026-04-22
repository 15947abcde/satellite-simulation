#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块三：根据云衰减计算每秒SNR，并结合模块二的阈值表，给每秒分配调制级别（带滞回的状态机判决）。
输出逐秒详表：时间、云衰减、SNR、所属60s窗口、该窗口的阈值（含 final_low/high）、Δ指标、以及调制级别。
level 含义：5=16QAM, 4=8QAM, 3=QPSK, 2=BPSK, 1=RF
"""

import pandas as pd
import numpy as np

# ===== 固定链路参数（可按需修改） =====
TX_POWER_DBM     = 30    # 发射功率 dBm
NOISE_POWER_DBM  = -90   # 接收端噪声功率 dBm
FS_LOSS_DB       = 170   # 星地自由空间损耗 dB
ATM_LOSS_DB      = 10    # 大气损耗 dB（若要 30 dB 无云SNR，则设为 0）
TX_GAIN_DB       = 40    # 发射端增益 dB
RX_GAIN_DB       = 40    # 接收端增益 dB

# 预计算固定项（除云衰减外的链路预算）
FIXED_SNR_NO_CLOUD_DB = (
    TX_POWER_DBM + TX_GAIN_DB + RX_GAIN_DB - FS_LOSS_DB - ATM_LOSS_DB
) - NOISE_POWER_DBM
print(f"[INFO] 基准无云SNR = {FIXED_SNR_NO_CLOUD_DB:.1f} dB "
      f"(Pt={TX_POWER_DBM} dBm, Gt={TX_GAIN_DB} dB, Gr={RX_GAIN_DB} dB, "
      f"Lfs={FS_LOSS_DB} dB, Latm={ATM_LOSS_DB} dB, N={NOISE_POWER_DBM} dBm)")

# ===== 公共键 =====
MCS_KEY_ORDER = ["16-QAM", "8-QAM", "QPSK", "BPSK"]  # 高→低
REQUIRED_THR_COLS = (
    ["window_start_s", "window_end_s", "Delta_star_dB", "Delta_down_dB", "Delta_up_dB"] +
    [f"{k}_final_low_dB"  for k in MCS_KEY_ORDER] +
    [f"{k}_final_high_dB" for k in MCS_KEY_ORDER]
)

# Level 与 MCS 的映射
LEVEL_FROM_MCS = {"16-QAM": 5, "8-QAM": 4, "QPSK": 3, "BPSK": 2}
MCS_FROM_LEVEL = {v: k for k, v in LEVEL_FROM_MCS.items()}
MCS_ORDER_DESC = ["16-QAM", "8-QAM", "QPSK", "BPSK"]  # 高→低

# ================== 校验与SNR计算 ==================
def _ensure_required_columns_stage1(df: pd.DataFrame) -> None:
    req = {"valid_time_seconds", "cloud_path_atten_dB"}
    missing = req - set(df.columns)
    if missing:
        raise KeyError(f"模块三：缺少模块一输出所需列：{missing}")

def _ensure_required_columns_stage2(df: pd.DataFrame) -> None:
    missing = set(REQUIRED_THR_COLS) - set(df.columns)
    if missing:
        raise KeyError(f"模块三：缺少模块二阈值所需列：{missing}")

def compute_snr_series(cloud_df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：包含 [valid_time_seconds, cloud_path_atten_dB]
    输出：增加 'snr_dB' 一列（float）
    公式：SNR[dB] = 基准无云SNR - cloud_path_atten_dB
    """
    _ensure_required_columns_stage1(cloud_df)
    out = cloud_df[["valid_time_seconds", "cloud_path_atten_dB"]].copy()
    out["snr_dB"] = FIXED_SNR_NO_CLOUD_DB - out["cloud_path_atten_dB"].astype(float)
    return out

# ================== 带滞回的状态机判决（核心） ==================
def _decide_with_hysteresis_for_window(
    sec_df: pd.DataFrame,
    thr_row: pd.Series,
    *,
    dwell_sec: int = 0  # 最小驻留秒（0 表示不限制）
) -> np.ndarray:
    """
    在一个 60s 窗口内做逐秒判级（stateful）：
      - 降档：SNR < 当前档 final_low
      - 升档：SNR ≥ 当前档 final_high；若当前为 RF(level=1)，使用 BPSK_final_high
      - dwell_sec：最小驻留时间，限制过于频繁的切换
    返回：该窗口内每秒的 level 数组（5..1；不足BPSK时为 1）
    """
    snr = sec_df["snr_dB"].to_numpy(dtype=float)
    levels_out = np.empty(len(sec_df), dtype=int)

    # 初始档位：选“能满足的最高档”（按 final_low 判断）
    lvl = 1
    for mcs in MCS_ORDER_DESC:
        if snr[0] >= float(thr_row[f"{mcs}_final_low_dB"]):
            lvl = LEVEL_FROM_MCS[mcs]
            break
    levels_out[0] = lvl
    dwell = 0  # 当前档位已驻留的连续秒数

    for i in range(1, len(sec_df)):
        changed = False

        # ---- 降档：若低于当前档 final_low，可逐级往下（直到 RF=1）----
        while lvl > 1:
            cur_mcs  = MCS_FROM_LEVEL[lvl]
            thr_down = float(thr_row[f"{cur_mcs}_final_low_dB"])
            if snr[i] < thr_down and (dwell >= dwell_sec):
                lvl -= 1
                changed = True
                dwell = 0
            else:
                break

        # ---- 升档：只看“本地 final_high”；若在 RF，则用 BPSK 的 final_high 作为门槛 ----
        while lvl < 5:
            if lvl == 1:
                # RF -> BPSK 的门槛
                thr_up = float(thr_row["BPSK_final_high_dB"])
            else:
                cur_mcs = MCS_FROM_LEVEL[lvl]
                thr_up  = float(thr_row[f"{cur_mcs}_final_high_dB"])
            if snr[i] >= thr_up and (dwell >= dwell_sec):
                lvl += 1
                changed = True
                dwell = 0
            else:
                break

        if not changed:
            dwell += 1

        levels_out[i] = lvl

    return levels_out

# ================== 主流程（按窗口广播阈值，并进行状态机判级） ==================
def run_module3(cloud_df: pd.DataFrame, threshold_df: pd.DataFrame) -> pd.DataFrame:
    """
    主入口（供主模块调用）
    输入：
      cloud_df     —— 模块一输出：按秒云衰减 [valid_time_seconds, cloud_path_atten_dB]
      threshold_df —— 模块二输出：每60s窗口阈值（含 *_final_low_dB / *_final_high_dB 与 Δ 指标）
    输出：逐秒详表 DataFrame（列顺序保持不变）
    """
    _ensure_required_columns_stage1(cloud_df)
    _ensure_required_columns_stage2(threshold_df)

    # 1) 每秒 SNR
    per_sec = compute_snr_series(cloud_df)

    # 2) 按窗口进行“带滞回状态机判决”，并把窗口阈值/Δ广播到每秒
    thr_df = threshold_df.reset_index(drop=True).copy()
    rows = []

    for _, w in thr_df.iterrows():
        s, e = int(w["window_start_s"]), int(w["window_end_s"])
        mask = (per_sec["valid_time_seconds"] >= s) & (per_sec["valid_time_seconds"] <= e)
        block = per_sec.loc[mask].copy()
        if block.empty:
            continue

        levels = _decide_with_hysteresis_for_window(block, w, dwell_sec=0)
        block["modulation_level"] = levels

        # 阈值与Δ广播
        block["window_start_s"] = s
        block["window_end_s"]   = e
        block["Delta_star_dB"]  = float(w["Delta_star_dB"])
        block["Delta_down_dB"]  = float(w["Delta_down_dB"])
        block["Delta_up_dB"]    = float(w["Delta_up_dB"])
        for mcs in MCS_KEY_ORDER:
            block[f"{mcs}_final_low_dB"]  = float(w[f"{mcs}_final_low_dB"])
            block[f"{mcs}_final_high_dB"] = float(w[f"{mcs}_final_high_dB"])
        rows.append(block)

    if not rows:
        # 兜底：不在任何窗口，全部视为 RF
        out = per_sec.copy()
        out["window_start_s"] = np.nan
        out["window_end_s"]   = np.nan
        out["Delta_star_dB"]  = np.nan
        out["Delta_down_dB"]  = np.nan
        out["Delta_up_dB"]    = np.nan
        for mcs in MCS_KEY_ORDER:
            out[f"{mcs}_final_low_dB"]  = np.nan
            out[f"{mcs}_final_high_dB"] = np.nan
        out["modulation_level"] = 1
        cols = [
            "valid_time_seconds", "cloud_path_atten_dB", "snr_dB",
            "window_start_s", "window_end_s",
            "Delta_star_dB", "Delta_down_dB", "Delta_up_dB",
            "16-QAM_final_low_dB", "16-QAM_final_high_dB",
            "8-QAM_final_low_dB",  "8-QAM_final_high_dB",
            "QPSK_final_low_dB",   "QPSK_final_high_dB",
            "BPSK_final_low_dB",   "BPSK_final_high_dB",
            "modulation_level",
        ]
        return out[cols]

    out = pd.concat(rows, ignore_index=True).sort_values("valid_time_seconds").reset_index(drop=True)
    cols = [
        "valid_time_seconds",
        "cloud_path_atten_dB",
        "snr_dB",
        "window_start_s", "window_end_s",
        "Delta_star_dB", "Delta_down_dB", "Delta_up_dB",
        "16-QAM_final_low_dB", "16-QAM_final_high_dB",
        "8-QAM_final_low_dB",  "8-QAM_final_high_dB",
        "QPSK_final_low_dB",   "QPSK_final_high_dB",
        "BPSK_final_low_dB",   "BPSK_final_high_dB",
        "modulation_level",
    ]
    return out[cols]
