#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quantile Dual-Threshold + Range Output (fixed paths, no arguments)
"""

import math
from typing import List, Tuple

import numpy as np
import pandas as pd

# ====== 固定路径与参数 ======
INPUT_PATH  = "attenuation.csv"   # 改成你的CSV路径（仅 main() 使用）
OUTPUT_PATH = "threshold.csv"     # 改成你想要的输出路径（仅 main() 使用）

DELAY_SEC   = 2.0
GUARD_SEC   = 7.0
SIGMA_DB    = 0.5
HYST_DB     = 2.0 #滞回阈值
PDOWN       = 0.90 #下置信区间
PUP         = 0.95 #上置信区间

# ====== 固定基础阈值（dB，按阶次） ======
MCS_ORDER = ["BPSK", "QPSK", "8-QAM", "16-QAM"]
GAMMA_STAR_DB = {
    "BPSK": 8.20,
    "QPSK": 12.97,
    "8-QAM": 16.65,
    "16-QAM": 19.96,
}

# ====== 标准正态分位点近似 ======
def invnorm_approx(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 0.97575
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

# ====== 滚动最坏衰减增量 Δ* ======
def rolling_worst_increment(values: np.ndarray, W: int) -> float:
    n = len(values)
    worst = 0.0
    if n <= 1 or W <= 0:
        return 0.0
    for j in range(n):
        max_k = min(W, n - 1 - j)
        base = values[j]
        for k in range(1, max_k + 1):
            inc = values[j + k] - base
            if inc > worst:
                worst = inc
    return max(worst, 0.0)

def compute_thresholds_for_window(vals: np.ndarray,
                                  L: int,
                                  H: int,
                                  sigma: float,
                                  h: float,
                                  p_down: float,
                                  p_up: float) -> Tuple[float, float, float]:
    W = max(int(L) + int(H), 0)
    Delta_star = rolling_worst_increment(vals, W)
    z_down = invnorm_approx(p_down)
    z_up   = invnorm_approx(p_up)
    Delta_down = float(Delta_star + z_down * sigma)
    Delta_up   = float(Delta_star + z_up   * sigma + h)
    return float(Delta_star), Delta_down, Delta_up

# ====== 60s 分窗 ======
def partition_windows(df: pd.DataFrame, win_sec: int = 60) -> List[pd.DataFrame]:
    if df.empty:
        return []
    t0     = int(math.floor(df["valid_time_seconds"].iloc[0]))
    t_last = int(math.floor(df["valid_time_seconds"].iloc[-1]))
    windows = []
    start = t0
    while start <= t_last:
        end = start + win_sec - 1  # inclusive
        mask = (df["valid_time_seconds"] >= start) & (df["valid_time_seconds"] <= end)
        w = df.loc[mask]
        if not w.empty:
            windows.append(w)
        start += win_sec
    return windows

# ====== 主流程（仍支持 CSV） ======
def main():
    usecols = ["valid_time_seconds", "cloud_path_atten_dB"]
    df = pd.read_csv(INPUT_PATH, usecols=usecols)
    df = df.dropna(subset=usecols).copy()
    df["valid_time_seconds"] = pd.to_numeric(df["valid_time_seconds"], errors="coerce")
    df["cloud_path_atten_dB"]          = pd.to_numeric(df["cloud_path_atten_dB"],          errors="coerce")
    df = df.dropna(subset=usecols).sort_values("valid_time_seconds").reset_index(drop=True)
    if df.empty:
        raise SystemExit("输入为空：解析后无有效数据行。")

    L = int(math.ceil(DELAY_SEC / 1.0))
    H = int(math.ceil(GUARD_SEC / 1.0))
    SIGMA = float(SIGMA_DB)
    HYST  = float(HYST_DB)

    wins = partition_windows(df, win_sec=60)
    if not wins:
        raise SystemExit("未形成任何 60s 窗口，请检查时间戳。")

    rows = []
    for w in wins:
        t_start = float(w["valid_time_seconds"].iloc[0])
        t_end   = float(w["valid_time_seconds"].iloc[-1])
        vals    = w["cloud_path_atten_dB"].to_numpy(dtype=float)

        Delta_star, Delta_down, Delta_up = compute_thresholds_for_window(
            vals, L=L, H=H, sigma=SIGMA, h=HYST, p_down=PDOWN, p_up=PUP
        )

        out = {
            "window_start_s": t_start,
            "window_end_s":   t_end,
            "num_samples":    int(len(vals)),
            "Delta_star_dB":  round(Delta_star, 6),
            "Delta_down_dB":  round(Delta_down, 6),
            "Delta_up_dB":    round(Delta_up,   6),
            "p_down": PDOWN,
            "p_up":   PUP,
        }

        for i, mcs in enumerate(MCS_ORDER):
            gamma_i   = GAMMA_STAR_DB[mcs]
            base_low  = gamma_i
            final_low = gamma_i + Delta_down

            if i + 1 < len(MCS_ORDER):
                gamma_next = GAMMA_STAR_DB[MCS_ORDER[i + 1]]
                base_high  = gamma_next
                final_high = gamma_next + Delta_up
            else:
                base_high  = 1e9
                final_high = 1e9

            prefix = mcs.replace(" ", "").replace("/", "-")
            out[f"{prefix}_base_low_dB"]   = round(base_low,  6)
            out[f"{prefix}_final_low_dB"]  = round(final_low, 6)
            out[f"{prefix}_base_high_dB"]  = round(base_high, 6)
            out[f"{prefix}_final_high_dB"] = round(final_high,6)

        rows.append(out)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out_df)} window rows to: {OUTPUT_PATH}")

# ===== 新增：内存式联通接口（不改动 main()） =====
def compute_thresholds_from_df(stage1_df: pd.DataFrame) -> pd.DataFrame:
    """
    输入：DataFrame，需包含：
         - valid_time_seconds
         - cloud_path_atten_dB
    输出：阈值窗口结果的 DataFrame（与写 CSV 前的 out_df 一致），不落盘。
    """
    usecols = ["valid_time_seconds", "cloud_path_atten_dB"]
    df = stage1_df[usecols].copy()
    df = df.dropna(subset=usecols).copy()
    df["valid_time_seconds"]  = pd.to_numeric(df["valid_time_seconds"], errors="coerce")
    df["cloud_path_atten_dB"] = pd.to_numeric(df["cloud_path_atten_dB"], errors="coerce")
    df = df.dropna(subset=usecols).sort_values("valid_time_seconds").reset_index(drop=True)
    if df.empty:
        raise SystemExit("输入为空：解析后无有效数据行。")

    L = int(math.ceil(DELAY_SEC / 1.0))
    H = int(math.ceil(GUARD_SEC / 1.0))
    SIGMA = float(SIGMA_DB)
    HYST  = float(HYST_DB)

    wins = partition_windows(df, win_sec=60)
    if not wins:
        raise SystemExit("未形成任何 60s 窗口，请检查时间戳。")

    rows = []
    for w in wins:
        t_start = float(w["valid_time_seconds"].iloc[0])
        t_end   = float(w["valid_time_seconds"].iloc[-1])
        vals    = w["cloud_path_atten_dB"].to_numpy(dtype=float)

        Delta_star, Delta_down, Delta_up = compute_thresholds_for_window(
            vals, L=L, H=H, sigma=SIGMA, h=HYST, p_down=PDOWN, p_up=PUP
        )

        out = {
            "window_start_s": t_start,
            "window_end_s":   t_end,
            "num_samples":    int(len(vals)),
            "Delta_star_dB":  round(Delta_star, 6),
            "Delta_down_dB":  round(Delta_down, 6),
            "Delta_up_dB":    round(Delta_up,   6),
            "p_down": PDOWN,
            "p_up":   PUP,
        }

        for i, mcs in enumerate(MCS_ORDER):
            gamma_i   = GAMMA_STAR_DB[mcs]
            base_low  = gamma_i
            final_low = gamma_i + Delta_down

            if i + 1 < len(MCS_ORDER):
                gamma_next = GAMMA_STAR_DB[MCS_ORDER[i + 1]]
                base_high  = gamma_next
                final_high = gamma_next + Delta_up
            else:
                base_high  = 1e9
                final_high = 1e9

            prefix = mcs.replace(" ", "").replace("/", "-")
            out[f"{prefix}_base_low_dB"]   = round(base_low,  6)
            out[f"{prefix}_final_low_dB"]  = round(final_low, 6)
            out[f"{prefix}_base_high_dB"]  = round(base_high, 6)
            out[f"{prefix}_final_high_dB"] = round(final_high,6)

        rows.append(out)

    return pd.DataFrame(rows)

if __name__ == "__main__":
    main()
