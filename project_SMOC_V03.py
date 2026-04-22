#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块一（V04）：按可见段逐段计算云路径衰减（只提供“Facility输入→分段输出”的接口）
- 输入：Facilityzz-Satellitexxyy.csv（列：VTW, elevation）
- 自动匹配：site{zz}.csv（列与旧模块一一致：时间、云量、CLWC、温度、可选气压）
- 处理：按 VTW 相邻差=1 切分为多个“可见段”，每段独立对齐云数据、仰角，计算逐秒云路径衰减
- 输出：list[DataFrame]，每个 DataFrame 含两列：
    [valid_time_seconds, cloud_path_atten_dB]
"""

import os
import re
import numpy as np
import pandas as pd

# ===== 列名候选 =====
TIME_COLS  = ["valid_time_seconds", "seconds", "time_seconds", "VTW", "vtw", "time"]
COVER_COLS = ["Cloud_cover", "cloud_cover", "cc", "Fraction_of_cloud_cover"]
CLWC_COLS  = ["Cloud_Liquid_Water_Content", "Cloud_liquid_water_content", "clwc"]
PRES_COLS  = ["pressure_level", "pressure", "p"]
TEMP_COLS  = ["temperature", "t", "temp", "T"]

# ===== 物理常数与参数 =====
R_d   = 287.0
rho_w = 1000.0
r_e   = 10e-6      # m
ell_c = 1.0        # km，有效穿云厚度
CUBE_H = 1.0      # km，高度
CUBE_W = 1.0      # km，水平边长
EPS   = 1e-12
EPS_SIN = 1e-3
DEFAULT_PRESSURE_HPA = 500.0
use_cover_as_weight = True

# 与旧模块一保持一致的裁剪
CLIP_COVER = (0.0, 1.0)
CLIP_CLWC  = (0.0, 1e-2)

# 文件名解析：Facilityzz-Satellitexxyy.csv
_FNAME_PAT = re.compile(r"^Facility(?P<zz>\d{2})-Satellite(?P<xx>\d{2})(?P<yy>\d{2})\.csv$", re.IGNORECASE)

# ---------- 小工具 ----------
def _pick(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None

def _req(cols, candidates):
    c = _pick(cols, candidates)
    if c is None:
        raise KeyError(f"缺少列，候选={candidates}，现有={list(cols)}")
    return c

def _load_cloud_csv(path):
    df = pd.read_csv(path)
    tcol = _req(df.columns, TIME_COLS)
    ccol = _req(df.columns, COVER_COLS)
    lcol = _req(df.columns, CLWC_COLS)
    Tcol = _req(df.columns, TEMP_COLS)
    pcol = _pick(df.columns, PRES_COLS)

    df[tcol] = pd.to_numeric(df[tcol], errors="coerce").astype("Int64")
    for c in [ccol, lcol, Tcol] + ([pcol] if pcol else []):
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = (df.dropna(subset=[tcol])
            .drop_duplicates(subset=[tcol])
            .sort_values(tcol))

    df[ccol] = df[ccol].clip(*CLIP_COVER)
    df[lcol] = df[lcol].clip(*CLIP_CLWC)

    df = df.rename(columns={
        tcol: "valid_time_seconds",
        ccol: "Cloud_cover",
        lcol: "Cloud_Liquid_Water_Content",
        Tcol: "temperature"
    })
    if pcol:
        df = df.rename(columns={pcol: "pressure_level"})
    return df

def _load_visibility_csv(path):
    df = pd.read_csv(path)
    tcol = _req(df.columns, ["VTW","vtw","valid_time_seconds","seconds","time"])
    ecol = _req(df.columns, ["elevation","Elevation","elev_deg","elev"])
    df[tcol] = pd.to_numeric(df[tcol], errors="coerce").astype("Int64")
    df[ecol] = pd.to_numeric(df[ecol], errors="coerce")

    df = (df.dropna(subset=[tcol])
            .drop_duplicates(subset=[tcol])
            .sort_values(tcol))
    return df.rename(columns={tcol:"valid_time_seconds", ecol:"elevation_deg"})[[
        "valid_time_seconds","elevation_deg"
    ]]

def _segment_passes(vtw_arr):
    vtw = np.asarray(vtw_arr, dtype=int)
    if vtw.size == 0:
        return []
    cuts = [0]
    for i in range(1, len(vtw)):
        if int(vtw[i]) - int(vtw[i-1]) != 1:
            cuts.append(i)
    cuts.append(len(vtw))
    segs = []
    for i in range(len(cuts)-1):
        s, e = cuts[i], cuts[i+1]-1
        if e >= s:
            segs.append((s, e))
    return segs

def _rho_air(p_hpa, T_K):
    return (np.asarray(p_hpa, float)*100.0) / (R_d*(np.asarray(T_K, float)+EPS))

def _alpha_from_LWC(LWC_kg_m3):
    sigma = 3.0*LWC_kg_m3 / (2.0*rho_w*r_e + EPS)
    return 4.343*sigma*1000.0  # dB/km

def _atten_for_pass(cloud_df, pass_times, pass_elev_deg):
    """在一个可见段上计算云路径衰减（两列 DataFrame）"""
    seconds = pd.Index(np.asarray(pass_times, dtype=int), name="valid_time_seconds")
    df = (cloud_df.set_index("valid_time_seconds")
                  .reindex(seconds)
                  .interpolate(limit_direction="both")
                  .reset_index())

    has_p = ("pressure_level" in df.columns) and (not df["pressure_level"].isna().all())
    rho = _rho_air(df["pressure_level"].to_numpy() if has_p else DEFAULT_PRESSURE_HPA,
                   df["temperature"].to_numpy())

    LWC = df["Cloud_Liquid_Water_Content"].to_numpy() * rho
    alpha = _alpha_from_LWC(LWC)

    # 路径长度与仰角（立方体约束）
    elev_rad = np.deg2rad(np.asarray(pass_elev_deg, float))
    sin_e = np.maximum(np.sin(elev_rad), EPS_SIN)
    cos_e = np.maximum(np.cos(elev_rad), EPS)

    d_max_h = CUBE_H / sin_e       # 受高度限制
    d_max_w = (CUBE_W/2.0) / cos_e # 受水平边长限制
    d_max = np.minimum(d_max_h, d_max_w)

    # 最终有效路径：取 min(原模型, 立方体限制)
    path_geom = ell_c / sin_e
    path_km = np.minimum(path_geom, d_max)

    cover = df["Cloud_cover"].to_numpy()
    attenuation = (cover * alpha * path_km) if use_cover_as_weight else (alpha * path_km)

    return pd.DataFrame({
        "valid_time_seconds": seconds.values.astype(int),
        "cloud_path_atten_dB": attenuation
    })

# ---------- 对外主入口 ----------
def run_stage1_from_facility_csv_segmented(vis_csv_path, cloud_root="."):
    base = os.path.basename(vis_csv_path)
    m = _FNAME_PAT.match(base)
    if not m:
        raise ValueError(f"文件名不符合规则：{base}")
    site_idx = int(m.group("zz"))
    site_csv = os.path.join(cloud_root, f"site{site_idx}.csv")
    if not os.path.exists(site_csv):
        raise FileNotFoundError(f"未找到云数据：{site_csv}")

    vis = _load_visibility_csv(vis_csv_path)
    cloud = _load_cloud_csv(site_csv)

    vtw  = vis["valid_time_seconds"].to_numpy(dtype=int)
    elev = vis["elevation_deg"].to_numpy(dtype=float)
    segs = _segment_passes(vtw)

    out = []
    for s, e in segs:
        pass_times = vtw[s:e+1]
        pass_elev  = elev[s:e+1]
        out.append(_atten_for_pass(cloud, pass_times, pass_elev))
    return out
