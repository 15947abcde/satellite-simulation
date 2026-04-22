#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模块四（V04）：逐段碎片化合并（支持非连续全局时间轴；段内时间连续）
- 对“单个可见段”的逐秒等级做：RLE → 碎片化迭代合并 → 短段吸收 → 按该段原时间还原
- 只返回两列：valid_time_seconds, modulation_level
- 提供 segmented 入口：list[DataFrame] -> list[DataFrame]
"""
import numpy as np
import pandas as pd
from typing import Tuple, List

def _rle(vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if vals.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    chg = np.diff(vals, prepend=vals[0]-1) != 0
    starts = np.flatnonzero(chg)
    lens = np.diff(np.append(starts, vals.size))
    return vals[starts].astype(int), lens.astype(int)

def _calc_thr(lens: np.ndarray, q: float) -> float:
    return float(np.percentile(lens.astype(float), q*100)) if len(lens) else 0.0

def _FI(lens: np.ndarray, thr: float, alpha: float, K: int) -> float:
    if len(lens) == 0:
        return 0.0
    L = sorted([float(x) for x in lens], reverse=True)
    topK = L[:K]
    S = [l for l in L if l < thr]
    sumM = float(np.sum(lens))
    if sumM <= 0:
        return 0.0
    return 1.0 - (sum(topK)/sumM) * (1.0 - alpha*(sum(S)/sumM))

def _merge_once(lens: np.ndarray, pris: np.ndarray, thr: float):
    L = lens.astype(float).copy()
    P = pris.astype(int).copy()
    for i in range(len(L)):
        if L[i] < thr:
            li, ri = i-1, i+1
            left_ok  = (li >= 0)        and (P[li] < P[i])
            right_ok = (ri < len(L))    and (P[ri] < P[i])
            if not (left_ok or right_ok):
                continue
            if left_ok and right_ok:
                ld, rd = abs(P[li]-P[i]), abs(P[ri]-P[i])
                if ld < rd: j = li
                elif rd < ld: j = ri
                else: j = li if L[li] >= L[ri] else ri
            else:
                j = li if left_ok else ri

            newL = L[i] + L[j]
            newP = P[i] if L[i] >= L[j] else P[j]
            if j < i:
                L[j] = newL; P[j] = newP
                L = np.delete(L, i); P = np.delete(P, i)
            else:
                L[i] = newL; P[i] = newP
                L = np.delete(L, j); P = np.delete(P, j)
            return L.astype(int), P.astype(int), True
    return L.astype(int), P.astype(int), False

def _simulate(lens: np.ndarray, pris: np.ndarray, q: float, beta: float, alpha: float, K: int, max_iter: int):
    it = 0
    L = lens.astype(int).copy()
    P = pris.astype(int).copy()
    while it < max_iter and len(L) > 1:
        thr = _calc_thr(L, q)
        fi  = _FI(L, thr, alpha, K)
        if fi <= beta:
            break
        L, P, merged = _merge_once(L, P, thr)
        if not merged:
            break
        it += 1
    return L, P

def _absorb_short(L: np.ndarray, P: np.ndarray, min_len=10):
    L = L.astype(int).copy()
    P = P.astype(int).copy()
    if len(L) <= 1:
        return L, P
    changed = True
    while changed and len(L) > 1:
        changed = False
        for i in range(len(L)):
            if L[i] >= min_len:
                continue
            li, ri = i-1, i+1
            if 0 <= li and ri < len(L):
                j = li if L[li] > L[ri] else (ri if L[ri] > L[li] else (li if P[li] < P[ri] else ri))
            elif 0 <= li:
                j = li
            elif ri < len(L):
                j = ri
            else:
                continue
            newL = L[i] + L[j]
            newP = P[j] if L[j] >= L[i] else P[i]
            if j < i:
                L[j] = newL; P[j] = newP
                L = np.delete(L, i); P = np.delete(P, i)
            else:
                L[i] = newL; P[i] = newP
                L = np.delete(L, j); P = np.delete(P, j)
            changed = True
            break
    return L, P

def _expand(times_sorted: np.ndarray, lens: np.ndarray, pris: np.ndarray) -> pd.DataFrame:
    assert int(np.sum(lens)) == len(times_sorted), "RLE长度与时间不符"
    levels = (6 - pris.astype(int)).astype(int)
    out = np.empty(len(times_sorted), dtype=int)
    pos = 0
    for L, lv in zip(lens.astype(int), levels.astype(int)):
        out[pos:pos+L] = lv
        pos += L
    return pd.DataFrame({
        "valid_time_seconds": times_sorted.astype(int),
        "modulation_level":   out
    })

# ---------- 对单段 ----------
def run_module4_one_segment(per_sec_df: pd.DataFrame, *, q: float, beta: float, alpha: float, K: int=2, max_iter: int=100) -> pd.DataFrame:
    """
    对一个“可见段”的逐秒等级进行碎片化处理并返回两列输出。
    约定：per_sec_df 时间升序且连续（段内），列至少有 [valid_time_seconds, modulation_level]。
    """
    df = per_sec_df.sort_values("valid_time_seconds").reset_index(drop=True)
    times = df["valid_time_seconds"].to_numpy(dtype=int)
    lv    = df["modulation_level"].to_numpy(dtype=int)

    vals, lens = _rle(lv)
    pris = (6 - vals).astype(int)

    L, P = _simulate(lens, pris, q, beta, alpha, K, max_iter)
    L, P = _absorb_short(L, P, min_len=10)

    return _expand(times, L, P)

# ---------- 对多段 ----------
def run_module4_segmented(per_second_segments: List[pd.DataFrame], *, q: float, beta: float, alpha: float, K: int=2, max_iter: int=100) -> List[pd.DataFrame]:
    outs = []
    for seg in per_second_segments:
        outs.append(run_module4_one_segment(seg, q=q, beta=beta, alpha=alpha, K=K, max_iter=max_iter))
    return outs
