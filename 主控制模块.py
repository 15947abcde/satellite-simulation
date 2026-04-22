# main_pipeline.py
import os
import re
import pandas as pd

# ====== 路径配置 ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== 路径配置（全部相对于 Satellite experiment） ======
VIS_ROOT = os.path.join(BASE_DIR, "星地可见性关系-STK清洗数据")
CLOUD_ROOT = os.path.join(BASE_DIR, "星地可见性关系-云数据")  # 如果你有这个目录
OUT_DIR = os.path.join(BASE_DIR, "星地可见性关系-调制格式分级")

os.makedirs(OUT_DIR, exist_ok=True)

# ====== 导入各模块（按你给定的版本号） ======
import project_SMOC_V03 as smoc               # 模块一：按可见段输出 list[pd.DataFrame]（两列：time, cloud_path_atten_dB）
import project_Threshold_V02 as thr           # 模块二：对“单段 DataFrame”计算 60s 阈值
import project_ModulationSelector_V03 as mod3 # 模块三：单段逐秒 SNR→modulation_level（带滞回）
import project_Fragmentation_V03 as mod4      # 模块四：碎片化（提供单段/多段接口）

# ====== 文件名匹配 ======
_FNAME_PAT = re.compile(
    r"^Facility(?P<zz>\d{2})-Satellite(?P<xx>\d{2})(?P<yy>\d{2})\.csv$",
    re.IGNORECASE
)

def _iter_facility_files(root: str):
    """遍历目录下所有 Facility??-Satellite????.csv"""
    if not os.path.isdir(root):
        return
    for name in sorted(os.listdir(root)):
        if _FNAME_PAT.match(name):
            yield os.path.join(root, name)

def _basename_of(path: str) -> str:
    """Facilityzz-Satellitexxyy 不带扩展名"""
    return os.path.splitext(os.path.basename(path))[0]

# ====== 单个 Facility（按“VTW 可见段”逐段处理） ======
def run_full_pipeline_for_facility(
    vis_csv_path: str, *,
    q: float = 0.6, beta: float = 0.15, alpha: float = 0.6,
    K: int = 2, max_iter: int = 1000
) -> pd.DataFrame:
    """
    输入：单个 Facilityzz-Satellitexxyy.csv（列：VTW, elevation）
    处理：按“可见段”（相邻 VTW 差=1）逐段执行 模块1→2→3→4，段间互不影响
    输出：拼接后的逐秒结果（两列：valid_time_seconds, modulation_level）
    """

    # 模块一：得到“每段”的逐秒云衰减列表（每段两列：time, cloud_path_atten_dB）
    stage1_segments = smoc.run_stage1_from_facility_csv_segmented(
        vis_csv_path, cloud_root=CLOUD_ROOT
    )
    if not stage1_segments:
        return pd.DataFrame(columns=["valid_time_seconds", "modulation_level"])

    # —— 逐段跑模块二、三 —— #
    stage3_segments = []
    for s1 in stage1_segments:
        if s1.empty:
            continue
        # 模块二：该段 60s 窗口阈值
        s2 = thr.compute_thresholds_from_df(s1)
        # 模块三：该段逐秒 SNR→等级（带滞回；阈值广播到秒）
        s3 = mod3.run_module3(s1, s2)
        stage3_segments.append(s3)

    if not stage3_segments:
        return pd.DataFrame(columns=["valid_time_seconds", "modulation_level"])

    # —— 模块四：分段碎片化（输入为 list[pd.DataFrame]，输出 list[pd.DataFrame]）—— #
    merged_segments = mod4.run_module4_segmented(
        stage3_segments, q=q, beta=beta, alpha=alpha, K=K, max_iter=max_iter
    )

    # 拼接全部段（不跨段再合并）
    out = pd.concat(merged_segments, ignore_index=True)
    out = out.sort_values("valid_time_seconds").reset_index(drop=True)
    return out[["valid_time_seconds", "modulation_level"]]

def save_timeslots_csv(per_second_df: pd.DataFrame, base_name: str, out_dir: str = OUT_DIR) -> str:
    """
    将逐秒两列结果落盘；输出文件名与 Facility 同名。
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{base_name}.csv")
    per_second_df[["valid_time_seconds", "modulation_level"]].to_csv(out_path, index=False)
    return out_path

# ====== 批处理入口：遍历目录中所有 Facility 文件 ======
def run_batch(
    vis_root: str = VIS_ROOT, *,
    q: float = 0.6, beta: float = 0.15, alpha: float = 0.6,
    K: int = 2, max_iter: int = 1000
) -> pd.DataFrame:
    """
    遍历 vis_root 下所有 Facility??-Satellite????.csv，逐个按“VTW 分段流程”处理并落盘。
    返回汇总表：文件名、输出秒数、输出路径、错误信息。
    """
    files = list(_iter_facility_files(vis_root))
    if not files:
        print(f"[WARN] 在目录 {vis_root} 下未找到 Facility??-Satellite????.csv")
        return pd.DataFrame(columns=["facility", "num_seconds", "output_path", "error"])

    rows = []
    for i, vis_csv in enumerate(files, 1):
        base = _basename_of(vis_csv)
        try:
            per_sec = run_full_pipeline_for_facility(
                vis_csv, q=q, beta=beta, alpha=alpha, K=K, max_iter=max_iter
            )
            out_path = save_timeslots_csv(per_sec, base_name=base, out_dir=OUT_DIR)
            rows.append({"facility": base, "num_seconds": len(per_sec), "output_path": out_path, "error": ""})
            print(f"[{i}/{len(files)}] ✅ {base}: 写出 {out_path}（{len(per_sec)} 秒）")
        except Exception as e:
            rows.append({"facility": base, "num_seconds": 0, "output_path": "", "error": str(e)})
            print(f"[{i}/{len(files)}] ❌ {base}: {e}")

    return pd.DataFrame(rows)

# ====== 可选：查看“单段”的第一个 60s 窗口详情（调试辅助） ======
def show_first_window_view_for_segment(stage3_df: pd.DataFrame) -> None:
    """
    显示单段数据的第一个 60s 窗口阈值与逐秒详情（前提：模块三已广播窗口阈值到秒）。
    """
    if stage3_df.empty or "window_start_s" not in stage3_df.columns:
        print("[WARN] 无窗口广播数据可展示。")
        return

    first_ws = stage3_df["window_start_s"].dropna().astype(int).min()
    block = stage3_df[stage3_df["window_start_s"] == first_ws].copy()
    if block.empty:
        print("[WARN] 未找到任何窗口记录。")
        return

    def fmt(x):
        try:
            return f"{float(x):.3f}"
        except Exception:
            return "NaN"

    one = block.iloc[0]
    print("\n=== 第一个 60s 窗口的切换阈值（当前段） ===")
    print(f"窗口区间: [{int(one['window_start_s'])}, {int(one['window_end_s'])}]")
    for mcs in ["BPSK", "QPSK", "8-QAM", "16-QAM"]:
        low  = fmt(one.get(f"{mcs}_final_low_dB"))
        high = fmt(one.get(f"{mcs}_final_high_dB"))
        print(f"{mcs:7s} : low={low} dB, high={high} dB")
    print(f"Delta*: {fmt(one.get('Delta_star_dB'))} dB, "
          f"Delta_down: {fmt(one.get('Delta_down_dB'))} dB, "
          f"Delta_up: {fmt(one.get('Delta_up_dB'))} dB")

    view_cols = [c for c in ["valid_time_seconds", "cloud_path_atten_dB", "snr_dB", "modulation_level"]
                 if c in block.columns]
    print("\n=== 该窗口内逐秒（窗口全体或前 60 行） ===")
    print(block[view_cols].to_string(index=False))

# ====== 示例运行 ======
if __name__ == "__main__":
    summary = run_batch(
        vis_root=VIS_ROOT,
        q=0.6,     # 分位数
        beta=0.5, # 碎片化阈值
        alpha=0.6, # alpha
        K=2,
        max_iter=1000
    )
    print("\n=== 批处理汇总（前 20 行） ===")
    print(summary.head(20))
