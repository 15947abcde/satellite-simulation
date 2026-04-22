import os
import random
import csv
import shutil
import re
from collections import defaultdict

# ======================================================
# 配置参数
# ======================================================

# ✅ 场景：每颗卫星固定多少条业务（你要对比的核心变量）
TASKS_PER_SAT = 50  # 改成 70/80/90/100 做不同业务规模场景对比

# 输出目录：业务文件（带到达时间）
BUSINESS_OUT_DIR = f"业务部署-30卫星-{TASKS_PER_SAT}业务"

# 可见性文件目录（可能有多级子目录）
VISIBILITY_DIR = "星地可见性关系-调制格式分级"

# 输出目录：卫星部署（复制来的可见性文件，不带到达时间）
SAT_DEPLOY_DIR = f"卫星部署-30卫星-{TASKS_PER_SAT}业务"

TOTAL_SELECTED_SATS = 30 #卫星数量

# ======================================================
# ✅【仅修改这一部分逻辑】BUSINESS_TRAFFIC 随 TASKS_PER_SAT 动态递增
# 50业务: {400: 15, 600: 10, 800: 5}
# 60业务: {450: 15, 650: 10, 850: 5}
# 70业务: {500: 15, 700: 10, 900: 5}
# ... 每增加10条业务 -> 每个key + 50
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

# ✅ 动态生成 BUSINESS_TRAFFIC（原来写死的那一行被替换为这一行）
BUSINESS_TRAFFIC = build_dynamic_business_traffic(TASKS_PER_SAT)

MIN_ARRIVAL_TIME = 32400
MAX_ARRIVAL_TIME = 36000

MAX_TRAFFIC_PER_ROW = 20.0   # 每条业务最大 20GB

# ✅ 是否要求 Facility–Satellite 组合唯一（防止复制覆盖导致数量不对）
REQUIRE_UNIQUE_VISIBILITY_PAIR = True

# ✅ 如果选到某颗卫星在目录里没有任何可见 facility，是否允许换卫星
ALLOW_REPLACE_SAT_IF_NO_FACILITY = True


# ======================================================
# 生成卫星全集
# ======================================================

def create_all_satellites():
    """生成320颗卫星的唯一标识（16轨道×20卫星），如 '0101'"""
    return [f"{orbit:02d}{sat:02d}" for orbit in range(1, 17) for sat in range(1, 21)]


# ======================================================
# 业务总量分配（30份，恰好对应30颗卫星）
# ======================================================

def build_total_traffic_list():
    lst = []
    for traffic, count in BUSINESS_TRAFFIC.items():
        lst.extend([traffic] * count)
    random.shuffle(lst)
    return lst


# ======================================================
# ✅ 生成：每颗卫星固定 TASKS_PER_SAT 条业务，总量精确守恒
# ======================================================

def generate_fixed_count_tasks(total_traffic_gb: float, n_tasks: int):
    """
    返回 rows: List[[value, traffic_gb], ...] 长度严格等于 n_tasks
    约束：
      - 每条 traffic_gb ∈ [0.01, MAX_TRAFFIC_PER_ROW]
      - sum(traffic_gb) == total_traffic_gb（保留两位小数）
    """
    total_traffic_gb = round(float(total_traffic_gb), 2)

    # 可行性检查：n_tasks 条，每条最多 20GB
    if total_traffic_gb > n_tasks * MAX_TRAFFIC_PER_ROW:
        raise ValueError(
            f"不可行：总量 {total_traffic_gb}GB 超过 n_tasks*MAX_TRAFFIC_PER_ROW = "
            f"{n_tasks}*{MAX_TRAFFIC_PER_ROW}={n_tasks*MAX_TRAFFIC_PER_ROW}GB"
        )
    # 也要保证每条至少 0.01GB
    if total_traffic_gb < n_tasks * 0.01:
        raise ValueError(
            f"不可行：总量 {total_traffic_gb}GB 小于 n_tasks*0.01 = {n_tasks*0.01}GB"
        )

    rows = []
    remaining = total_traffic_gb

    for i in range(n_tasks - 1):
        remaining_slots = n_tasks - i

        # 后面每条至少 0.01
        min_remain_after = (remaining_slots - 1) * 0.01
        # 后面每条最多 20
        max_remain_after = (remaining_slots - 1) * MAX_TRAFFIC_PER_ROW

        # 当前这条的可取范围：
        # 至少要让“剩余”不超过后面最大承载
        min_needed = max(0.01, remaining - max_remain_after)
        # 至多要让“剩余”不少于后面最小承载
        max_allowed = min(MAX_TRAFFIC_PER_ROW, remaining - min_remain_after)

        # 数值保护
        min_needed = round(min_needed, 2)
        max_allowed = round(max_allowed, 2)
        if min_needed > max_allowed:
            # 极端情况下由于四舍五入导致区间颠倒，直接取边界
            traffic = max_allowed
        else:
            traffic = round(random.uniform(min_needed, max_allowed), 2)

        rows.append([random.randint(1, 10), traffic])
        remaining = round(remaining - traffic, 2)

    # 最后一条兜底
    last = round(remaining, 2)
    if last < 0.01:
        last = 0.01
    if last > MAX_TRAFFIC_PER_ROW:
        last = MAX_TRAFFIC_PER_ROW
    rows.append([random.randint(1, 10), last])

    # 为了严格守恒，做一次微调（把误差加到最后一条）
    s = round(sum(r[1] for r in rows), 2)
    diff = round(total_traffic_gb - s, 2)
    if abs(diff) >= 0.01:
        # 尝试把 diff 加到最后一条（仍需满足范围）
        new_last = round(rows[-1][1] + diff, 2)
        if 0.01 <= new_last <= MAX_TRAFFIC_PER_ROW:
            rows[-1][1] = new_last
        else:
            # 如果最后一条加不了，就把 diff 分摊给多条（尽量简单处理）
            for k in range(len(rows)):
                newv = round(rows[k][1] + diff, 2)
                if 0.01 <= newv <= MAX_TRAFFIC_PER_ROW:
                    rows[k][1] = newv
                    break

    # 排序（你原来双重降序）
    rows_sorted = sorted(rows, key=lambda x: (-x[0], -x[1]))
    rows_sorted.insert(0, ["价值量", "业务量(GB)"])
    return rows_sorted


# ======================================================
# 可见性索引（递归扫描）
# ======================================================

VIS_PATTERN = re.compile(
    r"Facility\s*[_-]?(\d{1,2})\s*[_-]?Satellite\s*[_-]?(\d{3,4})\.csv",
    re.IGNORECASE
)

def index_visibility_files(root_dir):
    vis_map = defaultdict(list)  # vis_map[sat_id] -> [(facility_id, path), ...]
    vis_path_map = {}           # vis_path_map[(facility_id, sat_id)] -> path

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
# ✅ 选择 30 颗卫星并为每颗卫星找到一个真实存在的 facility（找满为止）
# ======================================================

def select_30_sat_facility_pairs(all_sats, total_traffic_list, vis_map):
    """
    返回 records: [(sat_id, total_traffic, facility_id, arrival_time), ...] 长度=30
    """
    need = len(total_traffic_list)  # 30

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
            # 该卫星可选facility都用完了，尝试换卫星
            if not ALLOW_REPLACE_SAT_IF_NO_FACILITY:
                raise RuntimeError(f"卫星 {sat_id} 找不到可用 facility（唯一性约束下）")
            # 换卫星直到找到可用
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

def generate_and_copy(records, vis_path_map):
    os.makedirs(BUSINESS_OUT_DIR, exist_ok=True)
    os.makedirs(SAT_DEPLOY_DIR, exist_ok=True)

    for idx, (sat_id, total_traffic, facility_id, arrival_time) in enumerate(records, 1):
        orbit_xx = sat_id[:2]
        sat_yy = sat_id[2:]

        # 业务文件（带到达时间）
        business_file_name = f"Facility{facility_id}-Satellite{orbit_xx}{sat_yy}-{arrival_time}.csv"
        business_file_path = os.path.join(BUSINESS_OUT_DIR, business_file_name)

        csv_data = generate_fixed_count_tasks(total_traffic, TASKS_PER_SAT)
        with open(business_file_path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(csv_data)

        # 可见性文件（不带时间）
        visibility_name = f"Facility{facility_id}-Satellite{orbit_xx}{sat_yy}.csv"
        src_vis_path = vis_path_map.get((facility_id, sat_id))
        if not src_vis_path or not os.path.exists(src_vis_path):
            raise RuntimeError(f"内部错误：预期存在可见性文件但未找到：{visibility_name}")

        dst_vis_path = os.path.join(SAT_DEPLOY_DIR, visibility_name)
        shutil.copy2(src_vis_path, dst_vis_path)

        print(
            f"[{idx}/{len(records)}] 业务: {business_file_name} "
            f"(总量{total_traffic}GB, 条数{TASKS_PER_SAT}) | 可见性: {visibility_name} ✅"
        )

    print("\n========== 完成 ==========")
    print(f"业务文件输出目录：{os.path.abspath(BUSINESS_OUT_DIR)}")
    print(f"可见性文件来源目录：{os.path.abspath(VISIBILITY_DIR)}")
    print(f"卫星部署输出目录：{os.path.abspath(SAT_DEPLOY_DIR)}")
    print(f"成功复制可见性文件：{len(records)} 个（已找满）")
    print(f"每颗卫星业务条数：{TASKS_PER_SAT}（严格固定）")


# ======================================================
# main
# ======================================================

if __name__ == "__main__":
    vis_map, vis_path_map = index_visibility_files(VISIBILITY_DIR)

    total_traffic_list = build_total_traffic_list()  # 30份：动态 BUSINESS_TRAFFIC
    all_sats = create_all_satellites()

    records = select_30_sat_facility_pairs(all_sats, total_traffic_list, vis_map)

    generate_and_copy(records, vis_path_map)
