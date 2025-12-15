# make_features.py
# db_drawing.csv の points(JSON) から特徴量を作り、円らしさフィルタをかけて保存する（筆圧なし）
# 重要：
#  - 曲率系：空なら 0.0（NaNにしない） ← A案
#  - early/late 速度比：末尾停止に強い計算（NaNにしない）

import pandas as pd
import numpy as np
import json
import math

# ======================================
# 設定
# ======================================

CSV_PATH_IN = "db_drawing.csv"
CSV_PATH_OUT_ALL = "features_drawings.csv"
CSV_PATH_OUT_FILT = "features_drawings_filtered.csv"

# 円らしさフィルタ（roundnessは制限しない）
LINEARITY_MIN = 0.2
CLOSURE_MAX = 0.8


# ======================================
# ユーティリティ
# ======================================

def parse_points(points_str):
    """points列(JSON文字列)を list[dict] に安全に変換"""
    if not isinstance(points_str, str):
        return []
    try:
        pts = json.loads(points_str)
        return pts if isinstance(pts, list) else []
    except Exception:
        return []


def is_trap_points(pts):
    """trap試行を points だけで判定"""
    if not pts:
        return False

    p0 = pts[0]
    if isinstance(p0, dict) and p0.get("x") == -1 and p0.get("y") == -1:
        return True

    for p in pts:
        if isinstance(p, dict) and p.get("trap") is True:
            return True

    return False


def safe_skew(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def safe_kurtosis(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size < 3:
        return 0.0
    m = x.mean()
    s = x.std()
    if s == 0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 4))


# ======================================
# 1描画分の特徴量計算（筆圧なし）
# ======================================

def compute_features_for_points(pts):
    if len(pts) < 3:
        return None

    # ---- 生データ配列に変換 ----
    try:
        xs = np.array([p["x"] for p in pts], dtype=float)
        ys = np.array([p["y"] for p in pts], dtype=float)
    except Exception:
        return None

    # 時刻 t（ない場合は NaN）
    ts = np.array([p.get("t", np.nan) for p in pts], dtype=float)

    # =====================================
    # 幾何形状系
    # =====================================

    cx = float(xs.mean())
    cy = float(ys.mean())

    dists = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2)
    r_min = float(dists.min())
    r_max = float(dists.max())
    r_std = float(dists.std())

    if r_max <= 0:
        return None

    roundness = float(r_min / r_max)

    x_min, x_max = float(xs.min()), float(xs.max())
    y_min, y_max = float(ys.min()), float(ys.max())
    bbox_w = float(x_max - x_min)
    bbox_h = float(y_max - y_min)
    bbox_area = float(bbox_w * bbox_h)
    aspect_ratio = float(bbox_w / bbox_h) if bbox_h > 0 else 0.0

    # 離心率近似（共分散の固有値）
    cov = np.cov(np.vstack([xs, ys]))
    eigvals, _ = np.linalg.eig(cov)
    eigvals = np.sort(np.real(eigvals))
    if eigvals[1] > 0:
        eccentricity = float(np.sqrt(max(0.0, 1.0 - eigvals[0] / eigvals[1])))
    else:
        eccentricity = 0.0

    # 閉じ具合
    start_x, start_y = float(xs[0]), float(ys[0])
    end_x, end_y = float(xs[-1]), float(ys[-1])
    closure_dist = float(math.hypot(start_x - end_x, start_y - end_y))
    closure_ratio = float(closure_dist / r_max)

    # 直線性（x,y分散比）
    std_x = float(xs.std())
    std_y = float(ys.std())
    if max(std_x, std_y) > 0:
        linearity = float(min(std_x, std_y) / max(std_x, std_y))
    else:
        linearity = 0.0

    # パス長
    pts_xy = np.stack([xs, ys], axis=1)
    step_vec = np.diff(pts_xy, axis=0)
    step_dist = np.linalg.norm(step_vec, axis=1)
    path_length = float(step_dist.sum())

    # 面積（shoelace）
    try:
        area = 0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1)))
        area = float(area)
    except Exception:
        area = 0.0

    # 円形度（4πA / P^2）
    if path_length > 0:
        circularity = float(4 * math.pi * area / (path_length ** 2))
    else:
        circularity = 0.0

    # =====================================
    # 時間・速度系
    # =====================================

    n_points = int(len(xs))

    if np.all(np.isfinite(ts)):
        duration = float(ts.max() - ts.min())
        dt_raw = np.diff(ts)
        dt_for_stroke = dt_raw.copy()
        dt_safe = np.where(dt_raw > 0, dt_raw, 1.0)  # dt<=0は保険
    else:
        duration = 0.0
        dt_for_stroke = None
        dt_safe = np.ones_like(step_dist)

    if len(step_dist) > 0:
        speed = step_dist / dt_safe
        mean_speed = float(speed.mean())
        median_speed = float(np.median(speed))
        max_speed = float(speed.max())
        min_speed = float(speed.min())
        speed_std = float(speed.std())
        speed_cv = float(speed_std / mean_speed) if mean_speed > 0 else 0.0
        speed_skew = safe_skew(speed)
        speed_kurt = safe_kurtosis(speed)
    else:
        speed = np.array([], dtype=float)
        mean_speed = median_speed = max_speed = min_speed = speed_std = 0.0
        speed_cv = speed_skew = speed_kurt = 0.0

    # 加速度
    if speed.size > 1:
        accel = np.diff(speed)
        accel_mean = float(accel.mean())
        accel_std = float(accel.std())
    else:
        accel_mean = 0.0
        accel_std = 0.0

    # jerk
    if speed.size > 2:
        jerk = np.diff(np.diff(speed))
        jerk_mean = float(jerk.mean()) if jerk.size else 0.0
        jerk_std = float(jerk.std()) if jerk.size else 0.0
    else:
        jerk_mean = 0.0
        jerk_std = 0.0

    # ---- early/late（末尾停止に強い版：NaNを作らない）----
    eps = 1e-6
    stop_speed_thresh = 1e-6
    min_valid = 5

    if speed.size >= min_valid:
        speed_move = speed[speed > stop_speed_thresh]

        if speed_move.size < min_valid:
            early_speed_mean = float(mean_speed)
            late_speed_mean = 0.0
            early_late_speed_ratio = 0.0
        else:
            k = max(1, speed_move.size // 10)
            early_speed_mean = float(speed_move[:k].mean())
            late_speed_mean = float(speed_move[-k:].mean())
            early_late_speed_ratio = float(early_speed_mean / (late_speed_mean + eps))
    else:
        early_speed_mean = 0.0
        late_speed_mean = 0.0
        early_late_speed_ratio = 0.0

    # =====================================
    # 曲率・方向変化系（A案：空なら 0.0）
    # =====================================

    curvatures = []
    direction_change_count = 0
    angle_threshold = np.deg2rad(20)

    for i in range(1, len(pts_xy) - 1):
        p_prev = pts_xy[i - 1]
        p_curr = pts_xy[i]
        p_next = pts_xy[i + 1]

        v1 = p_curr - p_prev
        v2 = p_next - p_curr

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = math.acos(cos_angle)
        curvatures.append(angle)

        if abs(angle) >= angle_threshold:
            direction_change_count += 1

    if len(curvatures) > 0:
        curvatures = np.array(curvatures, dtype=float)
        curvature_mean = float(curvatures.mean())
        curvature_std = float(curvatures.std())
        curvature_abs_mean = float(np.abs(curvatures).mean())
    else:
        curvature_mean = 0.0
        curvature_std = 0.0
        curvature_abs_mean = 0.0

    # =====================================
    # ストローク・ポーズ系
    # =====================================

    if dt_for_stroke is not None and dt_for_stroke.size > 0 and np.all(np.isfinite(dt_for_stroke)):
        pause_threshold = 200  # ms想定
        pauses = dt_for_stroke > pause_threshold
        pause_count = int(np.sum(pauses))
        stroke_count = int(1 + pause_count)
    else:
        pause_count = 0
        stroke_count = 1

    return {
        # 幾何・形状系
        "centroid_x": cx,
        "centroid_y": cy,
        "inner_radius": r_min,
        "outer_radius": r_max,
        "radius_std": r_std,
        "roundness": roundness,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "bbox_area": bbox_area,
        "aspect_ratio": aspect_ratio,
        "eccentricity": eccentricity,
        "closure_dist": closure_dist,
        "closure_ratio": closure_ratio,
        "linearity": linearity,
        "path_length": path_length,
        "area": area,
        "circularity": circularity,

        # 時間・速度系
        "n_points": n_points,
        "duration": duration,
        "mean_speed": mean_speed,
        "median_speed": median_speed,
        "max_speed": max_speed,
        "min_speed": min_speed,
        "speed_std": speed_std,
        "speed_cv": speed_cv,
        "speed_skew": speed_skew,
        "speed_kurtosis": speed_kurt,
        "accel_mean": accel_mean,
        "accel_std": accel_std,
        "jerk_mean": jerk_mean,
        "jerk_std": jerk_std,
        "early_speed_mean": early_speed_mean,
        "late_speed_mean": late_speed_mean,
        "early_late_speed_ratio": early_late_speed_ratio,

        # 曲率・方向変化系
        "curvature_mean": curvature_mean,
        "curvature_std": curvature_std,
        "curvature_abs_mean": curvature_abs_mean,
        "direction_change_count": direction_change_count,

        # ストローク・ポーズ系
        "stroke_count": stroke_count,
        "pause_count": pause_count,
    }


# ======================================
# メイン処理
# ======================================

def main():
    df = pd.read_csv(CSV_PATH_IN)
    print("Loaded rows:", len(df))

    records = []

    for idx, row in df.iterrows():
        val = row.get("valence")
        aro = row.get("arousal")

        # ラベルがない行は学習に使わない
        if pd.isna(val) or pd.isna(aro):
            continue

        pts = parse_points(row.get("points"))

        # trap 除外
        if is_trap_points(pts):
            continue

        feats = compute_features_for_points(pts)
        if feats is None:
            continue

        rec = {
            "row_index": idx,
            "id": row.get("id", np.nan),
            "user_id": row.get("user_id", np.nan),
            "trial_index": row.get("trial_index", np.nan),
            "valence": float(val),
            "arousal": float(aro),
        }
        rec.update(feats)
        records.append(rec)

    df_feat = pd.DataFrame(records)
    print("Usable feature rows:", len(df_feat))
    print("Feature columns:", len(df_feat.columns))
    print("Columns:", list(df_feat.columns))

    # フィルタ前
    df_feat.to_csv(CSV_PATH_OUT_ALL, index=False)
    print("Saved ALL features to:", CSV_PATH_OUT_ALL)

    # 円らしさフィルタ（roundnessは制限しない）
    mask = pd.Series(True, index=df_feat.index)
    mask &= df_feat["linearity"].notna()
    mask &= df_feat["closure_ratio"].notna()
    mask &= df_feat["linearity"] >= LINEARITY_MIN
    mask &= df_feat["closure_ratio"] <= CLOSURE_MAX

    df_feat_filt = df_feat[mask].copy()
    print("Filtered rows:", len(df_feat_filt))

    df_feat_filt.to_csv(CSV_PATH_OUT_FILT, index=False)
    print("Saved FILTERED features to:", CSV_PATH_OUT_FILT)

    # NaNチェック（重要列）
    check_cols = ["curvature_mean", "curvature_std", "curvature_abs_mean", "early_late_speed_ratio"]
    if all(c in df_feat_filt.columns for c in check_cols):
        print("\n[NaN Check in filtered CSV]")
        print(df_feat_filt[check_cols].isna().sum())


if __name__ == "__main__":
    main()
