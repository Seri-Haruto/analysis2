import math
import numpy as np
import joblib

MODEL_PATH_AROUSAL = "rf_arousal.pkl"
MODEL_PATH_VALENCE = "rf_valence.pkl"

LINEARITY_MIN = 0.2
CLOSURE_MAX = 0.8

# 学習に使った特徴量の順番（random_forest_models.pyのfeature_colsと一致させる）
FEATURE_COLS = [
    "centroid_x", "centroid_y",
    "inner_radius", "outer_radius", "radius_std", "roundness",
    "bbox_w", "bbox_h", "bbox_area", "aspect_ratio",
    "eccentricity",
    "closure_dist", "closure_ratio", "linearity",
    "path_length", "area", "circularity",
    "n_points", "duration",
    "mean_speed", "median_speed", "max_speed", "min_speed",
    "speed_std", "speed_cv", "speed_skew", "speed_kurtosis",
    "accel_mean", "accel_std", "jerk_mean", "jerk_std",
    "early_speed_mean", "late_speed_mean", "early_late_speed_ratio",
    "curvature_mean", "curvature_std", "curvature_abs_mean",
    "direction_change_count", "stroke_count", "pause_count"
]

_rf_arousal = None
_rf_valence = None

def _load_models():
    global _rf_arousal, _rf_valence
    if _rf_arousal is None:
        _rf_arousal = joblib.load(MODEL_PATH_AROUSAL)
    if _rf_valence is None:
        _rf_valence = joblib.load(MODEL_PATH_VALENCE)

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

def _compute_features_for_points(points):
    """
    points: [{"x":..., "y":..., "t":..., "pressure":...}, ...]
    pressureは今回使わない（無視）
    """
    if len(points) < 3:
        return None

    try:
        xs = np.array([p["x"] for p in points], dtype=float)
        ys = np.array([p["y"] for p in points], dtype=float)
    except Exception:
        return None

    ts = np.array([p.get("t", np.nan) for p in points], dtype=float)

    # ---------- 幾何 ----------
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

    cov = np.cov(np.vstack([xs, ys]))
    eigvals, _ = np.linalg.eig(cov)
    eigvals = np.sort(np.real(eigvals))
    if eigvals[1] > 0:
        eccentricity = float(np.sqrt(max(0.0, 1.0 - eigvals[0] / eigvals[1])))
    else:
        eccentricity = 0.0

    start_x, start_y = float(xs[0]), float(ys[0])
    end_x, end_y = float(xs[-1]), float(ys[-1])
    closure_dist = float(math.hypot(start_x - end_x, start_y - end_y))
    closure_ratio = float(closure_dist / r_max)

    std_x = float(xs.std())
    std_y = float(ys.std())
    linearity = float(min(std_x, std_y) / max(std_x, std_y)) if max(std_x, std_y) > 0 else 0.0

    pts_xy = np.stack([xs, ys], axis=1)
    step_vec = np.diff(pts_xy, axis=0)
    step_dist = np.linalg.norm(step_vec, axis=1)
    path_length = float(step_dist.sum())

    # polygon area (shoelace)
    try:
        area = float(0.5 * abs(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
    except Exception:
        area = 0.0

    circularity = float(4 * math.pi * area / (path_length ** 2)) if path_length > 0 else 0.0

    # ---------- 時間・速度 ----------
    n_points = int(len(xs))

    if np.all(np.isfinite(ts)):
        duration = float(ts.max() - ts.min())
        dt_raw = np.diff(ts)
        dt_for_stroke = dt_raw.copy()
        dt_safe = np.where(dt_raw > 0, dt_raw, 1.0)
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
        mean_speed = median_speed = max_speed = min_speed = 0.0
        speed_std = speed_cv = 0.0
        speed_skew = speed_kurt = 0.0

    if speed.size > 1:
        accel = np.diff(speed)
        accel_mean = float(accel.mean())
        accel_std = float(accel.std())
    else:
        accel = np.array([], dtype=float)
        accel_mean = accel_std = 0.0

    if accel.size > 1:
        jerk = np.diff(accel)
        jerk_mean = float(jerk.mean())
        jerk_std = float(jerk.std())
    else:
        jerk_mean = jerk_std = 0.0

    if speed.size >= 5:
        k = max(1, speed.size // 10)
        early_speed_mean = float(speed[:k].mean())
        late_speed_mean = float(speed[-k:].mean())
        # 0割回避：lateが0なら ratio=0
        early_late_speed_ratio = float(early_speed_mean / late_speed_mean) if late_speed_mean != 0 else 0.0
    else:
        early_speed_mean = late_speed_mean = early_late_speed_ratio = 0.0

    # ---------- 曲率・方向変化 ----------
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
        curv = np.array(curvatures, dtype=float)
        curvature_mean = float(curv.mean())
        curvature_std = float(curv.std())
        curvature_abs_mean = float(np.abs(curv).mean())
    else:
        curvature_mean = curvature_std = curvature_abs_mean = 0.0

    # ---------- stroke/pause ----------
    if dt_for_stroke is not None and dt_for_stroke.size > 0 and np.all(np.isfinite(dt_for_stroke)):
        pause_threshold = 200.0
        pauses = dt_for_stroke > pause_threshold
        pause_count = int(np.sum(pauses))
        stroke_count = int(1 + pause_count)
    else:
        pause_count = 0
        stroke_count = 1

    feats = {
        "centroid_x": cx, "centroid_y": cy,
        "inner_radius": r_min, "outer_radius": r_max, "radius_std": r_std, "roundness": roundness,
        "bbox_w": bbox_w, "bbox_h": bbox_h, "bbox_area": bbox_area, "aspect_ratio": aspect_ratio,
        "eccentricity": eccentricity,
        "closure_dist": closure_dist, "closure_ratio": closure_ratio, "linearity": linearity,
        "path_length": path_length, "area": area, "circularity": circularity,
        "n_points": n_points, "duration": duration,
        "mean_speed": mean_speed, "median_speed": median_speed, "max_speed": max_speed, "min_speed": min_speed,
        "speed_std": speed_std, "speed_cv": speed_cv, "speed_skew": speed_skew, "speed_kurtosis": speed_kurt,
        "accel_mean": accel_mean, "accel_std": accel_std, "jerk_mean": jerk_mean, "jerk_std": jerk_std,
        "early_speed_mean": early_speed_mean, "late_speed_mean": late_speed_mean, "early_late_speed_ratio": early_late_speed_ratio,
        "curvature_mean": curvature_mean, "curvature_std": curvature_std, "curvature_abs_mean": curvature_abs_mean,
        "direction_change_count": direction_change_count,
        "stroke_count": stroke_count, "pause_count": pause_count,
    }
    return feats

def _categorize(score: float):
    """
    あなたのスコアが -10〜+10 想定のため、ざっくり3分類
    """
    if score <= -3:
        return "low"
    if score >= 3:
        return "high"
    return "mid"

def predict_va_from_points(points):
    _load_models()

    feats = _compute_features_for_points(points)
    if feats is None:
        return {"ok": False, "error": "特徴量の計算に失敗しました（点の形式が不正かも）。"}

    # フィルタ判定（円っぽくない入力は弾く）
    if feats["linearity"] < LINEARITY_MIN or feats["closure_ratio"] > CLOSURE_MAX:
        return {
            "ok": False,
            "error": "円らしさフィルタにより除外されました（線形すぎる or 閉じていない）。もう少し円っぽく描いてください。",
            "debug": {
                "linearity": feats["linearity"],
                "closure_ratio": feats["closure_ratio"],
                "thresholds": {"LINEARITY_MIN": LINEARITY_MIN, "CLOSURE_MAX": CLOSURE_MAX}
            }
        }

    # 予測用ベクトル
    x = np.array([[float(feats[c]) for c in FEATURE_COLS]], dtype=float)

    arousal = float(_rf_arousal.predict(x)[0])
    valence = float(_rf_valence.predict(x)[0])

    return {
        "ok": True,
        "arousal": arousal,
        "valence": valence,
        "arousal_cat": _categorize(arousal),
        "valence_cat": _categorize(valence),
        # デバッグや後で可視化したいなら返してもOK（不要なら消してOK）
        "features": feats
    }
