# random_forest_models.py
#  make_features.py → points から特徴量を計算してCSV化
#  random_forest_models.py → 特徴量CSVから RandomForest で arousal / valence を学習

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib

# ======================================
# 設定
# ======================================

CSV_PATH = "features_drawings_filtered.csv"

MODEL_PATH_AROUSAL = "rf_arousal.pkl"
MODEL_PATH_VALENCE = "rf_valence.pkl"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# RandomForest パラメータ（まずはベースライン）
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

# ======================================
# データ読み込み
# ======================================

df = pd.read_csv(CSV_PATH)
print("Loaded:", CSV_PATH, "shape:", df.shape)
print("Columns:", df.columns.tolist())

# ======================================
# 説明変数・目的変数の設定
# ======================================

# モデルに使わない列（ID類＋目的変数）
EXCLUDE_COLS = [
    "row_index",
    "id",
    "user_id",
    "trial_index",
    "valence",
    "arousal",
]

# 数値列だけ抽出
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# 数値列の中から、除外対象以外を特徴量として採用
feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS]

# --------------------------------------
# ★筆圧系を全て「使わない」設定（CSVに残っていても除外）
# 例: mean_pressure, max_pressure, pressure_std, pressure_slope, pressure_peak_count
# --------------------------------------
feature_cols = [c for c in feature_cols if ("pressure" not in c.lower())]

print("\nNumber of feature columns:", len(feature_cols))
print("Feature columns:", feature_cols)

# ======================================
# 欠損チェック（何が dropna で消えるのか見える化）
# ======================================

needed_cols = feature_cols + ["arousal", "valence"]

# 欠損数（列ごと）
na_counts = df[needed_cols].isna().sum().sort_values(ascending=False)
na_counts = na_counts[na_counts > 0]

print("\n[NaN Diagnostics] 欠損がある列（上位）")
if len(na_counts) == 0:
    print("  欠損は見つかりませんでした（needed_cols内）")
else:
    for col, cnt in na_counts.head(30).items():
        print(f"  {col:30s} : {int(cnt)}")

# 欠損を含む行の数（行単位で1つでも欠損がある行）
rows_with_any_nan = df[needed_cols].isna().any(axis=1).sum()
print(f"\nRows that have ANY NaN in needed_cols: {int(rows_with_any_nan)} / {len(df)}")

# ======================================
# 欠損を含むサンプルは除外（まずはシンプルに）
# ======================================

df_model = df.dropna(subset=needed_cols).copy()
print("\nAfter dropna, usable rows:", df_model.shape[0])

if df_model.shape[0] < 10:
    print("\n[Warning] サンプル数が少ないので、R²は安定しません。")
    print("          フィルタ条件（linearity, closure_ratio）を緩めるか、欠損補完も検討してください。")

# ======================================
# RF学習＋評価をまとめた関数
# ======================================

def train_and_eval_rf(target_name: str, model_path: str):
    """
    target_name: 'arousal' or 'valence'
    model_path : 保存先ファイル名
    """
    print("\n==============================")
    print(f" Target: {target_name}")
    print("==============================")

    X = df_model[feature_cols].values
    y = df_model[target_name].values

    # Train/Test 分割
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # RandomForest 回帰モデル
    rf = RandomForestRegressor(**RF_PARAMS)

    # 学習
    rf.fit(X_train, y_train)

    # 予測
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    # 評価指標
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    # RMSE（sqrt(MSE)）
    mse_test = mean_squared_error(y_test, y_test_pred)
    rmse_test = float(np.sqrt(mse_test))

    print(f"Train R² : {r2_train:.3f}")
    print(f"Test  R² : {r2_test:.3f}")
    print(f"Test MAE : {mae_test:.3f}")
    print(f"Test RMSE: {rmse_test:.3f}")

    # 特徴量重要度
    importances = rf.feature_importances_
    fi = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)

    print("\n[Feature Importances] 上位20")
    for name, imp in fi[:20]:
        print(f"{name:30s} : {imp:.4f}")

    # モデル保存
    joblib.dump(rf, model_path)
    print(f"\nSaved model → {model_path}")

    return rf


# ======================================
# 実行：arousal / valence 別々に学習
# ======================================

rf_arousal = train_and_eval_rf("arousal", MODEL_PATH_AROUSAL)
rf_valence = train_and_eval_rf("valence", MODEL_PATH_VALENCE)
