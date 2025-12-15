# visualize_results.py
# ------------------------------------------------------------
# 目的:
#  - 学習済み RandomForest モデル（rf_arousal.pkl / rf_valence.pkl）を読み込む
#  - テストデータで予測し、可視化する
#    ① 実測値 vs 予測値（散布図 + 45度線）
#    ② 残差プロット（予測値 vs 残差）
#    ③ 特徴量重要度（上位Nの棒グラフ）
# ------------------------------------------------------------

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# =============================
# 設定
# =============================
CSV_PATH = "features_drawings_filtered.csv"
MODEL_AROUSAL = "rf_arousal.pkl"
MODEL_VALENCE = "rf_valence.pkl"

TEST_SIZE = 0.2
RANDOM_STATE = 42

TOP_N_IMPORTANCE = 20  # 重要度表示の上位数

# =============================
# 関数: 特徴量重要度プロット
# =============================
def plot_feature_importance(model, feature_names, title, top_n=20):
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1][:top_n]

    plt.figure(figsize=(7, 7))
    plt.barh(
        [feature_names[i] for i in idx][::-1],
        importances[idx][::-1]
    )
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# =============================
# メイン
# =============================
def main():
    # ---------- データ読み込み ----------
    df = pd.read_csv(CSV_PATH)
    print("Loaded:", CSV_PATH, "shape:", df.shape)

    # ---------- 特徴量列を決める ----------
    EXCLUDE_COLS = ["row_index", "id", "user_id", "trial_index", "valence", "arousal"]
    feature_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in EXCLUDE_COLS]

    print("Number of feature columns:", len(feature_cols))
    print("Feature columns:", feature_cols)

    # ここでNaNがあるなら落とす（あなたの最新版では0件のはず）
    needed_cols = feature_cols + ["arousal", "valence"]
    df_model = df.dropna(subset=needed_cols).copy()
    print("After dropna:", df_model.shape)

    X = df_model[feature_cols].values
    y_arousal = df_model["arousal"].values
    y_valence = df_model["valence"].values

    # ---------- 同じ分割を再現する ----------
    # ※ random_forest_models.py と同じ seed / test_size にすることで一致しやすい
    X_train, X_test, ya_train, ya_test = train_test_split(
        X, y_arousal, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    _, _, yv_train, yv_test = train_test_split(
        X, y_valence, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # ---------- モデル読み込み ----------
    rf_arousal = joblib.load(MODEL_AROUSAL)
    rf_valence = joblib.load(MODEL_VALENCE)
    print("Loaded models:", MODEL_AROUSAL, MODEL_VALENCE)

    # ---------- 予測 ----------
    ya_pred = rf_arousal.predict(X_test)
    yv_pred = rf_valence.predict(X_test)

    # ============================================================
    # ① 実測 vs 予測
    # ============================================================
    plt.figure(figsize=(12, 5))

    # --- Arousal ---
    plt.subplot(1, 2, 1)
    plt.scatter(ya_test, ya_pred, alpha=0.6)
    lo = min(ya_test.min(), ya_pred.min())
    hi = max(ya_test.max(), ya_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True Arousal")
    plt.ylabel("Predicted Arousal")
    plt.title("Arousal: True vs Predicted")

    # --- Valence ---
    plt.subplot(1, 2, 2)
    plt.scatter(yv_test, yv_pred, alpha=0.6)
    lo = min(yv_test.min(), yv_pred.min())
    hi = max(yv_test.max(), yv_pred.max())
    plt.plot([lo, hi], [lo, hi], linestyle="--")
    plt.xlabel("True Valence")
    plt.ylabel("Predicted Valence")
    plt.title("Valence: True vs Predicted")

    plt.tight_layout()
    plt.show()

    # ============================================================
    # ② 残差プロット（予測値 vs 残差）
    # ============================================================
    # --- Arousal ---
    residuals_a = ya_test - ya_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(ya_pred, residuals_a, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Arousal")
    plt.ylabel("Residual (True - Pred)")
    plt.title("Residual Plot (Arousal)")
    plt.tight_layout()
    plt.show()

    # --- Valence ---
    residuals_v = yv_test - yv_pred
    plt.figure(figsize=(6, 4))
    plt.scatter(yv_pred, residuals_v, alpha=0.6)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted Valence")
    plt.ylabel("Residual (True - Pred)")
    plt.title("Residual Plot (Valence)")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # ③ 特徴量重要度（上位N）
    # ============================================================
    plot_feature_importance(
        rf_arousal, feature_cols,
        f"Feature Importance (Arousal) Top{TOP_N_IMPORTANCE}",
        top_n=TOP_N_IMPORTANCE
    )

    plot_feature_importance(
        rf_valence, feature_cols,
        f"Feature Importance (Valence) Top{TOP_N_IMPORTANCE}",
        top_n=TOP_N_IMPORTANCE
    )


if __name__ == "__main__":
    main()
