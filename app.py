from flask import Flask, request, jsonify, render_template
from inference import predict_va_from_points
from llm_advice import generate_advice

app = Flask(__name__)

@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def api_predict():
    data = request.get_json(force=True, silent=True) or {}
    points = data.get("points", [])

    # 1) 入力チェック（HTTPは常に200、okで返す）
    if not isinstance(points, list) or len(points) < 3:
        return jsonify({
            "ok": False,
            "reason": "too_few_points",
            "message": "点が少なすぎます（最低3点）。もう少し描いて送信してください。",
            "state": "–",
            "quality": "低",
            "arousal": None,
            "valence": None,
            "advice": "まずはゆっくりでOKなので、もう少し長めに描いてみてください。",
            "debug": {"n_points": len(points)}
        }), 200

    # 2) RF 推論
    pred = predict_va_from_points(points)

    # predの想定:
    # ok: bool
    # valence/arousal: float
    # valence_cat/arousal_cat: str
    # state/quality: str（あるなら）
    # debug: dict（あるなら）
    if not isinstance(pred, dict):
        return jsonify({
            "ok": False,
            "reason": "bad_pred_return",
            "message": "推定処理の返り値が不正です（predがdictではありません）。",
            "state": "–",
            "quality": "不明",
            "arousal": None,
            "valence": None,
            "advice": "内部エラーです。サーバログを確認してください。",
            "debug": {"type": str(type(pred))}
        }), 200

    if not pred.get("ok", False):
        # フィルタで弾かれた・特徴量作れない等
        # → UI向けにやさしく message を付けて返す
        debug = pred.get("debug", {})
        # よくある：closure_ratioが閾値超え
        if "closure_ratio" in debug and "thresholds" in debug:
            c = debug.get("closure_ratio")
            mx = debug.get("thresholds", {}).get("CLOSURE_MAX")
            msg = (
                "推定できませんでした（円が閉じきっていない可能性があります）。\n"
                f"closure_ratio={c:.3f} / 条件: closure_ratio <= {mx}\n"
                "もう少しだけ「最後の終点を始点に近づける」ように描いてみてください。"
            )
        else:
            msg = pred.get("error") or pred.get("message") or "推定できませんでした。描き方を変えて再度お試しください。"

        # predに state/quality がなければ簡易で補完
        pred_out = {
            "ok": False,
            "reason": pred.get("reason", "predict_failed"),
            "message": msg,
            "state": pred.get("state", "–"),
            "quality": pred.get("quality", "要改善"),
            "arousal": None,
            "valence": None,
            "advice": "一筆で、ゆっくり大きめに描くと安定しやすいです。",
            "debug": debug
        }
        return jsonify(pred_out), 200

    # 3) LLM アドバイス（失敗してもRF結果は返す）
    advice_text = ""
    try:
        advice_text = generate_advice(
            valence=pred["valence"],
            arousal=pred["arousal"],
            valence_cat=pred.get("valence_cat", "–"),
            arousal_cat=pred.get("arousal_cat", "–"),
        )
    except Exception as e:
        advice_text = "アドバイス生成に失敗しました（LLM）。もう一度お試しください。"
        pred["llm_error"] = str(e)

    # 4) UIが期待する形に整形
    out = {
        "ok": True,
        "valence": float(pred["valence"]),
        "arousal": float(pred["arousal"]),
        "valence_cat": pred.get("valence_cat", "–"),
        "arousal_cat": pred.get("arousal_cat", "–"),
        "state": pred.get("state") or "推定完了",
        "quality": pred.get("quality") or "OK",
        "advice": advice_text,
        "debug": pred.get("debug", {})
    }
    return jsonify(out), 200


if __name__ == "__main__":
    # タブレット検証しやすいようにLAN公開
    app.run(host="0.0.0.0", port=5001, debug=True)
