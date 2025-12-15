import os

def generate_advice(valence: float, arousal: float, valence_cat: str, arousal_cat: str) -> str:
    """
    LLMに「寄り添い口調」の短いアドバイスを作らせる。
    APIキーが無い/失敗したらフォールバック文を返す。
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "（OpenAI APIキーが設定されていません）まず深呼吸して、今の気持ちを表す円を考えてみよう。"

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system = (
            "あなたはユーザーに寄り添うメンタルケアのアシスタントです。"
            "診断や断定はせず、優しく短く、具体的に一歩だけ行動提案します。"
            "医療行為や危機対応はしません。"
        )

        user = (
            f"円描画から推定された気分スコアです。\n"
            f"- Valence（快-不快）: {valence:.2f}（カテゴリ: {valence_cat}）\n"
            f"- Arousal（鎮静-活発）: {arousal:.2f}（カテゴリ: {arousal_cat}）\n\n"
            f"この情報だけを使って、ユーザーが受け取りやすい言葉で、"
            f"100〜180文字くらいのアドバイスを日本語で1つ生成してください。"
            f"最後に、短い問いかけを1つ添えてください。"
        )

        # モデル名は環境に合わせて調整してOK（例：gpt-4.1-mini 等）
        resp = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.8,
        )
        return resp.choices[0].message.content.strip()

    except Exception:
        return "うまく言葉にできない時ほど、まずは水を一口飲んで肩の力を抜こう。今の体の緊張はどこにある？"
