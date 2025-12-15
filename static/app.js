// static/app.js
(() => {
  const canvas = document.getElementById("drawCanvas");
  const btnPredict = document.getElementById("btnPredict");
  const btnClear = document.getElementById("btnClear");
  const btnUndo = document.getElementById("btnUndo");

  const kpiArousal = document.getElementById("kpiArousal");
  const kpiValence = document.getElementById("kpiValence");
  const pillState = document.getElementById("pillState");
  const pillQuality = document.getElementById("pillQuality");
  const adviceBox = document.getElementById("adviceBox");
  const debugBox = document.getElementById("debugBox");

  if (!canvas) {
    console.error("Canvas #drawCanvas not found");
    return;
  }

  const ctx = canvas.getContext("2d", { willReadFrequently: false });

  // ====== 描画設定 ======
  const STROKE_WIDTH = 8.0;
  const STROKE_ALPHA = 0.95;

  // ====== strokes: [ [ {x,y,t,pressure}, ... ], ... ] ======
  const strokes = [];
  let currentStroke = null;
  let isDrawing = false;

  const nowMs = () => performance.now();

  function log(obj) {
    if (!debugBox) return;
    debugBox.textContent = typeof obj === "string" ? obj : JSON.stringify(obj, null, 2);
  }

  function setResultUI({ arousal = null, valence = null, state = "–", quality = "–", advice = "" } = {}) {
    if (kpiArousal) kpiArousal.textContent = (arousal === null || arousal === undefined) ? "–" : Number(arousal).toFixed(2);
    if (kpiValence) kpiValence.textContent = (valence === null || valence === undefined) ? "–" : Number(valence).toFixed(2);
    if (pillState) pillState.textContent = `状態: ${state}`;
    if (pillQuality) pillQuality.textContent = `入力品質: ${quality}`;
    if (adviceBox) adviceBox.textContent = advice || "ここにアドバイスが表示されます。";
  }

  // ====== Canvasサイズ調整（ぼやけ防止） ======
  function resizeCanvasToDisplaySize() {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    const cssW = Math.max(1, Math.floor(rect.width));
    const cssH = Math.max(1, Math.floor(rect.height));

    const pxW = Math.floor(cssW * dpr);
    const pxH = Math.floor(cssH * dpr);

    if (canvas.width !== pxW || canvas.height !== pxH) {
      canvas.width = pxW;
      canvas.height = pxH;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      redrawAll();
    }
  }

  window.addEventListener("resize", resizeCanvasToDisplaySize);
  resizeCanvasToDisplaySize();

  function clearCanvas() {
    const rect = canvas.getBoundingClientRect();
    ctx.clearRect(0, 0, rect.width, rect.height);
  }

  function drawStroke(points) {
    if (!points || points.length === 0) return;

    ctx.save();
    ctx.lineWidth = STROKE_WIDTH;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.globalAlpha = STROKE_ALPHA;
    ctx.strokeStyle = "rgba(0,0,0,0.95)";

    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function redrawAll() {
    clearCanvas();
    for (const s of strokes) drawStroke(s);
    if (currentStroke) drawStroke(currentStroke);
  }

  function getCanvasPointFromEvent(e) {
    const rect = canvas.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    let pressure = e.pressure;
    if (pressure === 0 && e.pointerType === "mouse") pressure = 0.5;
    if (!Number.isFinite(pressure)) pressure = 0.5;

    return { x, y, t: nowMs(), pressure };
  }

  function onPointerDown(e) {
    if (e.pointerType === "mouse" && e.button !== 0) return;

    e.preventDefault();
    canvas.setPointerCapture(e.pointerId);

    isDrawing = true;
    currentStroke = [];
    const p = getCanvasPointFromEvent(e);
    currentStroke.push(p);
    redrawAll();
  }

  function onPointerMove(e) {
    if (!isDrawing || !currentStroke) return;

    e.preventDefault();
    const p = getCanvasPointFromEvent(e);

    const last = currentStroke[currentStroke.length - 1];
    const dx = p.x - last.x;
    const dy = p.y - last.y;
    const dist2 = dx * dx + dy * dy;

    if (dist2 >= 4) {
      currentStroke.push(p);

      ctx.save();
      ctx.lineWidth = STROKE_WIDTH;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.globalAlpha = STROKE_ALPHA;
      ctx.strokeStyle = "rgba(0,0,0,0.95)";
      ctx.beginPath();
      ctx.moveTo(last.x, last.y);
      ctx.lineTo(p.x, p.y);
      ctx.stroke();
      ctx.restore();
    }
  }

  function endStroke(e) {
    if (!isDrawing) return;
    e.preventDefault();

    isDrawing = false;

    if (currentStroke && currentStroke.length >= 3) {
      strokes.push(currentStroke);
      log({ strokes: strokes.length, lastStrokePoints: currentStroke.length });
    } else {
      log("stroke too short → ignored");
    }

    currentStroke = null;
    redrawAll();
  }

  canvas.addEventListener("pointerdown", onPointerDown, { passive: false });
  canvas.addEventListener("pointermove", onPointerMove, { passive: false });
  canvas.addEventListener("pointerup", endStroke, { passive: false });
  canvas.addEventListener("pointercancel", endStroke, { passive: false });
  canvas.addEventListener("pointerleave", (e) => { if (isDrawing) endStroke(e); }, { passive: false });

  btnClear?.addEventListener("click", () => {
    strokes.length = 0;
    currentStroke = null;
    isDrawing = false;
    redrawAll();
    setResultUI({ arousal: null, valence: null, state: "–", quality: "–", advice: "" });
    log("cleared");
  });

  btnUndo?.addEventListener("click", () => {
    if (strokes.length > 0) {
      strokes.pop();
      redrawAll();
      log({ undo: true, strokes: strokes.length });
    } else {
      log("nothing to undo");
    }
  });

  function formatErrorForUser(data) {
    // サーバの統一フォーマット: ok:false, message, reason, debug
    const msg = data?.message || data?.error || "推定できませんでした。";
    const reason = data?.reason ? `（理由: ${data.reason}）` : "";
    return `${msg}\n${reason}`.trim();
  }

  btnPredict?.addEventListener("click", async () => {
    if (strokes.length === 0) {
      log("no strokes");
      return;
    }

    const pts = strokes[strokes.length - 1];
    const payload = {
      points: pts.map(p => ({ x: p.x, y: p.y, t: p.t, pressure: p.pressure }))
    };

    log({ action: "predict", points: payload.points.length });

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      // ★ ここ重要：HTTPステータスに依存せず JSON を読む
      let data = null;
      try {
        data = await res.json();
      } catch (e) {
        const text = await res.text();
        setResultUI({
          arousal: null,
          valence: null,
          state: "通信エラー",
          quality: "–",
          advice: "サーバ応答がJSONではありません。ログを確認してください。",
        });
        log({ error: "non-json response", status: res.status, body: text });
        return;
      }

      // ok:false の場合も “文章” で表示する
      if (!data.ok) {
        setResultUI({
          arousal: null,
          valence: null,
          state: "推定不可",
          quality: data.quality ?? "要改善",
          advice: formatErrorForUser(data),
        });
        log({ ok: false, status: res.status, ...data });
        return;
      }

      // ok:true
      setResultUI({
        arousal: data.arousal,
        valence: data.valence,
        state: data.state ?? "推定完了",
        quality: data.quality ?? "OK",
        advice: data.advice ?? "",
      });

      log({ ok: true, status: res.status, ...data });
    } catch (err) {
      setResultUI({
        arousal: null,
        valence: null,
        state: "通信エラー",
        quality: "–",
        advice: "通信に失敗しました。サーバが起動しているか確認してください。",
      });
      log({ error: String(err) });
    }
  });

  setResultUI();
  log("ready: draw a circle");
})();
