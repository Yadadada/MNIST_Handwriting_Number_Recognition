import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
from flask import Flask, request, jsonify, render_template_string
import io
import base64

app = Flask(__name__)

# ─── 模型定义 ─────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.model(x)

model = MLP()
model.load_state_dict(torch.load("./mnist_model.pth", map_location="cpu"))
model.eval()

# ─── 图像预处理 ───────────────────────────────────────
def preprocess(image_bytes):
    import numpy as np

    img = Image.open(io.BytesIO(image_bytes)).convert("L")

    # 白底黑字 → 反转为黑底白字（与 MNIST 一致）
    pixels = list(img.getdata())
    if sum(pixels) / len(pixels) > 127:
        img = ImageOps.invert(img)

    # 自动裁剪空白边距
    arr = np.array(img)
    rows = np.any(arr > 20, axis=1)
    cols = np.any(arr > 20, axis=0)
    if rows.any() and cols.any():
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        pad = max((rmax - rmin), (cmax - cmin)) // 8
        rmin = max(0, rmin - pad)
        rmax = min(arr.shape[0], rmax + pad)
        cmin = max(0, cmin - pad)
        cmax = min(arr.shape[1], cmax + pad)
        img = img.crop((cmin, rmin, cmax, rmax))

    # 保持宽高比缩放到 20×20，居中 padding 到 28×28
    img.thumbnail((20, 20), Image.LANCZOS)
    padded = Image.new("L", (28, 28), 0)
    padded.paste(img, ((28 - img.width) // 2, (28 - img.height) // 2))

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    tensor = transform(padded).unsqueeze(0)

    buf = io.BytesIO()
    padded.save(buf, format="PNG")
    preview_b64 = base64.b64encode(buf.getvalue()).decode()
    return tensor, preview_b64

# ─── HTML 页面 ────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>MNIST 手写数字识别</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #f0f2f5; display: flex; justify-content: center; min-height: 100vh; padding: 40px 16px; }
  .container { width: 100%; max-width: 720px; }
  h1 { font-size: 24px; color: #1a1a2e; margin-bottom: 6px; }
  .subtitle { color: #666; font-size: 14px; margin-bottom: 24px; }

  .canvas-wrap { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; }
  canvas#draw-canvas { background: black; border-radius: 12px; cursor: crosshair; touch-action: none; }
  .canvas-side { flex: 1; min-width: 160px; }
  .preview-28 { width: 112px; height: 112px; image-rendering: pixelated; border: 2px solid #e0e4ea; border-radius: 8px; display: block; margin-bottom: 6px; background: black; }
  .preview-label { font-size: 12px; color: #aaa; margin-bottom: 16px; }
  .btn-row { display: flex; gap: 8px; margin-top: 16px; }
  .btn { flex: 1; padding: 12px; border: none; border-radius: 10px; font-size: 15px; cursor: pointer; transition: background 0.2s; }
  .btn-primary { background: #4f8ef7; color: white; }
  .btn-primary:hover { background: #3a7be0; }
  .btn-primary:disabled { background: #a0b8e8; cursor: not-allowed; }
  .btn-secondary { background: #f0f2f5; color: #555; border: 1.5px solid #d0d5e0; }
  .btn-secondary:hover { border-color: #4f8ef7; color: #4f8ef7; }

  #result-section { display: none; margin-top: 28px; background: white; border-radius: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.08); padding: 28px; }
  .result-header { font-size: 14px; color: #888; margin-bottom: 4px; }
  .result-row { display: flex; align-items: center; gap: 24px; margin-bottom: 20px; }
  .result-number { font-size: 80px; font-weight: 800; color: #1a1a2e; line-height: 1; }
  .result-meta { flex: 1; }
  .confidence-label { font-size: 13px; color: #888; }
  .confidence-value { font-size: 28px; font-weight: 700; color: #4f8ef7; }
  .bar-chart { display: flex; flex-direction: column; gap: 5px; }
  .bar-row { display: flex; align-items: center; gap: 8px; }
  .bar-label { width: 16px; text-align: right; font-size: 12px; color: #555; font-weight: 600; }
  .bar-track { flex: 1; background: #eef0f5; border-radius: 4px; height: 20px; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 4px; background: #c5d8fc; transition: width 0.5s ease; display: flex; align-items: center; justify-content: flex-end; padding-right: 5px; }
  .bar-fill.top { background: #4f8ef7; }
  .bar-pct { font-size: 10px; color: white; font-weight: 600; }
  .bar-pct-out { font-size: 10px; color: #bbb; margin-left: 4px; min-width: 36px; }

  .loading { text-align: center; padding: 16px; color: #888; display: none; }
  .spinner { display: inline-block; width: 20px; height: 20px; border: 3px solid #e0e4ea; border-top-color: #4f8ef7; border-radius: 50%; animation: spin 0.7s linear infinite; margin-right: 8px; vertical-align: middle; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="container">
  <h1>手写数字识别</h1>
  <p class="subtitle">在画布上用鼠标写一个数字（0–9），点击识别查看结果</p>

  <div class="canvas-wrap">
    <canvas id="draw-canvas" width="280" height="280"></canvas>
    <div class="canvas-side">
      <img id="preview-28" class="preview-28" src="" alt="">
      <div class="preview-label">模型看到的 28×28</div>
      <div class="btn-row">
        <button class="btn btn-primary" id="predict-btn" onclick="predictDraw()">识别</button>
        <button class="btn btn-secondary" onclick="clearCanvas()">清除</button>
      </div>
    </div>
  </div>

  <div class="loading" id="loading"><span class="spinner"></span>识别中...</div>

  <div id="result-section">
    <div class="result-header">识别结果</div>
    <div class="result-row">
      <div class="result-number" id="result-number">-</div>
      <div class="result-meta">
        <div class="confidence-label">置信度</div>
        <div class="confidence-value" id="confidence">-</div>
      </div>
    </div>
    <div class="bar-chart" id="bar-chart"></div>
  </div>
</div>

<script>
const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
let drawing = false;

ctx.strokeStyle = 'white';
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';

function getPos(e) {
  const r = canvas.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return { x: src.clientX - r.left, y: src.clientY - r.top };
}

canvas.addEventListener('mousedown',  e => { drawing = true; ctx.beginPath(); const p = getPos(e); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('mousemove',  e => { if (!drawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); updatePreview(); });
canvas.addEventListener('mouseup',    () => drawing = false);
canvas.addEventListener('mouseleave', () => drawing = false);
canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing = true; ctx.beginPath(); const p = getPos(e); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('touchmove',  e => { e.preventDefault(); if (!drawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); updatePreview(); });
canvas.addEventListener('touchend',   () => drawing = false);

function updatePreview() {
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = 28;
  tmp.getContext('2d').drawImage(canvas, 0, 0, 28, 28);
  document.getElementById('preview-28').src = tmp.toDataURL();
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  document.getElementById('preview-28').src = '';
  document.getElementById('result-section').style.display = 'none';
}

async function predictDraw() {
  const tmp = document.createElement('canvas');
  tmp.width = tmp.height = 28;
  tmp.getContext('2d').drawImage(canvas, 0, 0, 28, 28);
  const blob = await (await fetch(tmp.toDataURL('image/png'))).blob();
  const form = new FormData();
  form.append('file', blob, 'draw.png');

  document.getElementById('predict-btn').disabled = true;
  document.getElementById('loading').style.display = 'block';
  document.getElementById('result-section').style.display = 'none';

  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    const data = await resp.json();
    if (data.error) { alert('识别失败：' + data.error); return; }
    document.getElementById('preview-28').src = 'data:image/png;base64,' + data.preview;
    document.getElementById('result-number').textContent = data.predicted;
    document.getElementById('confidence').textContent = data.confidence.toFixed(1) + '%';
    renderBars(data.probs, data.predicted);
    document.getElementById('result-section').style.display = 'block';
  } catch(e) {
    alert('请求失败，请检查服务是否运行');
  } finally {
    document.getElementById('loading').style.display = 'none';
    document.getElementById('predict-btn').disabled = false;
  }
}

function renderBars(probs, top) {
  const chart = document.getElementById('bar-chart');
  chart.innerHTML = '';
  probs.forEach((p, i) => {
    const pct = (p * 100).toFixed(1);
    const isTop = i === top;
    const showInside = p > 0.1;
    const row = document.createElement('div');
    row.className = 'bar-row';
    row.innerHTML = `
      <div class="bar-label">${i}</div>
      <div class="bar-track">
        <div class="bar-fill ${isTop ? 'top' : ''}" style="width:${Math.max(p*100,0.5)}%">
          ${showInside ? `<span class="bar-pct">${pct}%</span>` : ''}
        </div>
      </div>
      <span class="bar-pct-out">${showInside ? '' : pct+'%'}</span>
    `;
    chart.appendChild(row);
  });
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "没有收到文件"})
    file = request.files["file"]
    try:
        image_bytes = file.read()
        tensor, preview_b64 = preprocess(image_bytes)
        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.softmax(outputs, dim=1)[0].tolist()
        predicted = int(probs.index(max(probs)))
        return jsonify({
            "predicted": predicted,
            "confidence": max(probs) * 100,
            "probs": probs,
            "preview": preview_b64
        })
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("启动服务：http://127.0.0.1:5000")
    app.run(host="0.0.0.0", debug=False)
