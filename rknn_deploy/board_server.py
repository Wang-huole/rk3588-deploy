"""板端 Web 演示服务（ONNX CPU 推理，浏览器在连板电脑上打开）。

只依赖标准库 + 板端已装的 numpy / opencv / pillow / onnxruntime，无需 FastAPI/uvicorn。

启动（板子上）:
    python3 rknn_deploy/board_server.py --port 8000

电脑浏览器打开（板子 IP，与板子同一局域网）:
    http://192.168.31.103:8000

功能:
    - 选择类别（pill / toothbrush）
    - 上传图片 或 摄像头抓帧
    - 返回: OK/NG 判定、异常分数、阈值、耗时、FPS、叠加热力图

推理: 复用 ort_infer.OrtPatchCore（ONNX Runtime CPU + CPU KNN）。
"""
from __future__ import annotations

import argparse
import base64
import cgi
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

import sys
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))

from ort_infer import OrtPatchCore, imread_unicode  # noqa: E402

MODELS: dict[str, OrtPatchCore] = {}
DEPLOY_ROOT = PROJECT_ROOT / "deploy"
DEFAULT_CATEGORY = "pill"

PAGE = """<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>工业缺陷检测 · 板端演示</title>
<style>
  :root{--bg:#0f1420;--card:#1b2233;--line:#2c3650;--txt:#e8edf7;--mut:#8fa0c0;--ok:#3ddc84;--ng:#ff5d5d;--acc:#4ea1ff}
  *{box-sizing:border-box}
  body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--txt)}
  .wrap{max-width:980px;margin:0 auto;padding:24px}
  h1{font-size:20px;margin:0 0 4px}
  .sub{color:var(--mut);font-size:13px;margin-bottom:18px}
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:16px;margin-bottom:16px}
  .row{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
  select,button,input[type=file]{background:#111726;color:var(--txt);border:1px solid var(--line);border-radius:10px;padding:10px 12px;font-size:14px}
  button{cursor:pointer;background:var(--acc);border-color:var(--acc);color:#06121f;font-weight:600}
  button.ghost{background:transparent;color:var(--txt)}
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:14px}
  .stat{background:#111726;border:1px solid var(--line);border-radius:12px;padding:12px}
  .stat .k{color:var(--mut);font-size:12px}
  .stat .v{font-size:20px;font-weight:700;margin-top:4px}
  .verdict{font-size:22px;font-weight:800}
  img{max-width:100%;border-radius:12px;border:1px solid var(--line)}
  .thumb{display:flex;gap:16px;flex-wrap:wrap;margin-top:14px}
  .thumb div{flex:1;min-width:280px}
  .thumb .cap{color:var(--mut);font-size:12px;margin-bottom:6px}
  canvas{display:none}
  .warn{color:#ffc86b;font-size:12px;margin-top:8px}
</style>
</head>
<body>
<div class="wrap">
  <h1>工业缺陷检测 · 板端演示</h1>
  <div class="sub">板子 ONNX CPU 推理 · 结果在本浏览器显示</div>

  <div class="card">
    <div class="row">
      <label>类别
        <select id="category"></select>
      </label>
      <input type="file" id="file" accept="image/*">
      <button onclick="predictFile()">上传检测</button>
      <button class="ghost" onclick="openCamera()">摄像头抓帧</button>
      <button class="ghost" onclick="snap()">拍摄当前帧</button>
    </div>
    <canvas id="cam"></canvas>
    <video id="video" playsinline style="display:none"></video>
    <div class="warn" id="warn"></div>
  </div>

  <div class="card">
    <div class="stats">
      <div class="stat"><div class="k">判定</div><div class="v verdict" id="verdict">--</div></div>
      <div class="stat"><div class="k">异常分数</div><div class="v" id="score">--</div></div>
      <div class="stat"><div class="k">阈值</div><div class="v" id="threshold">--</div></div>
      <div class="stat"><div class="k">耗时 / FPS</div><div class="v" id="ms">--</div></div>
    </div>
    <div class="thumb">
      <div><div class="cap">输入原图</div><img id="orig" alt=""></div>
      <div><div class="cap">缺陷热力图</div><img id="heat" alt=""></div>
    </div>
  </div>
</div>

<script>
let stream=null;
async function loadCats(){
  const r=await fetch('/api/status'); const d=await r.json();
  const sel=document.getElementById('category');
  sel.innerHTML=d.categories.map(c=>`<option value="${c}">${c}</option>`).join('');
  if(d.default_category) sel.value=d.default_category;
}
function encodeCanvas(){
  const v=document.getElementById('video'); const c=document.getElementById('cam');
  c.width=v.videoWidth; c.height=v.videoHeight;
  c.getContext('2d').drawImage(v,0,0);
  return new Promise(res=>c.toBlob(b=>res(b),'image/jpeg',0.9));
}
async function predict(blob){
  if(!blob) return;
  const fd=new FormData();
  fd.append('category',document.getElementById('category').value);
  fd.append('file',blob,'shot.jpg');
  const r=await fetch('/api/predict',{method:'POST',body:fd});
  if(!r.ok){document.getElementById('warn').textContent='请求失败: '+r.status;return;}
  const d=await r.json();
  document.getElementById('verdict').textContent=d.verdict;
  document.getElementById('verdict').style.color=d.verdict==='缺陷'?'var(--ng)':'var(--ok)';
  document.getElementById('score').textContent=d.score;
  document.getElementById('threshold').textContent=d.threshold;
  document.getElementById('ms').textContent=`${d.ms} ms · ${d.fps} FPS`;
  document.getElementById('orig').src='data:image/jpeg;base64,'+d.image_b64;
  document.getElementById('heat').src='data:image/jpeg;base64,'+d.heatmap_b64;
  document.getElementById('warn').textContent=d.note||'';
}
function predictFile(){
  const f=document.getElementById('file').files[0]; if(f) predict(f);
}
async function openCamera(){
  stream=await navigator.mediaDevices.getUserMedia({video:true});
  const v=document.getElementById('video'); v.srcObject=stream; v.style.display='block'; await v.play();
}
function snap(){ encodeCanvas().then(predict); }
loadCats();
</script>
</body>
</html>"""


def _jpeg_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""


def _decode_image(raw: bytes) -> np.ndarray:
    """先 OpenCV 解码，失败再用 PIL 兜底；都失败抛异常。"""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is not None:
        return img
    from PIL import Image
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(im)[:, :, ::-1].copy()


def _load_models() -> list[str]:
    cats = []
    for d in sorted(DEPLOY_ROOT.iterdir()):
        if d.is_dir() and (d / "backbone.onnx").exists():
            try:
                MODELS[d.name] = OrtPatchCore(d.name, deploy_root=str(DEPLOY_ROOT))
                cats.append(d.name)
            except Exception as e:
                print(f"[跳过] {d.name}: {e}")
    return cats


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[http]", fmt % args)

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/status":
            cats = list(MODELS.keys())
            self._send_json({
                "categories": cats,
                "default_category": cats[0] if cats else DEFAULT_CATEGORY,
                "loaded": cats,
            })
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path != "/api/predict":
            self._send_json({"error": "not found"}, 404)
            return
        ctype = self.headers.get("Content-Type", "")
        try:
            form = cgi.FieldStorage(
                fp=self.rfile, headers=self.headers,
                environ={"REQUEST_METHOD": "POST", "CONTENT_TYPE": ctype})
            category = form.getfirst("category", DEFAULT_CATEGORY)
            if category not in MODELS:
                self._send_json({"error": f"未知类别: {category}"}, 400)
                return
            model = MODELS[category]

            item = form["file"] if "file" in form else None
            if item is None or not item.file:
                self._send_json({"error": "未收到图片"}, 400)
                return
            raw = item.file.read()
            try:
                img = _decode_image(raw)
            except Exception:
                self._send_json({"error": "图片解码失败：文件可能损坏/不完整（PNG 校验错误），请换一张或另存为 JPG"}, 400)
                return
            if img is None or img.size == 0:
                self._send_json({"error": "图片为空"}, 400)
                return

            score, amap, stats = model.predict(img)
            threshold = float(model.cfg.get("threshold", 0.5))
            verdict = "缺陷" if score > threshold else "正常"
            vis = model.overlay(img, amap)

            self._send_json({
                "category": category,
                "score": round(float(score), 4),
                "verdict": verdict,
                "threshold": round(threshold, 4),
                "ms": round(stats["total_ms"], 1),
                "fps": round(stats["fps"], 1),
                "backend_ms": round(stats["backend_ms"], 1),
                "knn_ms": round(stats["knn_ms"], 1),
                "image_b64": _jpeg_b64(img),
                "heatmap_b64": _jpeg_b64(vis),
                "note": "板端 ONNX CPU 推理；NPU 未启用",
            })
        except Exception as e:
            self._send_json({"error": f"推理异常: {type(e).__name__}: {e}"}, 500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print("加载板端模型...")
    cats = _load_models()
    if not cats:
        raise SystemExit("deploy/ 下没有可用模型")
    print(f"就绪: {cats}")

    server = HTTPServer((args.host, args.port), Handler)
    print(f"服务已启动: http://<板子IP>:{args.port}  (本机访问 http://127.0.0.1:{args.port})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n停止服务")
        server.server_close()


if __name__ == "__main__":
    main()
