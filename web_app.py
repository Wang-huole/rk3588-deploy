"""PatchCore 实时缺陷检测 Web 服务（FastAPI）。

- 后端：加载 deploy/<category>/ 部署包（ONNX + memory bank），复用 deploy_infer.DeployPatchCore
- 前端：static/index.html（摄像头实时检测 + 图片上传）
- 运行：C:\\Users\\1\\anaconda3\\envs\\2401A-torch-base\\python.exe web_app.py
        浏览器打开 http://127.0.0.1:8000
"""
from __future__ import annotations

import base64
import html as html_esc
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Body
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from deploy_infer import DeployPatchCore
from classify_infer import DeployClassifier

DEPLOY_ROOT = Path(__file__).parent / "deploy"
STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_CATEGORY = "toothbrush"
DEFAULT_THRESHOLD = 0.85  # 演示默认；正式上线用现场正常样本标定后写回 config.json

# 药品（pill）缺陷类型 taxonomy（MVTec AD pill 子类）+ 中文对照
PILL_DEFECT_TYPES = ["good", "color", "combined", "contamination", "crack",
                     "faulty_imprint", "pill_type", "scratch"]
DEFECT_TYPE_ZH = {
    "good": "正常",
    "color": "颜色异常",
    "combined": "复合缺陷",
    "contamination": "污染",
    "crack": "裂纹",
    "faulty_imprint": "刻印错误",
    "pill_type": "药品类型错误",
    "scratch": "划痕",
}
# 像素->毫米标定比例；为 None 时仅显示像素尺寸。可在 deploy/<cat>/config.json 写入 "mm_per_px" 启用物理尺寸。
DEFAULT_MM_PER_PX = None

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """启动时加载部署包。

    写在 lifespan 里而不是 __main__，这样 `uvicorn web_app:app`、容器启动等
    任何启动方式都会真正加载模型（否则 /api/status 会返回空列表）。
    """
    print("加载部署包...")
    _load_models()
    print(f"就绪，共 {len(_MODELS)} 个类别: {list(_MODELS)}")
    yield


app = FastAPI(title="PatchCore 实时缺陷检测", lifespan=lifespan)

# ---------------------------------------------------------------------------
# 模型管理：启动时加载全部已导出类别，按需切换
# ---------------------------------------------------------------------------
_MODELS: dict[str, DeployPatchCore] = {}
# 缺陷类型分类器（可选）：存在 classify_config.json 时启用，缺失则回退启发式
_CLASSIFIERS: dict[str, DeployClassifier] = {}


def _load_models() -> None:
    if not DEPLOY_ROOT.exists():
        return
    for d in sorted(DEPLOY_ROOT.iterdir()):
        if d.is_dir() and (d / "backbone.onnx").exists():
            try:
                m = DeployPatchCore(d.name, deploy_root=str(DEPLOY_ROOT))
                _MODELS[d.name] = m
                print(f"  [模型] {d.name}: bank={m.cfg['memory_bank_size']:,} dim={m.cfg['feature_dim']} "
                      f"input={m.size}x{m.size} k={m.k}")
            except Exception as e:  # 单类别失败不影响其他
                print(f"  [跳过] {d.name}: {e}")
            # 缺陷类型分类器（独立可选，失败则用启发式）
            try:
                c = DeployClassifier(d.name, deploy_root=str(DEPLOY_ROOT))
                _CLASSIFIERS[d.name] = c
                acc = c.cfg.get("cv_accuracy_mean")
                print(f"  [分类器] {d.name}: {c.n_models} 折集成, 输入{c.cfg['image_size']}px, "
                      f"CV 准确率 {acc if acc is not None else '—'}")
            except FileNotFoundError:
                pass
            except Exception as e:
                print(f"  [分类器不可用] {d.name}: {e}")


def _get_model(category: str) -> DeployPatchCore:
    if category not in _MODELS:
        raise HTTPException(404, f"类别 {category} 未加载（可用: {list(_MODELS)}）")
    return _MODELS[category]


def _threshold_of(category: str, override: float | None) -> float:
    if override is not None:
        return float(override)
    m = _MODELS[category]
    if "threshold" in m.cfg:
        return float(m.cfg["threshold"])
    return DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/api/status")
def status() -> dict:
    cats = []
    for name, m in _MODELS.items():
        clf = _CLASSIFIERS.get(name)
        cats.append({
            "category": name,
            "memory_bank_size": m.cfg["memory_bank_size"],
            "feature_dim": m.cfg["feature_dim"],
            "image_size": m.cfg["image_size"],
            "threshold": float(m.cfg["threshold"]) if "threshold" in m.cfg else None,
            "default_threshold": DEFAULT_THRESHOLD,
            "has_classifier": clf is not None,
            "classifier_cv_acc": (clf.cfg.get("cv_accuracy_mean") if clf is not None else None),
        })
    return {"categories": cats, "loaded": list(_MODELS), "default_category": DEFAULT_CATEGORY}


# ---------------------------------------------------------------------------
# 附加分析：样品尺寸测量 + 缺陷类型启发式推测
# （PatchCore 仅输出异常分数，以下两项为在其之上叠加的轻量分析）
# ---------------------------------------------------------------------------
def measure_size(img_bgr: np.ndarray, mm_per_px: float | None = None) -> dict:
    """测量样品（前景最大连通域）的包围盒尺寸。

    基于“药片亮、背景暗”的阈值分割；返回像素尺寸、宽高比、面积占比，
    若给定 mm_per_px 标定值则一并给出物理尺寸（mm）。
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {"w_px": 0, "h_px": 0, "aspect": 0.0, "area_ratio": 0.0,
                "w_mm": None, "h_mm": None, "mm_per_px": mm_per_px}
    c = max(cnts, key=cv2.contourArea)
    _, _, bw, bh = cv2.boundingRect(c)
    area_ratio = float(cv2.contourArea(c)) / float(w * h)
    return {
        "w_px": int(bw), "h_px": int(bh),
        "aspect": round(bw / max(bh, 1), 3),
        "area_ratio": round(area_ratio, 3),
        "w_mm": (round(bw * mm_per_px, 2) if mm_per_px else None),
        "h_mm": (round(bh * mm_per_px, 2) if mm_per_px else None),
        "mm_per_px": mm_per_px,
    }


def classify_defect_type(img_bgr: np.ndarray, amap: np.ndarray) -> dict:
    """启发式缺陷类型推测（注意：非训练分类器，仅用于演示 / 占位）。

    PatchCore 输出的 anomaly map  baseline 较高（约 0.54~0.70），直接做相对阈值会
    把整图都当成异常。这里采用“值域归一化”(v-min)/(max-min) + 0.8 阈值，把真正
    的热点分离出来，再按 blob 数量、离心率、面积、位置等规则归入 7 类之一。

    真实部署应替换为专门的分类 / 分割模型；此函数结构便于整体替换。
    """
    h, w = img_bgr.shape[:2]
    amap_u = cv2.resize(amap.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    amin, amax = float(amap_u.min()), float(amap_u.max())
    if amax <= 1e-6 or (amax - amin) < 1e-6:
        return {"defect_type": "—", "confidence": 0.0, "heuristic": True}

    # 值域归一化：把 anomaly map 自身范围拉满后再取 top-20% 热点
    anorm = (amap_u - amin) / max(amax - amin, 1e-6)
    mask = (anorm > 0.8).astype(np.uint8)
    anomaly_ratio = float(mask.sum()) / float(w * h)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = w * h * 0.0001
    cnts = [c for c in cnts if cv2.contourArea(c) > min_area]
    n_blobs = len(cnts)

    eccs, areas, centroids = [], [], []
    for c in cnts:
        areas.append(float(cv2.contourArea(c)))
        if len(c) >= 5:
            ell = cv2.fitEllipse(c)
            major, minor = max(ell[1]), min(ell[1])
            eccs.append(float(np.sqrt(max(0.0, 1 - (minor / major) ** 2))) if major > 0 else 0.0)
        M = cv2.moments(c)
        if M["m00"] > 0:
            centroids.append((M["m10"] / M["m00"] / w, M["m01"] / M["m00"] / h))

    max_ecc = max(eccs) if eccs else 0.0
    mean_blob_area = float(np.mean(areas)) if areas else 0.0

    # 前景（药片）饱和度：pill_type 往往是不同颜色/材质的药片，饱和度明显偏高
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg_bool = fg_mask.astype(bool)
    fg_sat = 0.0
    if fg_bool.sum() > 0:
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        fg_sat = float(hsv[:, :, 1][fg_bool].mean())

    # 决策树（阈值针对 MVTec AD pill 448×448 样本校准，实际数据可再调）
    if anomaly_ratio > 0.03 or n_blobs >= 6:
        # 大范围 / 多连通域 -> 全局缺陷：pill_type 或 color
        if fg_sat > 15.0:
            t, conf = "pill_type", 0.65 + 0.15 * min(anomaly_ratio / 0.1, 1.0)
        else:
            t, conf = "color", 0.55 + 0.15 * min(anomaly_ratio / 0.1, 1.0)
    elif n_blobs == 0:
        t, conf = "—", 0.0
    elif n_blobs >= 3 and mean_blob_area < (w * h * 0.005):
        t, conf = "contamination", 0.6
    elif max_ecc > 0.85:
        # 线状 -> crack（较细、可能多段） / scratch（单段、面积较大）
        if n_blobs == 1 and mean_blob_area > (w * h * 0.005):
            t, conf = "scratch", 0.6
        else:
            t, conf = "crack", 0.6
    elif n_blobs >= 2:
        t, conf = "combined", 0.55
    elif centroids and all(0.3 < cx < 0.7 and 0.3 < cy < 0.7 for cx, cy in centroids):
        t, conf = "faulty_imprint", 0.6
    else:
        t, conf = "contamination", 0.45

    return {"defect_type": t, "confidence": round(float(conf), 3), "heuristic": True}


def _predict_from_image(category: str, img_bgr: np.ndarray, threshold_override: float | None) -> dict:
    model = _get_model(category)
    threshold = _threshold_of(category, threshold_override)

    t0 = time.time()
    score, amap = model.predict(img_bgr)
    patchcore_ms = (time.time() - t0) * 1000
    verdict = "缺陷" if score > threshold else "正常"

    # 附加分析：样品尺寸 + 缺陷类型（PatchCore 本身不输出这两项）
    mm_per_px = model.cfg.get("mm_per_px", DEFAULT_MM_PER_PX)
    size = measure_size(img_bgr, mm_per_px)

    clf = _CLASSIFIERS.get(category)
    dtype: dict = {}
    if verdict == "缺陷":
        if clf is not None:
            # 两阶段串联：PatchCore 判定为缺陷后，再由训练好的分类器细分类型
            try:
                r = clf.predict_defect_type(img_bgr, amap)
                dtype = {"defect_type": r["type"], "confidence": r["conf"], "source": "model",
                         "zh": r["zh"], "probs": r["probs"], "top3": r["top3"]}
            except Exception as e:
                print(f"  [分类器推理失败，回退启发式] {e}")
        if not dtype:
            h = classify_defect_type(img_bgr, amap)
            dtype = {"defect_type": h["defect_type"], "confidence": h["confidence"], "source": "heuristic",
                     "zh": DEFECT_TYPE_ZH.get(h["defect_type"], ""), "probs": {}, "top3": []}
    else:
        # 阈值内 = 正常样本 good
        dtype = {"defect_type": "good", "confidence": 0.0, "source": "threshold",
                 "zh": DEFECT_TYPE_ZH["good"], "probs": {}, "top3": []}

    # 输出图：原图 + 热力图叠加（JPEG 压缩，控制体积）
    ms = (time.time() - t0) * 1000  # 完整推理链路耗时（PatchCore + 分类器 + 尺寸测量）
    h, w = img_bgr.shape[:2]
    if (w, h) != (model.size, model.size):
        img_out = cv2.resize(img_bgr, (model.size, model.size), interpolation=cv2.INTER_LINEAR)
    else:
        img_out = img_bgr
    vis = model.overlay(img_out, amap)
    text = f"score={score:.4f} {verdict}"
    color = (0, 255, 0) if verdict == "正常" else (0, 0, 255)
    cv2.putText(vis, text, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

    def _jpeg_b64(img: np.ndarray) -> str:
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        return base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""

    return {
        "category": category,
        "score": round(float(score), 4),
        "verdict": verdict,
        "threshold": round(threshold, 4),
        "ms": round(ms, 1),
        "patchcore_ms": round(patchcore_ms, 1),
        "defect_type": dtype["defect_type"],
        "defect_conf": dtype["confidence"],
        "defect_source": dtype["source"],
        "defect_zh": dtype["zh"],
        "defect_probs": dtype["probs"],
        "defect_top3": dtype["top3"],
        "classifier_cv_acc": (clf.cfg.get("cv_accuracy_mean") if clf is not None else None),
        "size": size,
        "image_b64": _jpeg_b64(img_out),
        "heatmap_b64": _jpeg_b64(vis),
    }


@app.post("/api/predict")
async def predict(
    category: str = Form(DEFAULT_CATEGORY),
    threshold: float | None = Form(None),
    image_b64: str | None = Form(None),
    file: UploadFile | None = File(None),
) -> JSONResponse:
    """两种入参方式：
    1) multipart: file=<图片文件>, category, threshold
    2) form:      image_b64=<base64 JPEG/PNG>, category, threshold
    """
    if file is not None:
        raw = await file.read()
    elif image_b64:
        try:
            raw = base64.b64decode(image_b64)
        except Exception:
            raise HTTPException(400, "image_b64 不是合法的 base64")
    else:
        raise HTTPException(400, "必须提供 file 或 image_b64")

    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解码图片（仅支持 JPEG/PNG）")

    try:
        # 推理是 CPU 密集同步操作，丢到线程池避免阻塞 FastAPI 事件循环
        result = await run_in_threadpool(_predict_from_image, category, img, threshold)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"推理失败: {e}")
    return JSONResponse(result)


@app.get("/api/health")
def health() -> dict:
    return {"ok": True, "models": list(_MODELS)}


# ---------------------------------------------------------------------------
# 检测报告
# ---------------------------------------------------------------------------
_REPORT_CSS = """
:root { --ink:#1b1f24; --sub:#5b6570; --line:#e2e6ea; --bg:#f6f8fa; --ok:#1a7f37; --bad:#cf222e; }
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:"Segoe UI","Microsoft YaHei",sans-serif; color:var(--ink); background:#fff;
       padding:32px; font-size:13px; line-height:1.6; }
.wrap { max-width:900px; margin:0 auto; }
h1 { font-size:21px; font-weight:700; }
.sub { color:var(--sub); font-size:12px; margin-top:4px; }
.bar { height:3px; background:linear-gradient(90deg,#1a7f37,#2da44e,#cf222e); border-radius:2px; margin:14px 0 22px; }
h2 { font-size:14px; margin:24px 0 10px; padding-bottom:6px; border-bottom:1px solid var(--line); }
.grid { display:grid; grid-template-columns:repeat(3,1fr); gap:10px; }
.card { background:var(--bg); border:1px solid var(--line); border-radius:8px; padding:10px 12px; }
.card .k { font-size:11px; color:var(--sub); }
.card .v { font-size:16px; font-weight:700; margin-top:3px; }
.card .v.ok { color:var(--ok); } .card .v.bad { color:var(--bad); }
.card .v small { font-size:12px; font-weight:600; color:var(--sub); margin-left:4px; }
.imgs { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
.imgs figure { margin:0; }
.imgs img { width:100%; border:1px solid var(--line); border-radius:8px; display:block; }
.imgs figcaption { text-align:center; font-size:11px; color:var(--sub); margin-top:5px; }
table { width:100%; border-collapse:collapse; }
td { padding:7px 4px; border-bottom:1px solid var(--line); font-size:12px; }
td.k { color:var(--sub); width:38%; }
.concl { background:var(--bg); border:1px solid var(--line); border-left:3px solid var(--bad);
         border-radius:6px; padding:12px 14px; }
.concl.ok { border-left-color:var(--ok); }
.note { font-size:11px; color:var(--sub); margin-top:10px; line-height:1.7; }
footer { margin-top:26px; padding-top:12px; border-top:1px solid var(--line);
         font-size:11px; color:var(--sub); display:flex; justify-content:space-between; }
@media print { body { padding:0; } }
"""


def _render_report(payload: dict) -> str:
    """把一次检测结果渲染成自包含的 HTML 报告（可直接打印成 PDF）。"""
    e = html_esc.escape
    dtype = payload.get("defect_type") or "good"
    zh = DEFECT_TYPE_ZH.get(dtype, "未知")
    verdict = payload.get("verdict", "—")
    score = payload.get("score")
    threshold = payload.get("threshold")
    size = payload.get("size") or {}
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    fname = payload.get("_name") or "未命名"

    def _num(v, fmt="{:.4f}", dash="—"):
        return fmt.format(v) if isinstance(v, (int, float)) else dash

    size_txt = (f"{size.get('w_px', 0)} × {size.get('h_px', 0)} px")
    if size.get("w_mm"):
        size_txt += f"（{size.get('w_mm')} × {size.get('h_mm')} mm）"
    aspect = size.get("aspect")
    area_ratio = size.get("area_ratio")

    concl_ok = (verdict == "正常")
    source = payload.get("defect_source", "heuristic")
    type_desc = ("由训练分类模型给出（ResNet18 迁移学习微调，5 折集成）"
                 if source == "model" else
                 "为基于异常热力图空间特征的启发式推测，非训练所得分类模型")
    advice = ("样品各项指标正常，可放行。若需物理尺寸，请在部署包 config.json 中标定 mm_per_px。"
              if concl_ok else
              f"建议隔离该样品并人工复核：缺陷类型{type_desc}，判废前需人工确认。")

    img = payload.get("image_b64") or ""
    vis = payload.get("heatmap_b64") or ""
    img_tag = f'<img src="data:image/jpeg;base64,{img}">' if img else '<div class="card">无图像</div>'
    vis_tag = f'<img src="data:image/jpeg;base64,{vis}">' if vis else '<div class="card">无图像</div>'

    conf = payload.get("defect_conf")
    conf_txt = "—" if not isinstance(conf, (int, float)) or conf <= 0 else f"{conf:.2f}"
    type_line = ("正常样本（good）" if dtype == "good" else f"{dtype}（{zh}）")

    src_txt = {
        "model": "训练分类模型（两阶段：PatchCore 先判缺陷 → 分类器细分类型）",
        "heuristic": "启发式规则推测（非训练分类器）",
        "threshold": "判定阈值内，未触发分类",
    }.get(source, source)
    top3 = payload.get("defect_top3") or []
    top3_html = "".join(
        f'<tr><td class="k">{e(t.get("cls", ""))}（{e(t.get("zh", ""))}）</td>'
        f'<td>{t.get("prob", 0):.4f}</td></tr>' for t in top3
    ) or '<tr><td class="k">—</td><td>—</td></tr>'
    cv_acc = payload.get("classifier_cv_acc")
    cv_line = (f"分类器 5 折交叉验证平均准确率 {cv_acc:.2%}"
               if isinstance(cv_acc, (int, float)) else "分类器交叉验证准确率：—")

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>检测报告 {e(fname)}</title><style>{_REPORT_CSS}</style></head>
<body><div class="wrap">
  <h1>工业视觉缺陷检测报告</h1>
  <div class="sub">PatchCore 异常检测 · 样品尺寸测量 · 缺陷类型推测</div>
  <div class="bar"></div>

  <div class="grid">
    <div class="card"><div class="k">判定结果</div><div class="v {'ok' if concl_ok else 'bad'}">{e(verdict)}</div></div>
    <div class="card"><div class="k">异常分数</div><div class="v">{_num(score)}</div></div>
    <div class="card"><div class="k">判定阈值</div><div class="v">{_num(threshold)}</div></div>
    <div class="card"><div class="k">缺陷类型</div><div class="v {'ok' if concl_ok else 'bad'}">{e(dtype)}<small>{e(zh)}</small></div></div>
    <div class="card"><div class="k">类型置信度</div><div class="v">{conf_txt}</div></div>
    <div class="card"><div class="k">推理耗时</div><div class="v">{_num(payload.get('ms'), '{:.0f}')} ms</div></div>
  </div>

  <h2>检测图像</h2>
  <div class="imgs">
    <figure>{img_tag}<figcaption>输入画面</figcaption></figure>
    <figure>{vis_tag}<figcaption>检测热力图</figcaption></figure>
  </div>

  <h2>样品尺寸</h2>
  <table>
    <tr><td class="k">包围盒尺寸</td><td>{e(size_txt)}</td></tr>
    <tr><td class="k">宽高比</td><td>{e(str(aspect)) if aspect is not None else '—'}</td></tr>
    <tr><td class="k">前景面积占比</td><td>{e(f'{area_ratio:.1%}') if isinstance(area_ratio, (int, float)) else '—'}</td></tr>
    <tr><td class="k">毫米标定</td><td>{'已标定 (mm_per_px=' + e(str(size.get('mm_per_px'))) + ')' if size.get('w_mm') else '未标定（仅像素尺寸）'}</td></tr>
  </table>

  <h2>基本信息</h2>
  <table>
    <tr><td class="k">检测时间</td><td>{e(ts)}</td></tr>
    <tr><td class="k">模型类别</td><td>{e(payload.get('category', '—'))}</td></tr>
    <tr><td class="k">样本文件</td><td>{e(fname)}</td></tr>
    <tr><td class="k">缺陷类型（中/英）</td><td>{e(type_line)}</td></tr>
    <tr><td class="k">类型判定来源</td><td>{e(src_txt)}</td></tr>
    <tr><td class="k">{e(cv_line)}</td><td></td></tr>
  </table>

  <h2>候选类型概率</h2>
  <table>{top3_html}</table>

  <h2>结论与处置建议</h2>
  <div class="concl {'ok' if concl_ok else ''}">
    <div><b>结论：</b>该样品判定为「{e(verdict)}」，异常分数 {_num(score)}（阈值 {_num(threshold)}），缺陷类型 {e(type_line)}。</div>
    <div style="margin-top:6px"><b>建议：</b>{advice}</div>
  </div>
  <div class="note">
    说明：判定结果与异常分数来自 PatchCore 记忆库 KNN 异常分数（部署包内标定阈值）；
    缺陷类型{type_desc}，判废前须人工确认；
    样品尺寸为图像前景包围盒测量值，未标定 mm_per_px 时仅有像素意义。
  </div>

  <footer>
    <span>工业视觉缺陷检测系统 · 自动生成</span>
    <span>{e(ts)}</span>
  </footer>
</div></body></html>"""


@app.post("/api/report")
def report(payload: dict = Body(...)) -> Response:
    """生成单次检测的 HTML 报告：既落盘存档，也直接作为附件返回给前端下载。"""
    content = _render_report(payload)
    out_dir = Path(__file__).parent / "outputs" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    fp = out_dir / f"report_{stamp}.html"
    try:
        fp.write_text(content, encoding="utf-8")
    except Exception as e:  # 存档失败不影响下载
        print(f"  [报告存档失败] {e}")
        fp = None
    headers = {
        "Content-Disposition": f'attachment; filename="detection_report_{stamp}.html"',
        "X-Report-Path": str(fp) if fp else "",
    }
    return Response(content=content, media_type="text/html; charset=utf-8", headers=headers)


@app.get("/api/reports")
def list_reports() -> dict:
    """列出已存档的报告（倒序，最多 50 条）。"""
    out_dir = Path(__file__).parent / "outputs" / "reports"
    if not out_dir.exists():
        return {"reports": []}
    items = []
    for p in sorted(out_dir.glob("report_*.html"), reverse=True)[:50]:
        items.append({
            "name": p.name,
            "path": str(p),
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(p.stat().st_mtime)),
        })
    return {"reports": items}


# 静态资源（前端页面）
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


if __name__ == "__main__":
    import uvicorn

    print("浏览器打开: http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
