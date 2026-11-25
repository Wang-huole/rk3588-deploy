# PatchCore 工业缺陷检测项目（MVTec AD）

基于 **PatchCore: A User-Oriented Approach for Anomaly Detection**（Roth et al., 2022）的独立实现，
在 MVTec AD 数据集的 **toothbrush（牙刷）** 和 **pill（药片）** 两个类别上训练与评估。

## 项目原理

PatchCore 属于无监督异常检测：**只用正常样本训练**，缺陷样本只用于测试。

1. 用 ImageNet 预训练的 WideResNet50 作为冻结特征提取器，取 `layer2`、`layer3` 两层的 feature map；
2. 把 feature map 切成 patch 级特征向量，收集所有正常样本的 patch 特征组成 **memory bank**；
3. 用 **coreset greedy k-center** 对 memory bank 做稀疏采样（默认 10%），压缩到可管理规模；
4. 测试时把测试图的 patch 特征与 memory bank 做最近邻搜索，**距离越远越异常**；
5. 像素级：把 patch 距离图上采样回原图大小，得到缺陷热力图；
   图像级：对 patch 距离取最大值（max）聚合得到整张图的异常分数。

## 目录结构

```
industrial_inspection/
├── configs/patchcore_config.yaml   # 所有超参数
├── src/
│   ├── dataset.py                 # MVTec AD 数据集
│   ├── feature_extractor.py        # 冻结 backbone 多层特征提取
│   ├── patchcore.py                # Memory Bank + coreset + 推理
│   ├── metrics.py                  # AUROC / AUPRO
│   ├── visualize.py                # 热力图叠加
│   ├── train.py                   # 训练（建 memory bank）
│   ├── evaluate.py                # 测试集评估
│   ├── infer.py                    # 单图推理（PyTorch 版）
│   ├── export_onnx.py              # 导出部署包（ONNX + npy + json）
│   ├── export_tensorrt.py          # 可选：ONNX → TRT FP16 engine（需装 tensorrt）
│   ├── train_classifier.py         # 缺陷类型分类器训练（迁移学习 + 5 折交叉验证）
│   └── utils.py                    # 配置加载 / transform / dataloader
├── deploy/<category>/              # 部署包（导出后自动生成，可脱离训练环境）
├── outputs/                        # 模型与结果输出（自动生成，已 git 忽略）
├── static/index.html               # Web 前端
├── calibrate.py                    # 统一阈值标定（写回 config.json）
├── deploy_infer.py                # 部署推理（只依赖 onnxruntime，无 PyTorch）
├── classify_infer.py              # 缺陷类型分类器部署推理（只依赖 onnxruntime）
├── demo.py                        # 批量演示 + HTML 报告
├── web_app.py                      # FastAPI 实时检测服务
├── run.py                         # 一键入口（训练 + 评估 + 可视化）
└── .gitignore
```

> 更细的逐文件说明见 `目录结构.md`；项目整体说明（需求 / 难点 / 解决 / 拓展 / 优化 / 双新）见
> `项目整体说明.md`；答辩问答提纲见 `三性六讲提问清单.md`，技术分享大纲见 `技术讲解大纲.md`。

## 环境准备

训练/导出环境：**RTX 5060 Laptop（Blackwell，sm_120）** + CUDA 12.8+ 的 PyTorch。
本项目实际使用 conda 环境 `2401A-torch-base`（Python 3.12.13 / torch 2.13.0+cu130）：

```powershell
# 激活或直接全路径调用
C:\Users\1\anaconda3\envs\2401A-torch-base\python.exe --version
```

若从零搭建：先装 CUDA 12.8+ 版 PyTorch，再装其余依赖：

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

> 导出 ONNX 额外需要：`pip install onnx onnxscript`（已装）。
> 部署推理只需要：`onnxruntime` + `numpy` + `opencv-python` + `Pillow`，**不需要 PyTorch**。

## 运行

一键跑两个类别的训练 + 评估 + 热力图：

```powershell
cd C:\Users\1\Desktop\industrial_inspection
C:\Users\1\anaconda3\envs\2401A-torch-base\python.exe run.py --config configs\patchcore_config.yaml
```

单独跑某个类别或某一步：

```powershell
# 只训练 toothbrush
python -m src.train --category toothbrush

# 只评估（需要先训练好）
python -m src.evaluate --category pill

# 单图推理（PyTorch 版）
python -m src.infer --category pill --image path/to/test.jpg
```

## 输出

- `outputs/<category>/memory_bank.pt`：建好的 memory bank（下次直接加载，不用重训）
- `outputs/<category>/metrics.json`：图像级 / 像素级 AUROC、AUPRO
- `outputs/<category>/heatmaps/`：缺陷热力图对比（原图 / GT mask / 预测热力图）
- `outputs/summary.csv`：两个类别的指标汇总

## 部署与演示

### 1. 导出部署包（一次性的，训练环境执行）

```powershell
python -m src.export_onnx --category toothbrush   # 或 --category pill，或全类别
```

生成 `deploy/<category>/`：
- `backbone.onnx`：特征提取器（输入 [0,255] 图像，归一化已固化在图中）
- `memory_bank.npy`：coreset 后的 memory bank
- `config.json`：推理参数（含标定阈值）

> 已验证：同一张图 ONNX 推理分数与 PyTorch 完全一致（差 0.0000）。

### 2. 部署推理（不需要 PyTorch）

```powershell
# 单张图片（判定阈值默认读 config.json 里 calibrate.py 标定好的值）
python deploy_infer.py --category toothbrush --image path/to/img.png

# 批量文件夹（输出热力图到 outputs/deploy_demo/）
python deploy_infer.py --category pill --input path/to/folder
```

### 3. 阈值标定（上线前必须做）

异常分数是 L2 距离，**正常样本也有基础分数，不能用 0.5 当阈值**。两个类别分数尺度不同，
必须按类别分别标定，并且所有类别共用同一口径：

> **threshold = 正常样本分数的 P99**（误报率约 1%）
> 正常样本来源优先级：产线现场采集 > `test/good` > `train/good`

```powershell
# 全部类别统一标定（写回 deploy/<category>/config.json）
python calibrate.py

# 只标定一个类别
python calibrate.py --category toothbrush

# 用产线现场正常样本目录标定（推荐，最贴近真实部署域）
python calibrate.py --category toothbrush --normal_dir D:\normal_samples

# 只看结果不写回
python calibrate.py --dry_run
```

产物：`outputs/calibration_report.md`、`outputs/calibration_summary.csv`，
以及每个部署包 `config.json` 里的 `threshold`、`threshold_basis`、`threshold_source`、
`threshold_n`、`threshold_normal_stats`、`calibrated_at`。

> **域差异提示**：MVTec toothbrush 的 test/good 分数（0.67~0.84）明显高于 train/good（0.60~0.66）。
> 只用 train/good 的 P95 标定（旧做法，0.62）会把大批正常牙刷判成缺陷；改用统一 P99 口径后
> 阈值落在测试域正常样本上界附近。**正式上线请用产线相机实拍正常样本重新标定**。
> AUROC/AUPRO 与阈值无关，因此这一步不会影响精度指标。

### 4. 批量演示 + HTML 报告

```powershell
# 对测试集全部图片生成可分享的 HTML 报告（图片内嵌，双击即看）
# 不填 --threshold 时自动读 config.json 的标定阈值
python demo.py --category toothbrush

# 自定义输入目录
python demo.py --category pill --input <文件夹> --out outputs/demo
```

产物：`outputs/demo/<category>/result.html` + `vis/`（每张图的热力图叠加）。

## 实时检测 Web 页面

浏览器摄像头实时检测 + 图片上传检测，前后端都在本机：

```powershell
# 启动服务（加载 deploy/ 下全部已导出类别，约 10 秒）
C:\Users\1\anaconda3\envs\2401A-torch-base\python.exe web_app.py
```

然后浏览器打开 **http://127.0.0.1:8000**，支持：

- **摄像头实时检测**：页面直接调本机摄像头，可手动"检测当前帧"或开启自动检测（间隔可调，默认 2 秒）
- **图片上传检测**：点选或拖拽图片，即时出结果
- 类别切换（toothbrush / pill）、阈值实时调整、最近 20 条检测历史
- 每个结果返回：异常分数、正常/缺陷判定、推理耗时、热力图叠加
- **缺陷类型**：中英对照（`contamination 污染` 等），标注来源是分类模型还是启发式，并给出候选类型概率
- **样品尺寸**：药片包围盒像素尺寸、宽高比、面积占比（配置 `mm_per_px` 后额外给出毫米值）
- **检测报告**：一键导出自包含 HTML 报告（判定/分数/类型/尺寸/图像/结论建议），可打印成 PDF，
  同时自动存档到 `outputs/reports/`

文件：`web_app.py`（FastAPI 后端，复用 `deploy_infer.DeployPatchCore`）+ `static/index.html`（前端页面）。

> 推理在本地 CPU 进行，约 0.6~0.9 秒/帧（448 分辨率 backbone 占大头）；KNN 已走 Faiss，
> 比原 numpy 实现快 3.4×（1158ms → 338ms）。如需进一步加速可装 `onnxruntime-gpu`（RTX 5060 支持）。
> 摄像头实时检测需要浏览器授权摄像头权限（localhost 默认可授权）。

## 缺陷类型分类模型（两阶段）

PatchCore 是无监督异常检测，只能回答"**是不是缺陷**"。为了进一步回答"**是哪种缺陷**"，
在 pill 类别上额外训练了一个 8 类分类器（good + 7 类缺陷）。

**两阶段串联架构**：

1. **阶段1（PatchCore）**：异常分数与部署包内标定阈值比较 → 正常 / 缺陷。这一步很准（AUROC 0.98）
2. **阶段2（分类器）**：只有判为"缺陷"才调用分类器细分类型 → color / combined / contamination /
   crack / faulty_imprint / pill_type / scratch。正常样本直接返回 `good`，不浪费分类算力

**分类器设计**：

| 项 | 设置 |
|---|---|
| 主干 | ImageNet 预训练 ResNet18，全参数微调 |
| 输入 | **4 通道 = RGB(3) + PatchCore 异常热力图(1)**，异常图直接定位缺陷位置，对 crack / contamination 这类细小缺陷帮助很大 |
| 图像尺寸 | 384 × 384（比 224 更能保住小缺陷细节） |
| 训练技巧 | 类别不均衡加权 CE + label smoothing 0.05、AdamW + 余弦退火、RandomAffine/Flip/ColorJitter 增广 |
| 验证 | 5 折分层交叉验证，推理时 5 个模型 softmax 平均做集成 |

**数据划分（重要）**：`good` 类只用 `train/good`（267 张），**故意排除 `test/good`** ——
那 26 张正是阈值标定集，排除可避免判废阈值与分类器互相污染。7 类缺陷共 141 张来自 `test/<类>/`
（MVTec AD 只在 test 目录提供缺陷样本，这也是该方案的固有限制，见文末说明）。

**实测结果**：

- **5 折交叉验证准确率 87.75% ± 3.38%**（各折 0.878 / 0.902 / 0.841 / 0.926 / 0.840）← 对未见数据的诚实估计
- 全测试集端到端（167 张）：缺陷检出 **126/141 = 89.4%**，正常样本误报 **1/26**；
  已检出缺陷的类型判定 **126/126 = 100%**，混淆矩阵几乎全对角

> ⚠️ 上面的 100% 是**乐观估计**：交叉验证时每折模型都见过其中 4/5 的图。
> 对全新数据的类型判定准确率应按 **CV 的 87.75%** 来看。

**训练与产物**：

```powershell
C:\Users\1\anaconda3\envs\2401A-torch-base\python.exe -m src.train_classifier --category pill --folds 5 --epochs 18
```

产物：`deploy/pill/classify_fold{0..4}.onnx` + `classify_config.json`（类别表、归一化参数、CV 准确率）。
部署端只需 onnxruntime：`classify_infer.DeployClassifier`。`web_app.py` 检测到 `classify_config.json`
就自动启用分类器，没有则自动回退到启发式规则，功能不受影响。

**推理耗时**（本机 CPU）：

| 场景 | PatchCore | 分类器 | 合计 |
|---|---|---|---|
| 判为缺陷 | ~0.7 s | ~0.55 s | **~1.2 ~ 1.4 s** |
| 判为正常 | ~0.7 s | 不触发（两阶段省算力） | **~0.7 s** |

> 5 折集成让部署包增大约 224 MB（5 × 44.7 MB）。若对体积敏感，可只保留其中一折
> （单折准确率 0.841~0.926），把 `classify_config.json` 的 `files` 改成一个文件即可。

> **MVTec 数据集的固有局限**：官方没有单独的缺陷训练集，缺陷样本只存在于 test 目录，
> 因此这里存在"同分布评测"问题。做课程/竞赛演示完全够用；
> **正式上线必须用产线现场标注数据重训分类器并重新标定阈值**。


## 预期效果（MVTec AD 单类 PatchCore 参考值）

| 类别 | Image AUROC | Pixel AUROC | AUPRO |
|---|---|---|---|
| toothbrush | ~99% | ~98% | ~93% |
| pill | ~99% | ~98% | ~94% |

> 实际数字会受 coreset 比例、输入分辨率、最近邻个数影响，属正常波动。
> **本项目实测值见 `outputs/summary.csv`**（当前：toothbrush 0.988 / 0.993 / 0.967，
> pill 0.982 / 0.973 / 0.971，pill 使用 k=9）。

## 常见问题

- **CUDA out of memory**：把 `configs/patchcore_config.yaml` 里 `batch_size` 调小（如 8）。
- **想换类别**：把 `dataset.categories` 改成 MVTec AD 里其他类别（bottle/cable 等）即可，代码通用。
- **想换 backbone**：改 `model.backbone` 为 `resnet50`（更快但略低精度）。
- **数据集换了位置**：设环境变量 `PATCHCORE_DATA_ROOT` 覆盖 yaml 里的路径，不用改配置：
  `$env:PATCHCORE_DATA_ROOT = "D:\datasets\mvtec_anomaly_detection"`。
- **Web 页面类别列表为空**：模型在 FastAPI `lifespan` 里加载，任何启动方式（`python web_app.py`、
  `uvicorn web_app:app`）都会加载；若仍为空，检查 `deploy/<category>/` 下是否有 `backbone.onnx`。
- **TensorRT 加速**：`src/export_tensorrt.py` 生成 `deploy/<category>/backbone_fp16.engine` 后，
  `deploy_infer` 会自动检测并切到 TRT 后端（未装 tensorrt 或生成失败时回退 ONNX Runtime）；
  本机当前未安装 tensorrt，默认走 ONNX Runtime CPU。
- **改了阈值怎么生效**：`calibrate.py` 写回 `deploy/<category>/config.json`，
  `deploy_infer.py`、`demo.py`、`web_app.py` 都会读取；命令行 `--threshold` 可临时覆盖。

## RK3588S 板端部署（RKNPU2）

NanoPi-M6（Rockchip RK3588S，6 TOPS INT8 NPU）上的实测部署：ONNX 模型经 rknn-toolkit2 编译为 FP16 RKNN，
骨干跑 NPU，memory bank 近邻检索走 CPU 多线程。

### 板端环境

| 项 | 版本 / 配置 |
|---|---|
| 开发板 | NanoPi-M6（RK3588S，4×Cortex-A76 + 4×Cortex-A55，Mali-G610） |
| 系统 | Ubuntu 20.04 LTS aarch64，Python 3.8 |
| 模型转换 | rknn-toolkit2 2.3.2 |
| 板端推理 | rknn-toolkit-lite2 2.5.0 + librknnrt 2.5.0（RKNN Driver 0.9.8） |

### 转换与推理

```bash
# ONNX -> RKNN（FP16），产物在 rknn_deploy/rknn/<category>/backbone_fp16.rknn
python3 rknn_deploy/onnx2rknn.py --category toothbrush
python3 rknn_deploy/onnx2rknn.py --category pill

# 批量推理（60 张 toothbrush 正常样本）
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python3 -u rknn_deploy/rknn_infer.py \
  --category toothbrush --input ~/test_images/normal --model backbone_fp16.rknn
```

> rknn-toolkit2 2.3.2 的三个坑已在 `rknn_deploy/onnx2rknn.py` 中修好：
> `quantized_dtype` 必须是 `w16a16i`（不是 `w16a16`）；导出接口是 `export_rknn`（不是 `export_rknn_model`）；
> FP16 模式要删掉 `quantized_hybrid` 键。

### 实测数据

精度复核（toothbrush，60 张 train/good，全量 bank 18816×1536）：

| 指标 | 数值 |
|---|---|
| 正常样本分数范围 | 0.5973 ~ 0.6569 |
| 均值 / P99 | 0.6161 / 0.6565 |
| 原标定阈值 | 0.8429（余量 0.186，板端无需重标） |
| FP16 与 FP32 对照 | 同一测试图分数均为 1.0103，量化无损 |

性能实测：

| 配置 | 骨干 | KNN | 整帧 | FPS |
|---|---|---|---|---|
| ONNX FP32（CPU，25% bank） | 1351 ms | 1210 ms | 2577 ms | 0.39 |
| RKNN FP16（25% bank，单线程） | 489 ms | 4518 ms | 5029 ms | 0.20 |
| RKNN FP16（25% bank，4 线程） | 约 470 ms | 1232 ms | 1691 ms | 0.59 |
| RKNN FP16（全量 bank，4 线程） | 440~500 ms | 2666~2776 ms | 3160~3270 ms | 0.31 |

> 瓶颈在 CPU 端 KNN 检索（占整帧 81%~85%），骨干加速 2.9 倍但整帧只提升 1.6 倍；
> 压缩 bank 比换算力更有效，INT8 对本场景仅省约 0.2 s，收益有限。

### 文档与录屏

- `docs/RK3588S_PatchCore部署实验报告.docx`：完整实验报告（Word 版）
- `docs/RK3588S_PatchCore部署实验报告.md`：仓库内直接阅读版
- `media/board_demo.mp4`：板端部署全过程录屏（720p / 8fps 压缩版）
