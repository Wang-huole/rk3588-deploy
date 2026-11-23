# RK3588 部署与 NPU 量化加速指南（PatchCore 工业缺陷检测）

本目录把现有 ONNX 部署包迁移到 **RK3588（Rockchip NPU，6 TOPS）**：卷积 backbone 转到
**RKNN INT8** 走 NPU 加速，memory bank 最近邻留在板端 CPU 计算。

---

## 1. 现状盘点（已核对）

| 模型 | 文件 | 输入 | 输出 | 算子 | RKNN 可行性 |
|---|---|---|---|---|---|
| PatchCore backbone | `deploy/<cat>/backbone.onnx` | RGB `[1,3,448,448]`，像素 0–255 | feat_layer2 `[1,512,56,56]`、feat_layer3 `[1,1024,28,28]` | Conv/Add/Sub/Div/MaxPool/Relu | **可直接转 RKNN，NPU 友好** |
| 缺陷分类器 | `deploy/pill/classify_fold{0..4}.onnx` | 4 通道 `[1,4,384,384]`（RGB+异常图） | logits `[1,8]` | Conv/Gemm/MaxPool/ReduceMean/Relu/Reshape | **4 通道首层需特殊处理，见第 6 节** |

- 归一化（`/255`、ImageNet mean/std）已固化在 backbone ONNX 图内，板端/RKNN 无需再做。
- memory bank：`deploy/<cat>/memory_bank.npy`（toothbrush 18816×1536，pill 20000×1536，已 L2 归一化），是向量集合，**不参与 NPU 量化**，在 CPU 上做 KNN。
- 标定阈值已写在 `config.json` 的 `threshold` 字段，板端直接读取。

> 加速分工：**NPU 只跑卷积特征提取（backbone）**；KNN、上采样、分数聚合、可视化在 CPU。这是 PatchCore 在边缘端的标准切分——bank 是浮点向量库，量化到 NPU 既不划算也损失精度。

---

## 2. 目录与产物

```
rknn_deploy/
├── README_RK3588.md            # 本文件
├── requirements_board.txt      # 板端 Python 依赖
├── prepare_calib.py           # 生成 INT8 校准清单（PC 执行）
├── onnx2rknn.py               # ONNX -> RKNN INT8/FP16 转换（Linux 转换机执行）
├── check_onnx.py              # 转换前 ONNX 结构/算子核查（PC 执行）
├── rknn_infer.py              # 板端 RKNN 推理器（NPU + CPU KNN）
├── calib/
│   ├── toothbrush_calib.txt   # 校准图清单（已生成：72 张正常图）
│   └── pill_calib.txt          # 校准图清单（已生成：293 张正常图）
└── rknn/
    └── <category>/backbone_int8.rknn   # 转换后产物（转换机生成后拷回）
```

---

## 3. 整体流程（三段）

```
[PC 训练环境]                [Linux 转换机 x86/ARM64]            [RK3588 板端]
 现有 deploy/*.onnx  ──►  onnx2rknn.py 量化编译  ──►  *.rknn + *.npy + config.json
        │                         │                              │
  prepare_calib.py           生成 backbone_int8.rknn        rknn_infer.py 运行
  check_onnx.py 核查         （校准清单随包带入）             NPU 提特征 + CPU KNN
```

> RKNN-Toolkit2 **没有 Windows 版**。转换必须在 x86 Linux 或 ARM64 Linux 上做；
> 板端只装轻量的 `rknn_toolkit_lite2`（runtime），不做转换。

### 3.1 PC 侧：准备校准集（已完成，可重跑）

```powershell
cd C:\Users\1\Desktop\industrial_inspection
python rknn_deploy/prepare_calib.py
# 可选：把图复制进包，便于整目录拷到转换机
# python rknn_deploy/prepare_calib.py --copy
```

校准集必须用**正常样本**（与训练同分布、覆盖真实光照/角度）。当前取自 MVTec `train/good + test/good`；
**正式上线请换成产线相机实拍的正常图**，否则 INT8 激活值分布对不上，异常分数会偏移。

### 3.2 转换机：安装 Toolkit2 并转换

```bash
# 1) 安装 RKNN-Toolkit2（版本与板端 librknnrt 对齐，推荐 2.3.x / 2.4.x）
pip install rknn-toolkit2

# 2) 把项目 deploy/ 与 rknn_deploy/calib/ 拷到转换机后：
cd industrial_inspection

# 3) 转换 toothbrush / pill 的 backbone（INT8 量化）
python rknn_deploy/onnx2rknn.py --category toothbrush
python rknn_deploy/onnx2rknn.py --category pill

# 精度对照：先出一版 FP16 RKNN，跑同一批评测图对比异常分数/AUROC
python rknn_deploy/onnx2rknn.py --category pill --no-quant
```

产物：`rknn_deploy/rknn/<category>/backbone_int8.rknn`（以及可选 `backbone_fp16.rknn`）。

转换时的关键配置（见 `onnx2rknn.py`）：
- `target_platform="rk3588"`；
- `mean_values=0 / std_values=1`：归一化在 ONNX 图内，**不要重复减均值除标准差**；
- 固定 batch=1、448×448；
- 开启混合量化，特征图输出层保留精度，降低异常距离偏移。

### 3.3 板端：部署运行

把以下内容拷到 RK3588（保持目录相对结构，`deploy/` 与 `rknn_deploy/` 同级）：

```
deploy/<category>/config.json
deploy/<category>/backbone.onnx (+ .onnx.data)   # 转换 RKNN 时需要；板端推理只读 .rknn
deploy/<category>/memory_bank.npy
rknn_deploy/rknn/<category>/backbone_int8.rknn   # 转换机生成后放入
rknn_deploy/rknn_infer.py / board_pipeline.py / classify_infer.py / visualize.py
rknn_deploy/requirements_board.txt
```

#### 一条命令打包并传到板子（PC 侧 PowerShell）

已生成打包清单 `.ship_list.txt`（主链路，不含 5 折分类器权重）。在项目根目录执行：

```powershell
$base="C:\Users\1\Desktop\industrial_inspection"
C:\Windows\System32\tar.exe -czf "$base\rknn_deploy\ship_rk3588.tar.gz" -C $base -T "$base\rknn_deploy\.ship_list.txt"
C:\Windows\System32\OpenSSH\scp.exe "$base\rknn_deploy\ship_rk3588.tar.gz" pi@192.168.31.103:/home/pi/
# 板子上解压（SSH 过去执行）
C:\Windows\System32\OpenSSH\ssh.exe pi@192.168.31.103 "mkdir -p /home/pi/industrial_inspection && tar -xzf /home/pi/ship_rk3588.tar.gz -C /home/pi/industrial_inspection"
```

> 说明：`backbone.onnx(.data)` 与 memory bank 是转换/运行必需；5 折分类器 `classify_fold*.onnx(.data)`
> 属可选 CPU 第二阶段，主链路包已排除以减小体积（约 301MB）。需要分类器时再单独传 `deploy/pill/classify_*`。

板端安装依赖：

```bash
pip install -r rknn_deploy/requirements_board.txt
# 安装 SDK 自带 rknnlite（rknn_toolkit_lite2），并确认 librknnrt.so 在 LD_LIBRARY_PATH
```

运行单图 / 批量：

```bash
# 单图
python rknn_deploy/rknn_infer.py --category toothbrush --image /data/test/xxx.png

# 批量目录（输出热力图到 outputs/rknn_demo/）
python rknn_deploy/rknn_infer.py --category pill --input /data/test_images

# 指定模型（默认 backbone_int8.rknn；可用 FP16 对照）
python rknn_deploy/rknn_infer.py --category pill --image a.png --model backbone_fp16.rknn
```

推理接口：`score, amap, stats = model.predict(img_bgr)`，`stats` 含 `total_ms / fps / npu_ms / knn_ms`，可直接替换进
`web_app.py`（把 `from deploy_infer import DeployPatchCore` 换成 `RknnPatchCore`）。

### FPS 与耗时显示

- 单图命令行输出：`推理 XXms, X.X FPS | NPU XXms, KNN XXms`；可视化图上也叠加 `XXms X.X FPS`。
- 批量结束打印：平均耗时、平均 FPS、峰值/最低 FPS。
- `board_pipeline.py` 返回字段：`total_ms / fps`（端到端，含分类器）、`patchcore_ms / patchcore_fps`、
  `npu_ms / knn_ms / classifier_ms`。
- FPS 口径：`FPS = 1000 / 单帧总耗时(ms)`，含预处理、NPU 提特征、CPU KNN（分类器触发时另计）；
  连续帧测帧率时建议预热 1~2 帧再统计平均值，避免首帧加载/缓存抖动。

---

## 4. 量化精度校验（上线前必做）

INT8 量化会让特征距离产生偏移，**阈值不能直接照搬 FP32**。按下面顺序核对：

1. **同一批图对比三版输出**：原 ONNX(FP32) → RKNN FP16 → RKNN INT8，记录每张图的异常分数。
   板端可直接跑对照脚本（需同时有 deploy 包和 .rknn）：

   ```bash
   python rknn_deploy/compare_precision.py --category pill --input /data/test_images
   # FP16 对照: --model backbone_fp16.rknn
   ```

   脚本输出每张图两后端分数、绝对/相对偏差、Spearman 排序一致性、阈值判定一致率。
2. 看两个指标：
   - 分数相对偏差：正常图 INT8 分数相对 FP32 的偏移（建议正常样本分数均值偏差 < 3%）；
   - 排序一致性：AUROC / 缺陷召回率不明显掉点（参考原指标 toothbrush 图像 AUROC 0.988、pill 0.982）。
3. **用板端 INT8 输出的正常样本重新标定阈值**（P99），回写 `config.json` 的 `threshold`：
   板端或转换机上跑一批正常图，取分数 P99 作为新阈值；不要沿用 FP32 阈值，否则误报率会变。
4. 若 INT8 掉点明显：
   - 增加校准集数量/场景覆盖（建议 100–300 张正常图）；
   - 退回 FP16 RKNN（`--no-quant`）作为精度兜底；
   - 个别敏感层在转换配置里设为混合精度（保持 FP16）。

---

## 5. 性能与体积预期（RK3588）

- backbone 是 WideResNet50-2（448 输入），NPU INT8 下单帧特征提取通常为几十到百余毫秒量级；
  实测以板端 `rknn_infer.py` 打印的 `推理 ms` 为准（含 CPU KNN）。
- KNN 规模：pill 20000×1536、toothbrush 18816×1536，CPU 分块计算；如需进一步压缩，
  可调大 coreset 比例的反向（减少 bank 行数）或在板端用 Faiss-arm64 / 距离矩阵分块优化。
- 若实时性要求高（连续帧）：backbone 固定 batch=1，可在板端做帧缓存与异步 KNN；
  WideResNet50-2 较重，必要时把 backbone 换成 `resnet50`（需重训/重导 ONNX）以换帧率。

---

## 6. 缺陷类型分类器（pill，4 通道）的限制与可选方案

分类器首层是 **4 通道卷积（RGB 3 + 异常热力图 1）**，RKNN 对非 3 通道首层卷积的支持取决于
SDK 版本，不能保证整层落 NPU。三种处理方式，按优先级：

1. **板端 ONNX Runtime CPU 跑分类器（推荐，改动最小）**：两阶段架构本来就是“判为缺陷才调用分类器”，
   触发率低，CPU 跑 ResNet18 可接受；backbone 走 NPU，分类器保留原 `classify_infer.py`。
2. **拆模型重导出**：把首层 4 通道卷积拆成 3 通道卷积 + 第 4 通道单独支路，再分别转 RKNN；
   需要改 `build_model()` 与导出逻辑，属二次开发。
3. **降级为单 RGB 输入分类器**：去掉异常图通道重训，精度会下降（小缺陷定位信息丢失），仅在
   帧率优先、可接受精度回退时使用。

> 5 折集成在板端体积/算力代价较高（5×44.7MB）。边缘部署可只保留 CV 准确率最高的一折
> （pill fold1 0.9024），把 `classify_config.json` 的 `files` 改成单文件。

---

## 7. 常见问题

- **转换报不支持的算子**：看 build 日志 unsupported 层；当前 backbone 算子已粗筛通过，若版本差异
  出现不支持层，RKNN 会回退 CPU，日志会标注；必要时升 Toolkit2 版本。
- **板端 `from rknnlite` 导入失败**：确认 SDK 的 `rknn_toolkit_lite2/python` 在 `PYTHONPATH`，
  `librknnrt.so` 在库路径，且内核 NPU 驱动（rk3588 npu driver）已加载。
- **分数整体偏高/偏低、误报变多**：多半是校准集分布不符或阈值沿用了 FP32 值；换现场正常图
  重新校准并重标阈值。
- ** `.onnx.data` 外部权重**：本项目 backbone 权重已折叠进 ONNX（核查无外部初始化器）；分类器
  ONNX 带 `.onnx.data`，若转分类器需同目录携带。
- **想在板端保留 ONNX 兜底**：`rknn_infer.py` 目前只走 RKNN；需要回退可参考原 `deploy_infer.py`
  的 ORT 分支加一个后端选择。

---

## 8. 待你确认的环境信息

转换/落地前需要确认（影响 Toolkit2 版本与 runtime 对接）：

1. RK3588 板子的系统（Debian/Ubuntu 版本）与内核 NPU 驱动版本；
2. 手头是否已有 Rockchip RKNN SDK（Toolkit2 + Lite2）及其版本号；
3. 转换在哪台机器做：x86 Linux、板端本机，还是另一台 ARM64 设备；
4. 分类器在边缘端是否必须启用，还是第一阶段“是否缺陷”检测即可演示。
