"""冻结 backbone 多层特征提取。

PatchCore 用 WideResNet50 的 layer2 和 layer3 输出，
在 forward 时通过 forward hook 捕获这两层的 feature map。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
import torchvision.models as models


class FeatureExtractor(nn.Module):
    """从 ImageNet 预训练 backbone 中抽取指定层特征。

    Args:
        backbone_name: wide_resnet50_2 / resnet50
        layers: 要抽取的层名，如 ["layer2", "layer3"]
    """

    LAYER_MAP = {
        "wide_resnet50_2": models.wide_resnet50_2,
        "resnet50": models.resnet50,
    }

    def __init__(self, backbone_name: str = "wide_resnet50_2", layers: List[str] = ("layer2", "layer3")):
        super().__init__()
        if backbone_name not in self.LAYER_MAP:
            raise ValueError(f"不支持的 backbone: {backbone_name}，可选 {list(self.LAYER_MAP)}")

        # 加载预训练权重
        try:
            backbone = self.LAYER_MAP[backbone_name](weights="IMAGENET1K_V1")
        except TypeError:
            backbone = self.LAYER_MAP[backbone_name](pretrained=True)

        self.backbone_name = backbone_name
        self.layers = list(layers)
        self._features: Dict[str, torch.Tensor] = OrderedDict()

        # 注册 forward hook
        self._handles = []
        for name in self.layers:
            layer = dict([*backbone.named_modules()])[name]
            self._handles.append(layer.register_forward_hook(self._make_hook(name)))

        # 冻结全部参数
        self.model = backbone
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def _make_hook(self, name: str):
        def hook(_module, _inp, out):
            self._features[name] = out
        return hook

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        self._features.clear()
        with torch.no_grad():
            _ = self.model(x)
        # 按注册顺序返回
        return OrderedDict((name, self._features[name]) for name in self.layers)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
