import os

import torch.nn as nn
from torchvision.models import (
    EfficientNet_V2_L_Weights,
    EfficientNet_V2_M_Weights,
    EfficientNet_V2_S_Weights,
    efficientnet_v2_l,
    efficientnet_v2_m,
    efficientnet_v2_s,
)

_ARCHS = {
    "s": (efficientnet_v2_s, EfficientNet_V2_S_Weights.IMAGENET1K_V1),
    "m": (efficientnet_v2_m, EfficientNet_V2_M_Weights.IMAGENET1K_V1),
    "l": (efficientnet_v2_l, EfficientNet_V2_L_Weights.IMAGENET1K_V1),
}


class Baseline(nn.Module):
    """EfficientNetV2 (ImageNet) + the same MLP head as the ResNet50 baseline.

    Backbone: torchvision `efficientnet_v2_<s|m|l>` (`IMAGENET1K_V1`), without its classifier.
    Head:     Linear(n_features, 1024) -> LeakyReLU(0.2) -> Dropout(0.5) -> Linear(1024, 2).

    Parameters
    ----------
    pretrained : bool
        Load the ImageNet weights (downloaded and cached by torchvision on first use).
    arch : str, optional
        "s" (default), "m" or "l". If None, env var MAMMOBENCH_EFFNET is read.
    unfreeze_from : str, optional
        "none" (the default: backbone frozen, only the head trains, like the ResNet50
        baseline), "all", or K (e.g. "5" or "features.5") to train `features[K:]`
        (features has 8 blocks, 0 = stem). If None, env var MAMMOBENCH_UNFREEZE is read.
    freeze_bn_stats : bool
        Keep the BatchNorm layers of the frozen blocks in eval mode (their running
        statistics are buffers that FLamby's strategies do not aggregate). Default True.
    """

    def __init__(self, pretrained=True, arch=None, unfreeze_from=None, freeze_bn_stats=True):
        super(Baseline, self).__init__()
        arch = (arch or os.environ.get("MAMMOBENCH_EFFNET", "s")).lower()
        if unfreeze_from is None:
            unfreeze_from = os.environ.get("MAMMOBENCH_UNFREEZE", "none")
        if arch not in _ARCHS:
            raise ValueError(f"arch={arch!r}, use one of {sorted(_ARCHS)}")
        factory, weights = _ARCHS[arch]
        base = factory(weights=weights if pretrained else None)
        nftrs = base.classifier[-1].in_features
        base.classifier = nn.Identity()
        self.base_model = base
        self.arch = arch
        self.unfreeze_from = unfreeze_from
        self.freeze_bn_stats = freeze_bn_stats
        self.head = nn.Sequential(
            nn.Linear(nftrs, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
        )

        n_blocks = len(base.features)
        u = str(unfreeze_from).replace("features.", "")
        if u == "none":
            first = n_blocks
        elif u in ("all", "conv1"):
            first = 0
        elif u.isdigit() and int(u) < n_blocks:
            first = int(u)
        else:
            raise ValueError(f"unfreeze_from={unfreeze_from!r}: none, all or 0..{n_blocks - 1}")
        self._first_trainable = first
        for i, block in enumerate(base.features):
            for p in block.parameters():
                p.requires_grad = i >= first

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for block in list(self.base_model.features)[: self._first_trainable]:
                for m in block.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        return self

    def forward(self, image):
        return self.head(self.base_model(image))
