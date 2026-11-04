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
    """EfficientNetV2 (ImageNet), the counterpart of the EfficientNet-B0 baseline.

    Same recipe as `fed_mammobench`: torchvision `efficientnet_v2_<s|m|l>` with ImageNet
    weights, whole network trained, and only the last layer replaced by a Linear with 2
    outputs (benign, malignant); torchvision's dropout before it is kept (as B0 has its own).

    Parameters
    ----------
    pretrained : bool
        Load the ImageNet weights (downloaded and cached by torchvision on first use).
    arch : str, optional
        "s" (default), "m" or "l". If None, env var MAMMOBENCH_EFFNET is read (default "s");
        it is how the Docker runner picks it, because fed_benchmark.py calls Baseline()
        without arguments.
    """

    def __init__(self, pretrained=True, arch=None):
        super(Baseline, self).__init__()
        arch = (arch or os.environ.get("MAMMOBENCH_EFFNET", "s")).lower()
        if arch not in _ARCHS:
            raise ValueError(f"arch={arch!r}, use one of {sorted(_ARCHS)}")
        factory, weights = _ARCHS[arch]
        self.pretrained = pretrained
        self.arch = arch
        self.base_model = factory(weights=weights if pretrained else None)
        nftrs = self.base_model.classifier[-1].in_features
        print("Number of features output by EfficientNetV2", nftrs)
        self.base_model.classifier[-1] = nn.Linear(nftrs, 2)

    def forward(self, image):
        return self.base_model(image)
