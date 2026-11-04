import os

import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50

_BLOCKS = ["conv1", "bn1", "layer1", "layer2", "layer3", "layer4"]


class Baseline(nn.Module):
    """ResNet50 (ImageNet) + MLP head, the recipe of FedMammoBench's exp17.

    Backbone: torchvision `resnet50` with the `IMAGENET1K_V2` weights, without its `fc`.
    Head:     Linear(2048, 1024) -> LeakyReLU(0.2) -> Dropout(0.5) -> Linear(1024, 2).
    The 2 logits (benign, malignant) go to `BaselineLoss` and `metric` as in Fed-ISIC2019.

    Parameters
    ----------
    pretrained : bool
        Load the ImageNet weights (downloaded and cached by torchvision on first use).
    unfreeze_from : str, optional
        "none" (the default, as exp17: the whole backbone is frozen and only the head trains),
        or the first block that is trained: layer4, layer3, layer2, layer1, bn1, conv1.
        "all" is the same as "conv1". If None, the env var MAMMOBENCH_UNFREEZE is read
        (default "none"): it is how the Docker runner changes it, because
        fed_benchmark.py calls Baseline() without arguments.
    freeze_bn_stats : bool
        Keep the BatchNorm layers of the frozen blocks in eval mode. The running
        statistics are buffers, which FLamby's strategies do not aggregate, so letting each
        client update them would give every client a different backbone. Default True.
    """

    def __init__(self, pretrained=True, unfreeze_from=None, freeze_bn_stats=True):
        super(Baseline, self).__init__()
        if unfreeze_from is None:  # fed_benchmark.py builds Baseline() without arguments
            unfreeze_from = os.environ.get("MAMMOBENCH_UNFREEZE", "none")
        if unfreeze_from == "all":
            unfreeze_from = "conv1"
        if unfreeze_from != "none" and unfreeze_from not in _BLOCKS:
            raise ValueError(f"unfreeze_from={unfreeze_from!r}, use none or one of {_BLOCKS}")
        self.pretrained = pretrained
        self.unfreeze_from = unfreeze_from
        self.freeze_bn_stats = freeze_bn_stats

        base = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        nftrs = base.fc.in_features
        base.fc = nn.Identity()
        self.base_model = base
        self.head = nn.Sequential(
            nn.Linear(nftrs, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Linear(1024, 2),
        )

        trainable = False
        for name in _BLOCKS:
            trainable = trainable or name == unfreeze_from
            for p in getattr(base, name).parameters():
                p.requires_grad = trainable
        self._frozen_blocks = [b for b in _BLOCKS if not (
            unfreeze_from != "none" and _BLOCKS.index(b) >= _BLOCKS.index(unfreeze_from))]

    def train(self, mode=True):
        super().train(mode)
        if mode and self.freeze_bn_stats:
            for name in self._frozen_blocks:
                for m in getattr(self.base_model, name).modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        return self

    def forward(self, image):
        return self.head(self.base_model(image))
