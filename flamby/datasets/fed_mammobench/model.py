import torch.nn as nn


class Baseline(nn.Module):
    """Baseline model: EfficientNet-B0, as in Fed-ISIC2019 (and its binary version).

    We use the EfficientNets architecture that many participants in the ISIC competition
    identified to work best, via Luke Melas-Kyriazi's pytorch reimplementation
    (https://github.com/lukemelas/EfficientNet-PyTorch), with ImageNet weights. The whole
    network is trained (nothing is frozen). The last layer has 2 outputs (benign, malignant)
    so the loss, the metric and the reports work on (N, 2) logits.
    """

    def __init__(self, pretrained=True, arch_name="efficientnet-b0"):
        super(Baseline, self).__init__()
        # imported here so that fed_mammobench_efficientnetv2 does not need the package
        from efficientnet_pytorch import EfficientNet

        self.pretrained = pretrained
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        nftrs = self.base_model._fc.in_features
        print("Number of features output by EfficientNet", nftrs)
        self.base_model._fc = nn.Linear(nftrs, 2)

    def forward(self, image):
        out = self.base_model(image)
        return out
