import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class Baseline(nn.Module):
    """ResNet18 (ImageNet), the lightweight counterpart of the EfficientNet baselines.

    Same recipe as `fed_mammobench`: torchvision `resnet18` with `IMAGENET1K_V1` weights,
    the whole network trained, and only the last layer replaced by a Linear with 2 outputs
    (benign, malignant).

    Parameters
    ----------
    pretrained : bool
        Load the ImageNet weights (downloaded and cached by torchvision on first use).
    """

    def __init__(self, pretrained=True):
        super(Baseline, self).__init__()
        self.pretrained = pretrained
        self.base_model = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        )
        nftrs = self.base_model.fc.in_features
        print("Number of features output by ResNet18", nftrs)
        self.base_model.fc = nn.Linear(nftrs, 2)

    def forward(self, image):
        return self.base_model(image)
