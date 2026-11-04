"""Smoke test of Fed-MammoBench: 16 images per client, 1 FedAvg round. Run from the repo root:
    python mammobench/smoke_test.py [--v2 | --resnet18] [--scratch]
--v2 tests the EfficientNetV2 variant, --resnet18 the ResNet18 one, --scratch uses random weights (no download)."""
import sys

import torch
from torch.utils.data import DataLoader, Subset

if "--v2" in sys.argv:
    from flamby.datasets.fed_mammobench_efficientnetv2 import *  # noqa: F401,F403
elif "--resnet18" in sys.argv:
    from flamby.datasets.fed_mammobench_resnet18 import *  # noqa: F401,F403
else:
    from flamby.datasets.fed_mammobench import *  # noqa: F401,F403

m = Baseline(pretrained="--scratch" not in sys.argv)
print("trainable params", sum(p.numel() for p in m.parameters() if p.requires_grad),
      "of", sum(p.numel() for p in m.parameters()))
dls = [DataLoader(Subset(FedClass(center=c, train=True), range(16)), batch_size=8, shuffle=True)
       for c in range(NUM_CLIENTS)]
loss = BaselineLoss()
x, y = next(iter(dls[0]))
out = m(x)
print("logits", tuple(out.shape), "loss", float(loss(out, y)))
print("metric", metric(y.numpy(), out.detach().numpy()))
from flamby.strategies.fed_avg import FedAvg  # noqa: E402

FedAvg(dls, m, loss, torch.optim.SGD, 0.01, 2, 1).run()
print("FedAvg 1 round ok; default rounds", get_nb_max_rounds(100))
