"""Smoke test of Fed-MammoBench: 16 images per client, 1 FedAvg round. Run from the repo root:
    python mammobench/smoke_test.py            # ImageNet weights (needs internet the first time)
    python mammobench/smoke_test.py --scratch  # random weights, no download"""
import torch, numpy as np
from torch.utils.data import DataLoader, Subset
from flamby.datasets.fed_mammobench import *
import sys
pre = "--scratch" not in sys.argv
m = Baseline(pretrained=pre)
tr = [p for p in m.parameters() if p.requires_grad]
print("pretrained", pre, "trainable tensors", len(tr), "of", len(list(m.parameters())))
m.train(); print("bn of layer1 in train mode?", m.base_model.layer1[0].bn1.training)
m2 = Baseline(pretrained=False, unfreeze_from="layer4"); m2.train()
print("layer4 trainable:", all(p.requires_grad for p in m2.base_model.layer4.parameters()),
      "layer3 frozen:", not any(p.requires_grad for p in m2.base_model.layer3.parameters()),
      "bn layer3 eval:", not m2.base_model.layer3[0].bn1.training, "bn layer4 train:", m2.base_model.layer4[0].bn1.training)
dls = []
for c in range(NUM_CLIENTS):
    d = FedMammobench(center=c, train=True)
    dls.append(DataLoader(Subset(d, range(0, 16)), batch_size=8, shuffle=True, num_workers=0))
loss = BaselineLoss()
x, y = next(iter(dls[0])); out = m(x); print("logits", out.shape, "loss", float(loss(out, y)))
print("metric", metric(y.numpy(), out.detach().numpy()))
from flamby.strategies.fed_avg import FedAvg
s = FedAvg(dls, m, loss, torch.optim.SGD, 0.01, 2, 1)
s.run(); print("FedAvg 1 round ok; nrounds default", get_nb_max_rounds(100))
