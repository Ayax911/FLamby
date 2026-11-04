import torch

from flamby.datasets.fed_mammobench.dataset import FedMammobench

# One client per source database: cmmd, kau-bcmd, cdd-cesm, inbreast (see CENTERS in dataset.py).
NUM_CLIENTS = 4
BATCH_SIZE = 64
# 60 pooled epochs = the budget of FedMammoBench's federated runs (30 rounds x 2 local epochs).
NUM_EPOCHS_POOLED = 60
LR = 5e-4
Optimizer = torch.optim.Adam

FedClass = FedMammobench

# Images of the `train` partition (benign 4393 + malignant 2278). Regenerate with
# dataset_creation_scripts/make_manifests.py if the manifests change.
NUM_TRAIN_SAMPLES = 6671


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (NUM_TRAIN_SAMPLES // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
