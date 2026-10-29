import torch

from flamby.datasets.fed_isic2019_binary.dataset import FedIsic2019Binary

NUM_CLIENTS = 6
BATCH_SIZE = 64
NUM_EPOCHS_POOLED = 20
LR = 5e-4
Optimizer = torch.optim.Adam

FedClass = FedIsic2019Binary

# Train images with a binary label (the 18413 of the multiclass version minus the
# 528 AK, which are not used).
NUM_TRAIN_SAMPLES = 17885


def get_nb_max_rounds(num_updates, batch_size=BATCH_SIZE):
    return (NUM_TRAIN_SAMPLES // NUM_CLIENTS // batch_size) * NUM_EPOCHS_POOLED // num_updates
