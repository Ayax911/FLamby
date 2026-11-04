# Same task, data and training budget as fed_mammobench: only the backbone changes
# (ResNet18 instead of EfficientNet-B0), so the two are directly comparable.
from flamby.datasets.fed_mammobench.common import (  # noqa: F401
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    NUM_EPOCHS_POOLED,
    NUM_TRAIN_SAMPLES,
    Optimizer,
    get_nb_max_rounds,
)
from flamby.datasets.fed_mammobench_resnet18.dataset import (
    FedMammobenchResnet18,
)

FedClass = FedMammobenchResnet18
