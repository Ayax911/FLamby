# Same task, data and training budget as fed_mammobench: only the backbone changes
# (EfficientNetV2 instead of ResNet50), so the two are directly comparable.
from flamby.datasets.fed_mammobench.common import (  # noqa: F401
    BATCH_SIZE,
    LR,
    NUM_CLIENTS,
    NUM_EPOCHS_POOLED,
    NUM_TRAIN_SAMPLES,
    Optimizer,
    get_nb_max_rounds,
)
from flamby.datasets.fed_mammobench_efficientnetv2.dataset import (
    FedMammobenchEfficientnetv2,
)

FedClass = FedMammobenchEfficientnetv2
