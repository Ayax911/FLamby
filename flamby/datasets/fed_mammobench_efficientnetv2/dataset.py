from flamby.datasets.fed_mammobench.dataset import CENTERS, FedMammobench, MammobenchRaw  # noqa: F401


class FedMammobenchEfficientnetv2(FedMammobench):
    """Same data, partition and centers as `FedMammobench`. It only exists because
    `fed_benchmark.py` derives the dataset class name from the package name
    (fed_mammobench_efficientnetv2 -> FedMammobenchEfficientnetv2). Only the model differs."""
