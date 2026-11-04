from flamby.datasets.fed_mammobench.dataset import CENTERS, FedMammobench, MammobenchRaw  # noqa: F401


class FedMammobenchResnet18(FedMammobench):
    """Same data, partition and centers as `FedMammobench`. It only exists because
    `fed_benchmark.py` derives the dataset class name from the package name
    (fed_mammobench_resnet18 -> FedMammobenchResnet18). Only the model differs."""
