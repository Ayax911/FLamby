import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torchvision import transforms

# Client order (center=i). Must match CENTERS in dataset_creation_scripts/make_manifests.py.
CENTERS = ["cmmd", "kau-bcmd", "cdd-cesm", "inbreast"]

# ImageNet statistics for 1-channel images: the mean of the 3 ImageNet means / stds, the same
# values as FedMammoBench's exp17 (normalize_mean [0.449], normalize_std [0.226]). The TIFFs
# are float32 in [0, 1] (`norm_0_1`) so this is exactly what the torchvision weights expect.
NORM_MEAN = (0.449, 0.449, 0.449)
NORM_STD = (0.226, 0.226, 0.226)

_HERE = Path(os.path.realpath(__file__)).parent
_SPLIT_CSV = _HERE / "dataset_creation_scripts" / "mammobench_split.csv"
# <repo>/data/preproccesed_julian, where the extracted images live in this workspace.
_DEFAULT_IMAGE_ROOT = _HERE.parents[3] / "data" / "preproccesed_julian"


def resolve_image_root(data_path=None):
    """Folder that contains `norm_0_1/`. Order: `data_path`, env var `MAMMOBENCH_DATA`,
    `dataset_creation_scripts/dataset_location.yaml` (key `dataset_path`), the workspace
    default `<repo>/data/preproccesed_julian`."""
    if data_path is not None:
        if not os.path.isdir(data_path):
            raise ValueError(f"The string {data_path} is not a valid path.")
        return str(data_path)
    candidates = [os.environ.get("MAMMOBENCH_DATA")]
    yaml_file = _HERE / "dataset_creation_scripts" / "dataset_location.yaml"
    if yaml_file.exists():
        candidates.append(yaml.safe_load(yaml_file.read_text()).get("dataset_path"))
    candidates.append(str(_DEFAULT_IMAGE_ROOT))
    for c in candidates:
        if c and os.path.isdir(c):
            return str(c)
    raise FileNotFoundError(
        "Could not find the Mammo-Bench images. Pass data_path=, set MAMMOBENCH_DATA, or "
        "write dataset_path in dataset_creation_scripts/dataset_location.yaml "
        "(the folder that contains norm_0_1/)."
    )


def _read_split(path=_SPLIT_CSV):
    return pd.read_csv(path)


def get_augmentations(train):
    """Tensor pipeline for (3, 224, 224) float images in [0, 1]. Train augmentations are
    the ones of FedMammoBench's exp17 (hflip 0.5, vflip 0.2, rotation 15, blur 0.3)."""
    steps = []
    if train:
        steps += [
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(15),
            transforms.RandomApply(
                [transforms.GaussianBlur(3, sigma=(0.1, 0.6))], p=0.3
            ),
        ]
    steps.append(transforms.Normalize(NORM_MEAN, NORM_STD))
    return transforms.Compose(steps)


class MammobenchRaw(torch.utils.data.Dataset):
    """Pytorch dataset with all the images, labels and centers of Fed-MammoBench
    (benign 0 / malignant 1), 8341 images of 4 databases, 224x224, already preprocessed.

    Images are float32 TIFFs in [0, 1] (`norm_0_1`), one channel; they are replicated to 3
    channels and then normalized with the ImageNet statistics.

    Attributes
    ----------
    image_paths: list[str]
    targets: list[int]
    centers: list[int]
    splits: list[str]
        train / val / test, the FedMammoBench partition
    X_dtype, y_dtype: torch.dtype
    augmentations:
        callable applied to the (3, H, W) tensor
    data_path: str
        Folder that contains `norm_0_1/`. If None see `resolve_image_root`.
    """

    def __init__(
        self,
        X_dtype=torch.float32,
        y_dtype=torch.int64,
        augmentations=None,
        data_path=None,
    ):
        self.image_root = resolve_image_root(data_path)
        self.X_dtype = X_dtype
        self.y_dtype = y_dtype
        df = _read_split()
        self.image_paths = [os.path.join(self.image_root, p) for p in df.image]
        self.targets = df.binary_target.astype(int).tolist()
        self.centers = df.center.tolist()
        self.splits = df.split.tolist()
        self.augmentations = augmentations

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        # mode "F": do NOT .convert(), PIL would clip the floats into 0-255
        array = np.array(image, dtype=np.float32)[np.newaxis, ...]
        image = torch.from_numpy(array).repeat(3, 1, 1)
        target = self.targets[idx]

        if self.augmentations is not None:
            image = self.augmentations(image)

        return (
            image.to(self.X_dtype),
            torch.tensor(target, dtype=self.y_dtype),
        )


class FedMammobench(MammobenchRaw):
    """
    Pytorch dataset with, for each center (database), the images and labels of the
    Fed-MammoBench federated binary classification (benign vs malignant). One can
    instantiate it with train or test data coming from either of the 4 centers
    (0 cmmd, 1 kau-bcmd, 2 cdd-cesm, 3 inbreast) or all data pooled.
    The partition is the one of FedMammoBench (patient-disjoint), fixed in
    `dataset_creation_scripts/mammobench_split.csv`.

    Parameters
    ----------
    center : int, optional
        Default to 0
    train : bool, optional
        True -> "train" partition, False -> "test" partition. Default to True
    pooled : bool, optional
        Default to False
    debug : bool, optional
        Unused, kept for the FLamby signature
    X_dtype : torch.dtype, optional
        Default to torch.float32
    y_dtype : torch.dtype, optional
        Default to torch.int64
    data_path: str
        Folder that contains `norm_0_1/`. Default None (see `resolve_image_root`).
    split : str, optional
        "train", "val" or "test". When given it overrides `train`; this is how the
        validation partition is reached (FLamby itself only has train/test). Validation
        data gets no augmentation. Default None.
    """

    def __init__(
        self,
        center: int = 0,
        train: bool = True,
        pooled: bool = False,
        debug: bool = False,
        X_dtype: torch.dtype = torch.float32,
        y_dtype: torch.dtype = torch.int64,
        data_path: str = None,
        split: str = None,
    ):
        if split is None:
            split = "train" if train else "test"
        assert split in ("train", "val", "test"), f"split={split!r}"

        super().__init__(
            X_dtype=X_dtype,
            y_dtype=y_dtype,
            augmentations=get_augmentations(train=(split == "train")),
            data_path=data_path,
        )

        self.center = center
        self.split = split
        self.train_test = split
        self.pooled = pooled
        self.key = split + "_" + str(self.center)
        df = _read_split()

        if self.pooled:
            df2 = df.query("fold == '" + split + "' ").reset_index(drop=True)
        else:
            assert center in range(len(CENTERS))
            df2 = df.query("fold2 == '" + self.key + "' ").reset_index(drop=True)

        self.image_paths = [os.path.join(self.image_root, p) for p in df2.image]
        self.targets = df2.binary_target.astype(int).tolist()
        self.centers = df2.center.tolist()
        self.splits = df2.split.tolist()


if __name__ == "__main__":
    mydataset = MammobenchRaw()
    print(f"The dataset has {len(mydataset)} elements")
    for split in ("train", "val", "test"):
        d = FedMammobench(pooled=True, split=split)
        print(split, "pooled", len(d), "malignant", sum(d.targets))
        for c in range(len(CENTERS)):
            d = FedMammobench(center=c, split=split)
            print(f"  center {c} ({CENTERS[c]}):", len(d), "malignant", sum(d.targets))
    x, y = FedMammobench(center=0)[0]
    print(x.shape, x.dtype, float(x.min()), float(x.max()), y)
