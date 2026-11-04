"""Build the FLamby manifests of Fed-MammoBench from FedMammoBench's own manifest.

FedMammoBench ships one CSV (`manifests/fedmammobench_norm_0_1.csv`) with a patient-disjoint
train/val/test column. This script keeps that partition untouched and rewrites it in the
layout FLamby expects (the same one as `fed_isic2019/.../train_test_split`):

    image          path of the TIFF relative to the image root (`norm_0_1/<db>/<db>_<n>.tiff`)
    ID_image       FedMammoBench id (`CM000000`)
    patient_id     patient id (the split is disjoint on it)
    source_dataset original database (cmmd, kau-bcmd, cdd-cesm, inbreast)
    binary_target  0 = Benign, 1 = Malignant
    center         client index 0..3 (order of CENTERS)
    split          train | val | test           (partition of FedMammoBench, unchanged)
    fold           same as `split`              (pooled selection, as `fold` in ISIC)
    fold2          <split>_<center>             (per-client selection, as `fold2` in ISIC)

Outputs, next to this script:
    mammobench_split.csv                 all rows of the 4 clients
    manifests/<db>_<split>.csv           one file per client and split (handy to inspect)

    python make_manifests.py --source ../../../../../FedMammoBench/manifests/fedmammobench_norm_0_1.csv \
        --image-root ../../../../../data/preproccesed_julian --check

After a change copy the printed `NUM_TRAIN_SAMPLES` and `alpha` into common.py / loss.py.
`--normalization neg1_1` rebuilds the manifests for the [-1, 1] TIFFs instead of [0, 1].
"""
import argparse
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

# Client order: center=i of FedMammobench. Same order as FedMammoBench's federated nodes.
CENTERS = ["cmmd", "kau-bcmd", "cdd-cesm", "inbreast"]
LABELS = {"Benign": 0, "Malignant": 1}
SPLITS = ["train", "val", "test"]

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--source", required=True, help="fedmammobench_norm_0_1.csv")
parser.add_argument("--image-root", default=None, help="folder holding norm_0_1/ (for --check)")
parser.add_argument("--normalization", default="0_1", choices=["0_1", "neg1_1"])
parser.add_argument("--check", action="store_true", help="verify every TIFF exists")
parser.add_argument("--dry-run", action="store_true", help="only print the numbers")
args = parser.parse_args()

df = pd.read_csv(args.source)
needed = {"ID_image", "source_dataset", "preprocessed_image_path", "classification",
          "patient_id", "split"}
assert needed <= set(df.columns), f"missing columns: {sorted(needed - set(df.columns))}"
assert set(df.source_dataset) == set(CENTERS), f"unexpected databases: {set(df.source_dataset)}"
assert set(df.classification) <= set(LABELS), f"unexpected labels: {set(df.classification)}"
assert set(df.split) == set(SPLITS), f"unexpected splits: {set(df.split)}"
assert df.ID_image.is_unique, "ID_image is not unique"

# Patient-disjoint within each database (patients never leave their database).
per_patient = df.groupby("patient_id").split.nunique()
assert (per_patient == 1).all(), f"{(per_patient > 1).sum()} patients are in more than one split"

out = pd.DataFrame({
    "image": df.preprocessed_image_path.str.replace(
        "norm_0_1/", f"norm_{args.normalization}/", n=1, regex=False),
    "ID_image": df.ID_image,
    "patient_id": df.patient_id,
    "source_dataset": df.source_dataset,
    "binary_target": df.classification.map(LABELS),
    "center": df.source_dataset.map({c: i for i, c in enumerate(CENTERS)}),
    "split": df.split,
    "fold": df.split,
})
out["fold2"] = out.split + "_" + out.center.astype(str)
out = out.sort_values(["center", "split", "ID_image"]).reset_index(drop=True)

if args.check:
    root = args.image_root
    assert root, "--check needs --image-root"
    missing = [p for p in out.image if not os.path.exists(os.path.join(root, p))]
    assert not missing, f"{len(missing)} images missing, e.g. {missing[:3]}"
    print(f"all {len(out)} images found under {root}")

train = out[out.split == "train"]
n0, n1 = (train.binary_target == 0).sum(), (train.binary_target == 1).sum()
print(pd.crosstab([out.source_dataset, out.split], out.binary_target)
      .rename(columns={0: "benign", 1: "malignant"}).to_string())
print(f"\ntrain {len(train)} / val {(out.split == 'val').sum()} / test {(out.split == 'test').sum()}"
      f"   (benign/malignant in train: {n0}/{n1})")
print(f"\nCopy into the package:\n  NUM_TRAIN_SAMPLES = {len(train)}   (common.py)\n"
      f"  alpha=torch.tensor([{len(train) / n0:.4f}, {len(train) / n1:.4f}])   (loss.py)")

if args.dry_run:
    print("\n--dry-run: nothing written")
else:
    out.to_csv(os.path.join(HERE, "mammobench_split.csv"), index=False)
    mdir = os.path.join(HERE, "manifests")
    os.makedirs(mdir, exist_ok=True)
    for c, name in enumerate(CENTERS):
        for s in SPLITS:
            out[(out.center == c) & (out.split == s)].to_csv(
                os.path.join(mdir, f"{name}_{s}.csv"), index=False)
    print(f"\nWrote mammobench_split.csv and manifests/ to {HERE}")
