"""Regenerate the `binary_target` column of `train_test_split`.

Use it to change which ISIC 2019 classes are malignant / benign / unused, e.g. to move AK
into one of the groups. Every class must be in exactly one of the three groups. Rows of
unused classes get an empty `binary_target` and are dropped by `FedIsic2019Binary`.

    python make_binary_target.py                       # default: MEL BCC SCC vs NV BKL DF VASC
    python make_binary_target.py --malignant MEL BCC SCC AK --benign NV BKL DF VASC
    python make_binary_target.py --dry-run             # only print the numbers

After a change, copy the printed values into the package:
  - `NUM_TRAIN_SAMPLES` in common.py
  - the `alpha` default of `BaselineLoss` in loss.py   (N_train / n_class, benign first)
and rebuild the Docker image (the CSV is baked into it; run_isic2019_docker.sh rebuilds).
"""
import argparse
import os

import pandas as pd

# Order of the 8 one-hot columns == the multiclass `target` index (0 MEL ... 7 SCC).
CLASSES = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]
HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("--malignant", nargs="+", default=["MEL", "BCC", "SCC"], choices=CLASSES)
parser.add_argument("--benign", nargs="+", default=["NV", "BKL", "DF", "VASC"], choices=CLASSES)
parser.add_argument("--csv", default=os.path.join(HERE, "train_test_split"))
parser.add_argument("--dry-run", action="store_true", help="print the numbers, write nothing")
args = parser.parse_args()

both = set(args.malignant) & set(args.benign)
if both:
    parser.error(f"in both groups: {sorted(both)}")
unused = [c for c in CLASSES if c not in args.malignant + args.benign]

df = pd.read_csv(args.csv).drop(columns=["binary_target"], errors="ignore")
# The one-hot columns and `target` must agree, otherwise the CSV is not what we think.
from_onehot = df[CLASSES].values.argmax(axis=1)
assert (from_onehot == df.target.values).all(), "target does not match the one-hot columns"
group = {CLASSES.index(c): 1 for c in args.malignant}
group.update({CLASSES.index(c): 0 for c in args.benign})
df["binary_target"] = df.target.map(group)  # unused classes -> NaN -> empty cell

used = df[df.binary_target.notna()]
train, test = used[used.fold == "train"], used[used.fold == "test"]
n0, n1 = (train.binary_target == 0).sum(), (train.binary_target == 1).sum()
print(f"malignant (1): {args.malignant}\nbenign (0):    {args.benign}\nunused:        {unused}")
print(f"images used: {len(used)} of {len(df)}  (train {len(train)}, test {len(test)})")
print(f"train benign/malignant: {n0}/{n1}   test: "
      f"{(test.binary_target == 0).sum()}/{(test.binary_target == 1).sum()}")
print("\nPer center (train): size, % malignant")
print(train.groupby("center").binary_target.agg(["size", "mean"]).round(3).to_string())
print(f"\nCopy into the package:\n  NUM_TRAIN_SAMPLES = {len(train)}   (common.py)\n"
      f"  alpha=torch.tensor([{len(train) / n0:.4f}, {len(train) / n1:.4f}])   (loss.py)")
if args.dry_run:
    print("\n--dry-run: nothing written")
else:
    df.to_csv(args.csv, index=False)
    print(f"\nWrote binary_target to {args.csv}")
