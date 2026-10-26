#!/usr/bin/env bash
# End-to-end setup for Fed-ISIC2019: conda env -> pip deps -> download -> preprocess -> check.
#
# Usage (from anywhere):
#   bash scripts/setup_isic2019.sh [DATA_DIR]
#   DATA_DIR defaults to ~/datasets/fed_isic2019.
#   ENV_NAME=flamby EXTRAS=all_extra bash scripts/setup_isic2019.sh   # optional overrides
#   ZIP_DIR=/other/disk/tmp bash scripts/setup_isic2019.sh   # temp zip on another disk
#
# Every step is idempotent: an existing env is reused, and the FLamby download/resize scripts
# abort by themselves once dataset_location.yaml marks them complete. Re-running after a
# failure picks up where it left off (except a half-finished download, see step 3).
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${1:-$HOME/datasets/fed_isic2019}"
ENV_NAME="${ENV_NAME:-flamby}"
EXTRAS="${EXTRAS:-all_extra}"   # "isic2019,tests" is enough for this dataset alone
SCRIPTS_DIR="$REPO_DIR/flamby/datasets/fed_isic2019/dataset_creation_scripts"
# Temporary home of the 9.8 GB zip; put it on another disk when DATA_DIR's disk is short.
ZIP_DIR="${ZIP_DIR:-$DATA_DIR}"
# Zip (9.8 GB) and its extracted copy coexist until the zip is deleted.
ZIP_GB=11
EXTRACTED_GB=11
REQUIRED_GB=$((ZIP_GB + EXTRACTED_GB))

log() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- 1. conda env
CONDA_BASE="$(conda info --base 2>/dev/null)"
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"
PY="$ENV_DIR/bin/python"
if [[ -x "$PY" ]]; then
    log "Reusing conda env $ENV_DIR"
else
    log "Creating conda env $ENV_NAME from environment.yml"
    conda env create -n "$ENV_NAME" -f "$REPO_DIR/environment.yml"
fi

# ---------------------------------------------------------------- 2. python deps
# nnunet==1.7.0 depends on the deprecated "sklearn" PyPI package, which refuses to build
# without this flag (the project's CI sets it too).
log "Installing FLamby with extras [$EXTRAS]"
SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True \
    "$PY" -m pip install -e "$REPO_DIR[$EXTRAS]"
"$PY" -c "import torch, flamby; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

# ---------------------------------------------------------------- 3. download
mkdir -p "$DATA_DIR"
if "$PY" - "$SCRIPTS_DIR/dataset_location.yaml" <<'EOF'
import sys, yaml, os
p = sys.argv[1]
sys.exit(0 if os.path.exists(p) and yaml.safe_load(open(p)).get("download_complete") else 1)
EOF
then
    log "Download already complete, skipping"
else
    mkdir -p "$ZIP_DIR"
    free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
    mount_of() { df --output=target "$1" | tail -1; }
    need_space() {  # dir, GB
        if (( $(free_gb "$1") < $2 )); then
            echo "Only $(free_gb "$1") GB free on $(mount_of "$1"); need ~$2 GB." \
                 "Set ZIP_DIR to a folder on another disk to split the load." >&2
            exit 1
        fi
    }
    if [[ "$(mount_of "$ZIP_DIR")" == "$(mount_of "$DATA_DIR")" ]]; then
        need_space "$DATA_DIR" "$REQUIRED_GB"
    else
        need_space "$ZIP_DIR" "$ZIP_GB"
        need_space "$DATA_DIR" "$EXTRACTED_GB"
    fi
    # A previous interrupted run leaves a partial extraction that the download script
    # would mix with the new one; start from a clean folder.
    rm -rf "$DATA_DIR/ISIC_2019_Training_Input" "$ZIP_DIR/ISIC_2019_Training_Input.zip"
    log "Downloading ISIC 2019 into $DATA_DIR, zip in $ZIP_DIR (asks for license once)"
    # cd is required: the scripts import siblings (color_constancy) relatively.
    (cd "$SCRIPTS_DIR" && "$PY" download_isic.py --output-folder "$DATA_DIR" --zip-dir "$ZIP_DIR")
fi

# ---------------------------------------------------------------- 4. preprocess
log "Resizing images (shorter edge 224px, color constancy)"
(cd "$SCRIPTS_DIR" && "$PY" resize_images.py)

# ---------------------------------------------------------------- 5. sanity check
export NO_ALBUMENTATIONS_UPDATE=1
log "Checking the dataset loads per center"
"$PY" - <<'EOF'
from flamby.datasets.fed_isic2019 import FedIsic2019, NUM_CLIENTS
for c in range(NUM_CLIENTS):
    tr, te = FedIsic2019(center=c, train=True), FedIsic2019(center=c, train=False)
    print(f"center {c}: train={len(tr):5d} test={len(te):5d}")
x, y = FedIsic2019(center=0, train=True)[0]
print("sample:", tuple(x.shape), "label:", int(y))
EOF

log "Done. Data in $DATA_DIR"
