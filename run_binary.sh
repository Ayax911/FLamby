#!/usr/bin/env bash
# Fed-ISIC2019 binary (benign vs malignant, AK dropped): the whole experiment set, seed 42,
# the same one that was run for the 8-class task:
#   benchmark   Pooled + 6 Local + the 8 strategies of config_isic2019_binary.json
#   best        those 8 strategies again, one container each (seed42_best_*)
#   grid        the paper's search (Appendix H):
#                 client lr   10^{-3,-2.5,-2,-1.5,-1,-0.5}   (6)
#                 server lr   the same six + 1 + 10          (8)
#                 mu          {0.001, 0.01, 0.1, 1}          (4)
#               6 x (FedAvg, Cyclic, FedAvgFineTuning + 4 FedProx + 4 x 8 server-lr) = 234 runs,
#               minus FedAvgFineTuning with lr 0.316 (not part of the paper's grid) = 233.
# Unlike run.sh (8-class) nothing is left out for divergence: whether a binary run diverges
# is not known beforehand. Diverged runs end with NaN in their table; check them afterwards.
# Runs through scripts/run_isic2019_docker.sh with DATASET=fed_isic2019_binary, JOBS
# containers at a time. Finished runs are skipped, so re-running the same command resumes.
#
# Usage:
#   bash run_binary.sh            # run (re-run the same command to resume)
#   bash run_binary.sh stop       # pause; finished experiments are kept
#   bash run_binary.sh status     # PROGRESS.md
#   DRY_RUN=1 bash run_binary.sh  # print the docker commands without running them
#   DEBUG=1 OUT_DIR=<dir>-debug bash run_binary.sh   # 1 round/epoch smoke test
#
# Env: OUT_DIR (default outputs/isic2019_binary), JOBS (default 4), BENCH/BEST/GRID (default 1),
# plus anything the runner reads (DATA_DIR, CACHE_DIR, GPU, WANDB_*).
set -euo pipefail

MODE="${1:-run}"   # run | stop | status
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DATASET=fed_isic2019_binary
export OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/isic2019_binary}"
export SEEDS=42 GRID_SEED=42 BENCH="${BENCH:-1}" BEST="${BEST:-1}" GRID="${GRID:-1}"
export JOBS="${JOBS:-4}"

# Same spelling as the 8-class run names (grid/seed42_grid_<S>_lr<LR>_slr<SLR>.csv).
export GRID_LRS="0.001 0.0031622776601683 0.01 0.0316227766016837 0.1 0.316227766016838"
export GRID_MUS="0.001 0.01 0.1 1"
export GRID_SLRS="0.001 0.0031622776601683 0.01 0.0316227766016837 0.1 0.316227766016838 1.0 10.0"
export GRID_SKIP="seed42_grid_FedAvgFineTuning_lr0.316227766016838"

exec bash "$REPO_DIR/scripts/run_isic2019_docker.sh" "$MODE"
