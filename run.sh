#!/usr/bin/env bash
# Seed 42: the grid runs the paper's Fed-ISIC2019 search (Appendix H) asks for and that were
# never run, in the existing results folder (OUT_DIR, default outputs/isic2019):
#   client lr   10^{-3,-2.5,-2,-1.5,-1,-0.5}      -> new: 0.316
#   server lr   the same six + 1 + 10              -> new: 0.316 and 10
#   mu          {0.001, 0.01, 0.1, 1}              -> nothing new
# That is 78 new runs: FedAvg 1, Cyclic 1, FedProx 4, FedAdam/FedYogi/FedAdagrad/Scaffold 18 each.
# Already finished runs are skipped (their table exists in grid/). Left out on purpose:
#   - the 10 FedAdam/FedYogi runs with slr=1.0 and lr <= 0.1 that already diverged;
#   - FedAvgFineTuning with lr 0.316 (it is not part of the paper's grid).
# The paper's server-lr list is garbled ("1e-0.5, 1, 1e-0.5, 10"); if its third value was
# 10^0.5, add 3.1622776601683795 to GRID_SLRS below (24 more runs).
# Run through scripts/run_isic2019_docker.sh, JOBS containers at a time.
#
# Usage:
#   bash run.sh            # run (re-run the same command to resume)
#   bash run.sh stop       # pause; finished experiments are kept
#   bash run.sh status     # PROGRESS.md
#   DRY_RUN=1 bash run.sh  # print the docker commands without running them
#
# Env: OUT_DIR, JOBS (default 4), plus anything the runner reads (DATA_DIR, CACHE_DIR,
# GPU, WANDB_*).
set -euo pipefail

MODE="${1:-run}"   # run | stop | status
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/isic2019}"
export SEEDS=42 GRID_SEED=42 GRID=1 BEST=0 JOBS="${JOBS:-4}"

# Same spelling as the existing run names (grid/seed42_grid_<S>_lr<LR>_slr<SLR>.csv).
export GRID_LRS="0.001 0.0031622776601683 0.01 0.0316227766016837 0.1 0.316227766016838"
export GRID_MUS="0.001 0.01 0.1 1"
export GRID_SLRS="0.001 0.0031622776601683 0.01 0.0316227766016837 0.1 0.316227766016838 1.0 10.0"

skip=(seed42_grid_FedAvgFineTuning_lr0.316227766016838)
for s in FedAdam FedYogi; do
    for lr in 0.001 0.0031622776601683 0.01 0.0316227766016837 0.1; do
        skip+=("seed42_grid_${s}_lr${lr}_slr1.0")
    done
done
export GRID_SKIP="${skip[*]}"

exec bash "$REPO_DIR/scripts/run_isic2019_docker.sh" "$MODE"
