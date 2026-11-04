#!/usr/bin/env bash
# Fed-MammoBench, backbone EfficientNet-B0, the same experiment set as run_binary.sh (Fed-ISIC2019 binary):
# seed 42, DATASET=fed_mammobench.
#   benchmark   Pooled + 4 Local + the 8 strategies of flamby/config_mammobench.json
#   best        those 8 strategies again, one container each (seed42_best_*)
#   grid        the paper's search (Appendix H): client lr 10^{-3,-2.5,-2,-1.5,-1,-0.5} (6),
#               server lr the same six + 1 + 10 (8), mu {0.001, 0.01, 0.1, 1} (4):
#               6 x (FedAvg, Cyclic, FedAvgFineTuning + 4 FedProx + 4 x 8 server-lr) = 234 runs,
#               minus FedAvgFineTuning with lr 0.316 = 233. Diverged runs end with NaN.
# Finished runs are skipped, so re-running the same command resumes.
#
#   bash mammobench/run_docker_efficientnetb0.sh weights   # once: cache the ImageNet weights
#   bash mammobench/run_docker_efficientnetb0.sh           # run (re-run the same command to resume)
#   bash mammobench/run_docker_efficientnetb0.sh stop      # pause; finished experiments are kept
#   bash mammobench/run_docker_efficientnetb0.sh status    # PROGRESS.md
#   DRY_RUN=1 bash mammobench/run_docker_efficientnetb0.sh
#   DEBUG=1 OUT_DIR=<dir>-debug bash mammobench/run_docker_efficientnetb0.sh   # 1 round/epoch smoke test
#
# Env: DATA_DIR (folder with norm_0_1/, default <repo>/../data/preproccesed_julian), CACHE_DIR
# (default <repo>/../cache_torch), OUT_DIR (default outputs/mammobench), JOBS (4), GPU (0),
# BENCH/BEST/GRID (1), WANDB_API_KEY / WANDB_ENTITY / WANDB_PROJECT / WANDB_MODE,
# MAMMOBENCH_EFFNET (EfficientNetV2 only, so unused here).
set -euo pipefail

MODE="${1:-run}"   # run | weights | stop | status
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DATASET=fed_mammobench
export OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/mammobench}"
export SEEDS=42 GRID_SEED=42 BENCH="${BENCH:-1}" BEST="${BEST:-1}" GRID="${GRID:-1}"
export JOBS="${JOBS:-4}"

export GRID_LRS="0.001 0.0031622776601683 0.01 0.0316227766016837 0.1 0.316227766016838"
export GRID_MUS="0.001 0.01 0.1 1"
export GRID_SLRS="0.001 0.0031622776601683 0.01 0.0316227766016837 0.1 0.316227766016838 1.0 10.0"
export GRID_SKIP="seed42_grid_FedAvgFineTuning_lr0.316227766016838"

exec bash "$REPO_DIR/scripts/run_mammobench_docker.sh" "$MODE"
