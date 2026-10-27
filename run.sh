#!/usr/bin/env bash
# Seed 42, two tasks in the existing results folder (OUT_DIR, default outputs/isic2019):
#   1. the grid runs still pending (GRID=1; finished ones are skipped by FLamby);
#   2. the 8 strategies of flamby/config_isic2019.json again, named seed42_best_<Strategy>
#      (BEST=1): tables in best/ + results_isic2019_best.csv, reports in runs_best/,
#      W&B group best_config. They never touch the benchmark rows or runs/ folders.
# Both share the GPU through scripts/run_isic2019_docker.sh, JOBS containers at a time.
#
# Usage:
#   bash run.sh            # run (re-run the same command to resume)
#   bash run.sh stop       # pause; finished experiments are kept
#   bash run.sh status     # PROGRESS.md (benchmark, grid and best config)
#
# Env: OUT_DIR, JOBS (default 4), plus anything the runner reads (DATA_DIR, CACHE_DIR,
# GPU, WANDB_*).
set -euo pipefail

MODE="${1:-run}"   # run | stop | status
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/isic2019}"
export SEEDS=42 GRID_SEED=42 GRID=1 BEST=1 JOBS="${JOBS:-4}"

exec bash "$REPO_DIR/scripts/run_isic2019_docker.sh" "$MODE"
