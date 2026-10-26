#!/usr/bin/env bash
# Runs every Fed-ISIC2019 experiment inside Docker (GPU): for each seed, the pooled baseline,
# the 6 local (one-center) baselines and the 8 federated strategies with the paper's
# hyperparameters (flamby/config_isic2019.json). Optionally, a hyperparameter grid (GRID=1).
#
# Usage (from anywhere; the data must already be preprocessed, see scripts/setup_isic2019.sh):
#   bash scripts/run_isic2019_docker.sh
#   DEBUG=1 bash scripts/run_isic2019_docker.sh                # 1 round/epoch, checks the pipeline
#   SEEDS="42 43 44 45 46" GPU=1 bash scripts/run_isic2019_docker.sh
#   GRID=1 bash scripts/run_isic2019_docker.sh                 # also the grid, seed 42 only
#   DRY_RUN=1 bash scripts/run_isic2019_docker.sh              # print the commands only
#
# Outputs (OUT_DIR):
#   runs/fed_isic2019/seed<S>/<run>/   FedMammoBench-style reports for every run
#   results_isic2019.csv               FLamby's results table (all seeds, all runs)
#   logs/<...>.log                     stdout/stderr of every container
# Re-running is safe: experiments already in results_isic2019.csv are skipped by FLamby.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$HOME/datasets/fed_isic2019}"
DEBUG="${DEBUG:-0}"
if [[ "$DEBUG" == 1 ]]; then
    OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/isic2019_debug}"
else
    OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/isic2019}"
fi
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/torch}"   # EfficientNet weights, downloaded once
IMAGE="${IMAGE:-flamby-isic2019}"
GPU="${GPU:-0}"            # host GPU index the container gets (seen inside as GPU 0)
SEEDS="${SEEDS:-42}"
WORKERS="${WORKERS:-4}"    # dataloader workers per container
GRID="${GRID:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Grid (GRID=1): only on GRID_SEED, whatever SEEDS says. Log-spaced like the paper's search.
GRID_SEED="${GRID_SEED:-42}"
GRID_LRS="${GRID_LRS:-0.001 0.0031622776601683 0.01 0.0316227766016837 0.1}"
GRID_MUS="${GRID_MUS:-0.001 0.01 0.1 1}"
GRID_SLRS="${GRID_SLRS:-0.001 0.0031622776601683 0.01 0.0316227766016837 0.1 1.0}"

log() { printf '\n==> %s\n' "$*"; }

# ---------------------------------------------------------------- 1. checks
command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
if ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "Docker has no NVIDIA runtime: install nvidia-container-toolkit" \
         "and run 'sudo nvidia-ctk runtime configure --runtime=docker'." >&2
    exit 1
fi
for f in ISIC_2019_Training_Input_preprocessed ISIC_2019_Training_Metadata_FL.csv; do
    if [[ ! -e "$DATA_DIR/$f" ]]; then
        echo "Missing $DATA_DIR/$f: run scripts/setup_isic2019.sh first" \
             "or set DATA_DIR." >&2
        exit 1
    fi
done
mkdir -p "$OUT_DIR/logs" "$CACHE_DIR"

# ---------------------------------------------------------------- 2. image
log "Building image $IMAGE from $REPO_DIR"
[[ "$DRY_RUN" == 1 ]] || \
    docker build -f "$REPO_DIR/dockers/Dockerfile.isic2019" -t "$IMAGE" "$REPO_DIR"

# ---------------------------------------------------------------- 3. experiments
# run <log-name> <fed_benchmark.py args...>: one container per call.
run() {
    local name="$1"; shift
    local cmd=(docker run --rm --gpus "device=$GPU" --shm-size=16g
        --user "$(id -u):$(id -g)"
        -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" -v "$CACHE_DIR:/cache/torch"
        -e TORCH_HOME=/cache/torch
        "$IMAGE" "$@" --workers "$WORKERS"
        -rfp /out/results_isic2019.csv --runs-dir /out/runs)
    [[ "$DEBUG" == 1 ]] && cmd+=(--debug)
    log "$name"
    echo "${cmd[*]}"
    [[ "$DRY_RUN" == 1 ]] || "${cmd[@]}" 2>&1 | tee "$OUT_DIR/logs/$name.log"
}

for seed in $SEEDS; do
    # No -s/-scb: pooled + 6 locals + every strategy in the config, in one process.
    run "seed${seed}_benchmark" --seed "$seed"
done

if [[ "$GRID" == 1 ]]; then
    seed="$GRID_SEED"
    for lr in $GRID_LRS; do
        for s in FedAvg Cyclic; do
            run "seed${seed}_grid_${s}_lr${lr}" --seed "$seed" -s "$s" -lr "$lr"
        done
        # 3 fine-tuning steps, as in the config.
        run "seed${seed}_grid_FedAvgFineTuning_lr${lr}" \
            --seed "$seed" -s FedAvgFineTuning -lr "$lr" -nft 3
        for mu in $GRID_MUS; do
            run "seed${seed}_grid_FedProx_lr${lr}_mu${mu}" \
                --seed "$seed" -s FedProx -lr "$lr" -mu "$mu"
        done
        for slr in $GRID_SLRS; do
            for s in FedAdam FedYogi FedAdagrad Scaffold; do
                run "seed${seed}_grid_${s}_lr${lr}_slr${slr}" \
                    --seed "$seed" -s "$s" -lr "$lr" -slr "$slr"
            done
        done
    done
fi

log "Done. Reports in $OUT_DIR/runs/fed_isic2019, table in $OUT_DIR/results_isic2019.csv"
