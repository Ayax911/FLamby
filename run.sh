#!/usr/bin/env bash
# Extra seeds for the Fed-ISIC2019 benchmark (pooled + 6 locals + 8 strategies, no grid),
# in parallel on one GPU. Each seed gets its own OUT_DIR ($OUT_BASE/isic2019_seed<S>) and
# runs scripts/run_isic2019_docker.sh there: FLamby rewrites the whole results CSV, so
# seeds sharing one CSV would drop each other's rows.
#
# Usage:
#   bash run.sh            # seeds 43-46, all four at once
#   bash run.sh stop       # pause every seed (finished experiments are kept)
#   bash run.sh status     # PROGRESS.md of every seed
# Re-running `bash run.sh` resumes: only pending experiments run.
#
# Env: SEEDS (default "43 44 45 46"), JOBS (seeds at the same time, default 4), OUT_BASE,
# plus anything scripts/run_isic2019_docker.sh reads (DATA_DIR, CACHE_DIR, GPU, WANDB_*).
set -euo pipefail

MODE="${1:-run}"   # run | stop | status
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$REPO_DIR/scripts/run_isic2019_docker.sh"
SEEDS="${SEEDS:-43 44 45 46}"
JOBS="${JOBS:-4}"
OUT_BASE="${OUT_BASE:-$REPO_DIR/outputs}"
export GRID=0

out_dir() { echo "$OUT_BASE/isic2019_seed$1"; }

case "$MODE" in
    stop)
        # The runner's `stop` stops every flamby container, so mark all seeds as paused
        # first; otherwise the other seeds would report their cut runs as failures.
        for s in $SEEDS; do mkdir -p "$(out_dir "$s")"; touch "$(out_dir "$s")/STOP"; done
        docker ps -q --filter label=flamby-isic2019 | xargs -r docker stop -t 30 >/dev/null
        echo "Paused seeds: $SEEDS. Resume with: bash run.sh"
        exit 0 ;;
    status)
        for s in $SEEDS; do
            echo "================ seed $s"
            OUT_DIR="$(out_dir "$s")" SEEDS="$s" bash "$RUNNER" status | sed -n '1,25p'
        done
        exit 0 ;;
    run) ;;
    *) echo "unknown mode: $MODE (run | stop | status)" >&2; exit 1 ;;
esac

# Build once here so the parallel runners only hit the docker cache.
[[ "${DRY_RUN:-0}" == 1 ]] || docker build -f "$REPO_DIR/dockers/Dockerfile.isic2019" -t "${IMAGE:-flamby-isic2019}" \
    "$REPO_DIR"

mkdir -p "$OUT_BASE"
FAILED=0
for s in $SEEDS; do
    while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || FAILED=$((FAILED + 1)); done
    echo "==> seed $s -> $(out_dir "$s") (log: $(out_dir "$s").log)"
    OUT_DIR="$(out_dir "$s")" SEEDS="$s" JOBS=1 \
        bash "$RUNNER" > "$(out_dir "$s").log" 2>&1 &
    sleep 5   # stagger the image check / container start
done
for pid in $(jobs -p); do wait "$pid" || FAILED=$((FAILED + 1)); done

echo "==> Done: seeds $SEEDS in $OUT_BASE/isic2019_seed*/ (bash run.sh status)"
(( FAILED == 0 )) || { echo "$FAILED seed(s) failed; see $OUT_BASE/isic2019_seed*.log" >&2; exit 1; }
