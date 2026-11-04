#!/usr/bin/env bash
# Fed-MammoBench (benign vs malignant, 4 clients) on FLamby: pooled + 4 local baselines + the 8
# strategies of flamby/config_mammobench.json, seed 42. Run inside the conda env `flamby`.
#
#   bash mammobench/run.sh                                   # whole benchmark
#   bash mammobench/run.sh -s FedAvg -lr 0.01                # one strategy
#   DATA_DIR=/path/to/preproccesed_julian bash mammobench/run.sh
#   DEBUG=1 bash mammobench/run.sh                           # 1 round/epoch smoke test
#
# Env: DATA_DIR (folder holding norm_0_1/, default <repo>/data/preproccesed_julian),
#      SEED (42), RUNS_DIR (runs), GPU (0), WORKERS (4). Extra args go to fed_benchmark.py.
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "${DATA_DIR:-}" ] && export MAMMOBENCH_DATA="$DATA_DIR"
cd "$REPO_DIR/flamby/benchmarks"
EXTRA=()
[ "${DEBUG:-0}" = "1" ] && EXTRA+=(--debug)
exec python fed_benchmark.py --seed "${SEED:-42}" -cfp ../config_mammobench.json \
    --runs-dir "${RUNS_DIR:-$REPO_DIR/runs}" --GPU "${GPU:-0}" --workers "${WORKERS:-4}" "${EXTRA[@]}" "$@"
