#!/usr/bin/env bash
# Runs every Fed-MammoBench experiment (benign vs malignant, 4 clients) inside Docker (GPU): for
# each seed, the pooled baseline, the 4 local (one-database) baselines and the 8 federated
# strategies of the config. Optionally, a hyperparameter grid (GRID=1) and the best-config reruns
# (BEST=1). Derived from run_isic2019_docker.sh; same behaviour (pause/resume, JOBS, PROGRESS.md).
#
# DATASET picks the backbone (same data, partition and recipe, only the model changes):
#   fed_mammobench               EfficientNet-B0 (ImageNet) config_mammobench.json
#   fed_mammobench_efficientnetv2  EfficientNetV2 (ImageNet)  config_mammobench_efficientnetv2.json
#   fed_mammobench_resnet18      ResNet18 (ImageNet)        config_mammobench_resnet18.json
#
# Usage (from anywhere; mammobench/run_*.sh pin the settings of the full experiment):
#   bash scripts/run_mammobench_docker.sh weights               # cache the ImageNet weights (once)
#   DEBUG=1 bash scripts/run_mammobench_docker.sh               # 1 round/epoch, checks the pipeline
#   GRID=1 BEST=1 JOBS=4 bash scripts/run_mammobench_docker.sh
#   DATASET=fed_mammobench_efficientnetv2 bash scripts/run_mammobench_docker.sh
#   DRY_RUN=1 ... | bash ... stop | bash ... status
#
# Weights & Biases: project $WANDB_PROJECT (default "flamby-mammobench"; groups "<DATASET>_benchmark",
# "<DATASET>_grid", "<DATASET>_best_config"). Export WANDB_API_KEY (https://wandb.ai/authorize) or run `wandb login`
# on the host (~/.netrc is mounted); without credentials runs are saved offline in each run's
# folder (upload later with `wandb sync`). WANDB_PROJECT="" disables W&B.
#
# Model knob, passed into the container: MAMMOBENCH_EFFNET (s|m|l, EfficientNetV2 size; default s).
# Both backbones are trained end to end (ImageNet init), as Fed-ISIC2019.
#
# Outputs (OUT_DIR): runs/$DATASET/seed<S>/<run>/ reports, results_<name>.csv, grid/, best/,
# logs/, PROGRESS.md. Re-running is safe: experiments already in a results CSV are skipped.
set -euo pipefail

MODE="${1:-run}"   # run | weights | stop | status
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${DATASET:-fed_mammobench}"
case "$DATASET" in
    fed_mammobench)
        CONFIG_FILE="config_mammobench.json"; OUT_SUFFIX="mammobench" ;;
    fed_mammobench_efficientnetv2)
        CONFIG_FILE="config_mammobench_efficientnetv2.json"; OUT_SUFFIX="mammobench_efficientnetv2" ;;
    fed_mammobench_resnet18)
        CONFIG_FILE="config_mammobench_resnet18.json"; OUT_SUFFIX="mammobench_resnet18" ;;
    *) echo "Unknown DATASET=$DATASET (fed_mammobench | fed_mammobench_efficientnetv2 | fed_mammobench_resnet18)" >&2; exit 1 ;;
esac
# Folder that holds norm_0_1/ (the extracted preproccesed_julian.zip).
DATA_DIR="${DATA_DIR:-$REPO_DIR/../data/preproccesed_julian}"
RESULTS_CSV="results_${OUT_SUFFIX}.csv"
DEBUG="${DEBUG:-0}"
if [[ "$DEBUG" == 1 ]]; then
    OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/${OUT_SUFFIX}_debug}"
else
    OUT_DIR="${OUT_DIR:-$REPO_DIR/outputs/$OUT_SUFFIX}"
fi
CACHE_DIR="${CACHE_DIR:-$REPO_DIR/../cache_torch}"   # ImageNet weights (TORCH_HOME), downloaded once
IMAGE="${IMAGE:-flamby-mammobench}"
GPU="${GPU:-0}"            # host GPU index the containers get (seen inside as GPU 0)
SEEDS="${SEEDS:-42}"
BENCH="${BENCH:-1}"        # 0: skip the benchmark (only GRID/BEST runs)
WORKERS="${WORKERS:-4}"    # dataloader workers per container
GRID="${GRID:-0}"
JOBS="${JOBS:-1}"          # grid containers running at the same time on the GPU
DRY_RUN="${DRY_RUN:-0}"
WANDB_PROJECT="${WANDB_PROJECT-flamby-mammobench}"
EFFNET="${MAMMOBENCH_EFFNET:-s}"

# Grid (GRID=1): only on GRID_SEED, whatever SEEDS says. Log-spaced like the paper's search.
GRID_SEED="${GRID_SEED:-42}"
GRID_LRS="${GRID_LRS:-0.001 0.0031622776601683 0.01 0.0316227766016837 0.1 0.316227766016838}"
GRID_MUS="${GRID_MUS:-0.001 0.01 0.1 1}"
GRID_SLRS="${GRID_SLRS:-0.001 0.0031622776601683 0.01 0.0316227766016837 0.1 0.316227766016838 1.0 10.0}"
# Grid run names (seed<S>_grid_...) to leave out, e.g. runs known to diverge.
GRID_SKIP="${GRID_SKIP:-}"
# Best config (BEST=1): the 8 strategies of flamby/$CONFIG_FILE again, one container
# each on GRID_SEED, named seed<S>_best_<Strategy>. Own CSVs (best/), reports (runs_best/)
# and W&B group (best_config), so they never mix with the benchmark rows and folders.
BEST="${BEST:-0}"
BEST_CONFIG="$REPO_DIR/flamby/$CONFIG_FILE"

log() { printf '\n==> %s\n' "$*"; }

# grid_each <fn>: calls `fn <run-name> <fed_benchmark.py args...>` for every grid point.
grid_each() {
    local seed="$GRID_SEED" lr mu slr s
    for lr in $GRID_LRS; do
        for s in FedAvg Cyclic; do
            "$1" "seed${seed}_grid_${s}_lr${lr}" --seed "$seed" -s "$s" -lr "$lr"
        done
        # 3 fine-tuning steps, as in the config.
        "$1" "seed${seed}_grid_FedAvgFineTuning_lr${lr}" \
            --seed "$seed" -s FedAvgFineTuning -lr "$lr" -nft 3
        for mu in $GRID_MUS; do
            "$1" "seed${seed}_grid_FedProx_lr${lr}_mu${mu}" \
                --seed "$seed" -s FedProx -lr "$lr" -mu "$mu"
        done
        for slr in $GRID_SLRS; do
            for s in FedAdam FedYogi FedAdagrad Scaffold; do
                "$1" "seed${seed}_grid_${s}_lr${lr}_slr${slr}" \
                    --seed "$seed" -s "$s" -lr "$lr" -slr "$slr"
            done
        done
    done
}

# best_each <fn>: calls `fn <run-name> <fed_benchmark.py args...>` for every strategy in
# BEST_CONFIG, turning its hyperparameters into CLI flags.
best_each() {
    local seed="$GRID_SEED" s args
    while read -r -u 3 s args; do
        # shellcheck disable=SC2086  # args is a list of flags
        "$1" "seed${seed}_best_${s}" --seed "$seed" -s "$s" $args
    done 3< <(python3 - "$BEST_CONFIG" <<'EOF'
import json, sys

flags = {"learning_rate": "-lr", "server_learning_rate": "-slr", "mu": "-mu",
         "num_fine_tuning_steps": "-nft", "optimizer_class": "-opt"}
for s, hp in json.load(open(sys.argv[1]))["strategies"].items():
    args = [f"{flags[k]} {v}" for k, v in hp.items() if k in flags]
    if hp.get("deterministic_cycle"):
        args.append("-d")
    print(s, " ".join(args))
EOF
)
}

# write_report: $OUT_DIR/PROGRESS.md with what is done, running and pending. Finished =
# its row is in the results CSV (FLamby writes it as soon as the experiment ends).
write_report() {
    [[ "$DRY_RUN" == 1 ]] && return
    local tmp state running
    tmp="$(mktemp "$OUT_DIR/.PROGRESS.XXXX")"
    running="$(docker ps --filter label=flamby-mammobench \
        --format '{{.Label "flamby-mammobench"}}' 2>/dev/null | sort)"
    if [[ -n "$running" ]]; then state="en curso"
    elif [[ -f "$OUT_DIR/STOP" ]]; then state="pausado"
    else state="detenido (terminado, o cortado sin \`stop\`)"; fi
    {
        echo "# Progreso $DATASET"
        echo
        echo "- Actualizado: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "- Estado: **$state**"
        echo "- Último lanzamiento: \`$(cat "$OUT_DIR/.launch_cmd" 2>/dev/null || echo '?')\`"
        echo "- Para retomar: relanza ese mismo comando; solo corre lo pendiente."
        echo
        if [[ -n "$running" ]]; then
            echo "## En curso"
            echo
            sed 's/^/- /' <<< "$running"
            echo
        fi
        python3 - "$OUT_DIR/$RESULTS_CSV" "$SEEDS" <<'EOF'
import csv, os, sys

path, seeds = sys.argv[1], sys.argv[2].split()
done = set()
if os.path.exists(path):
    with open(path) as f:
        done = {(r["Method"], str(int(float(r["seed"])))) for r in csv.DictReader(f)}
strategies = "FedAvg FedProx FedAdam FedYogi FedAdagrad Scaffold Cyclic FedAvgFineTuning"
expected = (
    ["Pooled Training"]
    + [f"Local {i}" for i in range(4)]
    + [f"{s}100" for s in strategies.split()]
)
for seed in seeds:
    n = sum((m, seed) in done for m in expected)
    print(f"## Benchmark, semilla {seed}: {n}/{len(expected)}\n")
    for m in expected:
        print(f"- [{'x' if (m, seed) in done else ' '}] {m}")
    print()
EOF
        if [[ "$GRID" == 1 ]]; then
            local names=() name pending=() nskip=0
            collect() { names+=("$1"); }
            grid_each collect
            for name in "${names[@]}"; do
                if [[ " $GRID_SKIP " == *" $name "* ]]; then nskip=$((nskip + 1))
                elif [[ ! -s "$OUT_DIR/grid/$name.csv" ]]; then pending+=("$name"); fi
            done
            local total=$(( ${#names[@]} - nskip ))
            echo "## Malla, semilla $GRID_SEED: $(( total - ${#pending[@]} ))/$total"
            echo
            (( nskip > 0 )) && echo "($nskip omitidas a propósito con GRID_SKIP)" && echo
            if (( ${#pending[@]} > 0 )); then
                echo "Pendientes (incluye las que estén en curso):"
                echo
                printf -- '- %s\n' "${pending[@]}"
            fi
        fi
        if [[ "$BEST" == 1 ]]; then
            local best=() s todo=()
            collect_best() { best+=("$1"); }
            best_each collect_best
            echo
            echo "## Mejor config ($CONFIG_FILE), semilla $GRID_SEED:" \
                "$(for s in "${best[@]}"; do [[ -s "$OUT_DIR/best/$s.csv" ]] && echo; done \
                    | wc -l)/${#best[@]}"
            echo
            for s in "${best[@]}"; do
                if [[ -s "$OUT_DIR/best/$s.csv" ]]; then echo "- [x] $s"
                else echo "- [ ] $s"; fi
            done
        fi
    } > "$tmp"
    mv "$tmp" "$OUT_DIR/PROGRESS.md"
}

# stop / status: no image build needed. They read the settings of the last launch so the
# report knows the seeds and whether the grid was on.
if [[ "$MODE" == stop || "$MODE" == status ]]; then
    # shellcheck disable=SC1091
    [[ -f "$OUT_DIR/.launch_env" ]] && source "$OUT_DIR/.launch_env"
    mkdir -p "$OUT_DIR"
    if [[ "$MODE" == stop ]]; then
        touch "$OUT_DIR/STOP"
        log "Pausing: stopping the running containers (launches nothing else)"
        docker ps -q --filter label=flamby-mammobench | xargs -r docker stop -t 30 >/dev/null
    fi
    write_report
    cat "$OUT_DIR/PROGRESS.md"
    exit 0
fi

# One launch per OUT_DIR: two benchmark containers rewriting the same results CSV can read
# it half-written and retrain experiments that were already done (it happened once).
if [[ "$DRY_RUN" != 1 ]]; then
    mkdir -p "$OUT_DIR"
    exec 9>"$OUT_DIR/.lock"
    if ! flock -n 9; then
        echo "Another launch is already running on $OUT_DIR (see '$0 status')." >&2
        exit 1
    fi
fi

# ---------------------------------------------------------------- 1. image
command -v docker >/dev/null || { echo "docker not found" >&2; exit 1; }
if [[ "$DRY_RUN" != 1 ]] && ! docker info >/dev/null 2>&1; then
    echo "Cannot talk to the Docker daemon. If 'docker info' says permission denied:" \
         "sudo usermod -aG docker \$USER, then log out and back in." >&2
    exit 1
fi
log "Building image $IMAGE from $REPO_DIR"
[[ "$DRY_RUN" == 1 ]] || \
    docker build -f "$REPO_DIR/dockers/Dockerfile.mammobench" -t "$IMAGE" "$REPO_DIR"

# ---------------------------------------------------------------- 2. weights (optional)
if [[ "$MODE" == weights ]]; then
    mkdir -p "$CACHE_DIR"
    log "Caching the ImageNet weights of EfficientNet-B0, EfficientNetV2 and ResNet18 in $CACHE_DIR"
    [[ "$DRY_RUN" == 1 ]] || docker run --rm --user "$(id -u):$(id -g)" \
        -v "$CACHE_DIR:/cache/torch" -e TORCH_HOME=/cache/torch --entrypoint python "$IMAGE" -c "
from flamby.datasets.fed_mammobench.model import Baseline as R
from flamby.datasets.fed_mammobench_efficientnetv2.model import Baseline as E
from flamby.datasets.fed_mammobench_resnet18.model import Baseline as N
R(pretrained=True); E(pretrained=True, arch='s'); N(pretrained=True); print('weights cached')"
    log "Done."
    exit 0
fi

# ---------------------------------------------------------------- 3. checks
if ! docker info 2>/dev/null | grep -qi nvidia; then
    echo "Docker has no NVIDIA runtime: install nvidia-container-toolkit" \
         "and run 'sudo nvidia-ctk runtime configure --runtime=docker'." >&2
    exit 1
fi
if [[ ! -d "$DATA_DIR/norm_0_1" ]]; then
    echo "Missing $DATA_DIR/norm_0_1: set DATA_DIR to the extracted preproccesed_julian folder." >&2
    exit 1
fi
mkdir -p "$OUT_DIR/logs" "$OUT_DIR/grid" "$OUT_DIR/best" "$CACHE_DIR"
# A new launch clears a previous pause and records itself for `status` and PROGRESS.md.
rm -f "$OUT_DIR/STOP"
if [[ "$DRY_RUN" != 1 ]]; then
    echo "DATASET=$DATASET SEEDS=\"$SEEDS\" GRID=$GRID BEST=$BEST JOBS=$JOBS DEBUG=$DEBUG MAMMOBENCH_EFFNET=$EFFNET bash $0" \
        > "$OUT_DIR/.launch_cmd"
    declare -p DATASET CONFIG_FILE RESULTS_CSV BEST_CONFIG SEEDS GRID BEST GRID_SEED GRID_LRS GRID_MUS \
        GRID_SLRS GRID_SKIP DEBUG \
        > "$OUT_DIR/.launch_env"
fi

# ---------------------------------------------------------------- 4. experiments
# run <log-name> <fed_benchmark.py args...>: one container per call. RESULTS picks the
# results CSV (inside /out); QUIET=1 sends output only to the log (parallel grid).
run() {
    local name="$1"; shift
    [[ -f "$OUT_DIR/STOP" ]] && return 0   # paused: launch nothing else
    # --init forwards `docker stop` to python at once (PID 1 would ignore SIGTERM).
    local cmd=(docker run --rm --init --gpus "device=$GPU" --shm-size=16g
        --label "flamby-mammobench=$name"
        --user "$(id -u):$(id -g)"
        -v "$DATA_DIR:/data:ro" -v "$OUT_DIR:/out" -v "$CACHE_DIR:/cache/torch"
        -e TORCH_HOME=/cache/torch -e MAMMOBENCH_DATA=/data -e MAMMOBENCH_EFFNET="$EFFNET"
        # Passed by name only, so the key never shows in the printed command.
        -e WANDB_API_KEY -e WANDB_ENTITY -e WANDB_MODE)
    # `wandb login` on the host saves the key in ~/.netrc (HOME is /tmp inside).
    [[ -f "$HOME/.netrc" ]] && cmd+=(-v "$HOME/.netrc:/tmp/.netrc:ro")
    cmd+=("$IMAGE" "$@" --workers "$WORKERS" -cfp "../$CONFIG_FILE"
        -rfp "/out/${RESULTS:-$RESULTS_CSV}" --runs-dir "/out/${RUNS_DIR:-runs}")
    [[ "$DEBUG" == 1 ]] && cmd+=(--debug)
    [[ -n "$WANDB_PROJECT" ]] && cmd+=(--wandb-project "$WANDB_PROJECT"
        --wandb-group "${DATASET}_${WANDB_GROUP:-benchmark}")
    log "$name"
    echo "${cmd[*]}"
    [[ "$DRY_RUN" == 1 ]] && return
    local rc=0
    if [[ "${QUIET:-0}" == 1 ]]; then
        "${cmd[@]}" > "$OUT_DIR/logs/$name.log" 2>&1 || rc=$?
    else
        "${cmd[@]}" 2>&1 | tee "$OUT_DIR/logs/$name.log" || rc=$?
    fi
    if (( rc == 0 )); then
        echo "finished: $name"
    elif [[ -f "$OUT_DIR/STOP" ]]; then
        echo "paused: $name (its unfinished experiment reruns on resume)"
    else
        echo "FAILED: $name (see logs/$name.log)"
        write_report
        return 1
    fi
    write_report
}

[[ "$BENCH" == 1 ]] && for seed in $SEEDS; do
    # No -s/-scb: pooled + 4 locals + every strategy in the config, in one process.
    run "seed${seed}_benchmark" --seed "$seed"
done

# Each grid/best run writes its own results CSV: FLamby rewrites the whole file, so
# parallel runs sharing one CSV would drop each other's rows.
FAILED=0
# queue_run <csv-dir> <wandb-group> <runs-dir> <run-name> <args...>: up to JOBS at once.
queue_run() {
    local dir="$1" group="$2" runs="$3" name="$4"; shift 4
    # Finished = its table exists (FLamby writes it at the end); skip without a container.
    [[ -s "$OUT_DIR/$dir/$name.csv" ]] && return 0
    [[ "$dir" == grid && " $GRID_SKIP " == *" $name "* ]] && return 0
    if (( JOBS > 1 )) && [[ "$DRY_RUN" != 1 ]]; then
        while (( $(jobs -rp | wc -l) >= JOBS )); do wait -n || FAILED=$((FAILED + 1)); done
        RESULTS="$dir/$name.csv" WANDB_GROUP="$group" RUNS_DIR="$runs" QUIET=1 \
            run "$name" "$@" &
    else
        RESULTS="$dir/$name.csv" WANDB_GROUP="$group" RUNS_DIR="$runs" \
            run "$name" "$@" || FAILED=$((FAILED + 1))
    fi
}
grid_run() { queue_run grid grid runs "$@"; }
best_run() { queue_run best best_config runs_best "$@"; }

# merge_tables <csv-dir> <merged csv>
merge_tables() {
    log "Merging $1/ results into $OUT_DIR/$2"
    [[ "$DRY_RUN" == 1 ]] || docker run --rm --user "$(id -u):$(id -g)" -v "$OUT_DIR:/out" \
        --entrypoint python "$IMAGE" -c "
import glob, pandas as pd
files = sorted(glob.glob('/out/$1/*.csv'))
if files:
    pd.concat(map(pd.read_csv, files)).to_csv('/out/$2', index=False)
print(len(files), 'tables merged')"
}

[[ "$GRID" == 1 ]] && grid_each grid_run
[[ "$BEST" == 1 ]] && best_each best_run
for pid in $(jobs -p); do wait "$pid" || FAILED=$((FAILED + 1)); done
[[ "$GRID" == 1 ]] && merge_tables grid "${RESULTS_CSV%.csv}_grid.csv"
[[ "$BEST" == 1 ]] && merge_tables best "${RESULTS_CSV%.csv}_best.csv"

write_report
if [[ -f "$OUT_DIR/STOP" ]]; then
    log "Paused. Progress in $OUT_DIR/PROGRESS.md; re-run the same command to resume."
    exit 0
fi
log "Done. Reports in $OUT_DIR/runs/$DATASET (progress: $OUT_DIR/PROGRESS.md)"
if (( FAILED > 0 )); then
    echo "$FAILED grid/best run(s) failed; re-run the same command to retry only those." >&2
    exit 1
fi
