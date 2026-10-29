# Running Fed-ISIC2019 experiments

This covers the experiment you actually run on the workstation: pooled / per-partition
(local) / federated-strategy / grid training on Fed-ISIC2019, via either `fed_benchmark.py`
directly or the Docker runner. For the general FLamby architecture, see
[ARCHITECTURE.md](ARCHITECTURE.md).

**GPU note:** real runs happen on the workstation, never size `BATCH_SIZE` or other knobs to
the local laptop GPU (GTX 1650, 4 GB) — use FLamby's defaults (`BATCH_SIZE=64`). The laptop is
only for quick smoke tests in a scratch harness with tiny subsets, never by editing `common.py`.

## Route 1 — `fed_benchmark.py` directly (conda env)

```bash
conda activate flamby
cd FLamby/flamby/benchmarks
```

Every run writes FedMammoBench-style reports via `flamby/reporting.py` under `--runs-dir`
(default `runs`) → `runs/fed_isic2019/seed<seed>/<run>/`: `metrics.csv`, `plots/*_curve.png`,
`test/` (confusion matrix, ROC, per-center breakdown). `--runs-dir ""` disables reporting.

Dataset must already be preprocessed at `~/datasets/fed_isic2019` (see
`scripts/setup_isic2019.sh` for a fresh machine, or §"Preparing the data" below).

- **Pooled**: `python fed_benchmark.py -cfp ../config_isic2019.json --seed 42 -scb Pooled`
  → `pooled/`.
  Alternative official ISIC script (class-weighted loss + scheduler, differs from
  `fed_benchmark`'s pooled): `cd ../datasets/fed_isic2019 && python benchmark.py --GPU 0 --workers 4`
  → `runs/fed_isic2019/benchmark_pooled/`.
- **Per partition**: `-scb Local --nlocal <0..5>` → `local_<i>/` (trained on one center,
  tested on all centers).
- **Federated strategy**: `-s <FedAvg|FedProx|FedAdam|FedYogi|FedAdagrad|Scaffold|Cyclic|FedAvgFineTuning>`
  → `<Strategy>-num-updates100-lr...[-mu...]/{server,nodes/<center>,pooled_eval}/`.

  **Gotcha:** passing `-s` makes FLamby ignore the config's hyperparams — `lr` silently
  defaults to `5e-4`. Always pass the paper's values explicitly:
  | Strategy | flags |
  |---|---|
  | FedAvg | `-lr 0.01` |
  | FedProx | `-lr 0.01 -mu 0.001` |
  | FedAdam | `-lr 0.01 -slr 0.0031622776601683` |
  | Cyclic | `-lr 0.0031622776601683` |
  | others | see `flamby/config_isic2019.json` |

- **Everything at once** (pooled + 6 locals + all 8 config strategies, config hyperparams):
  omit `-s`/`-scb` entirely.
- **Grid**: a bash loop over `-s`, `-lr`, `-mu`, `-slr`, `--seed`; each combo gets its own
  folder (name built from its hyperparams). Number of rounds is **not** configurable —
  it's derived from `num_updates=100` (hardcoded) → 9 rounds at `BATCH_SIZE=64`.

**Gotcha — stale results CSV:** FLamby skips any experiment already present in the results
CSV (`results_isic42.csv` or whatever `-rfp` points to) — no training, no reports. Delete the
file or pass a fresh `-rfp <new>.csv` to force a rerun.

**Val = test:** FLamby has no validation split. `val_*` columns are computed on the test set;
`test/` always evaluates the final model; `best.json` is informative only, not a different
checkpoint.

## Route 2 — Docker runner (what's actually used on the workstation)

RTX 6000 Ada 48 GB, data under `/media/sosagro4c/DATA3/03_pregradojulian`, branch
`fedisic2019-reports-docker` of `Ayax911/FLamby`. Full bilingual walkthrough with exact
workstation paths and a troubleshooting table: `scripts/run_isic2019_docker.md`. Architecture
of the runner itself (env vars, idempotency, `stop`/`status`) is in
[ARCHITECTURE.md](ARCHITECTURE.md)'s Fed-ISIC2019 section — this is the "how do I actually
drive it for *this* experiment" cheat sheet.

```bash
# one-time, from the repo root on the workstation
export DATA_DIR=/media/sosagro4c/DATA3/03_pregradojulian/fed_isic2019
export OUT_DIR=/media/sosagro4c/DATA3/03_pregradojulian/outputs/isic2019
export CACHE_DIR=/media/sosagro4c/DATA3/03_pregradojulian/cache_torch
export WANDB_API_KEY=<key>
bash scripts/run_isic2019_docker.sh prepare     # download + preprocess (once), types "y" to ISIC license

# smoke test
DEBUG=1 OUT_DIR=$OUT_DIR-debug WANDB_PROJECT=flamby-debug bash scripts/run_isic2019_docker.sh

# full run: seed 42 only (your choice), benchmark + 155-run grid, 4 containers at a time
GRID=1 JOBS=4 bash scripts/run_isic2019_docker.sh

# or the pinned wrapper (same seed/GRID/BEST choices, OUT_DIR=outputs/isic2019)
bash run.sh

bash scripts/run_isic2019_docker.sh stop        # pause — use this, NOT Ctrl+C (leaks GPU containers)
bash scripts/run_isic2019_docker.sh status       # prints $OUT_DIR/PROGRESS.md
```

W&B project `flamby` (groups `benchmark`/`grid`), tracked automatically given
`WANDB_API_KEY`/`wandb login`; otherwise saved offline per-run, synced later with `wandb sync`.
Re-running is safe — same idempotency rule as Route 1 (skips rows already in the results CSV).

## Results layout (`OUT_DIR`, either route)

| Path | Contents |
|---|---|
| `runs/fed_isic2019/seed<S>/pooled/`, `local_<i>/` | `config.yaml`, `metrics.csv`, `plots/`, `test/` |
| `runs/fed_isic2019/seed<S>/<Strategy>-…/` | `server/`, `nodes/<center>/`, `pooled_eval/test/` |
| `results_isic2019.csv` | benchmark table (all seeds) |
| `grid/<run>.csv`, `results_isic2019_grid.csv` | per-grid-run table, and the merge |
| `best/<run>.csv`, `runs_best/`, `results_isic2019_best.csv` | Docker runner's `BEST=1` extra pass |
| `logs/<run>.log` | per-container stdout/stderr (Docker route) |
| `PROGRESS.md` | current state: done/in-progress/pending (Docker route) |

The official FLamby metric for ISIC2019 is **balanced accuracy**.

## Analyzing results

Use the `analizar-experimentos` skill to generate/compare reports across `results*/expNN`-style
folders (per-node and per-database metrics, weighted global performance).
