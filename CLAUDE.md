# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

FLamby is a benchmark suite for cross-silo federated learning on healthcare datasets with natural
partitions (NeurIPS 2022 Datasets & Benchmarks). It ships data loaders, per-dataset baseline
models/losses/metrics, and simulated FL strategies. **It does not ship datasets.** Each one has to be
downloaded separately by following `flamby/datasets/<name>/README.md`. This checkout is a fork
(`origin` = `Ayax911/FLamby`) of `owkin/FLamby`.

## Setup / commands

```bash
make install                      # conda env "flamby" (environment.yml) + pip install -e .[all_extra]
make enable=heart,tests install   # partial: cam16, heart, isic2019, ixi, kits19, lidc, tcga, docs, tests (comma, no spaces)
conda activate flamby
pre-commit install
```

Tests (CI runs only these two directories, on Python 3.8–3.11):
```bash
pytest tests/strategies/
pytest tests/benchmarks/
pytest tests/strategies/test_fed_avg.py::<test_name>   # single test
```
`tests/benchmarks/test_fed_benchmark.py::test_tcga` runs `fed_benchmark.py` end to end through
`subprocess`. It works without a download because the TCGA-BRCA data (`brca.csv`) is committed.
Tests for datasets that need a download are marked `pytest.mark.skip`.

Lint and format. CI checks these, and pre-commit applies them:
```bash
black --line-length=89 .
flake8 --max-line-length=89 --per-file-ignores="*/__init__.py:F401" ./flamby
isort --profile black .
```

Benchmark reproduction runs from inside `flamby/benchmarks` because the config paths are relative:
```bash
cd flamby/benchmarks
python fed_benchmark.py --seed 42 -cfp ../config_heart_disease.json
# add -s FedProx -mu 0.1 ... to run one strategy with CLI hyperparams instead of the config's list
```
Docs: `cd docs && make html` (Sphinx).

## Architecture

**Per-dataset contract.** Each `flamby/datasets/fed_<name>/` follows the same layout, and the
benchmark code finds everything by name through `__import__(f"flamby.datasets.{name}")`. Any new
dataset has to export the same symbols from its `__init__.py`:
- `common.py` defines `NUM_CLIENTS`, `BATCH_SIZE`, `LR`, `NUM_EPOCHS_POOLED`, `Optimizer`, `FedClass`,
  and `get_nb_max_rounds(num_updates)`, which converts the pooled-epoch budget into FL rounds.
- `dataset.py` has `Fed<Name>(center=..., train=..., pooled=...)`, a torch `Dataset` that returns
  one client's train/test split, or all clients pooled. It usually wraps a `<Name>Raw` class.
- `model.py` has `Baseline`, `loss.py` has `BaselineLoss`, and `metric.py` has `metric(y_true, y_pred)`.
- `benchmark.py` holds the per-dataset training glue.
- `dataset_creation_scripts/` contains `download.py` and `update_config.py`.

**Dataset location resolution.** `download.py` writes `dataset_location.yaml` (or
`dataset_location_debug.yaml` for `--debug` partial downloads) into that dataset's
`dataset_creation_scripts/` folder. `flamby.utils.check_dataset_from_config` reads it at dataset
construction. If only the debug config exists, it **silently falls back to debug data** and prints a
warning. Passing `data_path=` to the dataset bypasses the yaml. `get_config_file_path` asserts against
a hardcoded list of dataset names, so update that list when you add a dataset.

**Strategies** (`flamby/strategies/`): `FedAvg`, `FedProx`, `FedAdam`/`FedAdagrad`/`FedYogi`
(`fed_opt.py`), `Scaffold`, `Cyclic`, and `FedAvgFineTuning`. These are single-process simulations,
not networked FL. Each takes a list of per-client training dataloaders, a model, a loss, an
`optimizer_class`, `learning_rate`, `num_updates` (local steps per round), and `nrounds`. Calling
`.run()` returns the trained model(s). The shared machinery is in `strategies/utils.py`:
- `_Model` wraps one client's model copy with `_local_train`, `_prox_local_train`, and
  `_local_train_with_correction` (Scaffold).
- `DataLoaderWithMemory` iterates indefinitely, so a "round" is a number of batch updates rather
  than epochs.
- `check_exchange_compliance` caps the bytes exchanged per round.

Optional DP (`dp_target_epsilon`/`delta`/`max_grad_norm`) goes through opacus.

**Benchmark driver** (`flamby/benchmarks/`). `fed_benchmark.py` reads a JSON config
(`flamby/config_<dataset>.json`: `dataset`, `results_file`, `strategies` → per-strategy hyperparams).
For each strategy, `conf.py` resolves the dataset symbols and strategy kwargs. The driver trains
single-centric local baselines, a pooled baseline, and every FL strategy, then evaluates each model
on every client's local test set and on the pooled test set. Results are appended to a CSV.
`benchmark_utils.find_xps_in_df` skips experiments that already appear in that CSV, so stale results
files can make a rerun silently do nothing. Published results and plots are under `flamby/results/`.
`flamby/extract_config.py` turns a results CSV back into a config.

**Other directories.** `integration/` has glue examples for Fed-BioMed, FedML, and Substra.
`dockers/` has reproducible base and per-dataset images. `flamby/dp_example/` and
`flamby/personalization_example/` are paper-figure experiments.

## Fed-ISIC2019 Docker runner (fork addition, not upstream FLamby)

`scripts/run_isic2019_docker.sh` runs the full Fed-ISIC2019 experiment set (pooled baseline, 6
local baselines, 8 federated strategies from `flamby/config_isic2019.json`, and an optional 155-run
hyperparameter grid) inside Docker/GPU containers, outside conda, and writes FedMammoBench-style
reports (`metrics.csv`, curves, confusion matrix, ROC, per-center breakdown) plus Weights & Biases
tracking. `run.sh` at the repo root is a thin wrapper pinning seed 42 with `GRID=1 BEST=1` against
`outputs/isic2019`. Full walkthrough (workstation paths, one-time setup, troubleshooting table) in
`scripts/run_isic2019_docker.md` (Spanish).

```bash
bash scripts/run_isic2019_docker.sh prepare   # one-time: download + preprocess the ISIC2019 zip
DEBUG=1 bash scripts/run_isic2019_docker.sh   # 1 round/epoch smoke test
GRID=1 JOBS=4 bash scripts/run_isic2019_docker.sh   # full benchmark + grid, 4 containers at a time
bash scripts/run_isic2019_docker.sh stop      # pause: stop containers, keep finished runs
bash scripts/run_isic2019_docker.sh status    # print $OUT_DIR/PROGRESS.md
```

Key behaviors: re-running is safe/idempotent (FLamby skips experiments already present in a results
CSV, keyed off `OUT_DIR`'s CSVs); `stop` must be used instead of Ctrl+C to avoid leaking GPU
containers from the grid; `PROGRESS.md` is rewritten after every run and is the source of truth for
what's done/pending. Config is entirely via env vars (`DATA_DIR`, `OUT_DIR`, `CACHE_DIR`, `GPU`,
`SEEDS`, `GRID`/`GRID_SEED`, `JOBS`, `WANDB_PROJECT`, `DEBUG`, `DRY_RUN`) — see the variable table at
the bottom of `run_isic2019_docker.sh` or the md guide's §"Variables del script".

## How a federated run executes (`fed_benchmark.py` call chain)

1. `conf.check_config` validates the JSON; `conf.get_dataset_args("fed_isic2019")` pulls
   `BATCH_SIZE, LR, NUM_CLIENTS, NUM_EPOCHS_POOLED, Baseline, BaselineLoss, Optimizer,
   get_nb_max_rounds, metric` off the dataset package by name, and derives the dataset class
   name from the package name (`fed_isic2019` -> `FedIsic2019`; the `FedIxi` -> `FedIXITiny`
   special case lives there too). A new dataset package must be named so this derivation works.
2. `benchmark_utils.init_data_loaders` builds one train + one test `DataLoader` per center
   (`FedDataset(center=i, train=..., pooled=False)`) plus pooled ones. `Isic2019Raw` reads every
   image path from `dataset_creation_scripts/train_test_split`; `FedIsic2019` filters rows by the
   `fold2` column (`train_<center>`) or `fold` (pooled), and picks train/test augmentations.
3. `global_init = Baseline()` is created once; every experiment gets `copy.deepcopy(global_init)`
   and `set_seed(seed)` first so strategies start from the same weights.
4. Order per seed: Pooled -> Local 0..N-1 -> each strategy (sorted by name). Each strategy is
   built as `getattr(strats, sname)(**args)` where `args` = defaults (`SGD`, `LR`,
   `num_updates=100`, `nrounds=get_nb_max_rounds(100)`) overridden by the config entry.
5. Inside a strategy: `__init__` wraps each loader in `DataLoaderWithMemory` and creates one
   `_Model` per client (own deep-copied model + optimizer). `run()` calls `perform_round()`
   `nrounds` times. FedAvg round = per client snapshot params -> `_local_train` for `num_updates`
   batches -> delta -> restore snapshot -> sample-weighted average of deltas -> add to every
   client model. Subclasses only swap the pieces: `FedProx` overrides `_local_optimization`
   (`_prox_local_train`, adds `mu/2*||w-w0||^2`); `FedOpt` (`FedAdam/Yogi/Adagrad`) overrides
   `perform_round` to apply a server-side adaptive step on the averaged delta; `Scaffold` carries
   per-client correction state; `FedAvgFineTuning` runs FedAvg then `num_fine_tuning_steps`
   local steps per client, so its output models differ per client.
6. `s.run()[0]` (the first client's model) is evaluated on every center's test set and the pooled
   test set via `evaluate_model_on_local_and_pooled_tests`, then rows go to the results CSV.
   When `--runs-dir` is set and the dataset is in `reporting.DATASET_LABELS`, `perform_round` is
   monkey-patched to call `RunReporter.log_epoch` after each round (strategy code untouched).

**Where the grid is orchestrated.** `fed_benchmark.py` has no grid loop. With `-s <Strategy>` it runs
exactly one strategy with CLI hyperparams (ignoring the JSON's). The grid lives in bash:
`grid_each` in `scripts/run_isic2019_docker.sh` iterates `GRID_LRS` x strategies (plus `GRID_MUS`
for FedProx, `GRID_SLRS` for FedAdam/Yogi/Adagrad/Scaffold) and starts one container per point
running `fed_benchmark.py --seed 42 -s ... -lr ... [-mu|-slr ...]`, up to `JOBS` at once. Each point
writes `$OUT_DIR/grid/<name>.csv` (skipped if it exists; merged into `results_isic2019_grid.csv`).
To change the grid, edit those env vars / the script, not the Python.

**Where the actual training happens (all in-process, sequential, no network).**
- Federated: `_Model._local_train` in `flamby/strategies/utils.py`. A plain loop of
  `get_samples()` -> forward -> loss -> `backward()` -> `optimizer.step()` for `num_updates`
  batches, on `cuda:0` if available (device is hardcoded there, so `--GPU` only affects pooled/local
  evaluation via `use_gpu_idx`, not `_Model`). Clients run one after another, never in parallel;
  "federated" is simulated by snapshot/restore of parameters as numpy arrays.
- Pooled and Local baselines: `benchmark_utils.train_single_centric`, a classic epoch loop
  (`NUM_EPOCHS_POOLED` epochs) with the dataset's own `Optimizer`/`LR` (Adam, 5e-4 for ISIC),
  whereas strategies default to `torch.optim.SGD` unless the config/CLI says otherwise.
- `flamby/datasets/fed_isic2019/benchmark.py` is a separate, official pooled-only trainer
  (class-weighted loss + scheduler); it is not used by `fed_benchmark.py`.

**Binary variant (`flamby/datasets/fed_isic2019_binary/`, fork addition).** Benign (NV, BKL, DF,
VASC) vs malignant (MEL, BCC, SCC); AK is dropped. Same images/centers/split as `fed_isic2019`: its
own `train_test_split` copy has an extra `binary_target` column (empty for AK) and `dataset.py` filters
on it. Model head has 2 outputs, `BaselineLoss.alpha` has 2 weights, `check_dataset_from_config` still
points at `fed_isic2019` (shared data). Run with `config_isic2019_binary.json`; it needs its own
`DATASET_LABELS` entry in `reporting.py` or reports are silently skipped. Hyperparameters are
copied from the 8-class config. The Docker runner takes `DATASET=fed_isic2019_binary` (picks the
config, default `OUT_DIR` and `runs/$DATASET`); `dataset_creation_scripts/make_binary_target.py`
regenerates `binary_target` and prints the `NUM_TRAIN_SAMPLES`/`alpha` to copy. `run_binary.sh`
(repo root) is the binary counterpart of `run.sh`: benchmark + best + the 233-run paper grid. Full
rationale: `flamby/datasets/fed_isic2019_binary/MODIFICACIONES_BINARIO.md`.

Caveats seen while reading the code (verify before relying on them):
- `Cyclic.perform_round` only trains the current client's `_Model`; it never copies weights to
  the next client, and `fed_benchmark` evaluates `run()[0]`.
- `FedAvgFineTuning.__init__` passes `log, log_period, bits_counting_function` positionally into
  `FedAvg.__init__`, where those slots are `dp_target_*`/`dp_max_grad_norm`. Harmless today only
  because the DP check needs all three non-`None`.
- `ARCHITECTURE.md` and `EXPERIMENTS.md` (untracked) hold the longer general and Fed-ISIC2019
  walkthroughs; keep this file as the short router.
