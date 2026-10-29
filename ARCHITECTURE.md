# Architecture & how to use this repository

FLamby is a benchmark suite for cross-silo federated learning on healthcare datasets with
natural (not simulated) client partitions. It ships data loaders, per-dataset baseline
models/losses/metrics, and simulated FL strategies that run in a single process. **It does
not ship datasets** — each one is downloaded/preprocessed separately, following
`flamby/datasets/<name>/README.md`.

This checkout is a fork (`origin` = `Ayax911/FLamby`) of `owkin/FLamby`. The fork's own
addition — the Fed-ISIC2019 Docker runner — is documented in [EXPERIMENTS.md](EXPERIMENTS.md),
not here.

## Setup

```bash
make install                      # conda env "flamby" (environment.yml) + pip install -e .[all_extra]
make enable=heart,tests install   # partial install: cam16, heart, isic2019, ixi, kits19, lidc, tcga, docs, tests
conda activate flamby
pre-commit install
```

## Repository layout

```
flamby/
  datasets/<fed_name>/     per-dataset code (see "Per-dataset contract" below)
  strategies/              FL strategy implementations (single-process simulation)
  benchmarks/              fed_benchmark.py driver + per-dataset config JSONs
  utils.py                 dataset_location.yaml resolution, evaluate_model_on_tests
  reporting.py             FedMammoBench-style report writer (metrics.csv, plots, test/)
  results/                 published results & plots from the paper
  dp_example/, personalization_example/   paper-figure experiments
integration/               glue examples for Fed-BioMed, FedML, Substra
dockers/                   reproducible base + per-dataset Docker images (upstream)
docs/                      Sphinx docs source (`cd docs && make html`)
tests/strategies/, tests/benchmarks/   the only directories CI runs
```

## Per-dataset contract

Each `flamby/datasets/fed_<name>/` follows the same layout. Benchmark code finds everything
by name through `__import__(f"flamby.datasets.{name}")`, so any dataset must export the same
symbols from its `__init__.py`:

- `common.py` — `NUM_CLIENTS`, `BATCH_SIZE`, `LR`, `NUM_EPOCHS_POOLED`, `Optimizer`, `FedClass`,
  and `get_nb_max_rounds(num_updates)` (converts a pooled-epoch budget into FL rounds).
- `dataset.py` — `Fed<Name>(center=..., train=..., pooled=...)`, a torch `Dataset` returning
  one client's train/test split, or all clients pooled. Usually wraps a `<Name>Raw` class.
- `model.py` — `Baseline`. `loss.py` — `BaselineLoss`. `metric.py` — `metric(y_true, y_pred)`.
- `benchmark.py` — per-dataset training glue (an alternative entry point to `fed_benchmark.py`).
- `dataset_creation_scripts/` — `download.py` and `update_config.py`.

**Dataset location resolution.** `download.py` writes `dataset_location.yaml` (or
`dataset_location_debug.yaml` for a `--debug` partial download) into that dataset's
`dataset_creation_scripts/` folder. `flamby.utils.check_dataset_from_config` reads it at
dataset construction. If only the debug config exists, it **silently falls back to debug
data** and prints a warning. Passing `data_path=` to the dataset bypasses the yaml entirely.
`get_config_file_path` asserts against a hardcoded list of dataset names — update that list
when adding a dataset.

## Instantiating a dataset

```python
from flamby.datasets.fed_tcga_brca import TcgaBrcaRaw, FedTcgaBrca

raw = TcgaBrcaRaw()                                   # all metadata, less convenient
pooled_test = FedTcgaBrca(train=False, pooled=True)   # single-centric
center2_train = FedTcgaBrca(center=2, train=True, pooled=False)   # one client's data
```

Swapping dataset is normally two lines: the import and the `Fed<Name>` class name. See
`Quickstart.md` for the full local-training-loop example.

## Strategies (`flamby/strategies/`)

`FedAvg`, `FedProx`, `FedAdam`/`FedAdagrad`/`FedYogi` (`fed_opt.py`), `Scaffold`, `Cyclic`,
`FedAvgFineTuning`. All single-process simulations, not networked FL. Each takes a list of
per-client training dataloaders, a model, a loss, an `optimizer_class`, `learning_rate`,
`num_updates` (local steps per round), and `nrounds`. `.run()` returns the trained model(s).

Shared machinery in `strategies/utils.py`:
- `_Model` — wraps one client's model copy; `_local_train`, `_prox_local_train`,
  `_local_train_with_correction` (Scaffold).
- `DataLoaderWithMemory` — iterates indefinitely, so a "round" is a number of batch updates,
  not epochs.
- `check_exchange_compliance` — caps bytes exchanged per round.

Optional differential privacy (`dp_target_epsilon`/`delta`/`max_grad_norm`) goes through opacus.

## Benchmark driver (`flamby/benchmarks/fed_benchmark.py`)

Reads a JSON config (`flamby/config_<dataset>.json`: `dataset`, `results_file`, `strategies` →
per-strategy hyperparams). `conf.py` resolves the dataset symbols and strategy kwargs for each
strategy. The driver trains single-centric local baselines, a pooled baseline, and every FL
strategy, then evaluates each model on every client's local test set and on the pooled test
set. Results are appended to a CSV.

```bash
cd flamby/benchmarks
python fed_benchmark.py --seed 42 -cfp ../config_heart_disease.json
# -s FedProx -mu 0.1 ...   runs one strategy with CLI hyperparams instead of the config's list
```

Paths in the config are relative, so this only works from inside `flamby/benchmarks`.
`benchmark_utils.find_xps_in_df` skips experiments already present in `results_file`'s CSV —
**a stale results file can make a rerun silently do nothing**; delete it or pass a fresh
`-rfp`/`--results-file-path` to force a rerun. `flamby/extract_config.py` turns a results CSV
back into a config.

## Testing & linting

```bash
pytest tests/strategies/
pytest tests/benchmarks/
pytest tests/strategies/test_fed_avg.py::<test_name>   # single test

black --line-length=89 .
flake8 --max-line-length=89 --per-file-ignores="*/__init__.py:F401" ./flamby
isort --profile black .
```

CI runs on Python 3.8–3.11 and only touches `tests/strategies/` and `tests/benchmarks/`.
`test_fed_benchmark.py::test_tcga` runs `fed_benchmark.py` end-to-end via `subprocess` and
works without any download because TCGA-BRCA's `brca.csv` is committed. Tests for datasets
that need a download are marked `pytest.mark.skip`.
