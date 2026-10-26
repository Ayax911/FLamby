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
