# Meta-Dataset Pipeline

This folder contains the reproducible pipeline used to build the meta-dataset
shipped by `ml_meta_perf`.

The file-format abbreviations used below are comma-separated values (**CSV**) and
Attribute-Relation File Format (**ARFF**). The pretrained tabular models are Tabular
In-Context Learning (**TabICL**) and the Tabular Prior-Data Fitted Network (**TabPFN**).

## Folder Layout

```text
meta_dataset_pipeline/
  datasets/                  # Raw datasets. CSV/ARFF files are ignored by Git.
  results/                   # Generated stage outputs.
  utils/                     # Pipeline-only helpers and model wrappers.
  exp_setup.py               # Dataset, model, seed, and runtime configuration.
  exp_stage_dataset_desc.py  # Builds dataset meta-features.
  exp_stage_ml_tune.py       # Tunes model configurations.
  exp_stage_ml_eval.py       # Evaluates tuned model configurations.
  exp_stage_create_meta_dataset.py
  local_config.example.json  # Template for local secrets.
```

The package-level dataset remains:

```text
src/ml_meta_perf/meta_dataset.csv
```

## Installation

From the repository root, install the project with the meta-dataset pipeline dependencies
before running these stages:

```bash
python -m pip install -r requirements-meta-dataset.txt
```

This includes the regular `ml-meta-perf` dependencies plus the raw-corpus tooling used here,
such as `pymfe`, `pandas`, Optuna, LightGBM, XGBoost, and the tabular deep-learning model
libraries.

## Local Configuration

Secrets must stay outside the Python source files. For TabPFN, copy the example
configuration and fill in your local token:

```text
meta_dataset_pipeline/local_config.example.json
meta_dataset_pipeline/local_config.json
```

Expected format:

```json
{
  "TABPFN_TOKEN": "replace-with-your-tabpfn-token"
}
```

`local_config.json` is ignored by Git. You can also set `TABPFN_TOKEN` as an
environment variable; environment variables take precedence over the local file.

## Running The Pipeline

Run commands from the repository root:

```bash
python meta_dataset_pipeline/exp_stage_ml_tune.py
python meta_dataset_pipeline/exp_stage_dataset_desc.py
python meta_dataset_pipeline/exp_stage_ml_eval.py
python meta_dataset_pipeline/exp_stage_create_meta_dataset.py
```

The expected generated files are:

```text
meta_dataset_pipeline/results/results_stage_ml_tune.csv
meta_dataset_pipeline/results/results_stage_dataset_desc.csv
meta_dataset_pipeline/results/results_stage_ml_eval.csv
meta_dataset_pipeline/results/meta_dataset.csv
```

`results_stage_ml_tune.csv` is produced by `exp_stage_ml_tune.py` and is required by
both `exp_stage_dataset_desc.py` and `exp_stage_ml_eval.py`. The final creation stage
requires both `results_stage_dataset_desc.csv` and `results_stage_ml_eval.csv`.

The final stage computes `Processing Units Number` from each model's selected
hyperparameters. For TabICL and TabPFN it counts the loaded checkpoint parameters when the
corresponding library is available; otherwise, it uses the checkpoint counts recorded in
`exp_stage_create_meta_dataset.py`. Update those fallback counts whenever either checkpoint
changes.

## Publishing The Final Dataset

After rebuilding:

```text
meta_dataset_pipeline/results/meta_dataset.csv
```

compare it with:

```text
src/ml_meta_perf/meta_dataset.csv
```

If the regenerated file is the intended corpus update, copy it over the package
dataset:

```text
src/ml_meta_perf/meta_dataset.csv
```

The intermediate CSV files in `meta_dataset_pipeline/results/` are generated
artifacts. The raw dataset CSV/ARFF files under `datasets/` and the local secret
configuration are intentionally ignored by Git.
