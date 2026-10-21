# ![ml-meta-perf logo](assets/logo.svg) ml-meta-perf

**ml-meta-perf** fits short, readable equations that predict how well a model performs on a
dataset (a Matthews correlation coefficient, a training time, a number of bytes), from
meta-features of the dataset and of the model:

```
y = w0 + w1*t1 + w2*t2 + ... + wk*tk
```

Each `t` is a simple expression over one to three raw features — `f`, `log(f)`, `1/f`, `f^2`,
`f1/f2`, `f1*f2`, `(f1+f2)/f3` — and each `w` is a real number in the features' own units, so the
printed equation evaluates as written. The tool trades accuracy for explainability **on purpose**:
opaque regressors reach a higher in-sample score on this kind of meta-data, and nothing can be read
off them.

Everything about a problem — the corpus, the target and the range it can take, the features, the
protocols, the hyperparameters — is in one **study file** (`config/study.json`). Nothing about a
problem is in the code; a study of another target is another file of the same shape.

## The result of a study

Each study fits up to four equations, all under one configuration:

| equation | features | what it answers |
|---|---|---|
| **E1** | the dataset's | what the data alone explains (a per-dataset constant) |
| **E2** | the model's | what the model alone explains |
| **E3-Valid** | both | the smaller, explainable equation with similar performance: the one to read |
| **E3-MAX** | both, at the capability arity | the best score the configuration reaches: a ceiling, not an equation to read |

E3-Valid's arity and number of terms come from the knee of the **Pareto front** of cost (feature
slots, `arity * terms`) and score (the worst protocol), so the solution stops where the returns stop
and is not saturated ([chapter 7](assets/docs/07-generic-targets.md), section 7.6). The score of an
equation is its **worst** R² over the protocols the study declares:

- **IS** — in-sample, the fit itself;
- **LODO** — leave-one-dataset-out: a dataset the equation has never seen;
- **LOMO** — leave-one-model-out: a model it has never seen;
- **DHO** — doubly held out: neither the dataset nor the model of the cell.

A corpus with one grouping factor (runs of one network family, say) declares IS and LODO only.

## Install

```bash
python3 -m venv venv
venv/bin/pip install .          # polars, numpy, jsonargparse, kneeliverse
```

## Use

```bash
# fit the shipped MCC study (dataset/meta_dataset.csv, 484 rows)
ml-meta-perf fit

# fit another study file on another corpus
ml-meta-perf fit --config config/flexfl/total_time_s.json --data runs.csv --output out

# write a study file for a corpus CSV, to be edited
ml-meta-perf init runs.csv --dataset dataset --response total_time_s --bounds "[0, null]" --transform log1p

# search the shared hyperparameters and propose a replacement study file
ml-meta-perf sweep --config study.json --data runs.csv
```

`fit` writes one folder, `<output>/<csv stem>/<target>[-tags]/`:

```
e1/ e2/ e3/ e3-max/   equation.json, equation.txt, curve.csv for each equation fitted
scores.csv            every equation, protocol and scale: r2, mae, smape
term_effects.csv      what each term of E3-Valid is worth, in the target's units
group_shares.csv      how much of E3-Valid the dataset, the model, and both carry
report.md             generated from those numbers by arithmetic; nothing in it is narrated
manifest.json         the corpus (path, SHA-256, rows), the study (SHA-256), the version
```

A folder made from a different corpus content is not overwritten without `--force`. The study file is
read with `jsonargparse`: any field can be overridden as a dotted flag (`--search.penalty 3`,
`--selection.method pareto_elbow`), `--config` replaces the file and `--print_config` prints the
effective study.

The Python API is the same pipeline:

```python
from ml_meta_perf import load_config, run_study

report = run_study(load_config())        # the shipped MCC study
print(report.valid.equation)             # E3-Valid, as printed in equation.txt
print(report.scores.head())              # every equation, protocol and scale
```

The pieces are public too: the term library, the beam search and one equation by hand.

```python
from ml_meta_perf import build_library, columns_as_arrays, default_schema, load, load_config, search, target

schema = default_schema()          # the columns and features config/study.json describes
frame = load()
columns = columns_as_arrays(frame, schema.features)
library = build_library(schema.dataset_features, schema.model_features, columns, max_arity=2, max_abs_zscore=4.0)

config = load_config().search
equation = search(library, target(frame), max_terms=config.max_terms, penalty=config.penalty).best()
print(equation)                    # human-readable, with standardised betas
print(equation.to_latex())         # for a paper
```

## The study file

`config/study.json` describes the problem and holds every tuned value; the command line holds no
default of its own.

| section | keys | what it says |
|---|---|---|
| `corpus` | `path`, `dataset_column`, `model_column` | the CSV and the columns naming the groups that protocols hold out (`model_column` may be `null`: one grouping factor) |
| `corpus` (optional) | `row_filter`, `require`, `single_valued`, `drop_missing_response`, `drop_constant_features` | how to read a corpus that is not clean: keep a subset of rows, reject a stale corpus, insist that columns hold one value, drop rows with an empty target, remove features that never vary |
| `response` | `column`, `bounds`, `transform` | the target, the `[low, high]` it can take (either end may be `null`; a time is `[0, null]`, MCC `[-1, 1]`), and `none`, `log1p` (fit `log(1 + y)`) or `log` (fit `log(y)`); a transformed study also reports its scores on the target's own scale |
| `features` | `dataset`, `model` | which columns describe a dataset and which describe a model |
| `equations` | | `["E3"]`, `["E3", "E3-MAX"]`, `["E1", "E2", "E3"]` or all four |
| `protocols` | | from `IS`, `LODO`, `LOMO`, `DHO`; LOMO and DHO need a model column |
| `grammar` | `exclude` | per feature, the operations it may not take part in: `log`, `sqrt`, `sq`, or `inv` (never a divisor) |
| `search`, `selection`, `sweep` | | the hyperparameters below, and the grid `sweep` explores |

The mathematics of every rule is in [chapter 7](assets/docs/07-generic-targets.md).

| setting | value | what it does |
|---|---|---|
| `search.max_abs_zscore` | 3.5 | largest standard score a term may reach before it is rejected as a spike |
| `search.penalty` | 0.1 | ridge penalty on standardised terms |
| `search.pool_size` | 600 | terms surviving screening into the beam |
| `search.max_terms` | 30 | longest equation the search explores (the curve's horizon, not the published length) |
| `search.beam_width` | 6 | beam width |
| `search.max_arity` | 3 | raw features per term — see [chapter 2](assets/docs/02-additive-model.md) |
| `selection.delta` | 0.01 | largest smoothed-floor gain over the window that still counts as a plateau |
| `selection.window` | 4 | lengths after a candidate that must stay within `delta` |
| `selection.smoothing` | 3 | width of the running median applied to the worst-protocol curve |
| `selection.method` | plateau_knee | the rule that chooses E3-Valid: `plateau_knee` fits the configured arity; `pareto_knee`, `pareto_elbow`, `pareto_within_delta` and `plateau_pareto` fit every arity up to it and choose the arity too ([chapter 7](assets/docs/07-generic-targets.md), section 7.6) |
| `selection.capability_arity` | 3 | arity used by E3-MAX |

`ml-meta-perf sweep` fits a grid of `max_abs_zscore`, `penalty` and `max_arity`, applies the
study's own length rule to every point, keeps the readable ones tied on R², and proposes the best
under doubly-held-out ranking (for a bounded performance score) or the smallest equation (for a
cost). Adopting its `proposed_study.json` is a reviewed edit of the study file.

## Other corpora

`config/flexfl/` holds one complete study file per target of a federated-learning corpus
(times, bytes, epochs, performance): see [`FLEXFL.md`](FLEXFL.md). A cost is fitted on a log scale
(`response.transform`), is never clipped to the range its training rows span, and declares IS and
LODO only. `tests/synthetic.py` plants an equation with a known number of terms in a synthetic
corpus, which is how the length rules were compared.

## Layout

```
src/ml_meta_perf/
    config.py       the typed schema of the study file and its loader
    arguments.py    the flags every command shares
    data.py         the corpus loader and the target's scales
    terms.py        the grammar: atoms, products, ratios, sums; the term library
    search.py       screening, beam search, pruning
    fit.py          the ridge solve on standardised terms
    model.py        the Equation: evaluation, printing, JSON
    validate.py     the protocols and the baselines
    selection.py    the floor curve and the plateau rule
    lengths.py      the arity-and-length rules over the Pareto front
    experiment.py   one equation: search, cross-validate every length, publish one
    study.py        the equations a study lists, their scores
    attribution.py  what each term and each feature group is worth
    output.py       the result folder, the manifest, the generated report
    skeleton.py     `ml-meta-perf init`
    sweep.py        `ml-meta-perf sweep`
    cli.py          `ml-meta-perf fit | sweep | init`
config/             study.json, and flexfl/ with one study file per target
dataset/            meta_dataset.csv, the versioned MCC corpus
assets/docs/        the method chapters (2, 3, 5, 7)
meta_dataset_pipeline/  raw-corpus pipeline used to regenerate meta_dataset.csv
tests/              unittest suite (every unit test under one second)
```

## Development

`.pre-commit-config.yaml` is the gate and `.github/workflows/main.yml` runs the same five checks on
Python 3.12 and 3.14, reading their settings from `pyproject.toml`; CI then fits the shipped study.

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/pre-commit install
venv/bin/pre-commit run --all-files   # ruff, ruff format --check, basedpyright, vulture, unittest
```

The `dev` extra pins pre-commit and the three static-analysis tools at the versions CI installs.
`pyproject.toml` defines the supported dependency ranges; its `reproducibility` extra freezes the
runtime of the reference results.

## Citation

See [`CITATION.cff`](CITATION.cff). The paper release is the `v1.0.0` tag.

## License

MIT — see [`LICENSE`](LICENSE).
