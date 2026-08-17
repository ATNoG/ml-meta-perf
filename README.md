# ![metafit logo](assets/metafit_logo.svg) metafit

**metafit** fits short, readable equations that predict the **Matthews Correlation
Coefficient** a classifier will reach on a dataset, from meta-features of the dataset and
of the model:

```
MCC = w1*t1 + w2*t2 + ... + wk*tk
```

Each `t` is a simple expression over one to three raw features — `f`, `log(f)`, `1/f`,
`f^2`, `f1/f2`, `f1*f2`, `(f1+f2)/f3` — and each `w` is a real number in the features' own
units, so the printed equation evaluates as written.

The project trades accuracy for explainability on purpose. Opaque regressors reach higher
R² on this kind of meta-data but nothing can be read off them; the point here is an
equation a practitioner can inspect, argue with, and derive guidance from.

> **Status:** research prototype for an academic study. 20 datasets × 25 models = 476 rows,
> which is small. Every number is reported both in-sample and under leave-one-out
> cross-validation, because at this size the two differ a lot.

## Headline results

| | R² on all 476 rows |
|---|---|
| E1 — dataset features only, 5 terms | 0.337 |
| *ceiling: the true dataset means* | *0.354* |
| EM — model features only, 9 terms | 0.164 |
| *ceiling: the true model means* | *0.282* |
| **E2 — dataset + model, 24 terms** | **0.600** |
| *additive oracle* | *0.6605* |
| *additive + rank-1 interaction* | *0.7828* |

Leave-one-dataset-out R² for E2 is **0.466**; leave-one-model-out **0.489**. An
accuracy-leaning configuration reaches **0.668 in-sample** at 32 terms.

![Equations against their ceilings](assets/figures/equation_comparison.png)

Four findings the documentation develops:

- **A random split reports an equation that does not exist.** The same equation scores
  0.540 under random 10-fold and 0.466 under leave-one-dataset-out, because dataset
  features are constant within a dataset and a random fold puts the same dataset on both
  sides. See [chapter 4](assets/docs/04-evaluation.md).
- **The meta-features describe datasets far better than models.** The dataset equation
  reaches 95% of its ceiling; the model equation reaches 58%. See
  [chapter 6](assets/docs/06-results.md).
- **One interaction component is worth +0.122 R²** and the equation captures none of it —
  the clearest direction for future work. See [chapter 5](assets/docs/05-oracles.md).
- **Mixed dataset×model terms carry the equation.** 12 of 24 terms use features from both
  groups and drive **76%** of the output variance; dataset-only terms drive 10% and
  model-only terms 14%. "Which model suits which data" is where the signal is, not "how
  hard is this data" or "how good is this model". See
  [chapter 7](assets/docs/07-practices.md).

## Documentation

The rationale, methodology and full results live in
**[`assets/docs/`](assets/docs/index.md)**, written at the level of detail a paper needs
and carrying the generated figures.

| | | modules |
|---|---|---|
| 1 | [The problem and the data](assets/docs/01-problem.md) | `metafit.data` |
| 2 | [Equation form and term vocabulary](assets/docs/02-equation-form.md) | `metafit.terms`, `metafit.model` |
| 3 | [Search and fitting](assets/docs/03-search-and-fitting.md) | `metafit.fit`, `metafit.analysis`, `metafit.selection` |
| 4 | [Evaluation methodology](assets/docs/04-evaluation.md) | `metafit.validate`, `metafit.stats` |
| 5 | [Oracles and ceilings](assets/docs/05-oracles.md) | `metafit.validate` |
| 6 | [Results](assets/docs/06-results.md) | `metafit.experiment` |
| 7 | [From equation to practice](assets/docs/07-practices.md) | `metafit.practices`, `metafit.attribution` |
| 8 | [Limitations](assets/docs/08-limitations.md) | — |
| 9 | [Generated report](assets/docs/09-report.md) | `metafit.report` — **written by the code, not by hand** |

Related work and positioning: **[`RESEARCH.md`](RESEARCH.md)**.

The API reference is generated from the module docstrings with `pdoc`, and each module
links back to the chapter covering it:

```bash
make docs      # -> docs/index.html, with assets/ copied alongside
```

## Installation

Python 3.12+. Runtime dependencies are **polars**, **numpy**, **matplotlib** and
**kneeliverse**.

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-dev.txt
```

## Running it

**One command runs every phase**: screening, fitting E1/E2 and the two controls,
cross-validating under both protocols, extracting the practices, writing the figures, and
generating the report.

```bash
PYTHONPATH=src venv/bin/python -m metafit
```

That reproduces every number in this README, and writes:

| | |
|---|---|
| [`assets/docs/09-report.md`](assets/docs/09-report.md) | the generated report — equation, term analysis, practices |
| `results/e1.json`, `e2.json`, `em.json`, `e2_accurate.json` | the fitted equations, reloadable |
| `results/*.csv` | 17 tables — curves, baselines, oracles, stability, practices |
| `assets/figures/*.png` | the 14 figures |

Everything printed and written is derived from the run. **The report is generated by
`metafit.report`, not written by hand.** It ranks terms by standardised weight, writes a
sentence per major term, groups terms into the blocks that move together, and reports which
features and which grammar operations the search actually used — so a flat equation with no
dominant term is still readable, at a larger unit than one term.

Through `make`:

```bash
make study     # the full run above
make report    # the same without figures
make figures   # figures only
make quick     # seconds, not minutes; checks the wiring, not the numbers
make docs      # build the API reference
make test      # unittest
make lint      # the full CI gate
```

### Parameters

Every knob that was tuned during the study is a flag, and the defaults are the tuned
values, so a bare run is the reported study.

```bash
PYTHONPATH=src venv/bin/python -m metafit --data mine.csv --output runs/mine
```

| flag | default | what it does |
|---|---|---|
| `--data` | `data/meta_dataset.csv` | the meta-dataset to fit |
| `--output` | `results` | where the equations and CSV tables go |
| `--figures` | `assets/figures` | where the figures go |
| `--report` | `assets/docs/09-report.md` | where the generated report goes |
| `--terms` | 24 | terms in the published E2 equation |
| `--max-terms` | 32 | longest equation the search explores (drives the curve) |
| `--penalty` | 5.0 | ridge penalty on standardised terms |
| `--arity` | 3 | raw features allowed per term — see [chapter 2](assets/docs/02-equation-form.md) |
| `--pool` | 600 | terms surviving screening into the beam |
| `--beam` | 6 | beam width |
| `--zscore` | 3.0 | largest standard score a term may reach before it is rejected as a spike |
| `--phase` | all | `screen`, `equations`, `validation`, `practices`, `figures`, `report`; repeatable |
| `--quick` | off | a reduced configuration for smoke-testing; **not** the study |
| `--no-figures`, `--no-tables`, `--no-report`, `--quiet` | off | skip an output |

```bash
# a shorter, more heavily shrunk equation, no figures
PYTHONPATH=src venv/bin/python -m metafit --terms 12 --penalty 20 --no-figures

# four-feature terms: better fit, much worse transfer (chapter 2 measures this)
PYTHONPATH=src venv/bin/python -m metafit --arity 4 --pool 2000 --output runs/arity4

# just the screening table
PYTHONPATH=src venv/bin/python -m metafit --phase screen --no-figures
```

Two runnable examples:

```bash
PYTHONPATH=src venv/bin/python examples/run_experiment.py       # study + saved equations
PYTHONPATH=src venv/bin/python examples/predict_new_dataset.py  # per-term breakdown of one row
```

## Using it as a library

```python
from metafit import DATASET_FEATURES, MODEL_FEATURES, build_library, columns_as_arrays, fit, load, target

frame = load()
columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
library = build_library(DATASET_FEATURES, MODEL_FEATURES, columns)

equation = fit(library, target(frame), max_terms=14, penalty=20.0).best()

print(equation)                    # human-readable, with standardised betas
print(equation.to_latex())         # for the paper
equation.save("e2.json")           # round-trips exactly
predictions = equation.predict(columns)
```

Extracting guidance and figures from a fitted equation:

```python
from metafit.practices import best_practices, render
from metafit.plots import practice_effects

practices = best_practices(equation, columns)     # pass a stability table to rate them
print(render(practices))
practice_effects(practices, "practice_effects.png")
```

## Figures

`--figures <dir>` writes 14 figures. **None carries a title or an annotation** — they are
made for LaTeX `figure` environments where the caption does that work, and text baked into
a PNG cannot be restyled or translated. `metafit.figures.captions()` returns a suggested
caption per file, including the disclosures deliberately kept out of the images.

## Layout

```
src/metafit/
    data.py         loading, schema validation, aggregation, feature glossary
    terms.py        the term vocabulary and its admissibility rules
    stats.py        pearson, spearman, ranks, R2/MAE/RMSE/SMAPE (no scipy)
    analysis.py     correlation screening, redundancy clustering
    fit.py          standardisation, ridge solve, beam search, refinement
    model.py        the Equation object: predict, render, serialise
    validate.py     leave-one-group-out protocols, baselines, oracles
    selection.py    knee detection and Pareto fronts over equation length
    attribution.py  per-term effects, group shares, variance decomposition
    construct.py    agglomerative term building and the dendrogram cut
    practices.py    turning a fitted equation into written guidance
    plots.py        the figures (matplotlib, Agg, headless, no embedded text)
    figures.py      the figure set and suggested LaTeX captions
    report.py       the generated report: term importance and written analysis
    experiment.py   the end-to-end study and its tuned configurations
    cli.py          python -m metafit -- the entry point that runs every phase
assets/docs/        the study chapters
assets/figures/     generated figures
results/            generated equations, tables and report.md
data/               meta_dataset.csv
tests/              352 unittest tests
examples/           runnable entry points
```

## Development

`ci.sh` runs the full gate; `.pre-commit-config.yaml` runs the same checks at commit time.

```bash
venv/bin/pre-commit install
./ci.sh          # unittest + coverage, ruff, basedpyright, vulture
```

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
