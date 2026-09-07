# ![ml-meta-perf logo](assets/logo.svg) ml-meta-perf

**ml-meta-perf** fits short, readable equations that predict the **Matthews Correlation
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

| | in-sample | LOO-dataset | LOO-model |
|---|---|---|---|
| E1 — dataset features only, 7 terms | 0.349 | 0.341 | 0.306 |
| E2 — model features only, 6 terms | 0.248 | 0.185 | 0.228 |
| **E3 — both, 16 terms** | **0.665** | **0.627** | **0.622** |

The three equations differ only in which features they may draw on — **E1** sees the
dataset, **E2** sees the model, **E3** sees both. All three are fitted on the same 476 rows
by the same function and scored on the same 476 rows under the same two protocols, so the
gaps between them measure the features and nothing else.

**Their R² values share a scale but not a ceiling, and this is the most common way to
misread the table.** E1 predicts one value per dataset, so 0.354 — the variance of the true
per-dataset means — is the most it could *ever* reach, however good its terms were. E1 at
0.349 is not "much worse than E3"; it is finished. The comparable quantity is the fraction
of its own ceiling each equation attains:

| | reached | its ceiling | fraction |
|---|---|---|---|
| E1 (dataset features) | 0.349 | 0.354 | **99%** |
| E2 (model features) | 0.248 | 0.282 | **88%** |
| E3 (both) | 0.665 | *see below* | — |

E3 has no ceiling of that kind, because it is not constant within either group. The two
levels it can be read against are both *passed*, and passing them is the result:

| level | R² | what it bounds |
|---|---|---|
| additive oracle | 0.6605 | the best a per-dataset value **plus** a per-model value can do |
| all 75 single-feature terms | 0.6437 | the best a sum of per-feature functions can do |
| **E3, 16 terms** | **0.6651** | — |

Both bounds describe predictors that never combine a dataset feature with a model one. E3
does — 10 of its 16 terms are mixed — and clearing both by two independent routes is the
evidence that dataset×model *interaction* is what the equation is capturing.

![Equations against the levels they are read against](assets/figures/equation_comparison.png)

### What the documentation develops

- **One equation, refit — not one search per fold.** The equation's form is the claim; the
  folds recalibrate its constants and test whether the claim survives unseen data. Re-running
  term selection inside every fold answers a different question, and answering it as the
  first made transfer swing by 0.3 when the length changed by two.
  [Chapter 5](assets/docs/05-evaluation.md).
- **The corpus describes datasets far better than models.** As collected, its model
  descriptors reached 63% of their ceiling against the dataset side's 99%. Replacing them
  with a capability ordinal over the ten learner families and four mechanism gradings —
  all *asserted*, read off published descriptions rather than observed in a training run —
  takes the model side to 88%. What they cannot do is describe a method nobody has
  classified. [Chapter 7](assets/docs/07-limitations.md).
- **Mixed dataset×model terms carry the equation.** 10 of 16 terms use features from both
  groups and drive **53%** of the output variance; dataset-only terms drive 42% and
  model-only terms 5%. "Which model suits which data" is where the signal is, not "how hard
  is this data" or "how good is this model". [Chapter 6](assets/docs/06-practices.md).
- **No single meta-feature carries it either.** The strongest, `eq_num_attr`, reaches
  R² 0.142 alone; the transforms in the grammar are worth about +0.068 over entering the raw
  columns. There is no headline driver to quote, which is why the equation needs a dozen-odd
  terms rather than two. [Chapter 4](assets/docs/04-equation.md).
- **One interaction component is worth +0.122 R², and the equation reaches about a third of
  it** — alignment 0.31 in-sample and 0.27 out of fold, measured by stripping the additive
  part from both the truth and the prediction and correlating what is left. Only about
  **+0.018** of the +0.122 is reachable from meta-features on both sides, and the bottleneck
  is the model side. [Chapter 4](assets/docs/04-equation.md).
- **Ten best practices from the literature, weighed against the corpus** — 8 supported,
  1 qualified, 1 untestable here. The strongest: tree-based families average MCC **0.927**
  against **0.660** for neural ones on the datasets where every model ran, with plain MLPs
  and DNNs last of ten families at 0.454. [Chapter 10](assets/docs/10-report.md).

> **Status:** research prototype for an academic study. 476 rows is small, and every number
> is reported both in-sample and under leave-one-out cross-validation because at this size
> the two differ a lot. Current open work is in **[`TODO.md`](TODO.md)** — chiefly a full
> configuration sweep, and re-deriving the equation length on its result.

## Documentation

The rationale, methodology and full results live in
**[`assets/docs/`](assets/docs/index.md)**, written as a paper and carrying the generated
figures.

| | | modules |
|---|---|---|
| 0 | [Related work and positioning](assets/docs/00-related-work.md) | — |
| 1 | [The dataset](assets/docs/01-dataset.md) | `ml_meta_perf.data` |
| 2 | [The additive model](assets/docs/02-additive-model.md) | `ml_meta_perf.terms`, `ml_meta_perf.model` |
| 3 | [Term generation and selection](assets/docs/03-term-selection.md) | `ml_meta_perf.fit`, `ml_meta_perf.analysis`, `ml_meta_perf.selection` |
| 4 | [The equation](assets/docs/04-equation.md) | `ml_meta_perf.experiment`, `ml_meta_perf.validate` |
| 5 | [Evaluation](assets/docs/05-evaluation.md) | `ml_meta_perf.validate`, `ml_meta_perf.stats` |
| 6 | [Best practices against the equation](assets/docs/06-practices.md) | `ml_meta_perf.practices`, `ml_meta_perf.attribution`, `ml_meta_perf.guidance` |
| 7 | [Limitations and threats to validity](assets/docs/07-limitations.md) | `ml_meta_perf.identity` |
| 8 | [Appendix: what was measured and rejected](assets/docs/08-appendix.md) | — |
| 10 | [Generated report](assets/docs/10-report.md) | `ml_meta_perf.report` — **written by the code, not by hand** |

The API reference is generated from the module docstrings with `pdoc` and published to
GitHub Pages by `.github/workflows/docs.yml`; each module links back to the chapter
covering it.

## Installation

Python 3.12+. Runtime dependencies are **polars**, **numpy**, **matplotlib** and
**kneeliverse**.

```bash
python3 -m venv venv
venv/bin/pip install -e .
```

## Running it

**One command runs every phase**: screening, fitting E1/E3 and the two controls,
cross-validating under both protocols, extracting the practices, writing the figures, and
generating the report.

```bash
venv/bin/ml-meta-perf          # or: PYTHONPATH=src venv/bin/python -m ml_meta_perf
```

That takes about 20 seconds, reproduces every number in this README, and writes:

| | |
|---|---|
| [`assets/docs/10-report.md`](assets/docs/10-report.md) | the generated report — equation, term analysis, practices |
| `results/e1.json`, `e2.json`, `e3.json` | the fitted equations, reloadable |
| `results/*.csv` | 17 tables — curves, baselines, oracles, stability, practices |
| `assets/figures/*.png`, `*.pdf` | the 10 figures, raster and vector |

Everything printed and written is derived from the run. **The report is generated by
`ml_meta_perf.report`, not written by hand.** It ranks terms by standardised weight, writes a
sentence per major term, groups terms into the blocks that move together, and reports which
features and which grammar operations the search actually used — so a flat equation with no
dominant term is still readable, at a larger unit than one term.

Every phase can be skipped and every destination redirected, so the same entry point
serves a full study, a numbers-only run and a smoke test:

```bash
ml-meta-perf --no-figures                        # tables and report only
ml-meta-perf --quiet --no-tables --no-report     # figures only
ml-meta-perf --quick --output /tmp/check         # seconds, not minutes: wiring, not numbers
ml-meta-perf --output results --report report.md --figures figures
```

`ml-meta-perf --help` lists all of them.

**One note on threading.** The inner loop is ~87k solves of matrices no larger than 32×32,
far below the size where BLAS parallelism pays: threading buys no wall time and burns 3.5×
the CPU spinning. Setting `OPENBLAS_NUM_THREADS=1` costs nothing and saves the CPU; it is
left to the caller rather than forced from inside a library.

### Parameters

Every knob that was tuned during the study is a flag, and the defaults are the tuned
values, so a bare run is the reported study.

```bash
PYTHONPATH=src venv/bin/python -m ml_meta_perf --data mine.csv --output runs/mine
```

| flag | default | what it does |
|---|---|---|
| `--data` | the shipped corpus | the meta-dataset to fit |
| `--output` | `results` | where the equations and CSV tables go |
| `--figures` | `assets/figures` | where the figures go |
| `--report` | `assets/docs/10-report.md` | where the generated report goes |
| `--terms` | 24 | terms in the published E3 equation |
| `--max-terms` | 32 | longest equation the search explores (drives the curve) |
| `--penalty` | 5.0 | ridge penalty on standardised terms |
| `--arity` | 3 | raw features allowed per term — see [chapter 2](assets/docs/02-additive-model.md) |
| `--pool` | 600 | terms surviving screening into the beam |
| `--beam` | 6 | beam width |
| `--zscore` | 3.0 | largest standard score a term may reach before it is rejected as a spike |
| `--phase` | all | `screen`, `equations`, `validation`, `practices`, `figures`, `report`; repeatable |
| `--quick` | off | a reduced configuration for smoke-testing; **not** the study |
| `--no-figures`, `--no-tables`, `--no-report`, `--quiet` | off | skip an output |

```bash
# a shorter, more heavily shrunk equation, no figures
PYTHONPATH=src venv/bin/python -m ml_meta_perf --terms 12 --penalty 20 --no-figures

# four-feature terms: better fit, much worse transfer (chapter 2 measures this)
PYTHONPATH=src venv/bin/python -m ml_meta_perf --arity 4 --pool 2000 --output runs/arity4

# just the screening table
PYTHONPATH=src venv/bin/python -m ml_meta_perf --phase screen --no-figures
```

## Using it as a library

```python
from ml_meta_perf import DATASET_FEATURES, MODEL_FEATURES, build_library, columns_as_arrays, fit, load, target

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
from ml_meta_perf.practices import best_practices, render
from ml_meta_perf.plots import practice_effects

practices = best_practices(equation, columns)     # pass a stability table to rate them
print(render(practices))
practice_effects(practices, "practice_effects.png")
```

## Figures

`--figures <dir>` writes 10 figures, each as a PNG and a PDF on a transparent background. **None carries a title or an annotation** — they are
made for LaTeX `figure` environments where the caption does that work, and text baked into
a PNG cannot be restyled or translated. `ml_meta_perf.figures.captions()` returns a suggested
caption per file, including the disclosures deliberately kept out of the images.

## Layout

```
src/ml_meta_perf/
    data.py         loading, schema validation, aggregation, feature glossary
    terms.py        the term vocabulary and its admissibility rules
    stats.py        pearson, spearman, ranks, R2/MAE/RMSE/SMAPE (no scipy)
    analysis.py     correlation screening, redundancy clustering
    fit.py          standardisation, ridge solve, beam search, refinement
    model.py        the Equation object: predict, render, serialise
    validate.py     leave-one-group-out protocols, baselines, oracles
    selection.py    knee detection and Pareto fronts over equation length
    equation_search.py     the configuration sweep scored on seven components
    equation_search_cli.py `ml-meta-perf-search`, needs the `search` extra
    attribution.py  per-term effects, group shares, variance decomposition
    practices.py    per-feature associations measured from a fitted equation
    guidance.py     literature best practices, weighed against what the study measured
    plots.py        the figures (matplotlib, Agg, headless, no embedded text)
    figures.py      the figure set and suggested LaTeX captions
    report.py       the generated report: term importance and written analysis
    identity.py     per-model effects, measuring the ceiling on model descriptors
    experiment.py   the end-to-end study and its tuned configurations
    cli.py          the argparse pipeline: `ml-meta-perf`, `python -m ml_meta_perf`
    meta_dataset.csv  the corpus: 476 rows, shipped with the package
assets/docs/        the study chapters, 0-8 written and 10 generated
assets/figures/     generated figures
results/            generated equations, tables and report.md
tests/              unittest suite
scripts/            the Slurm batch script for the configuration sweep
.github/workflows/  CI on 3.12 and 3.14, and the published API reference
```

## Development

`.pre-commit-config.yaml` is the gate and `.github/workflows/main.yml` runs the same four
checks on Python 3.12 and 3.14, reading their settings from `pyproject.toml` so the two
cannot drift apart.

```bash
python3 -m venv venv                  # venv/, not .venv/ -- the hooks hard-code it
venv/bin/pip install -r requirements.txt
venv/bin/pre-commit install
venv/bin/pre-commit run --all-files   # ruff, basedpyright, vulture, unittest
```

`requirements.txt` installs the project editable with the `search` extra and pins the four
dev tools to exactly the versions CI installs, so a commit that passes the local hook passes
CI for the same reason. `pyproject.toml` remains the only source of the *project's*
dependencies — polars, numpy, matplotlib and kneeliverse, plus joblib under `search`.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
