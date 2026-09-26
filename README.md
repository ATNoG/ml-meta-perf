# ![ml-meta-perf logo](assets/logo.svg) ml-meta-perf

**ml-meta-perf** fits short, readable equations that predict the **Matthews Correlation
Coefficient (MCC)** a classifier will reach on a dataset, from meta-features of the dataset and
of the model:

```
MCC = w1*t1 + w2*t2 + ... + wk*tk
```

In the retained grammars, each `t` is a simple expression over one to three raw features — `f`, `log(f)`, `1/f`,
`f^2`, `f1/f2`, `f1*f2`, `(f1+f2)/f3` — and each `w` is a real number in the features' own
units, so the printed equation evaluates as written.

The evaluation protocols are in-sample (**IS**), leave-one-dataset-out (**LODO**),
leave-one-model-out (**LOMO**), and doubly held out (**DHO**).

The project prioritises a readable equation and tests whether that transparency costs transfer
accuracy. Opaque regressors fit these meta-data more closely under IS, but none beats the
equation under DHO. The result is an equation a practitioner
can inspect, challenge, and use to derive guidance.

> **Status:** research prototype for an academic study. The corpus contains 476 observed pairs
> from a grid of 20 datasets and 25 models, which is small. The headline equation scores are
> reported both under IS and held-out cross-validation because the two differ a lot
> at this size.

## Headline results

The equation labels are **E1** (dataset features only), **E2** (model features only),
**E3-Valid** (the published equation, using both feature groups), and **E3-MAX** (the same
configuration under a wider grammar, a capability bound rather than an equation to read).

The table below is **generated**: `ml-meta-perf` rewrites it on every run, like the result
sections of the chapters, so it cannot drift from the code.

<!-- generated: do not edit below -->

## The headline

The one table the study is summarised by, so that the summary cannot drift from the chapters. Every row is scored on the same 476 rows under the same protocols. E1, E2, and E3-Valid share the retained base configuration; E3-MAX uses the wider retained grammar as a capability bound.

| equation | features | terms | IS R2 | LODO R2 | LOMO R2 | DHO R2 | own ceiling | reached |
|---|---|---|---|---|---|---|---|---|
| E1 | dataset | 5 | 0.3394 | 0.3211 | 0.3022 | 0.2918 | 0.3539 | 0.9591 |
| E2 | model | 5 | 0.2537 | 0.1936 | 0.2312 | 0.1794 | 0.2821 | 0.8991 |
| E3-Valid | both | 17 | 0.6815 | 0.6513 | 0.6261 | 0.6151 |  |  |
| E3-MAX | both | 29 | 0.7271 | 0.6894 | 0.6538 | 0.6482 |  |  |

**Do not read these R² values as achievements against each other.** They share a scale but not a ceiling: E1 sees only dataset features, every row of a dataset shares one feature vector, and so E1 can predict nothing but a per-dataset constant. Its structural maximum is the `true dataset means` row, and reaching it means E1 is *done* rather than weak. The comparable quantity is the fraction of each equation's own ceiling, which the last column gives.

**E3-Valid and E3-MAX have blank ceiling cells because neither has a structural group-identity ceiling.** Nothing in the feature set stops an equation over both halves of the meta-data from predicting every cell, so there is no group-identity bound to divide by. The reference shown instead is the additive mean-based reference, in the comparison table of chapter 5. That reference bounds only an equation additive in dataset effect plus model effect; E3-Valid's mixed terms can represent interactions beyond it. The comparison table reports whether the current equation reaches or exceeds that reference.

<!-- end generated -->

The three primary equations differ only in which features they may draw on. All three are
fitted on the same 476 rows by the same function, under **one** hyperparameter configuration
(`config/study.json`), have their length chosen by **one** rule, and are scored on the same
476 rows under the same four protocols, so the gaps between them measure the features and
nothing else.

**Their coefficient of determination (R²) values share a scale but not a ceiling, and this
is the most common way to misread the table.** E1 predicts one value per dataset, so the
variance of the true per-dataset means is the most it could *ever* reach, however good its
terms were. E1 near that ceiling is not "much worse than E3"; it is finished. The comparable
quantity is the fraction of its own ceiling each equation attains — the `reached` column.
E3-Valid has no ceiling of that kind, because it is not constant within either group; it is
read instead against the best sum of per-feature functions and the best per-dataset value
plus per-model value, both of which it passes ([chapter 4](assets/docs/04-equation.md)).

**E3-Valid and E3-MAX have separate roles.** E3-Valid's length is the start of the first
sustained plateau of its worst-protocol curve: the first knee (found by multi-Kneedle,
[`kneeliverse`](https://github.com/mariolpantunes/knee)) of the smoothed curve after which it
stops improving, or its smoothed maximum when every knee is still followed by real gains.
E3-MAX takes the raw maximum of the same curve under the wider grammar
([chapter 3](assets/docs/03-term-selection.md)).

![Equations against the levels they are read against](assets/figures/01_equation_comparison.png)

### What the documentation develops

- **One equation, refit — not one search per fold.** The equation's form is the claim; the
  folds recalibrate its constants and test whether the claim survives unseen data. Re-running
  term selection inside every fold answers a different question about the search, and much
  of the transfer it reports belongs to choosing the terms on all rows — see the limitations
  of [chapter 3](assets/docs/03-term-selection.md) and [chapter 5](assets/docs/05-evaluation.md).
- **One configuration, chosen by the published equation's own criterion.** The shared
  hyperparameters come from a sweep (`ml-meta-perf-search`) that applies the study's length
  rule to every configuration and, among those tied on R², proposes the one whose
  doubly-held-out predictions rank the models best. Each equation searches every feature it
  may use; the configuration holds hyperparameters only.
- **The corpus describes datasets far better than models.** The model side's descriptors are
  a capability ordinal over the ten learner families and four mechanism gradings — all
  *asserted*, read from published descriptions rather than observed in a training run.
  What they cannot do is describe a method nobody has classified.
  [Chapter 1](assets/docs/01-dataset.md).
- **Mixed dataset×model terms carry a large share of the equation**, which is where "which
  model suits which data" lives; the shares by feature group are generated in
  [chapter 4](assets/docs/04-equation.md).
- **No single meta-feature carries it**, which is why the equation needs many terms rather
  than two, and **one interaction component is worth a measurable R², of which the equation
  reaches part** — both measured in [chapter 4](assets/docs/04-equation.md).
- **Ten best practices from the literature, weighed against the corpus** — which are
  supported, qualified or untestable here is generated in [chapter 6](assets/docs/06-practices.md).

## Documentation

The rationale, methodology and full results live in
**[`assets/docs/`](assets/docs/index.md)**, written as a paper and carrying the generated
figures.

| | | modules |
|---|---|---|
| 0 | [Related work and positioning](assets/docs/00-related-work.md) | — |
| 1 | [The dataset](assets/docs/01-dataset.md) | `ml_meta_perf.data` |
| 2 | [The additive model](assets/docs/02-additive-model.md) | `ml_meta_perf.terms`, `ml_meta_perf.model` |
| 3 | [Term generation and selection](assets/docs/03-term-selection.md) | `ml_meta_perf.search`, `ml_meta_perf.fit`, `ml_meta_perf.analysis`, `ml_meta_perf.selection` |
| 4 | [The equation](assets/docs/04-equation.md) | `ml_meta_perf.experiment`, `ml_meta_perf.validate` |
| 5 | [Evaluation](assets/docs/05-evaluation.md) | `ml_meta_perf.validate`, `ml_meta_perf.stats` |
| 6 | [Best practices against the equation](assets/docs/06-practices.md) | `ml_meta_perf.practices`, `ml_meta_perf.attribution`, `ml_meta_perf.guidance` |

Each chapter is self-contained: it ends with its own limitations, and chapters 4, 5 and 6
close with a **generated section** that `ml_meta_perf.report` rewrites on every run. Prose
above the marker is hand-written and describes the method; everything below it is computed
from the fitted equation, so no chapter can carry a stale table.

The application programming interface (API) reference is generated from the module docstrings with `pdoc` and published to
GitHub Pages by `.github/workflows/docs.yml`; each module links back to the chapter
covering it.

## Installation

Python 3.12+. Runtime dependencies are **Polars**, **NumPy**, **Matplotlib**,
**scikit-learn**, **joblib**, **jsonargparse** (reads `config/study.json`), and
**[kneeliverse](https://github.com/mariolpantunes/knee)** (knee detection for the
equation-length rule).

scikit-learn is there for one thing: the opaque-regressor comparison in
[chapter 5](assets/docs/05-evaluation.md), which prices the other side of the trade this
study is making. It brings SciPy with it, which the project refused for a long time on size
grounds — `ml_meta_perf.stats` exists because four statistics were not worth the dependency,
and that reasoning still holds for those four. What changed the answer is that the comparison
the chapters lean on hardest was measured once by hand and reproducible by nobody. An
unreproducible headline is a worse cost than a large wheel.

```bash
python3.12 -m venv venv
venv/bin/python -m pip install -e ".[reproducibility]"
```

On Windows PowerShell, create the environment with `py -3.12 -m venv venv` and use
`venv\Scripts\python.exe` or `venv\Scripts\pre-commit.exe` in place of the corresponding
`venv/bin/` command.

`pyproject.toml` is the only dependency manifest. The `reproducibility` extra freezes the
complete runtime used for the reference run; a plain `venv/bin/python -m pip install -e .`
installs the supported dependency ranges declared by the package instead.

## Meta-Dataset Generation

The generated corpus is versioned at
[`dataset/meta_dataset.csv`](dataset/meta_dataset.csv), outside the package. The study finds
it from a source checkout, or under the working directory when the package is installed and
run from a clone; anywhere else, pass `--data`. The scripts and
configuration needed to rebuild that corpus live in
[`meta_dataset_pipeline/`](meta_dataset_pipeline/).

The meta-dataset pipeline has additional dependencies, including `pymfe` and the tabular
model libraries used during the original model-evaluation stages. Install them explicitly:

```bash
venv/bin/python -m pip install -e ".[meta-dataset]"
```

On Windows PowerShell:

```powershell
venv\Scripts\python.exe -m pip install -e ".[meta-dataset]"
```

That folder is intentionally separate from the package code:

- `meta_dataset_pipeline/` contains the raw-corpus stages and can regenerate a candidate
  corpus at `meta_dataset_pipeline/results/meta_dataset.csv`;
- `src/ml_meta_perf/data.py` loads and validates the published corpus schema;
- `dataset/meta_dataset.csv` is the versioned corpus consumed by the library and
  command-line tool, and is updated only when the regenerated corpus is accepted.

See [`meta_dataset_pipeline/README.md`](meta_dataset_pipeline/README.md) for the stage
order, local token configuration, ignored raw dataset files, and instructions for comparing
or publishing a regenerated corpus.

## Running it

**One command runs every phase**: screening, fitting E1, E2, E3-Valid and E3-MAX, cross-validating under all four protocols, pricing the opaque comparison, extracting the
practices, writing the figures, and splicing the generated sections into the chapters.

```bash
venv/bin/ml-meta-perf          # or: PYTHONPATH=src venv/bin/python -m ml_meta_perf
```

That takes about six minutes on the reference Windows environment — nearly all of it the opaque-regressor comparison, which
refits a random forest once per held-out group and then once per *cell* for the
DHO protocol, 522 fits in all. The equation half of the pipeline is 14 seconds.
It reproduces every number in this README, and writes:

| | |
|---|---|
| [`assets/docs/`](assets/docs/index.md), `README.md` | the index, chapters 1, 3, 4, 5 and 6, and this README's headline — their generated sections rewritten in place |
| `results/e1.json`, `e2.json`, `e3.json` | the fitted equations, reloadable |
| `results/*.csv` | the study's tables — curves, baselines, oracles, stability, practices |
| `assets/figures/*.png`, `*.pdf` | the figure set, raster and vector, numbered in the order the study presents it |

Everything printed and written is derived from the run. **The report is generated by
`ml_meta_perf.report`, not written by hand.** It ranks terms by standardised weight, writes a
sentence per major term, groups terms into the blocks that move together, and reports which
features and which grammar operations the search actually used — so a flat equation with no
dominant term is still readable, at a larger unit than one term.

Every phase can be skipped and every destination redirected, so the same entry point
serves a full study and a numbers-only run:

```bash
ml-meta-perf --no-figures                        # tables and report only
ml-meta-perf --quiet --no-tables --no-report     # figures only
ml-meta-perf --output results --docs assets/docs --figures figures
```

`ml-meta-perf --help` lists all of them. There is no smoke-test flag: `--quick` was a preset
of three flags that already exist, and it quietly reached two things they did not, which is
how it came to advertise "seconds" while still paying for the full opaque comparison. The
wiring check is `python -m unittest discover -s tests`, which takes about a minute and checks more.

The shared configuration is chosen by a separate command, `ml-meta-perf-search`: it fits every
point of the grid in `config/study.json`'s `sweep` section once, applies the study's own length
rule to each, and among the configurations tied on R² at a readable length proposes the one
whose doubly-held-out predictions rank the models best. It writes `proposed_study.json`;
adopting it is a reviewed edit to `config/study.json`.

**One note on threading.** The inner loop is ~87k solves of matrices no larger than 32×32,
far below the size where Basic Linear Algebra Subprograms (BLAS) parallelism pays: threading
buys no wall time and burns 3.5× the central processing unit (CPU) spinning. Setting
`OPENBLAS_NUM_THREADS=1` costs nothing and saves the CPU; it is
left to the caller rather than forced from inside a library.

### Configuration

Every tuned value lives in **`config/study.json`** and nowhere else: the command line has no
defaults of its own for them. The file is read with `jsonargparse`, so `--config other.json`
replaces it, any field can be overridden as a dotted flag, and `--print_config` prints the
effective configuration. E1, E2 and E3 share every one of these values.

| setting | value | what it does |
|---|---|---|
| `search.max_abs_zscore` | 4.0 | largest standard score a term may reach before it is rejected as a spike |
| `search.penalty` | 0.3 | ridge penalty on standardised terms |
| `search.pool_size` | 600 | terms surviving screening into the beam |
| `search.max_terms` | 30 | longest equation the search explores (the curve's horizon, not the published length) |
| `search.beam_width` | 6 | beam width |
| `search.max_arity` | 2 | raw features per term — see [chapter 2](assets/docs/02-additive-model.md) |
| `selection.delta` | 0.01 | largest smoothed-floor gain over the window that still counts as a plateau |
| `selection.window` | 4 | lengths after a candidate that must stay within `delta` |
| `selection.smoothing` | 3 | width of the running median applied to the worst-protocol curve |
| `selection.capability_arity` | 3 | the wider grammar E3-MAX is fitted under |
| `opaque.forest_trees` | 100 | random forest size |
| `opaque.forest_max_features` | 0.33 | share of features considered at each forest split |
| `opaque.boosting_stages` | 50 | gradient boosting stages |
| `opaque.boosting_learning_rate` | 0.1 | gradient boosting learning rate |
| `opaque.boosting_max_depth` | 3 | gradient boosting tree depth |

The file's `sweep` section is the grid `ml-meta-perf-search` explores and how it chooses among
close candidates. A test checks this table against the file.

Run options are ordinary flags:

| flag | default | what it does |
|---|---|---|
| `--config` | `config/study.json` | the study configuration |
| `--data` | `dataset/meta_dataset.csv` | the meta-dataset to fit |
| `--output` | `results` | where the equations and comma-separated value (CSV) tables go |
| `--figures` | `assets/figures` | where the figures go |
| `--docs` | `assets/docs` | chapter directory whose generated sections are rewritten |
| `--readme` | `README.md` | README whose generated headline is rewritten |
| `--phase` | all | comma-separated subset of `screen`, `equations`, `validation`, `practices`, `figures`, `report` |
| `--no-figures`, `--no-tables`, `--no-report`, `--quiet` | off | skip an output |

```bash
# a shorter horizon and a heavier ridge, no figures. The published length is not a flag:
# it is derived from each equation's own curve
ml-meta-perf --search.max_terms 12 --search.penalty 10 --output runs/short --no-figures --no-report

# the wider grammar for every equation -- ratio-of-sums terms, harder to read
ml-meta-perf --search.max_arity 3 --output runs/arity3 --figures runs/arity3/figures --no-report

# just the screening table
ml-meta-perf --phase screen --output runs/screen --no-figures --no-report

# re-run the configuration sweep; writes results/sweep/proposed_study.json for review
ml-meta-perf-search --output results/sweep
```

## Using it as a library

```python
from ml_meta_perf import (
    DATASET_FEATURES, MODEL_FEATURES, build_library, columns_as_arrays, load, load_config, search, target,
)

frame = load()
columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
library = build_library(DATASET_FEATURES, MODEL_FEATURES, columns, max_arity=2, max_abs_zscore=4.0)

config = load_config().search
equation = search(library, target(frame), max_terms=config.max_terms, penalty=config.penalty).best()

print(equation)                    # human-readable, with standardised betas
print(equation.to_latex())         # for the paper
equation.save("e3.json")           # round-trips exactly
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

`--figures <dir>` writes the figure set, each as a Portable Network Graphics (PNG) file and
a Portable Document Format (PDF) file on a transparent
background. **Files are numbered by their position in the set** — `01_equation_comparison.png`,
`02_term_count_curve.png` and so on — so a figure can be named by its number in a review or a
caption. `ml_meta_perf.figures.FIGURE_ORDER` is the single place that order is written down,
and `generate` refuses to write a set that does not match it. **None carries a title or an
annotation** — they are
made for LaTeX `figure` environments where the caption does that work, and text baked into
a PNG cannot be restyled or translated. `ml_meta_perf.figures.captions()` returns a suggested
caption per file, including the disclosures deliberately kept out of the images.

## Layout

```
src/ml_meta_perf/
    data.py         loading, schema validation, aggregation, feature glossary
    terms.py        the term vocabulary and its admissibility rules
    stats.py        pearson, spearman, ranks, R2 (R²), mean absolute error (MAE),
                    root mean squared error (RMSE), and symmetric mean absolute percentage
                    error (SMAPE), written out
    opaque.py       the opaque-regressor comparison (scikit-learn), same protocols
    analysis.py     correlation screening, redundancy clustering
    search.py       screening, beam search, refinement, pruning
    fit.py          standardisation, the ridge solve, and reading weights back out
    model.py        the Equation object: predict, render, serialise
    validate.py     leave-one-group-out protocols, baselines, oracles
    selection.py    the equation-length rule (smoothed worst-protocol curve, multi-Kneedle,
                    first sustained plateau) and the E3-MAX bound
    attribution.py  per-term effects, group shares, variance decomposition
    practices.py    per-feature associations measured from a fitted equation
    guidance.py     literature best practices, weighed against what the study measured
    plots.py        the figures (Matplotlib, Agg, headless, no embedded text)
    figures.py      the figure set and suggested LaTeX captions
    report.py       the generated report: term importance and written analysis
    identity.py     per-model effects, measuring the ceiling on model descriptors
    experiment.py   the end-to-end study: one configuration, one length rule, E1/E2/E3/E3-MAX
    config.py       the typed schema of config/study.json and its loader
    sweep.py        `ml-meta-perf-search`: the configuration sweep that proposes study.json
    cli.py          the jsonargparse pipeline: `ml-meta-perf`, `python -m ml_meta_perf`
config/             study.json, every tuned hyperparameter
dataset/            meta_dataset.csv, the versioned corpus
assets/docs/        the study chapters, hand-written with generated sections spliced in
assets/figures/     generated figures
results/            generated equations and tables
meta_dataset_pipeline/  raw-corpus pipeline used to regenerate meta_dataset.csv
tests/              unittest suite
.github/workflows/  continuous integration (CI) on 3.12 and 3.14, and the published API reference
pyproject.toml      the package, its dependency ranges, and the extras: `dev` (pinned
                    tooling), `reproducibility` (exact reference runtime), `meta-dataset`
```

## Development

`.pre-commit-config.yaml` is the gate and `.github/workflows/main.yml` runs the same four
checks on Python 3.12 and 3.14, reading their settings from `pyproject.toml`. The hooks use
a pre-commit-managed environment, so the installed Git hook runs on both Windows and Linux.

```bash
python3 -m venv venv
venv/bin/pip install -e ".[dev]"
venv/bin/pre-commit install
venv/bin/pre-commit run --all-files   # ruff, basedpyright, vulture, unittest
```

The `dev` extra installs the project editable and pins pre-commit plus the three
static-analysis tools, at the versions continuous integration (CI) installs. `pyproject.toml`
defines the supported runtime dependency ranges; its `reproducibility` extra freezes the
complete environment used for the reference experiment.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
