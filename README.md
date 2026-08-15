# ![metafit logo](assets/metafit_logo.svg) metafit

**metafit** fits short, readable equations that predict the **Matthews Correlation
Coefficient** a classifier will reach on a dataset, from meta-features of the dataset
and of the model:

```
MCC = w1*t1 + w2*t2 + ... + wk*tk
```

where each `t` is a simple expression over one to three raw features — `f`, `log(f)`,
`1/f`, `f^2`, `f1/f2`, `f1*f2`, `(f1+f2)/f3` — and each `w` is a real number in the
features' own units, so the printed equation can be evaluated as written.

The project trades accuracy for explainability on purpose. Opaque regressors reach far
higher R² on this kind of meta-data, but nothing can be read off them; the point here is
an equation a practitioner can inspect, argue with, and derive guidance from.

> **Status:** research prototype for an academic study. The meta-dataset has 20 datasets
> and 25 models (476 rows), which is small. Every number below is reported both in-sample
> and under leave-one-out cross-validation, because on data this size the two differ a lot.

## The two equations

The study is built around a deliberate contrast.

**E1 — dataset features only.** Every model evaluated on a given dataset shares one
feature vector, so many MCC values map to a single input and least squares necessarily
lands on the per-dataset mean. That is not a defect to be fixed; it is the control. E1
measures how much of MCC is explained by *the data alone*.

**E2 — dataset + model features.** One input, one output. E2 can express variation
*within* a dataset, which is exactly what E1 structurally cannot do.

The gap between them is the evidence that capturing model capability matters.

## Results

### E1 vs E2, scored on the same 476 rows

R² depends on the variance in its denominator, so E1's own number (computed over 20
dataset means) and E2's (computed over 476 rows) are not comparable. Evaluating both
equations on every row puts them on one scale:

| | R² | MAE | Spearman |
|---|---|---|---|
| E1 (dataset only, 5 terms) | 0.308 | 0.226 | 0.640 |
| *E1's hard ceiling — the true dataset means* | *0.354* | *0.204* | *0.653* |
| **E2 (dataset + model, 12 terms)** | **0.556** | **0.169** | **0.780** |
| *additive oracle — true dataset + model effects* | *0.661* | *0.145* | *0.810* |

E1 is capped at **0.354** no matter how good the dataset equation becomes, because
per-dataset means explain only 35.4% of the total variance in MCC. E2 reaches 0.556.
Model features are not a refinement here — they are most of the signal.

The **additive oracle** row is the ceiling for this entire approach: give a model the
*exact* per-dataset and per-model effects and let it add them, and you get R² = 0.661.
No equation of the form `w1*t1 + w2*t2 + ...` can beat that, because that is what
additivity buys. E2 captures 84% of it.

### On their own scales

| | terms | in-sample R² | LOO-dataset R² | LOO-model R² |
|---|---|---|---|---|
| E1 (20 dataset means) | 5 | **0.953** | 0.506 | — |
| E2 (476 rows) | 12 | **0.556** | 0.371 | 0.455 |

### Accuracy versus number of terms

The explainability trade, made explicit rather than settled by taste. E2:

| terms | in-sample R² | LOO-dataset R² | LOO-model R² |
|---|---|---|---|
| 2 | 0.353 | 0.199 | 0.333 |
| 4 | 0.467 | 0.288 | 0.433 |
| 6 | 0.502 | 0.232 | 0.448 |
| 8 | 0.533 | 0.303 | 0.426 |
| 10 | 0.550 | 0.327 | 0.467 |
| **12** | **0.556** | **0.371** | **0.455** |
| 14 | 0.558 | 0.443 | 0.457 |
| 16 | 0.562 | 0.436 | 0.461 |

Past about 10 terms the in-sample curve is flat — the extra terms buy 0.006 R² between
12 and 16 — while the cross-validated column is still moving by more than that from fold
noise alone. Twelve is chosen from the flat part of the curve rather than from its peak.

A looser configuration (`LOOSE_E2`) pushes in-sample R² to **0.600** at 16 terms, but its
leave-one-dataset-out R² falls to about **0.23**. That direction is available and
documented; the defaults take the generalising side of the trade.

### The fitted equations

E1, on the 20 dataset means:

```
MCC = +1.21681
      -0.160398   * [log(eq_num_attr)] * [log(nr_class)]
      -0.0601461  * ([nr_cor_attr] + [log(ns_ratio)]) / [log(nr_class)]
      -0.0286345  * ([log(class_ent)] + [log(gravity)]) / [log(nr_attr)]
      +0.00728631 * [log(eq_num_attr)] * [log(inst_to_attr)]
      +0.000217397* [log(inst_to_attr)] * [nr_norm]
```

E2, on all 476 rows:

```
MCC = +0.829491
      -0.130089   * [log(eq_num_attr)] * [log(nr_class)]
      -0.0372985  * [log(gravity)] / [log(Training Operations)]
      -0.177795   * [nr_cor_attr] * [nr_norm]
      +0.0496212  * [log(Processing Units Number)] * [Robust to Outliers]
      -0.0327811  * ([log(eq_num_attr)] + [log(ns_ratio)]) / [log(nr_class)]
      +0.0946915  * sqrt(Prediction Operations)
      +0.0112845  * ([nr_bin] + [nr_norm]) / [log(nr_attr)]
      +0.0158504  * Training Operations
      -0.0335735  * [log(Prediction Operations)] * [Robust to Outliers]
      -0.0244605  * [log(inst_to_attr)] / [log(Processing Units Number)]
      +0.000902822* [log(inst_to_attr)] * [nr_norm]
      +0.103897   * [nr_cor_attr] * [log(Training Operations)]
```

Predictions are clipped to [-1, 1]: a linear form has no idea MCC stops at 1.0, and 17%
of the meta-dataset sits exactly there.

## Two things worth knowing before trusting any number like this

### Validation protocol changes the answer by more than the model does

Dataset features are constant across a dataset's 25 rows. A random k-fold split therefore
puts the same dataset on both sides of the fold, and the equation recognises the dataset
instead of generalising to it. The **same equation**, three protocols:

| protocol | R² | MAE |
|---|---|---|
| random 10-fold | **0.514** | 0.169 |
| leave-one-dataset-out | **0.371** | 0.199 |
| leave-one-model-out | **0.455** | 0.191 |

A reported R² near 0.5 on this data may be describing the split, not the model. This is
why leave-one-*group*-out is the only protocol `metafit` reports as a headline.

### Flexible models do *worse* here, not better

Under leave-one-dataset-out, on the same features:

| model | LOO-dataset R² |
|---|---|
| RandomForest (400 trees) | 0.067 |
| GradientBoosting | 0.049 |
| **metafit E2 (12-term equation)** | **0.371** |

With 20 dataset groups, flexible learners memorise dataset identity and generalise
poorly to an unseen dataset. The sparse equation is not a concession to interpretability
on this data — it is the better predictor. (Measured with scikit-learn during
exploration; it is not a dependency of this package.)

## Baselines

An equation only earns its place by beating the obvious alternatives:

| baseline | R² | MAE |
|---|---|---|
| global mean (loo-dataset) | -0.040 | 0.293 |
| per-model mean (loo-dataset) | 0.201 | 0.238 |
| global mean (loo-model) | -0.024 | 0.290 |
| per-dataset mean (loo-model) | 0.296 | 0.213 |

E2 beats all four on its respective protocol. **One honest caveat:** for *ranking* models
on a new dataset, the trivial "average MCC of this model elsewhere" baseline achieves a
higher mean per-dataset Spearman (0.70) than E2 (0.61), at comparable top-1 regret.
E2 wins on predicting the MCC *value*; it does not dominate on ranking.

## Method

**1. Correlation screening.** Every candidate term is scored by Pearson *and* Spearman
correlation against MCC. The two disagree informatively: when they agree the relation is
linear and a plain `f` is the right term; when Spearman is clearly larger the relation is
monotone but curved, which is the signal that a log, an inverse or a ratio will pay for
itself. Terms are additionally scored *within* group — both the term and the target are
centred inside each dataset first — so a term is credited only for variance that dataset
identity does not already explain.

**2. Term library.** Unary transforms over each feature, plus ratios, products and
`(f1+f2)/f3` combinations. Composite terms are built over log-compressed operands
wherever the feature is strictly positive, since `gravity` alone spans 1.6 to 1e16.

**3. Stability filtering.** Terms are rejected when they are effectively constant, when a
single row dominates their variance, or when a ratio's denominator approaches zero.
This matters more than it sounds — see below.

**4. Beam search.** Subset selection with a beam rather than greedy descent, ridge refit
at every step, a collinearity guard, and a swap-based refinement pass. Because the beam
records its best subset at every size, one search yields the whole term-count curve.

**5. Validation.** Leave-one-dataset-out and leave-one-model-out. Term *selection* runs
inside each fold, not once outside it — screening against the full target and then
cross-validating only the weights is a standard way to leak the held-out fold.

### Why the stability filters exist

Removing them raises in-sample R² to 0.611 and drives leave-one-dataset-out R² to **-1.7**.

Several features contain exact zeros (`nr_norm`, `nr_bin`, `nr_outliers`) and `log(f)`
hits zero whenever `f` reaches 1. Ratios dividing by these produce terms whose entire
variance is one spike. Standardisation hides the problem while fitting — the spike simply
becomes the scale — but the term explodes on a held-out dataset outside the training
range. A near-constant term causes a different failure: folding the standardisation back
into raw units divides its weight by a negligible spread, which once produced a
coefficient of `-1.5e9` against an intercept of `+1.5e9`. Correct arithmetic, unreadable
equation.

## Installation

Requires Python 3.12+. Dependencies are **polars** and **numpy** only.

```bash
python3 -m venv venv
venv/bin/pip install -r requirements-dev.txt
```

## Usage

Reproduce the whole study:

```bash
PYTHONPATH=src venv/bin/python -m metafit
```

```bash
PYTHONPATH=src venv/bin/python -m metafit --save results   # write e1.json / e2.json
PYTHONPATH=src venv/bin/python -m metafit --quick          # fast smoke run, not the study
```

As a library:

```python
from metafit import DATASET_FEATURES, MODEL_FEATURES, build_library, columns_as_arrays, fit, load, target

frame = load()
columns = columns_as_arrays(frame, DATASET_FEATURES + MODEL_FEATURES)
library = build_library(DATASET_FEATURES, MODEL_FEATURES, columns)

result = fit(library, target(frame), max_terms=12, penalty=20.0)
equation = result.best()

print(equation)                    # human-readable
print(equation.to_latex())         # for the paper
equation.save("e2.json")           # round-trips exactly
predictions = equation.predict(columns)
```

## Layout

```
src/metafit/
    data.py         loading, schema validation, per-dataset aggregation
    terms.py        the term vocabulary and its stability filters
    stats.py        pearson, spearman, rank data, R2/MAE/RMSE (no scipy)
    analysis.py     correlation screening, redundancy clustering
    fit.py          standardisation, ridge solve, beam search, refinement
    model.py        the Equation object: predict, render, serialise
    validate.py     leave-one-group-out protocols, baselines, oracles
    experiment.py   the end-to-end study and its tuned configurations
    cli.py          python -m metafit
data/               meta_dataset.csv
tests/              137 unittest tests
examples/           runnable entry point
```

## Development

`ci.sh` runs the full gate; `.pre-commit-config.yaml` runs the same checks at commit time.

```bash
./ci.sh                        # unittest, coverage, ruff, basedpyright, vulture
venv/bin/pre-commit install
```

## Limitations

- 20 datasets and 25 models is small. Leave-one-dataset-out R² varies by ±0.07 between
  adjacent term counts from fold noise alone; the curve should be read, not one cell.
- All datasets are networking/IoT/security tabular benchmarks. The equations should not
  be assumed to transfer to other domains.
- The additive form cannot express dataset×model interaction beyond what the supplied
  terms encode; the 0.661 oracle bounds the whole approach.
- E1's leave-one-dataset-out numbers rest on 20 points and are correspondingly unstable
  (negative at 1-2 terms, 0.51 at 5).

## Citation

See `CITATION.cff`. Related work and how this positions against it: `RESEARCH.md`.

## License

MIT — see `LICENSE`.
