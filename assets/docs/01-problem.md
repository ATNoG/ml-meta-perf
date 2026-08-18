# 1. The problem and the data

*Implemented in `metafit.data`.*

## The algorithm selection problem

Predicting how well an algorithm will perform on a dataset it has not seen is the
**algorithm selection problem**, framed by Rice (1976) as a mapping from a feature space
of problem instances to a performance space of algorithms. Meta-learning operationalises
it by describing each dataset with *meta-features* — statistical, information-theoretic
and complexity measures — and learning a regressor from those to a performance metric.

This study inverts the usual priority. The deliverable is the **mapping itself, in
readable form**, rather than its accuracy. An opaque regressor that predicts well but
cannot be inspected does not answer the question the practitioner is actually asking,
which is *why* one model suits one dataset.

## The target: MCC

The target is the **Matthews Correlation Coefficient**, chosen over accuracy or F1
because the datasets are imbalanced security and IoT benchmarks where accuracy is
misleading (Chicco & Jurman, 2020). MCC is bounded in [-1, 1], where 0 is chance and
negative values indicate anti-correlation with the labels.

Three properties of this target shape every downstream decision:

- **It is bounded.** A linear form is unaware of that, so predictions are clipped to
  [-1, 1] (`metafit.model.Equation.predict`).
- **It saturates.** 80 of the 476 rows (17%) sit at exactly 1.0. A logit or `atanh`
  transform of the target was tried to handle the bound and **failed badly** —
  cross-validated R² went negative, because the saturation point maps to infinity.
- **It passes through zero.** 15 rows are exactly 0.0 and one is negative. This is why
  MAE rather than SMAPE is the honest error headline: see
  [chapter 4](04-evaluation.md).

## The meta-dataset

`src/metafit/meta_dataset.csv` — 476 rows, no missing values. One row per (dataset, model) pair,
covering **20 datasets × 25 models** (24 of the 500 possible pairs are absent).

### Each row is the best of three seeds, not their mean

Audited against the corpus that produced it
([meta2perf-symbolic-regression](https://github.com/mariolpantunes/meta2perf-symbolic-regression),
`exp_stage_create_meta_dataset.py`). Every (dataset, model) pair was trained under three
random seeds, and the row kept here is the one with the **highest** MCC
(`groupby(["Dataset","Model"]).MCC.idxmax()`). Measured over the 348 pairs whose per-seed
records are published:

| | MCC |
|---|---|
| mean of (max − mean) across seeds | **0.0206** |
| mean of (max − min) across seeds | 0.0512 |
| 90th percentile of (max − min) | 0.1020 |
| pairs whose seeds spread more than 0.05 | 64 of 340 |
| mean within-run cross-validation fold std | 0.0267 |

Two consequences, and only the first is a caveat.

**The target is optimistic by about 0.02 MCC.** Every absolute statement in this study —
"tree families average 0.927", the go/no-go thresholds — describes the best of three runs
rather than a typical one. The bias is roughly constant across rows, so comparisons
between models and the fitted weights are barely affected; the level is.

**It sets a noise floor.** A quantity that moves by 0.05 between seeds of the *same*
configuration cannot be predicted more precisely than that by anything. The study's
leave-one-dataset-out MAE is 0.183, well above the floor, so the sample rather than the
target's own noise is what binds — but the floor is where an error curve would stop, and
it is worth knowing that it sits around 0.03–0.05.

The 80 rows at exactly 1.0 are not an artefact of the selection: of the 54 saturated pairs
with published seed records, 52 average above 0.99 across all three.

| | count | constant within |
|---|---|---|
| dataset meta-features | 12 | a dataset |
| model meta-features | 5 | — |
| **usable features** | **17** | |

Twenty CSV columns minus `Dataset`, `Model` and `MCC` leaves 17 features.

### Dataset features

`class_ent`, `eq_num_attr`, `gravity`, `inst_to_attr`, `nr_attr`, `nr_bin`, `nr_class`,
`nr_cor_attr`, `nr_inst`, `nr_norm`, `nr_outliers`, `ns_ratio`.

### Model features

`Processing Units Number`, `Training Operations`, `Prediction Operations`,
`Active Regularization Mechanisms`, `Robust to Outliers`.

`Robust to Outliers` is constant per model. The three operation counts are already
log-scaled and vary with the dataset as well as the model, since they are functions of
dataset size — which is why the model features are not purely model-level.

### How the runs were produced, and what that rules out

Two properties of the corpus are invisible in the CSV and change what may be asked of it.

**Every model was trained on a stratified sample capped at 100,000 rows.** Ten of the
twenty datasets are larger than that — up to 7.1 million — so `nr_inst` records the size of
the *source dataset*, not the size of the training set, and above the cap the training sets
are all the same size. **No question about the effect of more training data can be answered
here.** `nr_inst` and `inst_to_attr` remain perfectly good meta-features — they describe the
corpus a problem was drawn from, which is knowable before training — but a term over them
is not a statement about sample size.

**Runs that failed to train were discarded.** That is where the 24 missing cells went, and
they are not missing at random: eight models are absent from the same three datasets, which
are the three smallest in the corpus (165, 389 and 400 instances). The meta-dataset is
therefore a sample of *completed* runs, and every prediction is conditional on training
succeeding ([chapter 8](08-limitations.md)).

## Two facts that drive the whole design

**Dataset features are constant across a dataset's 25 rows.** Everything follows from
this. It means a dataset-only equation can predict nothing but a per-dataset constant
(chapter 6), and it means a random train/test split puts the same dataset on both sides
of the fold and inflates every score (chapter 4).

**The features span wildly different magnitudes.** `gravity` runs from 1.6 to 1.0e16;
`nr_inst` from 165 to 7.1 million. Composite terms are therefore built over
log-compressed operands wherever a feature is strictly positive, and the design matrix is
standardised before selection (chapter 2).

## Range of the target

| | value |
|---|---|
| minimum | -0.2898 (NSL-KDD / SGD, one row) |
| rows at exactly 0.0 | 15 |
| maximum | 1.0 |
| rows at exactly 1.0 | 80 |
| mean | 0.731 |

The single negative row is a legitimate measurement — a model that converged
anti-correlated with the labels — and is retained.
