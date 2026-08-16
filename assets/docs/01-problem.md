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
- **It passes through zero.** 38 rows are exactly 0.0 and one is negative. This is why
  MAE rather than SMAPE is the honest error headline: see
  [chapter 4](04-evaluation.md).

## The meta-dataset

`data/meta_dataset.csv` — 476 rows, no missing values. One row per (dataset, model) pair,
covering **20 datasets × 25 models** (24 of the 500 possible pairs are absent).

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
| rows at exactly 0.0 | 38 |
| maximum | 1.0 |
| rows at exactly 1.0 | 80 |
| mean | 0.731 |

The single negative row is a legitimate measurement — a model that converged
anti-correlated with the labels — and is retained.
