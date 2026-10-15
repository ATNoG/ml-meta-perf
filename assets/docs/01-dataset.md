# 1. The dataset

*Implemented in `ml_meta_perf.data`.*

The abbreviations used in this chapter are Matthews Correlation Coefficient (**MCC**), F1
score (**F1**), Internet of Things (**IoT**), coefficient of determination (**R²**), mean
absolute error (**MAE**), symmetric mean absolute percentage error (**SMAPE**),
comma-separated values (**CSV**), machine learning (**ML**), linear discriminant analysis
(**LDA**), quadratic discriminant analysis (**QDA**), stochastic gradient descent (**SGD**),
key performance indicator (**KPI**), and key quality indicator (**KQI**).
The equation labels used below are **E1** (dataset features only) and **E3-Valid** (the
plateau-selected equation using dataset and model features).
The evaluation protocols are in-sample (**IS**), leave-one-dataset-out (**LODO**), and
leave-one-model-out (**LOMO**).

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
because the corpus includes imbalanced network and service benchmarks where accuracy can be
misleading (Chicco & Jurman, 2020). MCC is bounded in [-1, 1], where 0 is chance and
negative values indicate anti-correlation with the labels.

Three properties of this target shape every downstream decision:

- **It is bounded.** Predictions are clipped to [-1, 1]
  (`ml_meta_perf.model.Equation.predict`).
- **It can saturate.** Perfect scores occur in the observed corpus; the generated table
  below reports their frequency. Transforms with a singularity at 1 need special care.
- **It passes through zero.** Zero and negative MCC values are valid observations and are
  retained. MAE is the error headline because SMAPE divides by `|truth| + |prediction|`;
  a zero target contributes 200% unless the prediction is also zero.

## The meta-dataset

`dataset/meta_dataset.csv` — one row per (dataset, model) pair, no missing values.
Its shape, and which pairs are absent, are in the generated section below.

### Each row is the best of five seeds, not their mean

Audited against [`meta_dataset_pipeline/`](../../meta_dataset_pipeline/), especially
`exp_stage_create_meta_dataset.py`. Every observed (dataset, model) pair has five seed
records (5, 7, 11, 13, 42), and the row kept here is the one with the **highest** MCC
(`groupby(["Dataset","Model"]).MCC.idxmax()`). The revised corpus contains 484 selected
pairs and 2,420 seed evaluation records.

The target is **best-observed performance**, rather than the mean or a typical seed.
Seed selection can raise absolute performance and affect comparisons between learners;
its variability is not modelled separately by the equation. The model descriptors in
that row come from the same selected run.

| | count | constant within |
|---|---|---|
| dataset meta-features | 12 | a dataset |
| model meta-features | 6 | a model, for five of the six |
| **usable features** | **18** | |

Twenty-one CSV columns minus `Dataset`, `Model` and `MCC` leaves 18 features.

### Dataset features

Twelve, all constant across a dataset's observed rows — which is the single most consequential
property of this corpus, and [chapter 5](05-evaluation.md) is built around it. They are
standard dataset meta-features in the sense of the pymfe/OpenML vocabulary.

| feature | what it measures |
|---|---|
| `nr_inst` | number of instances in the **source** dataset, before sampling |
| `nr_attr` | number of attributes |
| `nr_class` | number of classes |
| `nr_bin` | number of binary attributes |
| `nr_norm` | number of normally distributed attributes |
| `nr_outliers` | number of attributes containing outliers |
| `nr_cor_attr` | proportion of correlated attribute pairs |
| `inst_to_attr` | instances per attribute |
| `eq_num_attr` | equivalent number of attributes — an effective feature count, discounting redundancy |
| `class_ent` | class entropy: how evenly the labels are spread |
| `gravity` | separation between the majority and minority class centres |
| `ns_ratio` | noise-to-signal ratio |

Three of them carry warnings that matter downstream.

**`nr_inst` is the source size, not necessarily the training size.** Hyperparameter tuning
uses a sample capped at 100,000 rows, while performance evaluation uses the available source
with the pipeline's dataset and model limits. It is not a controlled training-size experiment — see [Training-set size is not a variable here](#training-set-size-is-not-a-variable-here).

**The set is deliberately redundant.** Some descriptors are strongly correlated, and
`inst_to_attr` is defined as `nr_inst / nr_attr`, so

```
log(inst_to_attr) + log(nr_attr)  ==  log(nr_inst)
```

holds up to floating-point precision, and the term grammar generates both sides. That is the price of *identifying*
all twenty datasets, which the corpus has to do before any equation over these features can
tell two of them apart; `terms.Library` de-duplicates on column values rather than names so
the pair cannot both enter an equation. **Any feature defined as a ratio of two others will
do this again**, which is a rule for whoever regenerates the corpus.

**`gravity` and `ns_ratio` span several orders of magnitude**, which
is why the vocabulary is built on `log` and why [chapter 2](02-additive-model.md) declines to
rescale the raw columns up front.

### Model features

Six model descriptors are retained. Their meanings differ: `Processing Units Number`
is a configuration-dependent structural proxy, while the other five are fixed study-specific
assignments motivated by descriptions of the learners.

| feature | what it records | source |
|---|---|---|
| `Processing Units Number` | formula-based count of nodes, trees or parameters, depending on the learner | selected configuration and dataset shape |
| `Model Capability` | assigned learner-family code (1–10) | heuristic study ordering |
| `Solution Stochasticity` | assigned fitting-randomness category (1–5) | learner descriptions |
| `Loss Margin Behaviour` | assigned loss-behaviour category (1–5) | learner descriptions |
| `Input Distribution Modelling` | assigned distribution-modelling category (1–5) | learner descriptions |
| `Fitting Regime` | assigned fitting-regime category (1–5) | learner descriptions |

`Processing Units Number` is not a common-unit cost or measured FLOP count: its formula
varies by learner and can depend on dataset shape. Composite terms apply the grammar's
natural logarithm once, so `log(Processing Units Number)` is one log of that proxy.

`Model Capability` is a bijection with the ten learner families. Its numerical spacing and
order are study choices, rather than calibrated capacity or a guaranteed performance
ordering. Terms over it describe associations under that encoding. The other assigned
categories likewise do not establish equal distances between mechanisms; categorical and
code-permutation controls assess sensitivity to those choices.

The category descriptions are motivated by the algorithms: stochastic fitting by Breiman
(1996, 2001), Ho (1998), and Geurts et al. (2006); distribution modelling by Ng & Jordan
(2001) and the LDA/QDA distinction; fitting regime from closed-form fitting through
amortised in-context prediction. These references motivate distinctions, not the exact
numeric scales.

All six columns are strictly positive. `tests/test_model_features.py` checks their allowed
values and the fixed assignments within a learner. Four earlier columns were retired:
`Training Operations`, `Prediction Operations`, `Active Regularization Mechanisms`, and
`Robust to Outliers`. They are not part of the current feature set.

### How the runs were produced, and what that rules out

The upstream pipeline tunes on a stratified sample capped at 100,000 rows, then evaluates
with the available source data under its existing limits (BoTNeT at one million rows;
TabICL/TabPFN context at 10,000). Tuning and cross-validation evaluation share the source
data; performance evaluation does not use the tuning-stage 20% holdout as an independent
test set. The base pipeline applies no feature scaling.

The 20 datasets cover network and service applications: six slicing/QoS datasets and
14 traffic/security datasets. The added [6G Network Slicing Classification source](https://github.com/robertbotez/6g-network-slicing-dataset)
is synthetic. The added [OCPP 1.6 source](https://zenodo.org/records/14887131) contributes
the prepared centralized APP-layer table; it does not turn this study into federated
learning. Their source-specific preparation is documented in the supplied dataset READMEs.
All datasets use the same evaluation pipeline, models, checkpoints, and five seeds.

Eight GPU/deep learners are excluded before fitting on KPI-KQI and UNAC by the minimum
training-size policy. These 16 absent cells leave 484 observed pairs; they should not be
described as failed training runs. The 18 complete datasets supply 450 pairs for matched
family comparisons.

## Two facts that drive the whole design

**Dataset features are constant across a dataset's observed rows.** Everything follows from
this. It means a dataset-only equation can predict nothing but a per-dataset constant
(chapter 4), and it means a random train/test split puts the same dataset on both sides
of the fold rather than testing transfer to an unseen dataset (chapter 5).

**The features span several orders of magnitude.** Source sizes and geometric
descriptors are on very different scales. Composite terms are therefore built over
log-compressed operands wherever a feature is strictly positive, and the design matrix is
standardised before selection (chapter 2).

## Range of the target

The shape of the corpus and the distribution of MCC are **generated from the file
itself**, in the section below. Zero and negative MCC values are valid measurements and
are retained. An observed zero does not identify a training crash.

<!-- generated: do not edit below -->

## The corpus

What the meta-dataset is, computed from the file the run was fitted on rather than transcribed into prose beside it.

| quantity | count |
|---|---|
| rows | 484 |
| datasets | 20 |
| models | 25 |
| cells absent from the dataset-by-model grid | 16 |
| dataset features | 12 |
| model features | 6 |

**The absent cells are not missing at random.** Every dataset short of models is one of the smallest in the corpus, which is what the instance counts show:

| dataset | nr_inst | models_absent |
|---|---|---|
| KPI-KQI | 165 | 8 |
| UNAC | 389 | 8 |

And how MCC is distributed over those rows:

| quantity | MCC | rows |
|---|---|---|
| mean | 0.7535 | 484 |
| standard deviation | 0.3346 | 484 |
| minimum | -0.2898 | 1 |
| maximum | 1.0000 | 105 |
| at exactly 1 |  | 105 |
| at exactly 0 |  | 13 |
| below 0 |  | 1 |

**The table counts observations exactly at 0 or 1, and those below zero.** That is what makes MAE rather than SMAPE the reported error: SMAPE divides by `|truth| + |prediction|`, so every row at exactly zero contributes the full 200% unless the prediction is exactly zero too, and the metric ends up dominated by the rows the equation is already known to handle worst.

Two things these tables cannot say, both of which bound every number in the study. Each row is the **best of five seeds**, not their mean, so the target is optimistic and subject to seed variation that is not modelled separately; and tuning uses a sample **capped at 100,000 rows**, while later performance evaluation uses available source data with dataset/model-specific limits. `nr_inst` is the source dataset's size rather than necessarily the training set's. These are properties of the corpus builder upstream, and are audited against it in the prose above.

<!-- end generated -->

## Limitations of the corpus

### Sample size

**Twenty datasets is the binding constraint on every cross-validated number here.**

- LODO R² can vary between adjacent term counts. The curve and its uncertainty
  should be read alongside any selected cell.
- E1's cross-validated result rests on only 20 folds; its per-fold errors and dispersion must
  be read alongside the pooled R².
- Every equation's length is read off a curve whose points carry that much noise. The length
  rule therefore smooths the curve with a running median and asks for a plateau over a forward
  window of lengths, rather than reacting to one adjacent point.

Twenty-five models is more comfortable but still small for the LOMO protocol.

### Domain

All 20 datasets concern network or service applications: six slicing/QoS datasets and
14 traffic/security datasets. The 6G slicing data are synthetic; the OCPP data are a
centralized APP-layer classification table. Nothing here establishes transfer to images,
text, or a representative sample of tabular problems from other domains.

### Training-set size is not a variable here

`nr_inst` describes source size. Tuning samples, cross-validation folds, and model-specific
context limits can differ from that number. Dataset sizes vary, but the corpus does not
intervene on training size while holding the other properties fixed. A term over `nr_inst`
or `inst_to_attr` therefore describes an association with the problem's source size,
rather than the effect of supplying more training examples.

### Every prediction is conditional on training succeeding

The observed population excludes the 16 cells ruled out by the minimum-size policy:
eight GPU/deep learners on KPI-KQI and UNAC. The equation models eligible observed runs,
and cannot estimate the probability that an arbitrary new training job will complete.

Missingness also means that means over all rows compare learners on different problem
sets. Family comparisons therefore use the 18 complete datasets (450 pairs), and the
generated diagnostics report the effect of this restriction. Observed zero-MCC rows
remain valid results, distinct from the excluded cells.

### The single negative row

The negative MCC observation is retained alongside the zero and positive observations.
Removing a difficult observed result would change the target population; bounded
prediction and error reporting therefore include it.

