# 1. The dataset

*Implemented in `ml_meta_perf.data`.*

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
  [-1, 1] (`ml_meta_perf.model.Equation.predict`).
- **It saturates.** A sixth of the rows sit at exactly 1.0 — the generated table below
  counts them. A logit or `atanh` transform of the target was tried to handle the bound and
  **failed badly**: cross-validated R² went negative, because the saturation point maps to
  infinity.
- **It passes through zero.** Some rows are exactly 0.0 and one is negative, again counted
  below. This is why MAE rather than SMAPE is the honest error headline: SMAPE divides by
  `|truth| + |prediction|`, so every row at exactly zero contributes the full 200% unless
  the prediction is exactly zero too. See [chapter 5](05-evaluation.md).

## The meta-dataset

`src/ml_meta_perf/meta_dataset.csv` — one row per (dataset, model) pair, no missing values.
Its shape, and which pairs are absent, are in the generated section below.

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
leave-one-dataset-out MAE — reported in [chapter 5](05-evaluation.md), where it is
generated — sits well above the floor, so the sample rather than the target's own noise is
what binds. The floor is still where an error curve would stop, and it is worth knowing that
it sits around 0.03–0.05.

The 80 rows at exactly 1.0 are not an artefact of the selection: of the 54 saturated pairs
with published seed records, 52 average above 0.99 across all three.

| | count | constant within |
|---|---|---|
| dataset meta-features | 12 | a dataset |
| model meta-features | 6 | a model, for five of the six |
| **usable features** | **18** | |

Twenty-one CSV columns minus `Dataset`, `Model` and `MCC` leaves 18 features.

### Dataset features

Twelve, all constant across a dataset's 25 rows — which is the single most consequential
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

**`nr_inst` is the source size, not the training size.** Every model was trained on a
stratified sample capped at 100,000 rows and ten of the twenty datasets exceed that cap, so
above it `nr_inst` describes a dataset nobody trained on. **No question about the effect of
more training data is testable here** — see [chapter 1](01-dataset.md).

**The set is deliberately redundant, and two of the redundancies are exact.** `nr_attr` and
`nr_outliers` correlate at 0.9995. And `inst_to_attr` is defined as `nr_inst / nr_attr`, so

```
log(inst_to_attr) + log(nr_attr)  ==  log(nr_inst)
```

holds to 2e-15, and the term grammar generates both sides. That is the price of *identifying*
all twenty datasets, which the corpus has to do before any equation over these features can
tell two of them apart; `terms.Library` de-duplicates on column values rather than names so
the pair cannot both enter an equation. **Any feature defined as a ratio of two others will
do this again**, which is a rule for whoever regenerates the corpus.

**`gravity` and `ns_ratio` span enormous ranges** — `gravity` runs from 1.6 to 1e16 — which
is why the vocabulary is built on `log` and why [chapter 2](02-additive-model.md) declines to
rescale the raw columns up front.

### Model features

Six, of which one is measured and five are asserted, and the split matters for how a term
over them may be read.

| feature | what it measures | source |
|---|---|---|
| `Processing Units Number` | log count of the units a learner fits — nodes, trees, parameters | measured per run |
| `Model Capability` | the learner family's rung on a ten-step capability ladder | asserted, from the literature |
| `Solution Stochasticity` | how much randomness the fitting procedure introduces | asserted, 1–5 |
| `Loss Margin Behaviour` | how hard the loss penalises points far from the boundary | asserted, 1–5 |
| `Input Distribution Modelling` | how much of the input distribution the learner models | asserted, 1–5 |
| `Fitting Regime` | closed form through to amortised, in-context | asserted, 1–5 |

**Only one of the six is a measurement, and none of them describes what a model is *good
at*.** `Processing Units Number` is a cost proxy and the other five are taxonomy. That is the
study's central limitation rather than an incidental one: [chapter 1](01-dataset.md)
measures the headroom a perfect model descriptor would buy, and records the five separate
attempts to recover it by re-encoding what the corpus already has, all of which failed.

`Processing Units Number` is the measured one. It is a count of the units a learner fits —
nodes, trees, parameters — and the corpus stores its natural log. That log is a modelling
claim rather than a formatting choice: capacity has bounded returns, so twice the units is
not twice the accuracy. Note that a term printed as `log(Processing Units Number)` therefore
applies a *second* log. It is also the one model feature that varies with the dataset as well
as the learner, since capacity scales with the data's shape — the model features are not
purely model-level.

`Model Capability` places a learner's family on a ten-rung ladder taken from the tabular-ML
literature. Chapter 7 records what it is worth and what it is not; the short version is that
it is a bijection with the ten families, so anything read off it is a claim about *family*.

The remaining four grade a mechanism, low to high, and are asserted from published
descriptions of the learners rather than measured from the runs:

| feature | 1 | 5 | grounding |
|---|---|---|---|
| `Solution Stochasticity` | deterministic | randomised splits | Breiman (1996, 2001), Ho (1998), Geurts et al. (2006) |
| `Loss Margin Behaviour` | squared / impurity | perceptron criterion | the standard robustness ordering over losses |
| `Input Distribution Modelling` | discriminative | generative, per-class covariance | Ng & Jordan (2001); the LDA/QDA hierarchy |
| `Fitting Regime` | closed form | amortised, in-context | — |

All six are strictly positive with every rung occupied, so the whole term grammar — `log`,
`sqrt`, `1/f`, `f^2` — is defined on all of them. `tests/test_model_features.py` pins that,
along with each ladder being gapless and constant within a model.

**Four earlier model columns were retired**: `Training Operations`, `Prediction Operations`,
`Active Regularization Mechanisms` and `Robust to Outliers`. Removing any of them improves
leave-one-model-out; three varied *within* a model, which made them partly dataset features
wearing a model feature's name; and two were zero-based, so no log, root or reciprocal was
defined on them at all.

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
succeeding ([chapter 1](01-dataset.md)).

## Two facts that drive the whole design

**Dataset features are constant across a dataset's 25 rows.** Everything follows from
this. It means a dataset-only equation can predict nothing but a per-dataset constant
(chapter 4), and it means a random train/test split puts the same dataset on both sides
of the fold and inflates every score (chapter 5).

**The features span wildly different magnitudes.** `gravity` runs from 1.6 to 1.0e16;
`nr_inst` from 165 to 7.1 million. Composite terms are therefore built over
log-compressed operands wherever a feature is strictly positive, and the design matrix is
standardised before selection (chapter 2).

## Range of the target

The shape of the corpus and the distribution of MCC across it are **generated from the file
itself**, in the section below, rather than transcribed into this prose. The single negative
row is a legitimate measurement — a model that converged anti-correlated with the labels —
and is retained; the rows at exactly 0.0 are not training failures, since those were
discarded when the corpus was built, but classifiers that converged and learned nothing.

<!-- generated: do not edit below -->

## The corpus

What the meta-dataset is, computed from the file the run was fitted on rather than transcribed into prose beside it.

| quantity | count |
|---|---|
| rows | 476 |
| datasets | 20 |
| models | 25 |
| cells absent of datasets x models | 24 |
| dataset features | 12 |
| model features | 6 |

**The absent cells are not missing at random.** Every dataset short of models is one of the smallest in the corpus, which is what the instance counts show:

| dataset | nr_inst | models_absent |
|---|---|---|
| KPI-KQI | 165 | 8 |
| UNAC | 389 | 8 |
| Social Network Ads | 400 | 8 |

And how MCC is distributed over those rows:

| quantity | MCC | rows |
|---|---|---|
| mean | 0.7305 | 476 |
| standard deviation | 0.3432 | 476 |
| minimum | -0.2898 | 1 |
| maximum | 1.0000 | 1 |
| at exactly 1 |  | 80 |
| at exactly 0 |  | 15 |
| below 0 |  | 1 |

**A third of the corpus is pinned at one end of the range or the other.** That is what makes MAE rather than SMAPE the reported error: SMAPE divides by `|truth| + |prediction|`, so every row at exactly zero contributes the full 200% unless the prediction is exactly zero too, and the metric ends up dominated by the rows the equation is already known to handle worst.

Two things these tables cannot say, both of which bound every number in the study. Each row is the **best of three seeds**, not their mean, so the target is optimistic and has a noise floor no predictor can go below; and every model was trained on a stratified sample **capped at 100,000 rows**, so `nr_inst` is the source dataset's size rather than the training set's. Both are properties of the corpus builder upstream, and are audited against it in the prose above.

<!-- end generated -->

## Limitations of the corpus

### Sample size

**Twenty datasets is the binding constraint on every cross-validated number here.**

- Leave-one-dataset-out R² varies by ±0.07 between adjacent term counts from fold noise
  alone. The curve should be read, never a single cell.
- E1's cross-validated numbers rest on 20 folds and are correspondingly unstable — its
  leave-one-dataset-out R² is negative at 1–4 terms and 0.217 at 7.
- The knee detector, the Pareto front and the "best cross-validated" rule all operate on a
  curve whose points carry that much noise. That they agree on 12–14 terms is reassuring,
  not conclusive.

Twenty-five models is more comfortable but still small for the leave-one-model-out
protocol.

### Domain

All twenty datasets are networking, IoT and security tabular benchmarks. Nothing here
should be assumed to transfer to images, text, or tabular data from other domains. The
extracted practices in particular are statements about this corpus.

### Training-set size is not a variable here

**Every model was trained on a stratified sample capped at 100,000 rows.** Ten of the
twenty datasets exceed that cap, so above it every training set is the same size and
`nr_inst` records the *source* dataset rather than what the model saw.

This is a hard boundary on what the study can be asked. Anything of the form "does X change
as the training set grows" is untestable here, and a split of the corpus by `nr_inst` is a
split by source size, which is not the same variable. A practice of the form *neural
architectures catch up on larger datasets* is therefore outside what this corpus can weigh,
and [chapter 6](06-practices.md) does not carry one — because the corpus cannot speak to it,
not because the answer came out one way or the other.

`nr_inst` and `inst_to_attr` remain legitimate meta-features — they are knowable before
training and they describe the problem a practitioner is facing — but no term over them
should be read as a statement about sample size.

### Every prediction is conditional on training succeeding

**Runs that failed to train were discarded when the meta-dataset was built.** That was a
deliberate choice — a crashed run has no MCC to record — but it means the corpus is a
sample of *completed* runs, and the equation is fitted on and can only speak about that
population.

Twenty-four of the 500 (dataset, model) cells are absent, and not at random: eight models
are missing from the same three datasets. So the exclusions are concentrated exactly where
one would expect a model to have struggled, which is the classic shape of selection bias.

Two consequences, and they point in different directions:

- **For the model comparison, the bias is measurable and small.** Restricting to the 17
  datasets where every model ran moves the model ranking by Spearman 0.975 and no mean by
  more than 0.07. Every family-level figure in [chapter 6](06-practices.md) is computed on
  that complete subset for this reason rather than on all rows.
- **For the go/no-go rule, the bias is not correctable.** A rule trained only on runs that
  completed answers "will this trained model be any good", not "should I try this at all".
  It has never seen a failure and cannot warn about one.

The 15 rows at exactly MCC = 0 are *not* the failures. They are classifiers that converged
and learned nothing useful — the majority-class predictor and its relatives. Predictions
never fall below 0.17, so those rows sit above the diagonal, but that is shrinkage toward
the middle of the observed range rather than an inability to recognise a failure mode the
data does not contain.

### The single negative row

One row (NSL-KDD / SGD, MCC = -0.29) is a legitimate anti-correlated result and is
retained. It is an outlier in a target that is otherwise non-negative, and it falls outside
the scatter's axes. Removing it was considered and rejected: dropping the worst observed
result biases the target upward.
