# 4. Evaluation methodology

*Implemented in `metafit.validate` and `metafit.stats`.*

## Protocols

| protocol | fitted on | scored on | question answered |
|---|---|---|---|
| **in-sample** | all 476 rows | the same 476 rows | how well does the equation *describe* the data |
| **leave-one-dataset-out** | 19 datasets | the held-out 20th, rotated | what will these models score on a dataset nobody has run |
| **leave-one-model-out** | 24 models | the held-out 25th, rotated | what will a new model score on datasets we know |

In-sample is reported as a first-class result rather than dismissed. Term count is capped
and terms are drawn from a screened pool, so this is **equation fitting, not model
fitting**: the capacity to memorise 476 rows with 20 terms is limited, and the gap between
in-sample and cross-validated columns is itself the diagnostic. For contrast, a
RandomForest reaches 0.910 in-sample and 0.067 leave-one-dataset-out on the same features.

Leave-one-dataset-out is the harder and more useful number, and is the one the study
leads with for transfer claims.

## Term selection happens inside the fold

Screening the library against the full target and then cross-validating only the weights
is a standard way to leak the held-out fold into the model. In `cross_validate_path`, the
standardiser, the screening pool, the beam search and the weights are all recomputed from
the training rows of each fold. Only the term *vocabulary* — which is a function of the
feature columns, not the target — is shared.

Sharing it is not a concession, it is required. The vocabulary is target-free, so building
it from every row leaks nothing; but rebuilding it per fold would make a term that is
*admissible* on the training rows — one whose denominator stays away from zero there —
undefined on the held-out rows, and the fold would score `NaN` rather than score badly.
Admissibility is a property of the feature columns and has to be decided once, globally.

## Random k-fold is a leak, not a protocol

Dataset meta-features are **constant across a dataset's 25 rows**. A random split
therefore puts the same dataset on both sides of the fold, and the equation recognises the
dataset rather than generalising to it.

The **same equation**, three protocols:

| protocol | R² | MAE |
|---|---|---|
| random 10-fold | **0.561** | 0.164 |
| leave-one-dataset-out | **0.478** | 0.179 |
| leave-one-model-out | **0.428** | 0.188 |

A reported R² near 0.5 on this kind of meta-data may be describing the split rather than
the model. `metafit.validate.random_kfold_groups` exists **only** to produce this
comparison; it is never used to score a result.

This is the same concern as subject-wise splitting in clinical machine learning, and the
same recommendation made by community reporting standards (Walsh et al., 2020).

## Metrics

| metric | units | why |
|---|---|---|
| **R²** | — | variance explained; the comparable headline |
| **MAE** | MCC | the honest error headline — what a practitioner actually asks |
| RMSE | MCC | outlier-sensitive companion to MAE |
| SMAPE | % | scale-free, but see the caveat below |
| Spearman | — | rank quality, insensitive to the MCC ceiling |

### A caveat on R²

R² is computed against the mean of the **evaluated** rows, so its denominator is the
variance of whatever is being scored. Two R² values computed over different row sets share
no denominator and **cannot be compared**.

This used to bite here. E1 was fitted and scored on the 20 aggregated per-dataset means,
which put its 0.506 on a twenty-point denominator beside E3's 0.478 on a 476-row one —
inviting exactly the comparison the caveat forbids, and in the direction that flatters the
control. All three equations are now fitted and scored on the same 476 rows
([chapter 6](06-results.md)), so the caveat is a general warning rather than a live hazard
in this study's own tables. E1's transfer figure on the common scale is 0.217.

### A caveat on SMAPE

SMAPE divides by $|y| + |\hat{y}|$, and **15 of the 476 rows have MCC exactly 0**. Each
contributes the full 200% unless the prediction is also exactly 0, so the metric is
dominated by the rows the equation is already known to handle worst rather than by its
typical error. It is reported because it is scale-free and was requested, but MAE is the
honest headline for a target that legitimately passes through zero.

![Mean absolute error against equation length](../figures/error_curve_mae.png)

## Ranking quality

For model *selection* the relevant questions are rank correlation within a held-out
dataset and **top-1 regret** — how much MCC is given up by picking the model the equation
ranks first.

![Per-dataset ranking quality](../figures/per_group_quality.png)

The mean (0.706) hides a wide spread across the twenty held-out datasets.

## Baselines

An equation earns its place only by beating the obvious alternatives:

| baseline | R² | MAE |
|---|---|---|
| global mean (loo-dataset) | -0.040 | 0.293 |
| per-model mean (loo-dataset) | 0.201 | 0.238 |
| global mean (loo-model) | -0.024 | 0.290 |
| per-dataset mean (loo-model) | 0.296 | 0.213 |

E3 beats all four on its respective protocol.

**The ranking caveat that stood through earlier drafts has closed.** For *ranking* models
on a new dataset, the trivial "average MCC of this model elsewhere" baseline used to beat
E3 on both measures. With `Model Capability` in the model side it no longer does:

| | mean per-dataset Spearman | mean top-1 regret |
|---|---|---|
| per-model mean (leave-one-dataset-out) | 0.703 | 0.011 |
| **E3** | **0.706** | **0.008** |

The margin on rank correlation is 0.003 and should be read as a tie rather than a win; the
regret figure, at less than half the baseline's, is the more meaningful of the two. What
changed is not the search but the feature: a column that says which family a model belongs
to is exactly what a per-model mean was standing in for.

E3 also wins clearly on predicting the MCC *value* — 0.478 against the baseline's 0.201,
and that was always the larger gap. Knowing which models are generally good is most of what
*ranking* needs, which is why the baseline was so hard to beat there; knowing *how well* a
particular model will do on a particular dataset is what needs the meta-features, and the
baseline has nothing to say about it. The practical reading is unchanged for anyone asking
"will this reach MCC 0.8 on my data" and improved for anyone asking "which of these 25
models should I run".

## Term stability

A fitted equation presents a term chosen in 19 of 20 folds and one chosen in 3
identically. `CrossValidation.stability()` counts selection frequency across folds, and no
extracted practice ([chapter 7](07-practices.md)) is published from a term below 50%.
