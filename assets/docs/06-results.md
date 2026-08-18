# 6. Results

*Produced by `ml_meta_perf.experiment`; reproduce with `PYTHONPATH=src venv/bin/python -m ml_meta_perf`.*

## The three equations

The study is built around a deliberate contrast. The three equations differ **only** in
which features they may draw on, so the gaps between them measure what each half of the
meta-data is worth.

**E1 — dataset features only.** Every model evaluated on a given dataset shares one
feature vector, so many MCC values map to a single input and least squares necessarily
lands on the per-dataset mean. That is not a defect to be corrected; it is the control. E1
measures how much of MCC is explained by *the data alone*.

**E2 — model features only.** The mirror of E1, added to settle whether model choice
outweighs dataset difficulty. There are only five model features and one is constant per
model, so E2 is short by necessity rather than by design.

**E3 — dataset and model features.** One input, one output. E3 can express variation
*within* a dataset, which is exactly what E1 structurally cannot do, and it is the
equation the study publishes.

### One process, three feature sets

All three are fitted by the same function on the same 476 rows, scored on the same 476
rows, under the same two protocols, with penalty and length chosen by the same rule.
`experiment.run_equation` is that function and `run_e1`, `run_e2` and `run_e3` are one
line each. The uniformity is not tidiness: the gaps between the three are only evidence
about what each half of the meta-data is worth if *nothing else* differs between them.

**An earlier version fitted E1 on the 20 aggregated per-dataset means.** The argument was
that a predictor constant inside a group can only predict that group's mean anyway, so
aggregating merely made the structure explicit. Both halves of that are true and the
choice was still wrong, for a reason that has nothing to do with the weights:

| E1 fitted on | in-sample R² (476 rows) | LOO-dataset R² | comparable with E3? |
|---|---|---|---|
| 20 dataset means | 0.337 | **0.506** *(on 20 points)* | no |
| **476 rows** | **0.349** | **0.217** *(on 476 rows)* | yes |

The 0.506 was computed against the variance of twenty numbers and the 0.466 it sat beside
was computed against the variance of 476. Printed in one table they invite the conclusion
that the dataset-only control transfers *better* than the full equation. On the common
scale it transfers half as well. Fitting on all rows costs nothing in fit — it gains 0.012
— and retires the caveat rather than restating it.

**Aggregating E2 the same way was measured, and is worse.** The mirror move is to fit E2
on 25 per-model means:

| E2 fitted on | in-sample R² (476 rows) | best out-of-fold | terms at the optimum |
|---|---|---|---|
| **476 rows** | **0.177** | **0.090** | 6 |
| 25 model means | 0.125 | 0.112 | **1** |

<sub>Both rows predate `Model Capability`; the comparison is between fitting scales, and
re-running it on the enlarged feature set would change both numbers without changing which
is larger.</sub>

Its optimum collapses to a single term and everything longer goes negative. The reason is
structural: three of the five model features are operation counts, which are functions of
the dataset size as well as the model, so they are **not** constant within a model.
Averaging them discards real variation, which is exactly what aggregating E1 does not do.
The two controls are not symmetric in the data even though they are symmetric in intent —
which is itself the argument for fitting both the same way and letting the optimizer
handle the difference.

## Headline comparison

Every equation is fitted on all 476 rows and scored on all 476 rows, so there is one
scale and one table:

| | terms | R² | MAE | Spearman |
|---|---|---|---|---|
| E1 (dataset only) | 7 | 0.349 | 0.210 | 0.657 |
| *E1's ceiling — the true dataset means* | | *0.354* | *0.204* | *0.653* |
| E2 (model only) | 12 | 0.281 | 0.227 | 0.521 |
| *E2's ceiling — the true model means* | | *0.282* | *0.226* | *0.487* |
| **E3 (dataset + model)** | **20** | **0.614** | **0.154** | **0.804** |
| *additive oracle* | | *0.6605* | *0.145* | *0.810* |

![Equations against their ceilings](../figures/equation_comparison.png)

### One scale is not one ceiling

Those R² values are now arithmetically comparable — same rows, same denominator. They are
still **not comparable as achievements**, and no change to the fitting could make them so.

E1 predicts one value per dataset. **0.354 is the most it could ever score**, however good
its terms were, because that is all the variance a per-dataset constant can reach. E1 at
0.349 is not "worse than E3 at 0.600"; it is at its own limit while E3 is not at its. The
same applies to E2 against 0.282.

So the comparable quantity is the *fraction of its own ceiling* each equation reaches, and
that is the column the next section reads.

## Under both protocols

| | terms | in-sample R² | LOO-dataset R² | LOO-model R² |
|---|---|---|---|---|
| E1 | 7 | 0.349 | 0.217 | 0.294 |
| E2 | 12 | 0.281 | 0.151 | 0.188 |
| **E3** | **20** | **0.614** | **0.478** | **0.428** |

Both controls are now reported under both protocols, which the aggregated E1 could not be.
The pattern is the one the design predicts and is worth checking rather than assuming: E1
transfers *better* across models (0.294) than across datasets (0.217), because it predicts
a per-dataset constant and a new model does not change it; E2 is the mirror.

**E3 transfers better across datasets than across models, and that is new.** Until
`Model Capability` was added the two were close (0.466 and 0.489). The column is constant
within a learner family and asserted by hand, so holding out a whole model removes a value
the equation cannot recompute — 0.489 falls to 0.428 while leave-one-dataset-out rises.
That asymmetry is a property of the feature, not of the search, and
[chapter 8](08-limitations.md) treats it as the cost it is.

E1 is unstable at short lengths — its leave-one-dataset-out R² is **negative at 1 to 4
terms** and only turns positive at 5. Twenty datasets is a small sample for a curve, and
this is what that looks like.

## Does model choice matter more than the dataset?

Not on this meta-dataset — but the reason is more interesting than the answer.

Dataset identity explains **0.354** of MCC variance against model identity's **0.282**, and
the equations widen the gap rather than closing it:

| | equation | own ceiling | captured |
|---|---|---|---|
| dataset features | 0.349 | 0.354 — the true dataset means | **98%** |
| model features, as the corpus ships them | 0.177 | 0.282 — the true model means | **63%** |
| model features **+ `Model Capability`** | 0.281 | 0.282 — the true model means | **99%** |
| both | 0.614 | 0.6605 — the additive oracle* | *93%* |

<sub>*The additive oracle bounds a two-way *additive* form. E3's mixed terms can cross it
given enough length ([chapter 5](05-oracles.md)), so its 91% is an indication rather than a
bound — unlike the two rows above it, which are hard ceilings.</sub>

**As the corpus ships, its meta-features describe datasets far better than models.**
Twelve dataset meta-features all but exhaust what dataset identity can explain — 98% of it,
so there is essentially nothing left for a better dataset descriptor to find. The five model
descriptors capture under two thirds of what model identity can.

That asymmetry is the study's central measurement, and the third row is what it took to
remove it. **One ten-level ordinal, asserted from the tabular-ML literature and absent from
the corpus, moves the model side from 63% to 99%.** The conclusion is therefore sharper
than "this meta-dataset lacks model features". It lacks them, the missing information is
worth roughly a third of what model identity explains, and *most of it is recoverable from
knowing only which of ten families a learner belongs to* — a fact about how coarse the
missing signal is, and an unusually cheap remedy.

Three qualifications keep that from being oversold, all of them developed in
[chapter 8](08-limitations.md): the ladder is asserted rather than measured, so it cannot be
evidence for the prior knowledge it encodes; the corpus agrees with it at only 6 of 9 steps,
with `generic NN` conspicuously misplaced; and it does not extend to a learner nobody has
classified, which is what the fall in leave-one-model-out records.

Collecting more meta-features of the *data* remains wasted effort — the equation is already
within 0.005 of what perfect knowledge of dataset identity would buy.

Two things create the opposite impression and are worth stating explicitly, because both
are artefacts:

- The **within-group correlation table is dataset-centred**, which removes dataset variance
  by construction. Only model terms can score well there. It is a diagnostic for model
  effects, not a statement of relative importance.
- **Model features carry more in combination than alone.** Adding them to E1 is worth
  +0.265 R² (0.349 → 0.614), beyond the 0.281 they achieve by themselves. The surplus is
  dataset×model interaction, which is why **11 of E3's 20 terms are mixed** and drive 59%
  of its output variance.

![Contribution shares](../figures/contribution_shares.png)

## One configuration, not two

Earlier drafts reported a second, accuracy-leaning configuration alongside the default:
arity 4 over a 4610-term library, 32 terms, reaching 0.668 in-sample. **It has been
dropped.** It existed to answer "how much fit is available if transfer is sacrificed", and
that question stopped being interesting once the default reached 0.6 — the extra fit cost
0.18 of leave-one-dataset-out R², which no reader of this study should want,
and reporting two headline equations invites quoting whichever suits the argument.

The arity trade it measured is still recorded, in the place it belongs:
[chapter 2](02-equation-form.md) sweeps the penalty and the length inside each arity and
reports the result as a design decision rather than as a second result.

`DEFAULT_E3` is the study:

| | `max_arity` | `max_abs_zscore` | `penalty` | headline terms |
|---|---|---|---|---|
| `DEFAULT_E3` | 3 (316-term library) | 3.0 | 20 | 20 |

| | terms | in-sample R² | LOO-dataset | LOO-model |
|---|---|---|---|---|
| `DEFAULT_E3` | 20 | **0.6141** | **0.4779** | 0.4276 |

It clears 0.6 *and* holds the best leave-one-dataset-out figure in the study — not the trade
the earlier configurations offered. Two changes got it there and neither was extra search:
the grammar fix, where symmetric `sum_ratio` gave the search 126 mixed dataset×model terms
in place of 19, and the sixth model descriptor.

**Why 20 terms and not 14, or 24?** Each was the optimum of a configuration that stopped
being current. Fourteen belonged to the arity-2 grammar; twenty-four to the arity-3 grammar
with five model descriptors. Adding a sixth descriptor moved it again:

| | in-sample | LOO-dataset | LOO-model |
|---|---|---|---|
| arity 2 implicit, 14 terms, λ=20 | 0.5582 | 0.4429 | 0.4561 |
| arity 3, 24 terms, λ=5, five model descriptors | 0.5998 | 0.4658 | 0.4887 |
| **arity 3, 20 terms, λ=20, six** | **0.6141** | **0.4779** | 0.4276 |

**The penalty moved with the length and that is not incidental.** λ is the ridge penalty on
a centred, standardised design whose Gram diagonal is exactly *n* = 476, so λ=20 is a 4%
shrinkage on an isolated weight — negligible as shrinkage. It matters because it also sits
in the subset score, `RSS − λ·wᵀw`, so it changes *which* terms the beam selects: half the
twenty-four-term equation's terms differ between λ=5 and λ=20. Judging an enlarged library
at the incumbent penalty would have credited the ridge with the new column's effect, or
blamed it for its cost.

The length and penalty were re-swept rather than carried over, which is the general lesson:
**a term budget tuned against one feature set is not evidence about another.**

## The fitted equations

E1, on all 476 rows:

```
MCC = +1.24602
      -0.109215   * [log(eq_num_attr)] * [log(nr_class)]
      -0.0284191  * ([log(gravity)] + [log(nr_class)]) / [log(nr_attr)]
      -0.415152   * ([log(eq_num_attr)] + [log(ns_ratio)]) / [log(nr_inst)]
      -0.128423   * [nr_cor_attr] * [nr_norm]
      -0.00254076 * ([log(gravity)] + [nr_norm]) / [log(nr_class)]
      +0.0272363  * ([nr_norm] + [log(ns_ratio)]) / [log(nr_attr)]
      +0.0057269  * ([nr_bin] + [nr_norm]) / [log(nr_attr)]
```

E2, over the five model features:

```
MCC = -0.00257295
      +0.081011   * [log(Processing Units Number)] * [Robust to Outliers]
      -0.0842406  * [log(Processing Units Number)] * [log(Prediction Operations)]
      +0.197874   * sqrt(Prediction Operations)
      +0.707659   * [log(Processing Units Number)] / [log(Training Operations)]
      -0.251995   * ([log(Processing Units Number)] + [Robust to Outliers]) / [log(Training Operations)]
      +0.0101446  * Training Operations
```

### How much of each control survives into E3

Counting terms the published equations share *exactly*:

| | shared with E3 | which |
|---|---|---|
| E2 → E3 | **4 of 6** | `sqrt(Prediction Operations)`, `Training Operations`, `log(PU)*[Robust to Outliers]`, `log(PU)*log(Prediction Operations)` |
| E1 → E3 | **1 of 7** | `[nr_cor_attr] * [nr_norm]` |

E3 adopts two thirds of E2's model-side vocabulary unchanged and rebuilds the dataset side
almost from scratch. That is not a defect in E1: E3 does not need dataset-only terms to
carry dataset information, because it carries it through the 12 mixed terms instead. E1's
`([log(gravity)] + [log(nr_class)]) / [log(nr_attr)]` reappears in E3 as
`([log(gravity)] + [nr_norm]) / [log(nr_attr)]` — same shape, same denominator, re-split
once a model feature is available to pair with.

So **the model side of the study is stable across equations and the dataset side is not**,
which is the inverse of the accuracy story: the dataset features are the ones that nearly
exhaust their ceiling, and they are the ones whose exact terms do not survive. Both facts
have the same cause — twelve dataset features admit many near-equivalent ways to say the
same thing, five model features admit few.

E3, on all 476 rows, is 20 terms and is **not reproduced here**. It is printed in full,
with its term-importance table and its analysis, in [chapter 10](10-report.md) — which is
regenerated with the equation on every run, so it cannot drift out of step with the code
the way a copy in this chapter would. An earlier draft of this chapter carried a 14-term
E3 that had stopped being the published equation several configurations earlier, which is
why the listing now lives on the generated side.

The shape of it, from that chapter: 11 of the 20 terms mix dataset and model features and
drive **59%** of the output variance; 4 are model-only (16%) and 5 are dataset-only (25%).
The weights are flat — they behave like **16.7 equally-weighted terms**, and the largest
carries under 10% of the mass.

![What each term is worth](../figures/term_effects.png)

## Where the equation is weakest

![Predicted versus actual MCC](../figures/predicted_vs_actual.png)

Predictions never fall below **0.17**, while 15 rows sit at exactly MCC = 0 — a short
column of points hanging above the diagonal on the left. The equation compresses toward
the middle of the range, as a shrunk linear fit will.

**This is not a claim that E3 cannot predict training failure, because the meta-dataset
contains no training failures.** Runs that failed to train were discarded when the corpus
was built, so every row is a model that trained and then scored. A row at MCC = 0 is a
classifier that converged and learned nothing useful — predicting the majority class, say
— not one that crashed. The equation is fitted on, and can only speak about, the
population of runs that completed.

What that costs is stated in [chapter 8](08-limitations.md): every prediction is implicitly
conditional on the training succeeding, and the study never measures how often that is
true. It is the main reason the go/no-go rule in [chapter 10](10-report.md) should be read
as "will this trained model be any good" rather than "should I try this at all".

The axes start at 0; the single negative row falls outside them and
`plots.count_below_floor()` returns the count for a caption.

## What was tried to push past 0.558, and failed

> **These were measured against the previous grammar**, whose default reached 0.5582 —
> before `sum_ratio` was made symmetric and the term budget re-tuned. They are reported
> unchanged rather than silently rebased, because none of them was re-run afterwards. What
> did beat 0.5582 was the grammar fix, not any of the six.

That earlier default was attacked from six directions. All keep the equation form
`MCC = Σ wᵢtᵢ` intact, and none beats it:

| attempt | in-sample R² |
|---|---|
| **baseline (ordinary least squares, uniform weights)** | **0.5582** |
| downweight the rows at MCC ∈ {0, 1} by 0.5 | 0.5380 |
| downweight them by 0.25 | 0.4147 |
| Huber IRLS, 8 iterations | 0.5526 |
| equal weight per dataset | 0.5509 |
| two-stage: 7 dataset terms, then 7 on the residual | 0.5532 |
| adding `f^3`, `1/sqrt(f)`, `f^0.25` to the vocabulary | 0.5582 (unchanged) |

Widening the operator set deserves its own note, since `f^2` and `sqrt(f)` are **already**
in the vocabulary. Adding `f^3`, `1/sqrt(f)` and `f^0.25` grows the library from 172 terms
to 182 — most of the 51 new candidates fail admissibility — and the beam then selects
**none of the ten that survive**. In-sample R² is identical to four decimal places at both
8, 14 and 20 terms. Under a looser arity-4 grammar the same extension is actively harmful,
dropping 20-term R² from 0.647 to 0.632. Higher powers are
near-duplicates of the ones already present, and the collinearity guard treats them as
such.

Reweighting was the most promising idea and is the clearest failure: R² is reported on all
476 rows with uniform weight, so any reweighting optimises a *different* objective and
necessarily scores worse on the one being reported. Downweighting the saturated rows in
particular removes 118 of 476 observations' worth of influence — the pile-ups at 0 and 1
are a third of the data, not outliers to be discounted.

Together with the earlier negatives — search strength ([chapter 3](03-search-and-fitting.md)),
transform vocabulary and feature scaling ([chapter 2](02-equation-form.md)), agglomerative
construction, and marginal-impact filtering — the additive form at this configuration is
exhausted.

**One thing does work and neither keeps the equation form intact.** The interaction the
equation cannot reach is worth **+0.122** on its own ([chapter 5](05-oracles.md)), and the
part of model capability the model features still miss is worth **+0.106** of
leave-one-dataset-out R² if it is tabulated per model rather than described
([chapter 9](09-model-effects.md)). The first is unreachable from these features; the
second replaces terms with a lookup table and is reported as a ceiling rather than as a
result. Both are measurements about where the headroom is, not attempts on it. Loosening the library and the
shrinkage also raises in-sample R², but only by giving up transfer at roughly six to one,
which is why that configuration is no longer reported as a result
([chapter 2](02-equation-form.md) measures the trade).

For that last gap the literature points at **GA2M / Explainable Boosting Machines** —
generalised additive models with explicit pairwise interaction terms (Lou et al., 2013;
GAMI-Net, arXiv:2003.07132) — which are exactly the model class that adds interaction while
staying inspectable. Billa et al. (arXiv:2601.00428) find EBMs and symbolic regression
dominate interpretable tabular regression. The trade is real: an EBM is a set of shape
functions rather than a closed-form equation, so it can be plotted but not written down.

## Flexible models do worse, not better

Standard regressors on the same raw features, under the same protocols:

| model | in-sample R² | LOO-dataset R² | LOO-model R² |
|---|---|---|---|
| RidgeCV (linear, 17 features) | 0.418 | **-2.002** | 0.328 |
| RandomForest (300 trees) | **0.910** | **0.067** | 0.465 |
| GradientBoosting | 0.820 | 0.049 | 0.354 |
| **ml-meta-perf E3 (20 terms)** | 0.614 | **0.478** | 0.428 |

Read the RandomForest row across. With 20 dataset groups a forest memorises dataset
identity almost perfectly and then transfers worse than a 20-term additive equation. This is also
the likely provenance of the R² ≈ 0.9 figures reported for opaque meta-models: an
in-sample or randomly-split forest reproduces them exactly, and the same forest is
near-useless on an unseen dataset.

> Measured with scikit-learn during exploration. It is not a dependency of the package.
