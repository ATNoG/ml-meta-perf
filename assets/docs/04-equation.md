# 4. The equation

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
outweighs dataset difficulty. There are six model features, five of them constant per model,
so E2 is short by necessity rather than by design.

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
structural: `Processing Units Number` is a function of the dataset's shape as well as the
model, so it is **not** constant within a model.
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
| E2 (model only) | 6 | 0.248 | 0.234 | 0.459 |
| *E2's ceiling — the true model means* | | *0.282* | *0.226* | *0.487* |
| **E3 (dataset + model)** | **15** | **0.658** | **0.137** | **0.819** |
| *additive oracle* | | *0.6605* | *0.145* | *0.810* |
| *E3 under the full grammar (arity 3)* | *23* | *0.707* | *0.126* | *0.849* |

![Equations against their ceilings](../figures/equation_comparison.png)

### One scale is not one ceiling

Those R² values are now arithmetically comparable — same rows, same denominator. They are
still **not comparable as achievements**, and no change to the fitting could make them so.

E1 predicts one value per dataset. **0.354 is the most it could ever score**, however good
its terms were, because that is all the variance a per-dataset constant can reach. E1 at
0.349 is not "worse than E3 at 0.658"; it is at its own limit while E3 is not at its. The
same applies to E2 against 0.282.

So the comparable quantity is the *fraction of its own ceiling* each equation reaches, and
that is the column the next section reads.

## Under both protocols

| | terms | in-sample R² | LOO-dataset R² | LOO-model R² |
|---|---|---|---|---|
| E1 | 7 | 0.349 | 0.341 | 0.306 |
| E2 | 6 | 0.248 | 0.185 | 0.228 |
| **E3** | **15** | **0.658** | **0.638** | **0.622** |

Both controls are now reported under both protocols, which the aggregated E1 could not be.
The pattern is the one the design predicts and is worth checking rather than assuming: E1
transfers *better* across models (0.294) than across datasets (0.217), because it predicts
a per-dataset constant and a new model does not change it; E2 is the mirror.

**E3's two transfer numbers are now close** — 0.638 and 0.622, against a fit of 0.658. An
equation that loses under 0.04 R² when a whole dataset or a whole learner is withheld is
transferring, not memorising, and the gap between the two protocols is small enough that
neither half of the meta-data is carrying the equation alone.

Five of the six model features are constant within a learner, so holding out a model removes
values the equation cannot recompute from the remaining rows. That it costs only 0.019 is the
measurement; [chapter 7](07-limitations.md) treats the asserted ladders as the cost they are.

## How much was there to explain? Three ceilings

*Implemented in `ml_meta_perf.validate.additive_oracle`, `interaction_oracle` and
`oracle_ladder`, and in `ml_meta_perf.analysis.grammar_ceiling`.*

An R² means nothing on its own. Before the equation can be judged, three questions have to
be answered about what was available to explain in the first place — and all three can be
answered without fitting anything the study reports.

An **oracle** here is not a model. It is fitted with the true target values, including
those of the rows it is scored on, uses no features, and can predict nothing about a
dataset it has not seen. It answers a structural question: *if a predictor of this shape
knew everything it could possibly know, how well would it do?*

### Ceiling 1 — what group identity alone can explain

$$\hat{y}_{dm} = \bar{y} + \underbrace{(\bar{y}_{d\cdot} - \bar{y})}_{\text{dataset effect}} + \underbrace{(\bar{y}_{\cdot m} - \bar{y})}_{\text{model effect}}$$

Take the true per-dataset and per-model mean MCC and add them. Even knowing perfectly how
hard every dataset is and how good every model is, **pure addition explains only 66% of
MCC**. The remaining third is dataset×model interaction — a specific model being unusually
well or badly suited to a specific dataset.

The same construction with one group at a time gives the two ceilings that bound E1 and E2:

| knowing only | R² |
|---|---|
| which **dataset** it is | 0.354 |
| which **model** it is | 0.282 |
| both, added | 0.6605 |

**This does not bound E3, and E3 passes it.** The additive oracle bounds a predictor that
is a per-dataset value *plus* a per-model value. Half of E3's terms are *mixed* — each
multiplying or dividing a dataset feature by a model feature — and those express precisely
the interaction the two-way additive form cannot. E3 reaches 0.658 in-sample against the
oracle's 0.6605 — just short of it — while the same features under the full grammar reach
**0.707**, comfortably past it, at a leave-one-dataset-out of 0.678.

> An earlier draft of this chapter read the crossing as a warning sign, on the grounds that
> E3 passed the oracle "at exactly the point its cross-validated score collapses". That was
> measured under the re-selecting protocol this study no longer reports, where the numbers
> either side of the crossing were -0.283 and -0.591. Under the fixed-form protocol the
> crossing happens with transfer intact. The claim has been withdrawn rather
> than restated: the crossing is what the mixed terms buy.

### Ceiling 2 — what the vocabulary can reach

The oracle bounds a sum of *group* effects. A different and equally useful bound is a sum
of *per-feature* functions — what an equation could explain if it never combined two
features in one term. `analysis.grammar_ceiling` computes it as three least-squares fits
over the term library, none of which needs the search to have run:

| level | terms | R² |
|---|---|---|
| every raw feature, untransformed | 18 | 0.476 |
| the best single-feature term per feature | 18 | 0.544 |
| every single-feature term at once | 75 | 0.644 |
| **E3, fitted (arity 2)** | **15** | **0.658** |
| **E3, full grammar (arity 3)** | **23** | **0.707** |

Two readings, and [chapter 10](10-report.md) regenerates both from the run:

**No single feature carries the equation.** The strongest, `eq_num_attr`, reaches R² 0.142
on its own; `Model Capability` 0.141; `Processing Units Number` 0.127. There is no dominant
driver to quote, which is why the equation needs a dozen-odd terms rather than two.

**The transforms earn about +0.068**, the gap between entering the raw columns and taking
the best single-feature term of each. `analysis.feature_reach` reports that per feature,
and it is concentrated where a straight line was the wrong shape: `class_ent` goes from
R² 0.002 raw to 0.050 as `1/class_ent`, `inst_to_attr` from 0.001 to 0.040 as
`1/inst_to_attr`, `gravity` from 0.002 to 0.031 under a log. For eight of the eighteen
features the grammar buys nothing at all — the raw column was already the best form of it.

**E3 passes this ceiling too**, with 15 terms against the 75 single-feature terms' 0.644,
and the full-grammar equation passes it by 0.063.
An equation cannot exceed it by describing features one at a time, so the excess is the
work the cross-feature terms do. That is the same conclusion the additive oracle reaches,
by an entirely independent route — one argues from group means, the other from the term
library — which is worth more than either on its own.

### Ceiling 3 — how fast interaction pays

The additive oracle's residual is not noise. It is a (dataset × model) matrix of
interactions, and interaction matrices are typically dominated by a few components. So
decompose it by SVD and add the leading components back:

$$\hat{y}^{(r)}_{dm} = \bar{y} + \alpha_d + \beta_m + \sum_{j=1}^{r} \sigma_j u_{dj} v_{mj}$$

This is the **AMMI model** — additive main effects, multiplicative interaction — long used
for genotype-by-environment trials in agronomy, which is structurally the same problem: a
grid of subjects crossed with conditions where particular pairings suit each other.
Unobserved cells (24 of 500 here) contribute zero residual, so they neither distort the
decomposition nor enter any score.

| interaction rank | R² | gain |
|---|---|---|
| 0 (additive oracle) | 0.6605 | — |
| **1** | **0.7828** | **+0.122** |
| 2 | 0.8537 | +0.071 |
| 3 | 0.8968 | +0.043 |
| 4 | 0.9278 | +0.031 |
| 8 | 0.9842 | +0.019 |
| 20 (full) | 1.0000 | — |

At full rank it reproduces every observed cell, so the question is not where the ladder
ends but **how fast it climbs**: the first interaction component alone is worth +0.122 R²,
more than the entire difference between a 2-term and a 14-term equation.

Does the equation reach any of it? Comparing two R² values cannot say — a number below
rank 0 is equally consistent with an equation that misses the pattern and one that finds it
but is inaccurate elsewhere. Comparing the two interaction *structures* can.
`validate.interaction_capture` lays both the truth and the equation's predictions on the
(dataset × model) grid, strips each of its own additive part, and correlates what is left:

| protocol | alignment with rank 1 | with ranks 1–2 |
|---|---|---|
| in-sample | 0.31 | 0.28 |
| leave-one-dataset-out | 0.27 | 0.21 |

The equation reaches **about a third** of the leading interaction pattern in-sample and a
little over a quarter out of fold — not most of it, and not none. Interaction is 36% of
MCC's variance once the additive part is removed, and the leading component carries 38% of
*that*, so the remaining headroom is real but smaller than +0.122 makes it look.

It is also not obviously reachable. A rank-1 interaction is a product of a dataset-side
latent and a model-side latent, both free numbers; the equation's mixed terms are products
of *single raw features*, which recover that structure only where it happens to align with
one feature pair. [Chapter 7](07-limitations.md) measures what is left once each latent has
to be predicted rather than handed over: **+0.018 of the +0.122**, with the shortfall almost
entirely on the model side.

## Does model choice matter more than the dataset?

Not on this meta-dataset — but the reason is more interesting than the answer.

Dataset identity explains **0.354** of MCC variance against model identity's **0.282**, and
the equations widen the gap rather than closing it:

| | equation | own ceiling | captured |
|---|---|---|---|
| dataset features | 0.349 | 0.354 — the true dataset means | **98%** |
| model features | 0.248 | 0.282 — the true model means | **88%** |
| both | 0.658 | 0.6605 — the additive oracle* | *100%* |

<sub>*The additive oracle bounds a two-way *additive* form, and **E3 now crosses it** —
0.658 against 0.661 for the published equation and 0.707 for the full-grammar one. That is
not an error: seven of E3's fifteen terms multiply or divide a
dataset feature by a model feature, and such a term expresses interaction the additive oracle
by construction cannot (this chapter). The two rows above it are hard ceilings;
this one is a reference level the equation is expected to pass.</sub>

**Both halves are now close to their own ceilings.** Twelve dataset meta-features all but
exhaust what dataset identity can explain — 98% of it, so there is essentially nothing left
for a better dataset descriptor to find. The six model features reach 88% of what model
identity can, which is the result of replacing the model side rather than a property of the
corpus as collected.

That replacement is the study's central measurement, and it is worth stating precisely.
**As originally collected, the corpus's model descriptors captured under two thirds of what
model identity explains.** They were counts of capacity and cost, and three of the five moved
with the dataset as well as the learner. What closes the gap is not more measurement — it is
**asserted taxonomy**: five ordinals placing each learner on a published ladder, none of them
observable in a training run, all of them free to write down. The conclusion is therefore
sharper than "this meta-dataset lacks model features". It lacked them, the missing information
was worth roughly a third of what model identity explains, and *nearly all of it is
recoverable from knowing what kind of learner this is* — how it randomises, what loss it
minimises, how much of the input distribution it models, and how it is fitted.

The cost of that remedy is that these columns are claims rather than observations, and
[chapter 7](07-limitations.md) is where that is paid for.

Three qualifications keep that from being oversold, all of them developed in
[chapter 7](07-limitations.md): the ladder is asserted rather than measured, so it cannot be
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
  +0.309 R² (0.349 → 0.658), beyond the 0.248 they achieve by themselves. The surplus is
  dataset×model interaction, which is why **7 of E3's 15 terms are mixed** and drive 60%
  of its output variance.

![Contribution shares](../figures/contribution_shares.png)

### The configuration that was dropped, and why this one is different

An earlier draft carried an accuracy-leaning arity-4 configuration — a 4610-term library, 32
terms, 0.668 in-sample — and dropped it. It cost 0.18 of leave-one-dataset-out R² for that
fit, which no reader of this study should want, and it was reported as a *second headline*,
which invites quoting whichever number suits the argument.

The capability equation below is not that. It gives up nothing on transfer — it is the better
equation on both protocols — and it is reported as the answer to one stated question rather
than as an alternative headline. The arity trade itself is recorded in
[chapter 2](02-additive-model.md), which sweeps the penalty and the length inside each arity.

## The two configurations the study reports

`DEFAULT_E3` is what the study recommends. `DEFAULT_E3_CAPABILITY` is the same corpus and the
same four model features under the **full** grammar, reported to answer a question the
published equation cannot answer about itself.

| | grammar | z-cap | penalty | terms | in-sample | LOO-dataset | LOO-model |
|---|---|---|---|---|---|---|---|
| `DEFAULT_E3` | arity 2 | 4.25 | 20 | **15** | 0.6578 | **0.6381** | 0.6218 |
| `DEFAULT_E3_CAPABILITY` | arity 3 | 4.25 | 3 | 23 | 0.7068 | 0.6781 | 0.6512 |

**Why two, and why this is not "quote whichever suits the argument".** An earlier draft
carried a second accuracy-leaning configuration and dropped it for exactly that reason. The
difference is that this one answers a specific question and is labelled as its answer:
*is the additive form out of room, or is the published equation short of it?* Without it,
the published R² can only be read against oracles and baselines, none of which is an
equation of this shape. With it, the answer is that **+0.049 of in-sample R² and +0.040 of
transfer are still available to the form** — so the published equation is not at the form's
limit, and what it pays for that gap is eight fewer terms, one operation fewer, and a form
that reselects far more often across folds. The capability equation is never analysed term by
term and never used for guidance.

**Every knob comes from the 2026-09-07 sweep** — 48,576 configurations over feature subsets ×
penalties × lengths × z-caps × arities. It confirmed arity 2 for the published equation
outright (best arity-2 objective 0.730 against 0.687 for arity 3, and the whole top band is
arity 2), and left penalty and z-cap unbeaten by any significant margin.

**The model-feature pool is four, not six, and that is compression rather than loss.** The
corpus carries six because the corpus is designed for *identification*
([chapter 1](01-dataset.md)); the equation is judged on *compression*, so it drops `Solution
Stochasticity` and `Loss Margin Behaviour` from its term pool. On the sweep's numbers the
four-feature pool dominates the full six on every axis — objective 0.730 against 0.722,
leave-one-dataset-out 0.686 against 0.672, leave-one-model-out 0.654 against 0.640. The
published equation goes further and uses **12 of the 16 features available to it**; an
equation that used all sixteen would be one that had failed to generalise.

**The length is chosen by a rule, not written down** — the argmax of the consensus curve,
`selection.best_length`. It selects 15 under arity 2 and 23 under arity 3 without either
number appearing anywhere in the code.
[Chapter 3](03-term-selection.md#stage-3--choosing-the-number-of-terms) sets out the rule,
every alternative that was computed and rejected, and why the geometric ones disagree.

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

E2, over the six model features:

```
MCC = +0.46554
      +0.559567   * 1/Loss Margin Behaviour
      +0.00549788 * Model Capability^2
      -0.196113   * sqrt(Processing Units Number)
      +0.0410675  * [log(Loss Margin Behaviour)] * [log(Model Capability)]
      +0.317787   * [log(Loss Margin Behaviour)] / [Model Capability]
      -0.381236   * [log(Model Capability)] / [log(Processing Units Number)]
      +0.385192   * [log(Model Capability)] / [Solution Stochasticity]
      +0.129776   * [log(Processing Units Number)] * [log(Solution Stochasticity)]
```

### How much of each control survives into E3

Counting terms the published equations share *exactly*:

| | shared with E3 | which |
|---|---|---|
| E2 → E3 | **0 of 8** | — |
| E1 → E3 | **1 of 7** | `[log(eq_num_attr)] * [log(nr_class)]` |

**Almost nothing survives, and that is the honest reading of the controls.** E2's terms are
built to say as much as possible using *only* model features, so they lean on ratios between
the model columns themselves — `log(Model Capability) / Solution Stochasticity`, and so on.
Once a dataset feature is available to pair with, none of those pairings is the best use of a
term slot, and E3 rebuilds the model side from scratch against dataset partners.

This does not weaken the controls; it clarifies what they measure. E1 and E2 bound **how much
of MCC each half of the meta-data explains**, and they do that whether or not their particular
terms reappear. They were never evidence that E3 would phrase things the same way, and an
earlier version of this chapter over-read a 4-of-6 overlap as though they were.

What it does show is that the equation is not a concatenation of its two halves. Nine of E3's
sixteen terms are mixed, and a mixed term is not available to either control by construction.

E3, on all 476 rows, is 15 terms and is **not reproduced here**. It is printed in full,
with its term-importance table and its analysis, in [chapter 10](10-report.md) — which is
regenerated with the equation on every run, so it cannot drift out of step with the code
the way a copy in this chapter would. An earlier draft of this chapter carried a 14-term
E3 that had stopped being the published equation several configurations earlier, which is
why the listing now lives on the generated side.

The shape of it, from that chapter: 7 of the 15 terms mix dataset and model features and
drive **60%** of the output variance; 3 are model-only (3%) and 5 are dataset-only (37%).
The weights are flat — they behave like **13.8 equally-weighted terms**, and the largest
carries under 11% of the mass.

![What each term is worth](../figures/term_effects.png)

## Reading the equation: what interpretability buys and what it costs

The equation is not the most accurate predictor available on this meta-data — an opaque
regressor reaches R² ≈ 0.9 in-sample, and [chapter 5](05-evaluation.md) shows what happens
to that number out of fold. It is the one that can be *read*, and this section says what
reading it actually gets you, because "interpretable" is a claim that ought to be cashed
out rather than asserted.

**Three properties make the form readable, and each is a constraint the search runs under.**

*It is a sum.* Every term contributes independently and additively, so a term can be
removed, quoted, or reasoned about without re-deriving the rest. Nothing in the equation is
conditional on anything else — there is no branch, no per-family slope, no lookup table.
That is why binary indicator features are excluded ([chapter 2](02-additive-model.md)):
inside a product, an indicator is identically zero on the rows where it is off, which turns
an additive term into a per-group slope and the equation into a piecewise model.

*Every term is a named quantity.* A term is one to three raw features under a transform the
vocabulary defines, so `[log(gravity)] / [log(Processing Units Number)]` reads as "class
separation, relative to model capacity" without a loadings table or a rotation matrix. This
is the constraint that has cost the study the most: a principal-coordinate embedding of the
model taxonomy transferred better than anything else measured here and was rejected because
two of its five columns had no plain-language reading.

*One term per combination of raw features.* Added to the protocol on 2026-09-07. Without
it the equation could carry two terms over the same feature pair — `[a] * [b]` and
`[a] / [b]`, say — which read as two independent findings and are one relationship
expressed twice. It is a constraint on form rather than on fit, and it was measured to cost
nothing: paired over the twenty folds, p = 0.503 with an interval spanning zero.

### The cost, stated plainly

The weights are **flat**. They behave like about 13.8 equally-weighted terms out of 16, and
the largest carries under 11% of the mass. There is no headline term to quote, and the
[ceilings section](#ceiling-2--what-the-vocabulary-can-reach) says why: the strongest single
feature reaches R² 0.142 on its own. MCC on this corpus is not driven by one thing.

That is a statement about the *unit of explanation*, not a defect. It does mean the honest
way to read this equation is in blocks rather than term by term — which features the search
reached for, which operations it needed to apply to them, and which terms move together —
and [chapter 10](10-report.md) generates all three readings from the fitted object.

### Why this matters for the next chapter but one

The point of a readable equation is that its terms can be **matched against claims made
elsewhere**. [Chapter 6](06-practices.md) does exactly that: it takes best practices stated
in the tabular-ML literature — "prefer tree ensembles on tabular data", "high class
imbalance hurts", and so on — and checks each against the terms of E3, asking whether the
equation independently arrived at the same relationship, contradicted it, or is silent.

That comparison is only possible because the terms are named physical quantities in a sum.
A forest with the same accuracy admits no such check, and neither would this equation if its
features were rotated coordinates. **The interpretability constraint is not a tax paid for
presentation; it is what makes the chapter-6 evaluation exist at all** — which is the
argument for accepting the accuracy it gives up.

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

What that costs is stated in [chapter 7](07-limitations.md): every prediction is implicitly
conditional on the training succeeding, and the study never measures how often that is
true. It is the main reason the go/no-go rule in [chapter 10](10-report.md) should be read
as "will this trained model be any good" rather than "should I try this at all".

The axes start at 0; the single negative row falls outside them and
`plots.count_below_floor()` returns the count for a caption.
