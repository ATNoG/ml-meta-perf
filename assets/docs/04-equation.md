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
| **E3 (dataset + model)** | **16** | **0.665** | **0.134** | **0.823** |
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
| E1 | 7 | 0.349 | 0.341 | 0.306 |
| E2 | 6 | 0.248 | 0.185 | 0.228 |
| **E3** | **16** | **0.665** | **0.627** | **0.622** |

Both controls are now reported under both protocols, which the aggregated E1 could not be.
The pattern is the one the design predicts and is worth checking rather than assuming: E1
transfers *better* across models (0.294) than across datasets (0.217), because it predicts
a per-dataset constant and a new model does not change it; E2 is the mirror.

**E3's two transfer numbers are now close** — 0.627 and 0.622, against a fit of 0.665. An
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
the interaction the two-way additive form cannot. E3 reaches 0.665 in-sample against the
oracle's 0.6605, and its leave-one-dataset-out score at that length is 0.627.

> An earlier draft of this chapter read the crossing as a warning sign, on the grounds that
> E3 passed the oracle "at exactly the point its cross-validated score collapses". That was
> measured under the re-selecting protocol this study no longer reports, where the numbers
> either side of the crossing were -0.283 and -0.591. Under the fixed-form protocol the
> crossing happens at 16 terms with transfer intact. The claim has been withdrawn rather
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
| **E3, fitted** | **16** | **0.665** |

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

**E3 passes this ceiling too**, with 16 terms against the 75 single-feature terms' 0.644.
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
| both | 0.665 | 0.6605 — the additive oracle* | *101%* |

<sub>*The additive oracle bounds a two-way *additive* form, and **E3 now crosses it** —
0.665 against 0.661. That is not an error: ten of E3's sixteen terms multiply or divide a
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
  +0.317 R² (0.349 → 0.665), beyond the 0.248 they achieve by themselves. The surplus is
  dataset×model interaction, which is why **9 of E3's 16 terms are mixed** and drive 56%
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
[chapter 2](02-additive-model.md) sweeps the penalty and the length inside each arity and
reports the result as a design decision rather than as a second result.

`DEFAULT_E3` is the study:

| | `max_arity` | `max_abs_zscore` | `penalty` | headline terms |
|---|---|---|---|---|
| `DEFAULT_E3` | 2 (270-term library) | 4.25 | 15 | 16 |

| | terms | in-sample R² | LOO-dataset | LOO-model |
|---|---|---|---|---|
| `DEFAULT_E3` | 16 | **0.6651** | **0.6270** | **0.6220** |

**All four knobs moved when the model features were replaced, and each for a reason.**

`max_arity` falls from 3 to 2. The third arity buys `(f1+f2)/f3`, and against the new model
side it is simply not selected — the best arity-2 configuration matches the best arity-3 one
to within 0.007 on both transfer protocols. A smaller grammar that scores the same is not a
trade.

`max_abs_zscore` rises from 3.0 to 4.25. The cap exists to stop a term being carried by a
handful of extreme rows. Loosening it is safe here in a way it would not have been before:
every one of the six model features is positive, bounded and occupied at every rung, with
none of the sparse tail that made a loose cap dangerous.

`penalty` falls from 20 to 15, and on fit alone would fall further. Under the previous
protocol the ridge did two jobs — shrinking the weights *and* scoring which subset the beam
chose, since λ sits in `RSS − λ·wᵀw`. With the equation's form now fixed and only its weights
refit ([chapter 5](05-evaluation.md)), it does only the first, so heavy shrinkage stopped
paying for itself.

**Why 16 terms and not 14, or 24?** Because leave-one-dataset-out is now *smooth* in length —
0.612 at 12 terms, 0.627 at 16, 0.638 at 20 — and past twenty it turns down, to 0.614 at 24
buys under 0.005. Sixteen is a knee. Under the previous protocol the same curve swung by 0.3
between adjacent lengths, because each length was a different equation refitted twenty times;
fixing the form removed that variance and made the knee readable.

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

E3, on all 476 rows, is 16 terms and is **not reproduced here**. It is printed in full,
with its term-importance table and its analysis, in [chapter 10](10-report.md) — which is
regenerated with the equation on every run, so it cannot drift out of step with the code
the way a copy in this chapter would. An earlier draft of this chapter carried a 14-term
E3 that had stopped being the published equation several configurations earlier, which is
why the listing now lives on the generated side.

The shape of it, from that chapter: 9 of the 16 terms mix dataset and model features and
drive **56%** of the output variance; 3 are model-only (3%) and 4 are dataset-only (41%).
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
