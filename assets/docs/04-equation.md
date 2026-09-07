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

**All three are fitted on the 476 rows, not on aggregated group means.** Fitting E1 on the
20 per-dataset means is tempting — a predictor constant inside a group can only predict that
group's mean anyway — but it puts E1's R² on a twenty-point denominator:

| E1 fitted on | in-sample R² | LOO-dataset R² | comparable with E3? |
|---|---|---|---|
| 20 dataset means | 0.337 | 0.506 *(on 20 points)* | no |
| **476 rows** | **0.349** | **0.341** *(on 476 rows)* | yes |

Two R² values computed over different row sets share no denominator. Printed in one table
they invite the conclusion that the dataset-only control transfers *better* than the full
equation, which on the common scale it does not. Fitting on all rows costs nothing and
removes the caveat instead of requiring one.

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
measurement; [chapter 4](04-equation.md) treats the asserted ladders as the cost they are.

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

> The crossing has to be read under the reported protocol. Under a re-selecting protocol —
> where every length is a separately searched equation — the transfer figures either side of
> it swing wildly and the crossing looks like a warning sign. Under the fixed-form protocol
> this study reports, it happens with transfer intact, and it is what the mixed terms buy.

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

Two readings, and the generated section below regenerates both from the run:

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
one feature pair. [chapter 4](04-equation.md) measures what is left once each latent has
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
[chapter 4](04-equation.md) is where that is paid for.

Three qualifications keep that from being oversold, all of them developed in
[chapter 4](04-equation.md): the ladder is asserted rather than measured, so it cannot be
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

### Why two configurations, and why that is not two headlines

Reporting two equations risks quoting whichever number suits the argument. The capability
equation avoids that in two ways: it gives up nothing on transfer — it is the better equation
on both protocols — and it is reported as the answer to one stated question rather than as an
alternative headline. It is never analysed term by term and never used for guidance. The
arity trade behind it is a design decision, set out in [chapter 2](02-additive-model.md),
which sweeps the penalty and the length inside each arity.

## The two configurations the study reports

`DEFAULT_E3` is what the study recommends. `DEFAULT_E3_CAPABILITY` is the same corpus and the
same four model features under the **full** grammar, reported to answer a question the
published equation cannot answer about itself.

| | grammar | z-cap | penalty | terms | in-sample | LOO-dataset | LOO-model |
|---|---|---|---|---|---|---|---|
| `DEFAULT_E3` | arity 2 | 4.25 | 20 | **15** | 0.6578 | **0.6381** | 0.6218 |
| `DEFAULT_E3_CAPABILITY` | arity 3 | 4.25 | 3 | 23 | 0.7068 | 0.6781 | 0.6512 |

The second row answers a question the first cannot answer about itself: *is the additive form
out of room, or is the published equation short of it?* Without it the published R² can only
be read against oracles and baselines, none of which is an equation of this shape. With it,
**+0.049 of in-sample R² and +0.040 of transfer are still available to the form** — so the
published equation is not at the form's limit, and what it pays for that gap is eight fewer
terms, one operation fewer, and a form that reselects far more often across folds.

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

What it does show is that the equation is not a concatenation of its two halves. Seven of
E3's fifteen terms are mixed, and a mixed term is not available to either control by construction.

E3, on all 476 rows, is 15 terms. It is printed in full, with its term-importance table and
its analysis, in the generated section at the end of this chapter — rewritten with the
equation on every run, so it cannot drift out of step with the code the way a hand-copied
listing would.

Its shape: 7 of the 15 terms mix dataset and model features and
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
and the generated section below generates all three readings from the fitted object.

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

What that costs is stated in [chapter 4](04-equation.md): every prediction is implicitly
conditional on the training succeeding, and the study never measures how often that is
true. It is the main reason the go/no-go rule in the generated section below should be read
as "will this trained model be any good" rather than "should I try this at all".

The axes start at 0; the single negative row falls outside them and
`plots.count_below_floor()` returns the count for a caption.

<!-- generated: do not edit below -->

## The equation

E3 uses **15 terms** over dataset and model meta-features, simplified and refitted after pruning, so it evaluates exactly as printed.

```
MCC = +1.35863
      +0.00770741 * [log(gravity)] * [log(Model Capability)]      # beta=+0.1515
      -0.290997 * [log(eq_num_attr)] / [log(Processing Units Number)]  # beta=-0.1408
      -0.000958087 * [log(nr_class)] * [nr_outliers]              # beta=-0.1229
      -0.0186651 * [log(gravity)] / [log(Processing Units Number)]  # beta=-0.1150
      -0.069639 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.0888
      +0.269778 * 1/Fitting Regime                                # beta=+0.0757
      -0.0197473 * [log(gravity)] / [log(nr_attr)]                # beta=-0.0731
      -0.00570359 * [log(gravity)] * [log(Fitting Regime)]        # beta=-0.0684
      -0.0715798 * [log(Input Distribution Modelling)] * [log(Processing Units Number)]  # beta=-0.0661
      -1.90277 * [nr_cor_attr] / [log(Processing Units Number)]   # beta=-0.0656
      +0.00125935 * [nr_bin] * [log(Model Capability)]            # beta=+0.0605
      -0.0485924 * [log(Processing Units Number)] / [log(nr_class)]  # beta=-0.0458
      -0.0126933 * class_ent^2                                    # beta=-0.0313
      +0.223455 * 1/gravity                                       # beta=+0.0310
      +0.00392383 * Fitting Regime^2                              # beta=+0.0258
```

LaTeX:

```latex
\mathrm{MCC} = +1.359 +0.007707 \cdot \mathrm{[log(gravity)] * [log(Model Capability)]} -0.291 \cdot \mathrm{[log(eq\_num\_attr)] / [log(Processing Units Number)]} -0.0009581 \cdot \mathrm{[log(nr\_class)] * [nr\_outliers]} -0.01867 \cdot \mathrm{[log(gravity)] / [log(Processing Units Number)]} -0.06964 \cdot \mathrm{[log(eq\_num\_attr)] * [log(nr\_class)]} +0.2698 \cdot \mathrm{1/Fitting Regime} -0.01975 \cdot \mathrm{[log(gravity)] / [log(nr\_attr)]} -0.005704 \cdot \mathrm{[log(gravity)] * [log(Fitting Regime)]} -0.07158 \cdot \mathrm{[log(Input Distribution Modelling)] * [log(Processing Units Number)]} -1.903 \cdot \mathrm{[nr\_cor\_attr] / [log(Processing Units Number)]} +0.001259 \cdot \mathrm{[nr\_bin] * [log(Model Capability)]} -0.04859 \cdot \mathrm{[log(Processing Units Number)] / [log(nr\_class)]} -0.01269 \cdot \mathrm{class\_ent\^{}2} +0.2235 \cdot \mathrm{1/gravity} +0.003924 \cdot \mathrm{Fitting Regime\^{}2}
```

## Equation analysis

The equation has 15 terms, of which **10** carry 83% of the standardised weight mass; the single largest carries 13.0%, and the weights behave like **12.1 equally-weighted terms** (inverse Simpson index of the shares).

That last number is the one to read for concentration, because it does not depend on where a threshold is drawn. At 81% of the term count the equation is **flat**: no single term dominates. That is a statement about the *unit of explanation*, not about the quality of the equation — MCC here is inferred by a set of terms acting together rather than by one or two that could be quoted on their own. Three readings follow, and the sections below give each one: read the terms in the blocks that move together, read which features the search reached for, and read which operations it needed to apply to them.

`beta` is the standardised weight — the MCC contributed per standard deviation of the term, which is what makes terms in unrelated units comparable. `effect` is the swing in predicted MCC across the middle 80% of the term's observed range. `stability` is the fraction of leave-one-dataset-out folds that selected the term.

| rank | term | group | features | weight | beta | effect | share | cumulative | major | stability | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [log(gravity)] * [log(Model Capability)] | mixed | gravity, Model Capability | 0.0077 | 0.1515 | 0.3891 | 0.1304 | 0.1304 | yes | 1.0000 | raises MCC |
| 2 | [log(eq_num_attr)] / [log(Processing Units Number)] | mixed | eq_num_attr, Processing Units Number | -0.2910 | -0.1408 | 0.3489 | 0.1212 | 0.2516 | yes | 0.7000 | lowers MCC |
| 3 | [log(nr_class)] * [nr_outliers] | dataset | nr_class, nr_outliers | -0.0010 | -0.1229 | 0.1892 | 0.1058 | 0.3573 | yes | 0.1500 | lowers MCC |
| 4 | [log(gravity)] / [log(Processing Units Number)] | mixed | gravity, Processing Units Number | -0.0187 | -0.1150 | 0.2940 | 0.0990 | 0.4563 | yes | 1.0000 | lowers MCC |
| 5 | [log(eq_num_attr)] * [log(nr_class)] | dataset | eq_num_attr, nr_class | -0.0696 | -0.0888 | 0.2984 | 0.0764 | 0.5327 | yes | 0.8500 | lowers MCC |
| 6 | 1/Fitting Regime | model | Fitting Regime | 0.2698 | 0.0757 | 0.2023 | 0.0651 | 0.5978 | yes | 0.9000 | raises MCC |
| 7 | [log(gravity)] / [log(nr_attr)] | dataset | gravity, nr_attr | -0.0197 | -0.0731 | 0.2126 | 0.0629 | 0.6606 | yes | 0.8000 | lowers MCC |
| 8 | [log(gravity)] * [log(Fitting Regime)] | mixed | gravity, Fitting Regime | -0.0057 | -0.0684 | 0.1457 | 0.0589 | 0.7195 | yes | 0.8500 | lowers MCC |
| 9 | [log(Input Distribution Modelling)] * [log(Processing Units Number)] | model | Input Distribution Modelling, Processing Units Number | -0.0716 | -0.0661 | 0.1358 | 0.0569 | 0.7764 | yes | 0.8000 | lowers MCC |
| 10 | [nr_cor_attr] / [log(Processing Units Number)] | mixed | nr_cor_attr, Processing Units Number | -1.9028 | -0.0656 | 0.1653 | 0.0564 | 0.8328 | yes | 0.5000 | lowers MCC |
| 11 | [nr_bin] * [log(Model Capability)] | mixed | nr_bin, Model Capability | 0.0013 | 0.0605 | 0.0710 | 0.0520 | 0.8848 | no | 0.8500 | raises MCC |
| 12 | [log(Processing Units Number)] / [log(nr_class)] | mixed | Processing Units Number, nr_class | -0.0486 | -0.0458 | 0.1239 | 0.0394 | 0.9243 | no | 0.5500 | lowers MCC |
| 13 | class_ent^2 | dataset | class_ent | -0.0127 | -0.0313 | 0.0666 | 0.0269 | 0.9512 | no | 0.2000 | lowers MCC |
| 14 | 1/gravity | dataset | gravity | 0.2235 | 0.0310 | 0.0155 | 0.0266 | 0.9778 | no | 0.5000 | raises MCC |
| 15 | Fitting Regime^2 | model | Fitting Regime | 0.0039 | 0.0258 | 0.0589 | 0.0222 | 1.0000 | no | 0.4000 | raises MCC |

#### The 8 largest terms, in words

(10 terms are flagged major; the leading 8 are written out, and the table above carries the rest.)

1. `[log(gravity)] * [log(Model Capability)]` (mixed) raises MCC: one standard deviation of this term is worth +0.152 MCC, it moves predicted MCC by 0.389 across the middle 80% of its observed range, it carries 13.0% of the equation's weight mass, and it was selected in every fold.
2. `[log(eq_num_attr)] / [log(Processing Units Number)]` (mixed) lowers MCC: one standard deviation of this term is worth -0.141 MCC, it moves predicted MCC by 0.349 across the middle 80% of its observed range, it carries 12.1% of the equation's weight mass, and it was selected in 70% of folds.
3. `[log(nr_class)] * [nr_outliers]` (dataset) lowers MCC: one standard deviation of this term is worth -0.123 MCC, it moves predicted MCC by 0.189 across the middle 80% of its observed range, it carries 10.6% of the equation's weight mass, and it was selected in 15% of folds.
4. `[log(gravity)] / [log(Processing Units Number)]` (mixed) lowers MCC: one standard deviation of this term is worth -0.115 MCC, it moves predicted MCC by 0.294 across the middle 80% of its observed range, it carries 9.9% of the equation's weight mass, and it was selected in every fold.
5. `[log(eq_num_attr)] * [log(nr_class)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.089 MCC, it moves predicted MCC by 0.298 across the middle 80% of its observed range, it carries 7.6% of the equation's weight mass, and it was selected in 85% of folds.
6. `1/Fitting Regime` (model) raises MCC: one standard deviation of this term is worth +0.076 MCC, it moves predicted MCC by 0.202 across the middle 80% of its observed range, it carries 6.5% of the equation's weight mass, and it was selected in 90% of folds.
7. `[log(gravity)] / [log(nr_attr)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.073 MCC, it moves predicted MCC by 0.213 across the middle 80% of its observed range, it carries 6.3% of the equation's weight mass, and it was selected in 80% of folds.
8. `[log(gravity)] * [log(Fitting Regime)]` (mixed) lowers MCC: one standard deviation of this term is worth -0.068 MCC, it moves predicted MCC by 0.146 across the middle 80% of its observed range, it carries 5.9% of the equation's weight mass, and it was selected in 85% of folds.

#### Large terms the folds disagreed on

1 of the major terms were selected by fewer than half of the folds. A large weight and a low selection frequency together mean the term is doing its work for *this* training set and would be replaced by something else on another, which the equation as printed does not show. **Do not build guidance on these.**

| rank | term | group | beta | share | stability |
|---|---|---|---|---|---|
| 3 | [log(nr_class)] * [nr_outliers] | dataset | -0.1229 | 0.1058 | 0.1500 |

#### Reading the terms in blocks

An additive form invites reading one term at a time, and that works when one or two weights dominate. When they do not, the honest unit is larger than a term and smaller than the equation: terms whose per-row contributions move together say the same thing about a row and can be read as one block. Grouping is on the contributions rather than on shared features, because two terms can share no feature and still track each other.

11 blocks over 15 terms, the largest holding 4 terms and 29% of the weight mass.

| group | n_terms | share | effect | direction | shared | terms |
|---|---|---|---|---|---|---|
| 1 | 4 | 0.2858 | 0.6113 | lowers MCC | gravity | 1/Fitting Regime ; [log(gravity)] / [log(nr_attr)] ; [log(gravity)] * [log(Fitting Regime)] ; [log(gravity)] / [log(Processing Units Number)] |
| 2 | 2 | 0.1976 | 0.4761 | lowers MCC | eq_num_attr | [log(eq_num_attr)] * [log(nr_class)] ; [log(eq_num_attr)] / [log(Processing Units Number)] |
| 3 | 1 | 0.1304 | 0.3891 | raises MCC | Model Capability, gravity | [log(gravity)] * [log(Model Capability)] |
| 4 | 1 | 0.1058 | 0.1892 | lowers MCC | nr_class, nr_outliers | [log(nr_class)] * [nr_outliers] |
| 5 | 1 | 0.0569 | 0.1358 | lowers MCC | Input Distribution Modelling, Processing Units Number | [log(Input Distribution Modelling)] * [log(Processing Units Number)] |
| 6 | 1 | 0.0564 | 0.1653 | lowers MCC | Processing Units Number, nr_cor_attr | [nr_cor_attr] / [log(Processing Units Number)] |
| 7 | 1 | 0.0520 | 0.0710 | raises MCC | Model Capability, nr_bin | [nr_bin] * [log(Model Capability)] |
| 8 | 1 | 0.0394 | 0.1239 | lowers MCC | Processing Units Number, nr_class | [log(Processing Units Number)] / [log(nr_class)] |
| 9 | 1 | 0.0269 | 0.0666 | lowers MCC | class_ent | class_ent^2 |
| 10 | 1 | 0.0266 | 0.0155 | raises MCC | gravity | 1/gravity |
| 11 | 1 | 0.0222 | 0.0589 | raises MCC | Fitting Regime | Fitting Regime^2 |

#### Which features the search reached for

**12 of 18** available meta-features appear in the equation. `share` sums the weight mass of every term a feature appears in, so a feature in two terms is credited both and the column does not sum to 1 — it answers how much of the equation touches this feature, not how much it owns. A feature the search declined to use after seeing every transform of it is itself a result.

| feature | meaning | n_terms | share | transforms | operations |
|---|---|---|---|---|---|
| gravity | gravity (separation between the majority and minority class centres) | 5 | 0.3777 | inv, log | atom, product, ratio |
| Processing Units Number | model capacity (log processing units) | 5 | 0.3729 | log | product, ratio |
| nr_class | number of classes | 3 | 0.2216 | log | product, ratio |
| eq_num_attr | equivalent number of attributes (effective feature count) | 2 | 0.1976 | log | product, ratio |
| Model Capability | learner family's capability rank in the tabular-ML literature (1-10) | 2 | 0.1824 | log | product |
| Fitting Regime | how the parameters are reached, closed form to in-context (1-5) | 3 | 0.1462 | inv, log, sq | atom, product |
| nr_outliers | number of attributes containing outliers | 1 | 0.1058 | id | product |
| nr_attr | number of attributes | 1 | 0.0629 | log | ratio |
| Input Distribution Modelling | how much of the input distribution the learner models (1-5) | 1 | 0.0569 | log | product |
| nr_cor_attr | proportion of correlated attribute pairs | 1 | 0.0564 | id | ratio |
| nr_bin | number of binary attributes | 1 | 0.0520 | id | product |
| class_ent | class entropy (how evenly the labels are spread) | 1 | 0.0269 | sq | atom |
| inst_to_attr | instances per attribute | 0 | 0.0000 |  |  |
| nr_inst | number of instances in the source dataset (before sampling) | 0 | 0.0000 |  |  |
| nr_norm | number of normally distributed attributes | 0 | 0.0000 |  |  |
| ns_ratio | noise-to-signal ratio | 0 | 0.0000 |  |  |
| Solution Stochasticity | how deep randomisation reaches into the fit (1-5) | 0 | 0.0000 |  |  |
| Loss Margin Behaviour | how hard the loss penalises points far from the boundary (1-5) | 0 | 0.0000 |  |  |

#### Which operations the equation needed

The vocabulary offers five operations and five transforms and the search is free to ignore any of them, so a row that was offered and went unused is a shape this data turned out not to need. Rows marked `offered = no` were kept out of the library by the arity cap and say nothing about the data:

| kind | name | offered | n_terms | share |
|---|---|---|---|---|
| operation | atom | yes | 4 | 0.1408 |
| operation | ratio | yes | 5 | 0.3789 |
| operation | product | yes | 6 | 0.4803 |
| operation | sum_ratio | no | 0 | 0.0000 |
| operation | ratio_of_sums | no | 0 | 0.0000 |
| transform | id | yes | 3 | 0.2142 |
| transform | log | yes | 11 | 0.8592 |
| transform | sqrt | yes | 0 | 0.0000 |
| transform | inv | yes | 2 | 0.0917 |
| transform | sq | yes | 2 | 0.0491 |

#### Where the equation's variance comes from

- terms over dataset meta-features alone (how hard is this data): 5 terms, 37.2% of the equation's output variance.
- terms over model meta-features alone (how capable is this model): 3 terms, 3.2% of the equation's output variance.
- terms mixing dataset and model features (which model suits which data): 7 terms, 59.6% of the equation's output variance.

| group | n_terms | share | effect_sum |
|---|---|---|---|
| dataset | 5 | 0.3724 | 0.3558 |
| model | 3 | 0.0316 | 0.0353 |
| mixed | 7 | 0.5960 | 0.5135 |

## Equation length

Where additional terms stop paying, by knee detection on the accuracy-versus-length curve and by Pareto dominance:

| rule | n_terms | r2_in_sample | r2_loo_dataset |
|---|---|---|---|
| pareto front, closest to ideal | 4 | 0.5390 | 0.5053 |
| pareto front, furthest from nadir | 4 | 0.5390 | 0.5053 |
| pareto front, furthest from chord | 4 | 0.5390 | 0.5053 |
| best loo-dataset | 15 | 0.6578 | 0.6381 |
| best consensus (the rule) | 15 | 0.6578 | 0.6381 |
| published | 15 | 0.6578 | 0.6381 |

| n_terms | r2_in_sample | mae_in_sample | smape_in_sample | r2_loo_dataset | mae_loo_dataset | smape_loo_dataset | r2_loo_model | mae_loo_model | smape_loo_model |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.2173 | 0.2440 | 46.8473 | 0.1654 | 0.2517 | 47.6765 | 0.1994 | 0.2472 | 47.1900 |
| 2 | 0.3679 | 0.2048 | 42.9474 | 0.3191 | 0.2122 | 43.9902 | 0.3553 | 0.2070 | 43.2309 |
| 3 | 0.4752 | 0.1843 | 40.9152 | 0.4277 | 0.1913 | 42.1394 | 0.4534 | 0.1881 | 41.3969 |
| 4 | 0.5390 | 0.1676 | 38.8722 | 0.5053 | 0.1733 | 39.8140 | 0.5173 | 0.1716 | 39.4355 |
| 5 | 0.5581 | 0.1635 | 38.2407 | 0.5315 | 0.1678 | 39.0066 | 0.5335 | 0.1682 | 38.9153 |
| 6 | 0.5903 | 0.1560 | 37.7603 | 0.5659 | 0.1617 | 38.1458 | 0.5652 | 0.1607 | 38.2592 |
| 7 | 0.6125 | 0.1499 | 36.0807 | 0.5876 | 0.1560 | 37.0900 | 0.5873 | 0.1548 | 36.9766 |
| 8 | 0.6243 | 0.1486 | 36.1415 | 0.5965 | 0.1545 | 36.4395 | 0.6014 | 0.1532 | 36.5711 |
| 9 | 0.6275 | 0.1482 | 36.0784 | 0.6059 | 0.1530 | 36.7461 | 0.6029 | 0.1534 | 36.7278 |
| 10 | 0.6344 | 0.1448 | 35.8198 | 0.6151 | 0.1492 | 36.5028 | 0.6068 | 0.1502 | 36.6987 |
| 11 | 0.6408 | 0.1435 | 35.7871 | 0.6187 | 0.1497 | 36.6836 | 0.6124 | 0.1493 | 36.6601 |
| 12 | 0.6452 | 0.1421 | 35.4886 | 0.6231 | 0.1476 | 36.1081 | 0.6080 | 0.1491 | 36.4758 |
| 13 | 0.6471 | 0.1416 | 35.4546 | 0.4829 | 0.1734 | 40.2100 | 0.6099 | 0.1488 | 36.2947 |
| 14 | 0.6497 | 0.1410 | 34.9209 | 0.5003 | 0.1715 | 39.5174 | 0.6102 | 0.1487 | 36.1729 |
| 15 | 0.6578 | 0.1371 | 34.4737 | 0.6381 | 0.1417 | 34.8803 | 0.6218 | 0.1449 | 35.4201 |
| 16 | 0.6619 | 0.1357 | 33.8603 | 0.6317 | 0.1451 | 34.8130 | 0.6263 | 0.1431 | 34.7298 |
| 17 | 0.6633 | 0.1357 | 34.0624 | 0.6299 | 0.1446 | 34.9541 | 0.6254 | 0.1435 | 34.8493 |
| 18 | 0.6639 | 0.1360 | 34.0860 | 0.6343 | 0.1433 | 34.7439 | 0.6254 | 0.1438 | 34.8883 |
| 19 | 0.6663 | 0.1350 | 33.8645 | 0.6289 | 0.1425 | 34.9076 | 0.6266 | 0.1430 | 34.9297 |
| 20 | 0.6679 | 0.1343 | 33.8598 | 0.6379 | 0.1433 | 34.5722 | 0.6282 | 0.1423 | 34.5920 |
| 21 | 0.6693 | 0.1342 | 33.9862 | 0.6337 | 0.1431 | 34.8863 | 0.6288 | 0.1422 | 34.7660 |
| 22 | 0.6701 | 0.1338 | 33.8926 | 0.6255 | 0.1461 | 35.0555 | 0.6284 | 0.1421 | 34.6898 |
| 23 | 0.6707 | 0.1335 | 33.8135 | 0.6306 | 0.1438 | 34.4890 | 0.6283 | 0.1419 | 34.8448 |
| 24 | 0.6707 | 0.1337 | 33.8867 | 0.3874 | 0.1749 | 42.0258 | 0.6265 | 0.1424 | 35.1649 |
| 25 | 0.6718 | 0.1341 | 33.8208 | 0.6331 | 0.1452 | 35.6574 | 0.6278 | 0.1429 | 34.9396 |
| 26 | 0.6716 | 0.1337 | 33.8379 | 0.5574 | 0.1618 | 37.2894 | 0.6249 | 0.1432 | 35.1182 |
| 27 | 0.6721 | 0.1333 | 33.7753 | 0.5683 | 0.1592 | 37.0522 | 0.6255 | 0.1427 | 35.2324 |
| 28 | 0.6727 | 0.1333 | 33.8031 | 0.4801 | 0.1717 | 39.2600 | 0.6253 | 0.1429 | 35.2727 |
| 29 | 0.6729 | 0.1334 | 33.8388 | 0.5165 | 0.1659 | 37.9845 | 0.6241 | 0.1433 | 35.3512 |
| 30 | 0.6732 | 0.1334 | 33.8136 | 0.5371 | 0.1633 | 37.6430 | 0.6236 | 0.1434 | 35.3303 |
| 31 | 0.6728 | 0.1334 | 33.7488 | 0.5373 | 0.1603 | 36.4489 | 0.6193 | 0.1439 | 35.3237 |
| 32 | 0.6730 | 0.1333 | 33.7138 | 0.4202 | 0.1798 | 39.1599 | 0.6193 | 0.1438 | 35.3273 |

## The dataset-only and model-only controls

E1 sees dataset meta-features only, so it can predict just one value per dataset; E2 sees model meta-features only. Together they show how much of MCC each half of the meta-data explains on its own.

**E1** (7 terms):

```
MCC = +1.24602
      -0.109215 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.1392
      -0.0284191 * ([log(gravity)] + [log(nr_class)]) / [log(nr_attr)]  # beta=-0.1072
      -0.415152 * ([log(eq_num_attr)] + [log(ns_ratio)]) / [log(nr_inst)]  # beta=-0.0860
      -0.128423 * [nr_cor_attr] * [nr_norm]                       # beta=-0.0728
      -0.00254076 * ([log(gravity)] + [nr_norm]) / [log(nr_class)]  # beta=-0.0523
      +0.0272363 * ([nr_norm] + [log(ns_ratio)]) / [log(nr_attr)]  # beta=+0.0465
      +0.0057269 * ([nr_bin] + [nr_norm]) / [log(nr_attr)]        # beta=+0.0342
```

**E2** (6 terms):

```
MCC = +0.374877
      +0.13883 * [log(Model Capability)] * [log(Processing Units Number)]  # beta=+0.2564
      -0.268522 * log(Fitting Regime)                             # beta=-0.1359
      -0.0315036 * Processing Units Number                        # beta=-0.1346
      +0.101647 * [log(Fitting Regime)] * [log(Processing Units Number)]  # beta=+0.1183
      +0.322689 * 1/Loss Margin Behaviour                         # beta=+0.0896
      +0.231211 * [log(Loss Margin Behaviour)] / [Model Capability]  # beta=+0.0519
```

<!-- end generated -->

## Limitations of the equation

### Model descriptors are thin, and one of them is asserted

The five model features the corpus originally shipped captured 63% of what model identity
explains. That gap is what the asserted ordinals were added to close, and they close most of
it: E2 goes from 63% to **88%** of its ceiling.

**Five of the six model features are asserted, not measured, and that is the sharpest
limitation in this chapter.** `Processing Units Number` is computed from a trained instance.
`Model Capability` ranks the ten learner families on a ladder taken from the tabular-ML
literature (Grinsztajn et al. 2022;
Shwartz-Ziv & Armon 2022; McElfresh et al. 2023; Hollmann et al. 2023). Three consequences
follow and none should be glossed:

- **It encodes prior knowledge, so it cannot be evidence for that knowledge.** Any result
  that amounts to "capable families do better" is partly built in. What the column can
  support is the *conditional* claim — how family capability interacts with dataset
  properties — which is not something the ladder asserts.
- **The corpus contradicts it in places.** Ordering families by observed mean MCC agrees
  with the ladder at only 6 of 9 steps, and `generic NN` sits at rung 6 of 10 while holding
  the *lowest* family mean here, 0.454. The overall association is nonetheless strong
  (Spearman 0.733 conditional, 0.391 marginal).
- **It does not extend to an unclassified model.** A new learner needs a human to place it
  on the ladder. So does each of `Solution Stochasticity`, `Loss Margin Behaviour`,
  `Input Distribution Modelling` and `Fitting Regime` — all four are read off a published
  description of the algorithm, not off a training run. Five of six model features therefore
  require an act of classification before the equation can be applied to a genuinely new
  method.

The four mechanism ordinals share the first and third caveats and are **weaker than
`Model Capability` on the second**, because they were not fitted against anything: each grades
one mechanism on a stated scale, with `Loss Margin Behaviour` the least defensible of them —
placing the perceptron criterion above hinge is a choice rather than a consensus. Any term
over one of these columns is evidence about the ordering claimed here, not about a quantity
anyone observed.

Richer *measured* descriptors — inductive bias, hypothesis-space characteristics, optimiser
behaviour — would carry none of these caveats and remain the better answer.

How much more is now measured rather than guessed. Tabulating that missing third per model
instead of describing it is worth **+0.106** of leave-one-dataset-out R²
(this chapter) — which is at most what a *perfect* set of extra
descriptors would be worth, since free per-model numbers are the best any descriptor set
could do at telling these 25 models apart. It is also why a table is not a substitute for
descriptors: it transfers to a new dataset and not to a new model, while a descriptor would
do both.

### The bound on what better model descriptors could buy

*Implemented in `ml_meta_perf.identity`. It measures the headroom the study cannot reach with
the descriptors the corpus records, which is a limitation rather than a result.*

*Implemented in `ml_meta_perf.identity`. **Not part of the reported study** — see the last
section for why it was measured and then withdrawn.*

[Chapter 4](04-equation.md) measures a gap and stops there. Dataset identity explains 0.354
of MCC variance and the twelve dataset meta-features recover **98%** of it; model identity
explains 0.282 and the six model meta-features recover **88%**. What is left is written down
nowhere in this corpus — the features record what a model *is*, its capacity and five
asserted facts about how it is built, but nothing about what it is good at.

**That gap was a third when this chapter was written, and it is now an eighth.** The four
asserted ordinals merged on 2026-09-05 closed most of it, which is why the ceiling measured
below is much smaller than the text originally described.

[Chapter 4](04-equation.md) measures the same gap from the other side: a rank-1 interaction
component is worth **+0.122 R²** over the additive oracle, and the equation reaches about a
third of that pattern in-sample and a quarter out of fold — measured, not inferred from an
R² comparison.

Both are statements that something is missing. Neither says **how much** a better set of
model descriptors would be worth, and that is a number a reader will want before deciding
whether to go and collect them. This chapter measures it.

### The measurement

Under leave-one-dataset-out **every one of the 25 models appears in every training fold**.
So the equation's residual can be averaged per model on the training rows and applied to
the held-out dataset without any leak. Doing that replaces the model descriptors with the
best possible substitute — the model's *identity*, fitted freely — and the improvement is
therefore an upper bound on what any descriptor set could achieve at telling these 25
classifiers apart.

Two numbers per model: a level, and a slope on one dataset feature chosen inside the fold.

$$\text{correction}_m = b_m + c_m \cdot \log(\mathrm{gravity})$$

| | LOO-dataset R² | MAE |
|---|---|---|
| E3, 15 terms | 0.6381 | 0.1425 |
| + levels only | 0.6693 | 0.1351 |
| **+ levels and slope** | **0.6693** | **0.1351** |
| *per-model mean baseline* | *0.2006* | *0.2382* |

**+0.017 of leave-one-dataset-out R² is what perfect model descriptors would still be
worth.** This chapter's whole reason for existing was that the number used to be **+0.106**,
and before `Model Capability` was added, **+0.121** against an E3 scoring 0.4658. It is now
small enough that the question it was asked to answer is closed.

That is the strongest evidence in the study that the model side is adequately described.
Free per-model numbers are the best any descriptor set could do at telling these 25 models
apart — they are model identity itself, fitted out of fold — so what they add on top of the
equation is exactly the information the descriptors are missing. Six columns now leave 0.017
of it on the table, where five columns left 0.106.

**The slope adds nothing at all**, where it used to add +0.065 on top of the levels. A
per-model slope on a dataset feature is an interaction the equation could not express; the
equation now expresses it directly, in nine mixed terms out of sixteen. The row is kept at
its unchanged value rather than deleted, because a measurement going to zero is the result.

The rank columns of this table are dropped rather than refreshed, because the ranking
comparison they fed now lives in [chapter 5](05-evaluation.md), where E3 beats the baseline
on every head-weighted measure.

### Why the model side, specifically

A purely predictive version of chapter 4's rank-1 oracle would estimate *both* latents from
meta-features. Replacing each latent with a ridge fit on its own features, scored
leave-one-out over the latent's own rows:

| latent | rows | best 1 feature | best 2 | best 3 |
|---|---|---|---|---|
| dataset, $u_d$ | 20 | **+0.597** (`gravity`) | +0.629 | +0.662 |
| model, $v_m$ | 25 | +0.227 (`Processing Units Number`) | +0.321 | +0.286 |

And the interaction those fitted latents reconstruct, scored on all 476 rows:

| interaction estimated as | R² | gain over additive |
|---|---|---|
| additive oracle, no interaction | 0.6605 | — |
| $\hat{u}$ from covariates, **true** $v$ | 0.7294 | +0.069 |
| **$\hat{u}$ and $\hat{v}$ both from covariates** | **0.6780** | **+0.018** |
| true $u$, true $v$ (the oracle) | 0.7828 | +0.122 |

**Of chapter 4's +0.122, about +0.018 is reachable from meta-features on both sides, and
the bottleneck is entirely the model side.** The dataset latent is largely predictable —
one feature gets 0.597 of it — and the model latent is not, at 0.32 from its best two.

This is the same conclusion as the +0.106 above, reached independently: the dataset half of
this meta-data is close to exhausted and the model half is not. Chapter 4 says it from the
ceilings, chapter 4 from the interaction ladder, and this chapter from both directions at
once. **Richer model descriptors — inductive bias, hypothesis-space characteristics,
optimiser behaviour — are the only open direction in this study with room left in it.**

### What the correction is, mechanically

Two stages, in this order. The level is the shrunk mean residual of the model's own rows;
the slope is then fitted to what the levels leave behind. Both are shrunk toward zero,
which is the empirical-Bayes estimator for a group effect under a common prior (Efron &
Morris, 1975), with the prior-to-noise ratio fixed rather than estimated — fixed because it
barely matters:

| intercept shrinkage | levels only | with slope |
|---|---|---|
| 0 | 0.5219 | 0.5882 |
| 2 | 0.5217 | 0.5886 |
| **5 (default)** | 0.5198 | **0.5874** |
| 10 | 0.5155 | 0.5836 |
| 20 | 0.5073 | 0.5756 |

R² moves by 0.015 across a range of 20; the slope's shrinkage moves it by 0.005 across the
same range. Nothing here was tuned against the reported figure.

The construction is a **factorial regression** in the sense used for genotype-by-environment
trials: a two-way table modelled with covariates on one margin and free coefficients on the
other (Denis, 1988; van Eeuwijk, Denis & Kang, 1996). In machine-learning terms it is the
collaborative half of a hybrid recommender, and algorithm selection has been posed as
collaborative filtering before (Mısır & Sebag, 2017; Fusi et al., arXiv:1705.05355; Yang et
al., arXiv:1808.03233).

### Two-stage, not backfitting

Fitting the two blocks jointly — backfitting, the standard scheme for an additive model
with a tabulated block (Hastie & Tibshirani, 1990) — is much worse, measured with levels
only:

| rounds | LOO-dataset R² | mixed terms in the equation |
|---|---|---|
| **1 (two-stage)** | **0.5198** | 11.4 / 24 |
| 2 | 0.4068 | 12.2 / 24 |
| 3 | 0.3751 | 12.2 / 24 |
| 5 | 0.3316 | 12.3 / 24 |

It fails in an informative direction. Once the table absorbs model capability the equation
stops needing to explain it and spends its terms on within-dataset detail instead — the
mixed-term count rises every round — and detail like that does not transfer. **The equation
has to be fitted against MCC to remain a statement about MCC.**

### A dense dataset-side score is worse than one feature

The general form of the dataset side is a score over several features, fitted with the
model side by alternating least squares:

| dataset score | LOO-dataset R² | MAE | mean regret |
|---|---|---|---|
| all 12 features | 0.5195 | 0.1627 | **0.016** |
| 3 features | 0.4760 | 0.1702 | 0.017 |
| 2 features | 0.5546 | 0.1610 | 0.030 |
| **1 feature (`gravity`), closed form** | **0.5874** | **0.1494** | 0.032 |

A twelve-loading score is worth no more than the levels alone. Twenty datasets do not
support twelve free loadings, and with a single feature the alternation is unnecessary —
each model's slope is a closed-form ridge fit of its own residuals.

### Negative result: a ranking objective ranks worse

[Chapter 5](05-evaluation.md) reports that the equation is indistinguishable from the
trivial "how well does this model usually do" baselines at ranking, and the natural response is
to fit for ranking rather than for squared error. Squared error over
all within-dataset *pairs* expands to ordinary least squares on a design and target both
centred inside each dataset — the within, or fixed-effects, transform (Mundlak, 1978) — so
a pairwise ranking objective costs one extra centring step and stays a closed-form ridge
solve. It was implemented and cross-validated:

| | mean ρ | mean regret |
|---|---|---|
| E3, squared error, 24 terms | 0.625 | 0.064 |
| **pairwise ranking objective, 24 terms** | **0.532** | **0.095** |
| per-model mean baseline | 0.703 | 0.011 |

<sub>Measured against the 24-term E3, before `Model Capability` joined the model side. The
comparison is between two *objectives* on one feature set, which is what makes it evidence
about the loss function; the published equation now reaches 0.706 with the extra descriptor
and squared error unchanged ([chapter 5](05-evaluation.md)), which is the same conclusion
from the other end.</sub>

**It ranks worse than the objective it was meant to beat.** Centring inside each dataset
removes the dataset main effect, which is most of what the terms explain; what remains is
the within-dataset ordering, for which the equation has only five model meta-features. The
loss function was never the limitation — the model descriptors are, which is this chapter's
conclusion arrived at by a third route.

### Why this is a measurement and not a fourth equation

The numbers are good enough to be tempting, so the reasons a per-model correction table is
not published as an equation are worth stating:

- **It is not a unified equation.** Fifty fitted numbers in a table cannot be read the way a
  term can, cannot be evaluated by hand, and support no term or term-group analysis. The
  additive form exists so that a reader can decompose the prediction; a lookup table
  refuses that at exactly the point where the study's contribution lives.
- **It yields per-model advice, not practices.** The study's unit of output is a statement
  about a *meta-feature* that generalises ([chapter 6](06-practices.md)). A table says "add
  0.24 for logistic regression on this corpus", which is a fact about these 25 runs and this
  equation's bias, not transferable guidance.
- **It cannot extrapolate to a new model.** Under leave-one-model-out the correction is
  empty and its predictions are bitwise E3's. Every number comes from rows where that model
  was already run.
- **It is still below what an opaque regressor reaches** on this kind of meta-data. Trading
  the readable equation for a table does not even win the accuracy argument outright, so it
  wins nothing worth the trade.

What survives is the **measurement**: +0.106 as the upper bound on better model descriptors,
and +0.018 as the covariate-reachable part of the interaction ladder. Those belong to
this chapter's argument about thin model descriptors, and they are the
reason that argument now carries a number.

`ml_meta_perf.identity` ships tested and is wired into no pipeline. It stays in the tree where
the agglomerative-construction experiment of [chapter 3](03-term-selection.md) did not,
because this one produces a number the study quotes — the +0.106 ceiling — rather than only
a conclusion. `ml_meta_perf.identity.correct_out_of_fold`, applied to a finished `cross_validate_fixed_form`, reproduces the table above.
