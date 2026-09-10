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
rows, under the same four protocols, from **one** configuration, with the length chosen by
the same rule.
`experiment.run_equation` is that function and `run_e1`, `run_e2` and `run_e3` are one
line each. The uniformity is not tidiness: the gaps between the three are only evidence
about what each half of the meta-data is worth if *nothing else* differs between them.

**All three are fitted on the 476 rows, not on aggregated group means.** Fitting E1 on the
20 per-dataset means is tempting — a predictor constant inside a group can only predict that
group's mean anyway — but it puts E1's R² on a twenty-point denominator:

| E1 fitted on | in-sample R² | LOO-dataset R² | comparable with E3? |
|---|---|---|---|
| 20 dataset means | 0.337 | 0.506 *(on 20 points)* | no |
| **476 rows** | **0.351** | **0.338** *(on 476 rows)* | yes |

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

<sub>**Measured before the 2026-09-05 protocol change**, unlike the E1 table above it, which
is current. Read the two rows against each other and not against any figure elsewhere in the
study: E2 on all 476 rows scores higher than 0.177 under the reported protocol — the
generated tables have the current value — and three model features that varied within a model
have since been retired, which is part of why. What the table supports is the *direction*,
and that has been re-checked: aggregating E2 to per-model means is still worse.</sub>

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

Each equation, each equation's own ceiling, and the additive oracle are in one generated
table — on the [index page](index.md), which carries the headline, and again with the full
metric set in [chapter 5](05-evaluation.md). The figure is the same comparison drawn:

![Equations against their ceilings](../figures/01_equation_comparison.png)

### One scale is not one ceiling

Those R² values are arithmetically comparable — same rows, same denominator. They are
still **not comparable as achievements**, and no change to the fitting could make them so.

E1 predicts one value per dataset, so **the true dataset means are the most it could ever
score**, however good its terms were: that is all the variance a per-dataset constant can
reach. E1 is not "worse than E3"; it is at its own limit while E3 is not at its. The same
applies to E2 against the true model means.

So the comparable quantity is the *fraction of its own ceiling* each equation reaches, which
is the `reached` column of the generated headline table and what the next section reads.

## Under all four protocols

All three equations are scored under every protocol
[chapter 5](05-evaluation.md#four-protocols) defines: in-sample, leave-one-dataset-out,
leave-one-model-out, and the doubly-held-out cell. The numbers are on the
[index page](index.md) and, with the full metric set, in [chapter 5](05-evaluation.md) —
this chapter does not keep a second copy of them.

What is worth stating here is what to look for in them.

**E3's transfer numbers should be close to each other and to its fit, and the gap between the
best and the worst is a reported quantity.** `selection.protocol_spread` is exactly that
gap — in-sample minus the floor over the four — and for the published equation it is 0.0416.
An equation that loses little R² when a whole dataset, a whole learner, or both are withheld
is transferring rather than memorising; a small spread says no single protocol is carrying
it. The arity-3 bound scores higher on every column and spreads wider, 0.0539, which is the
shape of a grammar leaning harder on what it has seen.

**The two controls are not symmetric, and not in the direction the design suggests.** E1
predicts a per-dataset constant, so holding out a *model* leaves its constant well estimated
while holding out a *dataset* asks it to extrapolate — the naive expectation is that E1
transfers better across models than across datasets. Measured, it is the other way round, and
E2 mirrors it. The reason is the denominator rather than the fit: pooled R² is taken against
the variance of all 476 rows under every protocol, and a fold that removes a whole model
removes rows spread across every dataset, which is a different perturbation from removing a
contiguous dataset block. Read the two controls each against its own ceiling — the `reached`
column — and not against each other across protocols.

Five of the six model features are constant within a learner, so holding out a model removes
values the equation cannot recompute from the remaining rows. That this costs so little is
the measurement; the [limitations](#limitations-of-the-equation) treat the asserted ladders
as the cost they are.

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

The same construction with one group at a time gives the two ceilings that bound E1 and E2 —
knowing only which dataset it is, and knowing only which model it is. All three are in the
generated section below, under *Where the variance is, before any equation*.

**This does not bound E3, although the current E3 remains below it.** The additive oracle
bounds a predictor that is a per-dataset value *plus* a per-model value. Half of E3's terms
are *mixed* — each
multiplying or dividing a dataset feature by a model feature — and those express precisely
the interaction the two-way additive form cannot. E3 reaches 0.641 in-sample and the wider
default grammar reaches 0.655, against the oracle's 0.661. Their direct alignment with the
oracle's interaction components, reported below, is therefore the evidence that the mixed
terms capture part of that structure.

### Ceiling 2 — what the vocabulary can reach

The oracle bounds a sum of *group* effects. A different and equally useful bound is a sum
of *per-feature* functions — what an equation could explain if it never combined two
features in one term. `analysis.grammar_ceiling` computes it as three least-squares fits
over the term library, none of which needs the search to have run. The ladder — raw columns,
the best single-feature term per feature, then every single-feature term at once — is in the
generated section below, with the fitted equations beside it. Two readings:

**No single feature carries the equation.** The strongest reaches R² of about 0.14 on its
own. There is no dominant driver to quote, which is why the equation needs a dozen-odd terms
rather than two, and the per-feature table in the generated section is where to check it.

**The transforms earn the gap between entering the raw columns and taking the best
single-feature term of each.** `analysis.feature_reach` reports that per feature, and it is
concentrated where a straight line was the wrong shape — `class_ent` and `inst_to_attr` are
each worth an order of magnitude more as reciprocals than raw, and `gravity` under a log. For
a good third of the features the grammar buys nothing at all: the raw column was already the
best form of it.

**E3 passes this ceiling too.**
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
decomposition nor enter any score. The ladder is in the generated section below, under
*How fast interaction pays*.

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
| in-sample | 0.32 | 0.27 |
| leave-one-dataset-out | 0.29 | 0.24 |

The equation reaches **about a third** of the leading interaction pattern in-sample and a
little over a quarter out of fold — not most of it, and not none. Interaction is 36% of
MCC's variance once the additive part is removed, and the leading component carries 38% of
*that*, so the remaining headroom is real but smaller than +0.122 makes it look.

It is also not obviously reachable. A rank-1 interaction is a product of a dataset-side
latent and a model-side latent, both free numbers; the equation's mixed terms are products
of *single raw features*, which recover that structure only where it happens to align with
one feature pair. The [limitations](#why-the-model-side-specifically) measure what is left
once each latent has to be predicted rather than handed over, and it is a small fraction of
the rank-1 gain, with the shortfall almost entirely on the model side.

## Does model choice matter more than the dataset?

Not on this meta-dataset — but the reason is more interesting than the answer.

Dataset identity explains more of MCC's variance than model identity does — the
decomposition is in the generated section above — and the equations widen the gap rather than
closing it: the dataset features recover almost all of their ceiling, the model features a
clearly smaller share of theirs. The `reached` column of the [index page](index.md)'s headline
table is that comparison in one place.

The additive oracle is the third reference, and it is **not** a ceiling for E3: six of the
twelve terms multiply or divide a dataset feature by a model feature, and such a term
expresses interaction a two-way additive form by construction cannot. The two group-identity
levels are hard ceilings; the oracle is a reference level that a mixed equation can pass.

**Both halves are now close to their own ceilings.** Twelve dataset meta-features all but
exhaust what dataset identity can explain — 99% of it, so there is essentially nothing left
for a better dataset descriptor to find. The six model features reach 87% of what model
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

The cost of that remedy is that these columns are claims rather than observations, and the
[limitations](#model-descriptors-are-thin-and-one-of-them-is-asserted) are where that is paid
for.

Three qualifications keep that from being oversold, all of them developed there: the ladder
is asserted rather than measured, so it cannot be
evidence for the prior knowledge it encodes; the corpus agrees with it at only 6 of 9 steps,
with `generic NN` conspicuously misplaced; and it does not extend to a learner nobody has
classified, which is what the fall in leave-one-model-out records.

Collecting more meta-features of the *data* remains low priority — E1 is already within
0.003 of what perfect knowledge of dataset identity would buy.

Two things create the opposite impression and are worth stating explicitly, because both
are artefacts:

- The **within-group correlation table is dataset-centred**, which removes dataset variance
  by construction. Only model terms can score well there. It is a diagnostic for model
  effects, not a statement of relative importance.
- **Model features carry more in combination than alone.** Adding them to E1 is worth
  +0.290 R² (0.351 → 0.641), beyond the 0.247 they achieve by themselves. The surplus is
  dataset×model interaction, which is why **6 of E3's 12 terms are mixed** and carry 44%
  of its standardised weight mass.

## The two equations the study reports, and why it no longer names either

Both come from **one search over grammars**, not from two configurations written down.
`experiment.search_grammars` fits the same corpus and the same four model features once per
arity under a single configuration — the arity is the only thing that differs between the runs
— and the selection rule then names them:

* **E3-Valid** is what the study recommends: the simplest grammar the larger one does not beat
  by more than the spread of that beating (`selection.best_configuration`).
* **E3-MAX** is the capability measurement: the best floor any searched grammar reaches
  (`selection.most_capable`).

Neither the length nor the arity is asserted anywhere. The length inside a grammar is the
argmax of that grammar's worst-protocol curve (`selection.floor_argmax`); the grammar across
them is the rule above. Every number the choice rests on — the floor, the drop from fit to
worst protocol, and the margin against the best — is written to `results/grammars.csv`, and
the generated section below reports them. **This chapter deliberately restates none of them.**
It used to carry its own copy of that table and went stale the moment the two grammars were
made to share one configuration.

The second equation answers a question the first cannot answer about itself: *is the additive
form out of room, or is the published equation short of it?* Without it the published R² can
only be read against oracles and baselines, none of which is an equation of this shape.

**Reporting two equations risks quoting whichever number suits the argument, and the rule is
what stops that.** E3-MAX scores higher on every protocol and is still not the recommendation,
because the rule asks whether it *beats* E3-Valid rather than whether it outscores it — see
[chapter 2](02-additive-model.md#the-arity-is-searched-not-set) for that comparison in
numbers. It is reported as the answer to one stated question, never analysed term by term,
never used for guidance, and it is kept out of every ranking and threshold comparison in
chapter 5 so that it cannot appear as a candidate a reader might pick.

**One caveat on reading the gap between them.** Until 2026-09-09 the bound was fitted with its
own, much lighter, ridge penalty, which made "the full grammar reaches further" partly a
statement about two hyperparameter sets. It is not any more, and the gap is correspondingly
smaller. That is the comparison becoming honest rather than the form shrinking.

**The search settings come from the 2026-09-07 sweep** — 48,576 configurations over feature
subsets × penalties × lengths × z-caps × arities. They are held fixed here so the change from
a stored log count to the raw processing-unit count is isolated. The full configuration sweep
has not been repeated after that data correction; the equation length and grammar are still
re-derived from the current validation curves.

**The model-feature pool is four, not six, and that is compression rather than loss.** The
corpus carries six because the corpus is designed for *identification*
([chapter 1](01-dataset.md)); the equation is judged on *compression*, so it drops `Solution
Stochasticity` and `Loss Margin Behaviour` from its term pool. That four-feature subset was
selected by the sweep and is retained in the corrected-data run. The published equation goes
further and uses fewer features than are available to it — the generated feature table below
counts them — and an equation that used every one would be one that had failed to generalise.

**The length is chosen by a rule, not written down** — the argmax of the worst-protocol curve,
`selection.floor_argmax`. It selects 12 under arity 2 and 14 under arity 3 without either
number appearing anywhere in the code.
[Chapter 3](03-term-selection.md#stage-3-choosing-the-number-of-terms) sets out the rule,
every alternative that was computed and rejected, and why the geometric ones disagree.

## The fitted equations

E1, on all 476 rows:

```
MCC = +1.10961
      -0.0384221 * [log(gravity)] / [log(nr_attr)]
      -0.0920537 * sqrt(eq_num_attr)
      -0.0671274 * [log(eq_num_attr)] * [log(nr_class)]
      -0.12376 * [nr_cor_attr] * [nr_norm]
      +0.000175192 * [nr_bin] * [log(nr_inst)]
      -0.000120213 * [log(inst_to_attr)] * [nr_outliers]
      +0.000891407 * [log(inst_to_attr)] * [nr_norm]
      +0.257987 * [log(inst_to_attr)] / [log(nr_inst)]
      +0.107512 * 1/ns_ratio
      -0.318387 * [log(nr_class)] / [log(nr_inst)]
```

E2, over the six model features:

```
MCC = +0.699974
      +0.00495734 * Model Capability^2
      -0.685729 * [log(Model Capability)] / [log(Processing Units Number)]
      +0.273273 * 1/Loss Margin Behaviour
      -0.207532 * [log(Input Distribution Modelling)] / [Model Capability]
      -0.0115725 * log(Processing Units Number)
      -0.00842549 * [log(Loss Margin Behaviour)] * [log(Processing Units Number)]
```

### How much of each control survives into E3

Counting terms the published equations share *exactly*:

| | shared with E3 | which |
|---|---|---|
| E2 → E3 | **0 of 6** | — |
| E1 → E3 | **2 of 10** | `[log(eq_num_attr)] * [log(nr_class)]`; `[log(gravity)] / [log(nr_attr)]` |

**Almost nothing survives, and that is the honest reading of the controls.** E2's terms are
built to say as much as possible using *only* model features, so they lean on ratios between
the model columns themselves — `log(Model Capability) / Solution Stochasticity`, and so on.
Once a dataset feature is available to pair with, none of those pairings is the best use of a
term slot, and E3 rebuilds the model side from scratch against dataset partners.

This does not weaken the controls; it clarifies what they measure. E1 and E2 bound **how much
of MCC each half of the meta-data explains**, and they do that whether or not their particular
terms reappear. They were never evidence that E3 would phrase things the same way, and an
earlier version of this chapter over-read a 4-of-6 overlap as though they were.

What it does show is that the equation is not a concatenation of its two halves. Six of
E3's twelve terms are mixed, and a mixed term is not available to either control by construction.

E3, on all 476 rows, is 12 terms. It is printed in full, with its term-importance table and
its analysis, in the generated section at the end of this chapter — rewritten with the
equation on every run, so it cannot drift out of step with the code the way a hand-copied
listing would.

Its shape: 6 of the 12 terms mix dataset and model features and carry **44%** of the
standardised weight mass; 2 are model-only (3%) and 4 are dataset-only (53%).
The weights are flat: they behave like far more equally-weighted terms than any headline
reading would suggest, and no single term carries a seventh of the mass. The generated
section below gives the current figures — an inverse Simpson index over the standardised
weight shares, and the largest term's share — rather than restating them here, because they
move with every configuration and this sentence has drifted from them before.

![What each term is worth](../figures/04_term_effects.png)

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

The weights are **flat**: the effective number of terms, by the inverse Simpson index in the
generated section below, is close to the number the equation has, and the largest carries a
small fraction of the mass. There is no headline term to quote, and the
[ceilings section](#ceiling-2-what-the-vocabulary-can-reach) says why: the strongest single
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

![Predicted versus actual MCC](../figures/03_predicted_vs_actual.png)

In-sample predictions never fall below **0.01**, while 15 rows sit at exactly MCC = 0 — a short
column of points hanging above the diagonal on the left. The equation compresses toward
the middle of the range, as a shrunk linear fit will.

**This is not a claim that E3 cannot predict training failure, because the meta-dataset
contains no training failures.** Runs that failed to train were discarded when the corpus
was built, so every row is a model that trained and then scored. A row at MCC = 0 is a
classifier that converged and learned nothing useful — predicting the majority class, say
— not one that crashed. The equation is fitted on, and can only speak about, the
population of runs that completed.

What that costs is stated in [chapter 1](01-dataset.md): every prediction is implicitly
conditional on the training succeeding, and the study never measures how often that is
true. It is the main reason the go/no-go rule in the generated section below should be read
as "will this trained model be any good" rather than "should I try this at all".

The axes start at 0; the single negative row falls outside them and
`plots.count_below_floor()` returns the count for a caption.

<!-- generated: do not edit below -->

## The equation

E3 uses **12 terms** over dataset and model meta-features, simplified and refitted after pruning, so it evaluates exactly as printed.

```
MCC = +1.21173
      -0.165894 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.2115
      -0.0620398 * [log(nr_attr)] / [log(nr_class)]               # beta=-0.1432
      +0.00604706 * [log(gravity)] * [log(Model Capability)]      # beta=+0.1189
      -0.0295834 * [log(gravity)] / [log(nr_attr)]                # beta=-0.1095
      -0.0482441 * [log(gravity)] / [log(Processing Units Number)]  # beta=-0.0996
      -0.00656639 * [log(gravity)] * [log(Fitting Regime)]        # beta=-0.0788
      +0.00160773 * [nr_bin] * [log(Model Capability)]            # beta=+0.0772
      +0.202801 * [log(class_ent)] / [log(inst_to_attr)]          # beta=+0.0709
      +0.0174389 * [log(nr_inst)] / [Input Distribution Modelling]  # beta=+0.0686
      +0.216163 * 1/Fitting Regime                                # beta=+0.0606
      -0.509458 * [log(nr_class)] / [log(Processing Units Number)]  # beta=-0.0594
      +0.00503447 * [log(Fitting Regime)] * [log(Processing Units Number)]  # beta=+0.0347
```

LaTeX:

```latex
\mathrm{MCC} = +1.212 -0.1659 \cdot \mathrm{[log(eq\_num\_attr)] * [log(nr\_class)]} -0.06204 \cdot \mathrm{[log(nr\_attr)] / [log(nr\_class)]} +0.006047 \cdot \mathrm{[log(gravity)] * [log(Model Capability)]} -0.02958 \cdot \mathrm{[log(gravity)] / [log(nr\_attr)]} -0.04824 \cdot \mathrm{[log(gravity)] / [log(Processing Units Number)]} -0.006566 \cdot \mathrm{[log(gravity)] * [log(Fitting Regime)]} +0.001608 \cdot \mathrm{[nr\_bin] * [log(Model Capability)]} +0.2028 \cdot \mathrm{[log(class\_ent)] / [log(inst\_to\_attr)]} +0.01744 \cdot \mathrm{[log(nr\_inst)] / [Input Distribution Modelling]} +0.2162 \cdot \mathrm{1/Fitting Regime} -0.5095 \cdot \mathrm{[log(nr\_class)] / [log(Processing Units Number)]} +0.005034 \cdot \mathrm{[log(Fitting Regime)] * [log(Processing Units Number)]}
```

## How far the form could reach

Two ceilings, both computed from the library alone and so available *before* an equation exists. Each is an expectation the fitted equation is then held against, rather than a number read off it.

#### How far the additive form reaches

The published equation is the **parsimonious** grammar (arity 2). The same features under the **full** grammar (arity 3), with the length chosen by the same rule, reach 14 terms at R² 0.6551 in-sample:

| | terms | in-sample | LOO-dataset | LOO-model |
|---|---|---|---|---|
| published (arity 2) | 12 | 0.6408 | 0.6234 | 0.6050 |
| capability (arity 3) | 14 | 0.6551 | 0.6266 | 0.6171 |

This is a **capability measurement, not a recommendation**. It answers the question the published equation cannot answer about itself — whether the additive form is out of room or whether this equation is short of it — and the answer is that +0.0143 of in-sample R² is still available to a longer equation over a wider grammar. What that costs is what the published equation is buying: more terms, an operation more, and a form that reselects far less often across folds.

#### What the vocabulary could reach, before any search

Three levels of what the vocabulary can explain, each a least-squares fit over the library and each computable before the search runs. They bound a *sum of per-feature functions*, which is a different question from the additive oracle above: that one bounds a per-dataset value plus a per-model value.

| level | terms | R² |
|---|---|---|
| every raw term admitted by the grammar | 17 | 0.4524 |
| the best single-feature term per feature | 18 | 0.5369 |
| every single-feature term at once | 72 | 0.6321 |
| **the fitted equation (E3)** | **12** | **0.6408** |

No individual feature carries much: the strongest is `eq_num_attr` at R² 0.142, so any accuracy beyond that is combination rather than a single dominant driver. Taking one best admissible term per feature is worth +0.085 over the admissible raw-term fit.

E3 reaches 0.6408 with 12 terms, **above** the 0.6321 that all 72 single-feature terms reach together. An equation cannot pass that level by describing features one at a time, so the excess is what the cross-feature terms buy — the same conclusion the additive oracle reaches, by an independent route.

#### Where the variance is, before any equation

Variance of MCC explained by group identity alone, with no equation involved:

| knowing only | n_groups | variance_explained |
|---|---|---|
| dataset identity | 20 | 0.3539 |
| model identity | 25 | 0.2821 |

#### How fast interaction pays

The additive form cannot represent dataset-by-model interaction beyond what its mixed terms reach. The ladder below adds interaction components to an oracle that is handed the true group means, so it measures the ceiling rather than any equation:

| interaction_rank | r2 | gain |
|---|---|---|
| 0 | 0.6605 |  |
| 1 | 0.7828 | 0.1223 |
| 2 | 0.8537 | 0.0709 |
| 3 | 0.8968 | 0.0431 |
| 4 | 0.9278 | 0.0310 |
| 6 | 0.9651 | 0.0373 |
| 8 | 0.9842 | 0.0191 |

That is a ceiling, not a score. Whether the equation reaches any of it is a separate question, and the answer is that it reaches some: below, `alignment` is the squared correlation between the equation's own interaction residual and the leading components of the oracle's, over observed cells. `leading_share` is how much of the interaction variance those components carry, and `interaction_share` how much of MCC's variance is interaction at all.

| protocol | rank | alignment | leading_share | interaction_share |
|---|---|---|---|---|
| in-sample | 1 | 0.3214 | 0.3757 | 0.3647 |
| in-sample | 2 | 0.2744 | 0.5658 | 0.3647 |
| leave-one-dataset-out | 1 | 0.2949 | 0.3757 | 0.3647 |
| leave-one-dataset-out | 2 | 0.2372 | 0.5658 | 0.3647 |

## Equation analysis

The equation has 12 terms, of which **8** carry 80% of the standardised weight mass; the single largest carries 18.7%, and the weights behave like **9.8 equally-weighted terms** (inverse Simpson index of the shares).

That last number is the one to read for concentration, because it does not depend on where a threshold is drawn. At 81% of the term count the equation is **flat**: no single term dominates. That is a statement about the *unit of explanation*, not about the quality of the equation — MCC here is inferred by a set of terms acting together rather than by one or two that could be quoted on their own. Three readings follow, and the sections below give each one: read the terms in the blocks that move together, read which features the search reached for, and read which operations it needed to apply to them.

`beta` is the standardised weight — the MCC contributed per standard deviation of the term, which is what makes terms in unrelated units comparable. `effect` is the swing in predicted MCC across the middle 80% of the term's observed range. `stability` is the fraction of leave-one-dataset-out folds that selected the term.

| rank | term | group | features | weight | beta | effect | share | cumulative | major | stability | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [log(eq_num_attr)] * [log(nr_class)] | dataset | eq_num_attr, nr_class | -0.1659 | -0.2115 | 0.7108 | 0.1867 | 0.1867 | yes | 0.5000 | lowers MCC |
| 2 | [log(nr_attr)] / [log(nr_class)] | dataset | nr_attr, nr_class | -0.0620 | -0.1432 | 0.3462 | 0.1264 | 0.3131 | yes | 0.0500 | lowers MCC |
| 3 | [log(gravity)] * [log(Model Capability)] | mixed | gravity, Model Capability | 0.0060 | 0.1189 | 0.3053 | 0.1050 | 0.4181 | yes | 0.5500 | raises MCC |
| 4 | [log(gravity)] / [log(nr_attr)] | dataset | gravity, nr_attr | -0.0296 | -0.1095 | 0.3185 | 0.0966 | 0.5147 | yes | 0.2000 | lowers MCC |
| 5 | [log(gravity)] / [log(Processing Units Number)] | mixed | gravity, Processing Units Number | -0.0482 | -0.0996 | 0.2514 | 0.0879 | 0.6026 | yes | 0.8000 | lowers MCC |
| 6 | [log(gravity)] * [log(Fitting Regime)] | mixed | gravity, Fitting Regime | -0.0066 | -0.0788 | 0.1677 | 0.0696 | 0.6722 | yes | 0.3500 | lowers MCC |
| 7 | [nr_bin] * [log(Model Capability)] | mixed | nr_bin, Model Capability | 0.0016 | 0.0772 | 0.0907 | 0.0681 | 0.7403 | yes | 0.7500 | raises MCC |
| 8 | [log(class_ent)] / [log(inst_to_attr)] | dataset | class_ent, inst_to_attr | 0.2028 | 0.0709 | 0.0479 | 0.0626 | 0.8029 | yes |  | raises MCC |
| 9 | [log(nr_inst)] / [Input Distribution Modelling] | mixed | nr_inst, Input Distribution Modelling | 0.0174 | 0.0686 | 0.1805 | 0.0606 | 0.8634 | no | 0.1500 | raises MCC |
| 10 | 1/Fitting Regime | model | Fitting Regime | 0.2162 | 0.0606 | 0.1621 | 0.0535 | 0.9169 | no | 0.7000 | raises MCC |
| 11 | [log(nr_class)] / [log(Processing Units Number)] | mixed | nr_class, Processing Units Number | -0.5095 | -0.0594 | 0.1544 | 0.0525 | 0.9694 | no | 0.3500 | lowers MCC |
| 12 | [log(Fitting Regime)] * [log(Processing Units Number)] | model | Fitting Regime, Processing Units Number | 0.0050 | 0.0347 | 0.0535 | 0.0306 | 1.0000 | no | 0.1500 | raises MCC |

#### The 8 largest terms, in words

1. `[log(eq_num_attr)] * [log(nr_class)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.211 MCC, it moves predicted MCC by 0.711 across the middle 80% of its observed range, it carries 18.7% of the equation's weight mass, and it was selected in 50% of folds.
2. `[log(nr_attr)] / [log(nr_class)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.143 MCC, it moves predicted MCC by 0.346 across the middle 80% of its observed range, it carries 12.6% of the equation's weight mass, and it was selected in 5% of folds.
3. `[log(gravity)] * [log(Model Capability)]` (mixed) raises MCC: one standard deviation of this term is worth +0.119 MCC, it moves predicted MCC by 0.305 across the middle 80% of its observed range, it carries 10.5% of the equation's weight mass, and it was selected in 55% of folds.
4. `[log(gravity)] / [log(nr_attr)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.109 MCC, it moves predicted MCC by 0.318 across the middle 80% of its observed range, it carries 9.7% of the equation's weight mass, and it was selected in 20% of folds.
5. `[log(gravity)] / [log(Processing Units Number)]` (mixed) lowers MCC: one standard deviation of this term is worth -0.100 MCC, it moves predicted MCC by 0.251 across the middle 80% of its observed range, it carries 8.8% of the equation's weight mass, and it was selected in 80% of folds.
6. `[log(gravity)] * [log(Fitting Regime)]` (mixed) lowers MCC: one standard deviation of this term is worth -0.079 MCC, it moves predicted MCC by 0.168 across the middle 80% of its observed range, it carries 7.0% of the equation's weight mass, and it was selected in 35% of folds.
7. `[nr_bin] * [log(Model Capability)]` (mixed) raises MCC: one standard deviation of this term is worth +0.077 MCC, it moves predicted MCC by 0.091 across the middle 80% of its observed range, it carries 6.8% of the equation's weight mass, and it was selected in 75% of folds.
8. `[log(class_ent)] / [log(inst_to_attr)]` (dataset) raises MCC: one standard deviation of this term is worth +0.071 MCC, it moves predicted MCC by 0.048 across the middle 80% of its observed range, it carries 6.3% of the equation's weight mass, and it was fold agreement not measured.

#### Large terms the folds disagreed on

3 of the major terms were selected by fewer than half of the folds. A large weight and a low selection frequency together mean the term is doing its work for *this* training set and would be replaced by something else on another, which the equation as printed does not show. **Do not build guidance on these.**

| rank | term | group | beta | share | stability |
|---|---|---|---|---|---|
| 2 | [log(nr_attr)] / [log(nr_class)] | dataset | -0.1432 | 0.1264 | 0.0500 |
| 4 | [log(gravity)] / [log(nr_attr)] | dataset | -0.1095 | 0.0966 | 0.2000 |
| 6 | [log(gravity)] * [log(Fitting Regime)] | mixed | -0.0788 | 0.0696 | 0.3500 |

#### Reading the terms in blocks

An additive form invites reading one term at a time, and that works when one or two weights dominate. When they do not, the honest unit is larger than a term and smaller than the equation: terms whose per-row contributions move together say the same thing about a row and can be read as one block. Grouping is on the contributions rather than on shared features, because two terms can share no feature and still track each other.

8 blocks over 12 terms, the largest holding 2 terms and 19% of the weight mass.

| group | n_terms | share | effect | direction | shared | terms |
|---|---|---|---|---|---|---|
| 1 | 2 | 0.1890 | 0.3645 | lowers MCC |  | [log(class_ent)] / [log(inst_to_attr)] ; [log(nr_attr)] / [log(nr_class)] |
| 2 | 1 | 0.1867 | 0.7108 | lowers MCC | eq_num_attr, nr_class | [log(eq_num_attr)] * [log(nr_class)] |
| 3 | 2 | 0.1845 | 0.5041 | lowers MCC | gravity | [log(gravity)] / [log(nr_attr)] ; [log(gravity)] / [log(Processing Units Number)] |
| 4 | 2 | 0.1655 | 0.4300 | raises MCC |  | [log(gravity)] * [log(Model Capability)] ; [log(nr_inst)] / [Input Distribution Modelling] |
| 5 | 2 | 0.1231 | 0.3242 | lowers MCC | Fitting Regime | 1/Fitting Regime ; [log(gravity)] * [log(Fitting Regime)] |
| 6 | 1 | 0.0681 | 0.0907 | raises MCC | Model Capability, nr_bin | [nr_bin] * [log(Model Capability)] |
| 7 | 1 | 0.0525 | 0.1544 | lowers MCC | Processing Units Number, nr_class | [log(nr_class)] / [log(Processing Units Number)] |
| 8 | 1 | 0.0306 | 0.0535 | raises MCC | Fitting Regime, Processing Units Number | [log(Fitting Regime)] * [log(Processing Units Number)] |

#### Which features the search reached for

**12 of 18** available meta-features appear in the equation. `share` sums the weight mass of every term a feature appears in, so a feature in two terms is credited both and the column does not sum to 1 — it answers how much of the equation touches this feature, not how much it owns. A feature the search declined to use after seeing every transform of it is itself a result.

| feature | meaning | n_terms | share | transforms | operations |
|---|---|---|---|---|---|
| nr_class | number of classes | 3 | 0.3656 | log | product, ratio |
| gravity | gravity (separation between the majority and minority class centres) | 4 | 0.3591 | log | product, ratio |
| nr_attr | number of attributes | 2 | 0.2230 | log | ratio |
| eq_num_attr | equivalent number of attributes (effective feature count) | 1 | 0.1867 | log | product |
| Model Capability | learner family's capability rank in the tabular-ML literature (1-10) | 2 | 0.1731 | log | product |
| Processing Units Number | model capacity (number of fitted processing units) | 3 | 0.1710 | log | product, ratio |
| Fitting Regime | how the parameters are reached, closed form to in-context (1-5) | 3 | 0.1537 | inv, log | atom, product |
| nr_bin | number of binary attributes | 1 | 0.0681 | id | product |
| class_ent | class entropy (how evenly the labels are spread) | 1 | 0.0626 | log | ratio |
| inst_to_attr | instances per attribute | 1 | 0.0626 | log | ratio |
| nr_inst | number of instances in the source dataset (before sampling) | 1 | 0.0606 | log | ratio |
| Input Distribution Modelling | how much of the input distribution the learner models (1-5) | 1 | 0.0606 | id | ratio |
| nr_cor_attr | proportion of correlated attribute pairs | 0 | 0.0000 |  |  |
| nr_norm | number of normally distributed attributes | 0 | 0.0000 |  |  |
| nr_outliers | number of attributes containing outliers | 0 | 0.0000 |  |  |
| ns_ratio | noise-to-signal ratio | 0 | 0.0000 |  |  |
| Solution Stochasticity | how deep randomisation reaches into the fit (1-5) | 0 | 0.0000 |  |  |
| Loss Margin Behaviour | how hard the loss penalises points far from the boundary (1-5) | 0 | 0.0000 |  |  |

#### Which operations the equation needed

The vocabulary offers five operations and five transforms and the search is free to ignore any of them, so a row that was offered and went unused is a shape this data turned out not to need. Rows marked `offered = no` were kept out of the library by the arity cap and say nothing about the data:

| kind | name | offered | n_terms | share |
|---|---|---|---|---|
| operation | atom | yes | 1 | 0.0535 |
| operation | ratio | yes | 6 | 0.4865 |
| operation | product | yes | 5 | 0.4600 |
| operation | sum_ratio | no | 0 | 0.0000 |
| operation | ratio_of_sums | no | 0 | 0.0000 |
| transform | id | yes | 2 | 0.1287 |
| transform | log | yes | 11 | 0.9465 |
| transform | sqrt | yes | 0 | 0.0000 |
| transform | inv | yes | 1 | 0.0535 |
| transform | sq | yes | 0 | 0.0000 |

#### Where the equation's variance comes from

- terms over dataset meta-features alone (how hard is this data): 4 terms, 53.1% of the equation's output variance.
- terms over model meta-features alone (how capable is this model): 2 terms, 2.8% of the equation's output variance.
- terms mixing dataset and model features (which model suits which data): 6 terms, 44.1% of the equation's output variance.

| group | n_terms | share | effect_sum |
|---|---|---|---|
| dataset | 4 | 0.5311 | 0.5442 |
| model | 2 | 0.0275 | 0.1227 |
| mixed | 6 | 0.4413 | 0.4800 |

## The ceiling on model descriptors

Under leave-one-dataset-out every model appears in every training fold, so the equation's residual can be averaged per model on the training rows and applied to the held-out dataset with no leak. That replaces the model descriptors with the best possible substitute -- the model's **identity**, fitted freely -- so what it adds is a ceiling on what any descriptor set could reach by telling these classifiers apart.

| correction | r2_loo_dataset | mae | gain | ci_low | ci_high | sign_p | wins | verdict |
|---|---|---|---|---|---|---|---|---|
| none (E3, 12 terms) | 0.6234 | 0.1472 | 0.0000 |  |  |  | 0 | baseline |
| per-model level | 0.6451 | 0.1421 | 0.0044 | -0.0015 | 0.0104 | 0.2632 | 13 | tie |
| per-model level and slope | 0.6779 | 0.1321 | 0.0146 | 0.0052 | 0.0257 | 0.2632 | 13 | real |

**The gap is 0.055 of leave-one-dataset-out R2**, of which a per-model *level* recovers 0.022 and the level-plus-slope form the remaining 0.033.

**Whether that is real is a paired question**, so each rung is compared with the uncorrected equation dataset by dataset, on absolute error, over the twenty held-out folds. The two rungs come back differently, and the difference is the finding:

* a per-model **level** is a **tie** -- it wins on 13 of the twenty folds and its interval spans zero. A constant shift per model, which is what a level is, adds nothing the equation does not already have.

* a per-model **slope** is **not** a tie: the bootstrap interval [+0.0052, +0.0257] lies entirely above zero, on 13 winning folds of twenty. Read it with the sign test beside it, which at p = 0.263 does **not** reach significance -- so the gain is carried by its size on the folds it wins rather than by winning nearly all of them. That is weaker evidence than the interval alone suggests, and stronger than a tie.

**So the question this chapter was written to close is not closed.** The half of the correction that survives is the one that lets a model's advantage depend on the data -- exactly what a *capability* descriptor would have to do, and exactly what none of the descriptors this corpus records does. The mixed terms were supposed to absorb that interaction and have absorbed only part of it.

Every model-side encoding the study tried and rejected was rejected for failing to recover this gap, so it is a property of the corpus rather than of the search -- and the one route to closing it that survives on the merits is measuring what a model is good at rather than asserting it.

## The dataset-only and model-only controls

E1 sees dataset meta-features only, so it can predict just one value per dataset; E2 sees model meta-features only. Together they show how much of MCC each half of the meta-data explains on its own.

**E1** (10 terms):

```
MCC = +1.10961
      -0.0384221 * [log(gravity)] / [log(nr_attr)]                # beta=-0.1422
      -0.0920537 * sqrt(eq_num_attr)                              # beta=-0.0962
      -0.0671274 * [log(eq_num_attr)] * [log(nr_class)]           # beta=-0.0856
      -0.12376 * [nr_cor_attr] * [nr_norm]                        # beta=-0.0701
      +0.000175192 * [nr_bin] * [log(nr_inst)]                    # beta=+0.0559
      -0.000120213 * [log(inst_to_attr)] * [nr_outliers]          # beta=-0.0444
      +0.000891407 * [log(inst_to_attr)] * [nr_norm]              # beta=+0.0413
      +0.257987 * [log(inst_to_attr)] / [log(nr_inst)]            # beta=+0.0351
      +0.107512 * 1/ns_ratio                                      # beta=+0.0321
      -0.318387 * [log(nr_class)] / [log(nr_inst)]                # beta=-0.0241
```

**E2** (6 terms):

```
MCC = +0.699974
      +0.00495734 * Model Capability^2                            # beta=+0.1622
      -0.685729 * [log(Model Capability)] / [log(Processing Units Number)]  # beta=-0.0896
      +0.273273 * 1/Loss Margin Behaviour                         # beta=+0.0759
      -0.207532 * [log(Input Distribution Modelling)] / [Model Capability]  # beta=-0.0730
      -0.0115725 * log(Processing Units Number)                   # beta=-0.0494
      -0.00842549 * [log(Loss Margin Behaviour)] * [log(Processing Units Number)]  # beta=-0.0305
```

#### The capability variant, in full

The same feature sets under the looser arity-3 grammar. It is **not** the study's recommendation and not what the term-by-term analysis above is about; it exists so that the published equation's accuracy can be read against what the additive *form* can do, rather than only against oracles and baselines. It is printed here in full because a ceiling quoted as a number and never shown is a ceiling a reader has to take on trust -- and because the reason it is not recommended is visible only in the reading: 14 terms over three-feature expressions is past the point where the equation can be reasoned about a term at a time, which is the whole thing this study is trading accuracy for.

It reaches **0.6551** in-sample against the published equation's 0.6408, and **0.6266** leave-one-dataset-out against 0.6234.

**E3 capability** (14 terms):

```
MCC = +1.52115
      -0.146517 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.1868
      +0.0065022 * [log(gravity)] * [log(Model Capability)]       # beta=+0.1278
      -0.047455 * ([log(nr_attr)] + [log(Input Distribution Modelling)]) / [log(nr_class)]  # beta=-0.1154
      -0.0465282 * ([log(gravity)] + [log(nr_class)]) / [log(Processing Units Number)]  # beta=-0.0972
      -0.0254783 * ([log(gravity)] + [log(nr_class)]) / [log(nr_attr)]  # beta=-0.0961
      -0.294034 * ([log(nr_class)] + [log(ns_ratio)]) / [log(Processing Units Number)]  # beta=-0.0891
      -0.47443 * ([log(class_ent)] + [log(Fitting Regime)]) / [log(Processing Units Number)]  # beta=-0.0851
      +0.220741 * [log(class_ent)] / [log(inst_to_attr)]          # beta=+0.0771
      -0.0062342 * [log(gravity)] * [log(Fitting Regime)]         # beta=-0.0748
      +0.0694345 * ([log(inst_to_attr)] + [log(Input Distribution Modelling)]) / [log(Processing Units Number)]  # beta=+0.0517
      +0.000861956 * [nr_bin] * [log(Model Capability)]           # beta=+0.0414
      +0.0219926 * [log(eq_num_attr)] * [log(Model Capability)]   # beta=+0.0369
      -0.0140379 * ([nr_cor_attr] + [log(Processing Units Number)]) / [Fitting Regime]  # beta=-0.0323
      -2.54045 * [nr_cor_attr] / [log(Processing Units Number)]   # beta=-0.0304
```

LaTeX:

```latex
\mathrm{MCC} = +1.521 -0.1465 \cdot \mathrm{[log(eq\_num\_attr)] * [log(nr\_class)]} +0.006502 \cdot \mathrm{[log(gravity)] * [log(Model Capability)]} -0.04745 \cdot \mathrm{([log(nr\_attr)] + [log(Input Distribution Modelling)]) / [log(nr\_class)]} -0.04653 \cdot \mathrm{([log(gravity)] + [log(nr\_class)]) / [log(Processing Units Number)]} -0.02548 \cdot \mathrm{([log(gravity)] + [log(nr\_class)]) / [log(nr\_attr)]} -0.294 \cdot \mathrm{([log(nr\_class)] + [log(ns\_ratio)]) / [log(Processing Units Number)]} -0.4744 \cdot \mathrm{([log(class\_ent)] + [log(Fitting Regime)]) / [log(Processing Units Number)]} +0.2207 \cdot \mathrm{[log(class\_ent)] / [log(inst\_to\_attr)]} -0.006234 \cdot \mathrm{[log(gravity)] * [log(Fitting Regime)]} +0.06943 \cdot \mathrm{([log(inst\_to\_attr)] + [log(Input Distribution Modelling)]) / [log(Processing Units Number)]} +0.000862 \cdot \mathrm{[nr\_bin] * [log(Model Capability)]} +0.02199 \cdot \mathrm{[log(eq\_num\_attr)] * [log(Model Capability)]} -0.01404 \cdot \mathrm{([nr\_cor\_attr] + [log(Processing Units Number)]) / [Fitting Regime]} -2.54 \cdot \mathrm{[nr\_cor\_attr] / [log(Processing Units Number)]}
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
instead of describing it is worth what the generated ceiling at the foot of this chapter
measures — which is at most what a *perfect* set of extra descriptors would be worth, since
free per-model numbers are the best any descriptor set could do at telling these 25 models
apart. It is also why a table is not a substitute for
descriptors: it transfers to a new dataset and not to a new model, while a descriptor would
do both.

### The bound on what better model descriptors could buy

*Implemented in `ml_meta_perf.identity`. It measures the headroom the study cannot reach with
the descriptors the corpus records, which is a limitation rather than a result.*

The chapter above measures a gap and stops there: the dataset meta-features recover almost
all of what dataset identity explains, and the model meta-features a clearly smaller share of
what model identity explains — the `reached` column of the generated headline has both. What
is left is written down
nowhere in this corpus — the features record what a model *is*, its capacity and five
asserted facts about how it is built, but nothing about what it is good at.

**That gap was a third when this chapter was written, and it is now an eighth.** The four
asserted ordinals merged on 2026-09-05 closed most of it, which is why the ceiling measured
below is much smaller than the text originally described.

The interaction ladder measures the same gap from the other side: the first interaction
component is worth substantially more than the equation reaches, and the equation recovers
about a third of that pattern in-sample and a quarter out of fold — measured, not inferred
from an R² comparison. Both numbers are in the generated section above.

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

The measurement is in the generated section above — both rungs, on
this run, against the equation the study publishes. A hand-written copy stood here and had
drifted badly enough to invert the chapter's conclusion: it recorded the two rungs as
*identical*, which no run of `identity.correct_out_of_fold` can produce, and reported a gap
of +0.017 where the measurement gives roughly three times that.

**And the answer is that the question is not closed.** The gap is not merely larger than the
chapter recorded; the half of it that matters survives the study's own paired test. Split into
its two rungs and compared with the uncorrected equation dataset by dataset:

- a per-model **level** — a constant shift for each learner — is a **tie**. The equation
  already has whatever that would add.
- a per-model **slope**, which lets a learner's advantage depend on the data, is **not**. Its
  bootstrap interval lies entirely above zero.

Read the second with its sign test beside it, as the generated section does: the sign test
does not reach significance, so the gain comes from its size on the folds it wins rather than
from winning nearly all of them. That is weaker evidence than a bare "significant" implies and
clearly stronger than a tie.

The current pooled gap is 0.055 of leave-one-dataset-out R²: the level recovers 0.022 and
the level-plus-slope form the remaining 0.033. **The residual is specifically the interaction
the mixed terms were supposed to absorb**, and they have absorbed only part of it.

Free per-model numbers are the best any descriptor set could do at telling these 25 models
apart — they are model identity itself, fitted out of fold — so what they add on top of the
equation is exactly the information the descriptors are missing. The rung that matters is the
**slope**: a level shifts every one of a model's rows by the same amount, which is a
per-model constant, while a slope lets a model's advantage depend on the data. That is what a
*capability* descriptor would have to do, and it is the half of the gap the level cannot
reach. It is also why the negative results in [the limitations](#limitations-of-the-equation)
all fail the same way.

**The slope still adds more than the level does**, which is the reading that had been lost.
A per-model slope on a dataset feature is an interaction, and the hope behind the mixed terms
was that expressing it directly would leave the slope nothing to recover; the generated table
says otherwise. That is the honest statement of where the study stands: the mixed terms
capture some of this interaction and not the part that varies per model.

The rank columns of this table are dropped rather than refreshed, because the ranking
comparison they fed now lives in [chapter 5](05-evaluation.md).

### Why the model side, specifically

The generated identity table isolates this directly. A free per-model level is a tie, while
a per-model slope on a dataset feature adds a measurable gain. The missing information is
therefore not just a constant ranking of learners; it is how a learner's advantage changes
with the dataset. The interaction-alignment table reaches the same conclusion from the
equation itself: its mixed terms recover part of the leading dataset-by-model pattern, with
substantial headroom remaining.

The current experiment does not fit a separate latent-covariate model, a backfitted model,
or a pairwise ranking objective. Numerical comparisons for those exploratory variants are
therefore outside the reported results.

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

What survives is the **measurement**: the generated per-model identity ceiling as the upper
bound on better model descriptors. It belongs to this chapter's argument about thin model
descriptors and is recomputed on every run.

`ml_meta_perf.identity` was for a long time tested and wired into no pipeline, and the
consequence was that the ceiling it measures was hand-copied into this chapter and went
stale by a factor of three. It is now run on every study:
`identity_ceiling` calls `identity.correct_out_of_fold` against the finished
`cross_validate_fixed_form` path, so the number costs no extra search and cannot drift from
the equation it is a ceiling for.
