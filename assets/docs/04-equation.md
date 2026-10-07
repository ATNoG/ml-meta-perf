# 4. The equation

*Produced by `ml_meta_perf.experiment`; reproduce with `PYTHONPATH=src venv/bin/python -m ml_meta_perf`.*

The abbreviations used in this chapter are Matthews Correlation Coefficient (**MCC**),
coefficient of determination (**R²**), in-sample (**IS**), leave-one-dataset-out (**LODO**),
leave-one-model-out (**LOMO**), doubly held out (**DHO**), singular value decomposition (**SVD**), additive main effects
and multiplicative interaction (**AMMI**), machine learning (**ML**), and neural network
(**NN**).
The equation labels are **E1** (dataset features only), **E2** (model features only), **E3**
(dataset and model features), **E3-Valid** (the plateau-selected E3 equation), and
**E3-MAX** (the maximum-capability E3 equation).

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
rows, evaluated under the same four protocols from **one** configuration
(`config/study.json`), and have their length chosen by **one** rule — the first sustained
plateau of the worst-protocol curve ([chapter 3](03-term-selection.md#stage-3-choosing-the-number-of-terms)).
Each searches every feature it is allowed to use.
`experiment.run_equation` is that function and `run_e1`, `run_e2` and `run_e3` are one
line each. The uniformity is not tidiness: the gaps between the three are only evidence
about what each half of the meta-data is worth if *nothing else* differs between them.

**All three are fitted on the 476 rows, not on aggregated group means.** Fitting E1 on the
20 per-dataset means is tempting — a predictor constant inside a group can only predict that
group's mean anyway — but it puts E1's R² on a twenty-point denominator:

| E1 fitted on | IS R² | LODO R² | comparable with E3? |
|---|---|---|---|
| 20 dataset means | 0.337 | 0.506 *(on 20 points)* | no |
| **476 rows** | **0.354** | **0.349** *(on 476 rows)* | yes |

Two R² values computed over different row sets share no denominator. Printed in one table
they invite the conclusion that the dataset-only control transfers *better* than the full
equation, which on the common scale it does not. Fitting on all rows costs nothing and
removes the caveat instead of requiring one.

**Aggregating E2 the same way was measured, and is worse.** The mirror move is to fit E2
on 25 per-model means:

| E2 fitted on | IS R² (476 rows) | best out-of-fold | terms at the optimum |
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

Each equation, each equation's own ceiling, and the additive mean-based reference are in one generated
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
[chapter 5](05-evaluation.md#four-protocols) defines: IS, LODO,
LOMO, and the DHO cell. The numbers are on the
[index page](index.md) and, with the full metric set, in [chapter 5](05-evaluation.md) —
this chapter does not keep a second copy of them.

What is worth stating here is what to look for in them.

**E3's transfer numbers should be close to each other and to its fit, and the gap between the
best and the worst is a reported quantity.** `selection.protocol_spread` is exactly that
gap — IS minus the floor over the four — and for the published equation it is 0.0684.
An equation that loses little R² when a whole dataset, a whole learner, or both are withheld
is transferring rather than memorising; a small spread says no single protocol is carrying
it. E3-Valid has a spread of 0.0684; E3-MAX is reported separately as the capability bound.

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

*Implemented in `ml_meta_perf.validate.additive_mean_reference`, `interaction_oracle` and
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

**This does not bound E3-Valid, and the current E3-Valid equation exceeds it.** The additive mean-based reference
bounds a predictor that is a per-dataset value *plus* a per-model value. Most of E3-Valid's terms
are *mixed* — each
multiplying or dividing a dataset feature by a model feature — and those express precisely
the interaction the two-way additive form cannot. E3-Valid reaches 0.679 under IS against the
reference value of 0.661. Its direct alignment with the interaction components of that reference, reported below,
is therefore the evidence that the mixed
terms capture part of that structure.

### Ceiling 2 — what the vocabulary can reach

The additive mean-based reference bounds a sum of *group* effects. A different and equally useful bound is a sum
of *per-feature* functions — what an equation could explain if it never combined two
features in one term. `analysis.grammar_ceiling` computes it as three least-squares fits
over the term library, none of which needs the search to have run. The ladder — raw columns,
the best single-feature term per feature, then every single-feature term at once — is in the
generated section below, with the fitted equations beside it. Two readings:

**No single feature carries the equation.** The strongest reaches R² of about 0.14 on its
own. There is no dominant driver to quote, which is why the equation needs many terms
rather than two, and the per-feature table in the generated section is where to check it.

**The transforms earn the gap between entering the raw columns and taking the best
single-feature term of each.** `analysis.feature_reach` reports that per feature, and it is
concentrated where a straight line was the wrong shape — `class_ent` and `inst_to_attr` are
each worth an order of magnitude more as reciprocals than raw, and `gravity` under a log. For
a good third of the features the grammar buys nothing at all: the raw column was already the
best form of it.

**E3 passes this ceiling too.**
An equation cannot exceed it by describing features one at a time, so the excess is the
work the cross-feature terms do. That is the same conclusion the additive mean-based reference reaches,
by an entirely independent route — one argues from group means, the other from the term
library — which is worth more than either on its own.

### Ceiling 3 — how fast interaction pays

The residual of the additive mean-based reference is not noise. It is a (dataset × model) matrix of
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
ends but **how fast it climbs**: the first interaction component alone is worth +0.122 R².

Does the equation reach any of it? Comparing two R² values cannot say — a number below
rank 0 is equally consistent with an equation that misses the pattern and one that finds it
but is inaccurate elsewhere. Comparing the two interaction *structures* can.
`validate.interaction_capture` lays both the truth and the equation's predictions on the
(dataset × model) grid, strips each of its own additive part, and correlates what is left:

| protocol | alignment with rank 1 | with ranks 1–2 |
|---|---|---|
| IS | 0.37 | 0.32 |
| LODO | 0.33 | 0.26 |

The equation reaches **about a third** of the leading interaction pattern under both readings.
Interaction is 36% of
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

The additive mean-based reference is the third reference, and it is **not** a ceiling for E3: ten of the
eighteen terms combine a dataset feature with a model feature, and such a term
expresses interaction a two-way additive form by construction cannot. The two group-identity
levels are hard ceilings; the additive reference is a level that a mixed equation can pass.

**Both halves are now close to their own ceilings.** Twelve dataset meta-features exhaust what
dataset identity can explain to the reported precision, so there is essentially nothing left
for a better dataset descriptor to find. The six model features reach 92% of what model
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
[limitations](#model-descriptors-are-thin-and-five-of-them-are-asserted) are where that is paid
for.

Three qualifications keep that from being oversold, all of them developed there: the ladder
is asserted rather than measured, so it cannot be
evidence for the prior knowledge it encodes; the corpus agrees with it at only 6 of 9 steps,
with `generic NN` conspicuously misplaced; and it does not extend to a learner nobody has
classified, which is what the fall in LOMO records.

Collecting more meta-features of the *data* remains low priority — E1 is already close to
what perfect knowledge of dataset identity would buy (the `reached` column of the headline).

Two things create the opposite impression and are worth stating explicitly, because both
are artefacts:

- The **within-group correlation table is dataset-centred**, which removes dataset variance
  by construction. Only model terms can score well there. It is a diagnostic for model
  effects, not a statement of relative importance.
- **Model features carry more in combination than alone.** Adding them to E1 is worth
  more than E2 achieves by itself (compare the E1, E2 and E3-Valid rows of the headline). The
  surplus is dataset×model interaction, which is why so many of E3's terms are mixed; the
  count and their share of the weight mass are generated at the end of this chapter.

## One configuration, two roles

* **E3-Valid** is the equation the study publishes. It uses the shared configuration and the
  shared length rule, exactly as E1 and E2 do.
* **E3-MAX** is the capability measurement: the same configuration under the wider grammar
  (arity 3, ratio-of-sums terms), at the raw maximum of its worst-protocol curve. It shows how
  far the additive form reaches when it is not asked to stay readable.

Downstream prediction, threshold, and model-ranking analyses use only E3-Valid.

The configuration — stability cap, ridge penalty, arity, pool, beam and horizon — is the one
the configuration sweep proposed ([chapter 3](03-term-selection.md#how-the-configuration-itself-was-chosen)):
among the configurations whose equation is readable and tied on R², the one whose
doubly-held-out predictions rank the models best. It holds hyperparameters only. The model
feature pool is all six descriptors; which of them the equation keeps is the search's
decision, not the configuration's.

## The fitted equations

E1, E2 and E3-Valid are printed in full in the generated sections at the end of this chapter,
with E3-Valid's term-importance table and analysis and a count of the terms the controls share
with it — rewritten on every run, so they cannot drift out of step with the code the way a
hand-copied listing would.

**Almost nothing of the controls survives into E3, and that is the honest reading of the
controls.** E2's terms are built to say as much as possible using *only* model features, so
they lean on ratios between the model columns themselves. Once a dataset feature is available
to pair with, none of those pairings is the best use of a term slot, and E3 rebuilds the model
side from scratch against dataset partners.

This does not weaken the controls; it clarifies what they measure. E1 and E2 bound **how much
of MCC each half of the meta-data explains**, and they do that whether or not their particular
terms reappear. What the overlap does show is that the equation is not a concatenation of its
two halves: a mixed term is available to neither control by construction.

The weights are flat: they behave like far more equally-weighted terms than any headline
reading would suggest. The generated section gives the current figures — an inverse Simpson
index over the standardised weight shares, and the largest term's share — rather than
restating them here, because they move with every configuration.

## Reading the equation: what interpretability buys and what it costs

The equation is not the most accurate predictor available on this meta-data — an opaque
regressor reaches R² ≈ 0.9 under IS, and [chapter 5](05-evaluation.md) shows what happens
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

IS predictions never fall below **0.01**, while 15 rows sit at exactly MCC = 0 — a short
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

E3-Valid uses **17 terms** over dataset and model meta-features, simplified and refitted after pruning, so it evaluates exactly as printed.

```
MCC = +1.76536
      -0.194111 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.2475
      -0.106651 * [log(nr_attr)] / [log(nr_class)]                # beta=-0.2461
      +0.11274 * [log(eq_num_attr)] * [log(Model Capability)]     # beta=+0.1892
      -0.0482114 * [log(gravity)] / [log(nr_attr)]                # beta=-0.1784
      +0.0236479 * [log(gravity)] / [Fitting Regime]              # beta=+0.1760
      +0.00840496 * [log(gravity)] * [log(Model Capability)]      # beta=+0.1653
      -0.0575455 * [log(gravity)] / [log(inst_to_attr)]           # beta=-0.1621
      -0.190385 * [log(eq_num_attr)] / [Input Distribution Modelling]  # beta=-0.1585
      -0.022032 * [log(nr_inst)] * [log(Input Distribution Modelling)]  # beta=-0.1258
      -0.0141436 * [log(gravity)] / [Solution Stochasticity]      # beta=-0.1185
      -0.0167132 * [log(inst_to_attr)] * [log(Model Capability)]  # beta=-0.1129
      -0.812793 * [log(nr_class)] / [log(Processing Units Number)]  # beta=-0.0948
      -0.0439052 * [log(gravity)] / [log(Processing Units Number)]  # beta=-0.0906
      +0.00341741 * [nr_bin] / [Input Distribution Modelling]     # beta=+0.0892
      +0.280505 * 1/Solution Stochasticity                        # beta=+0.0852
      +0.00125795 * [log(nr_attr)] * [nr_norm]                    # beta=+0.0609
      +0.000957842 * [log(inst_to_attr)] * [log(nr_inst)]         # beta=+0.0438
```

LaTeX:

```latex
\mathrm{MCC} = +1.765 -0.1941 \cdot \mathrm{[log(eq\_num\_attr)] * [log(nr\_class)]} -0.1067 \cdot \mathrm{[log(nr\_attr)] / [log(nr\_class)]} +0.1127 \cdot \mathrm{[log(eq\_num\_attr)] * [log(Model Capability)]} -0.04821 \cdot \mathrm{[log(gravity)] / [log(nr\_attr)]} +0.02365 \cdot \mathrm{[log(gravity)] / [Fitting Regime]} +0.008405 \cdot \mathrm{[log(gravity)] * [log(Model Capability)]} -0.05755 \cdot \mathrm{[log(gravity)] / [log(inst\_to\_attr)]} -0.1904 \cdot \mathrm{[log(eq\_num\_attr)] / [Input Distribution Modelling]} -0.02203 \cdot \mathrm{[log(nr\_inst)] * [log(Input Distribution Modelling)]} -0.01414 \cdot \mathrm{[log(gravity)] / [Solution Stochasticity]} -0.01671 \cdot \mathrm{[log(inst\_to\_attr)] * [log(Model Capability)]} -0.8128 \cdot \mathrm{[log(nr\_class)] / [log(Processing Units Number)]} -0.04391 \cdot \mathrm{[log(gravity)] / [log(Processing Units Number)]} +0.003417 \cdot \mathrm{[nr\_bin] / [Input Distribution Modelling]} +0.2805 \cdot \mathrm{1/Solution Stochasticity} +0.001258 \cdot \mathrm{[log(nr\_attr)] * [nr\_norm]} +0.0009578 \cdot \mathrm{[log(inst\_to\_attr)] * [log(nr\_inst)]}
```

## How far the form could reach

Two ceilings, both computed from the library alone and so available *before* an equation exists. Each is an expectation the fitted equation is then held against, rather than a number read off it.

#### How far the additive form reaches

The published equation uses arity 2. The same configuration under arity 3, at the raw maximum of its worst-protocol curve (E3-MAX), reaches 29 terms at R² 0.7271 under IS:

| | terms | IS | LODO | LOMO | DHO | form stability |
|---|---|---|---|---|---|---|
| E3-Valid (arity 2) | 17 | 0.6815 | 0.6513 | 0.6261 | 0.6151 | 0.35 |
| E3-MAX (arity 3) | 29 | 0.7271 | 0.6894 | 0.6538 | 0.6482 | 0.26 |

This is a **capability measurement, not a recommendation**. It answers the question the published equation cannot answer about itself — whether the additive form is out of room or whether this equation is short of it — and the answer is that +0.0456 of IS R² is still available to a longer equation over a wider grammar. Form stability is the mean share of LODO folds that re-select an equation's own terms when selection is re-run without the held-out dataset.

#### What the vocabulary could reach, before any search

Three levels of what the vocabulary can explain, each a least-squares fit over the library and each computable before the search runs. They bound a *sum of per-feature functions*, which is a different question from the additive mean-based reference above: that one bounds a per-dataset value plus a per-model value.

| level | terms | R² |
|---|---|---|
| every raw term admitted by the grammar | 11 | 0.4016 |
| the best single-feature term per feature | 18 | 0.5324 |
| every single-feature term at once | 55 | 0.6321 |
| **the fitted equation (E3-Valid)** | **17** | **0.6815** |

No individual feature carries much: the strongest is `eq_num_attr` at R² 0.142, so any accuracy beyond that is combination rather than a single dominant driver. Taking one best admissible term per feature is worth +0.131 over the admissible raw-term fit.

E3-Valid reaches 0.6815 with 17 terms, **above** the 0.6321 that all 55 single-feature terms reach together. An equation cannot pass that level by describing features one at a time, so the excess is what the cross-feature terms buy — the same conclusion the additive mean-based reference reaches, by an independent route.

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
| IS | 1 | 0.3822 | 0.3757 | 0.3647 |
| IS | 2 | 0.3320 | 0.5658 | 0.3647 |
| LODO | 1 | 0.3405 | 0.3757 | 0.3647 |
| LODO | 2 | 0.2677 | 0.5658 | 0.3647 |

## Equation analysis

The equation has 17 terms, of which **11** carry 80% of the standardised weight mass; the single largest carries 10.6%, and the weights behave like **14.5 equally-weighted terms** (inverse Simpson index of the shares).

That last number is the one to read for concentration, because it does not depend on where a threshold is drawn. At 85% of the term count the equation is **flat**: no single term dominates. That is a statement about the *unit of explanation*, not about the quality of the equation — MCC here is inferred by a set of terms acting together rather than by one or two that could be quoted on their own. Three readings follow, and the sections below give each one: read the terms in the blocks that move together, read which features the search reached for, and read which operations it needed to apply to them.

`beta` is the standardised weight — the MCC contributed per standard deviation of the term, which is what makes terms in unrelated units comparable. `effect` is the swing in predicted MCC across the middle 80% of the term's observed range. `stability` is the fraction of LODO folds that selected the term.

| rank | term | group | features | weight | beta | effect | share | cumulative | major | stability | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [log(eq_num_attr)] * [log(nr_class)] | dataset | eq_num_attr, nr_class | -0.1941 | -0.2475 | 0.8317 | 0.1055 | 0.1055 | yes | 0.4500 | lowers MCC |
| 2 | [log(nr_attr)] / [log(nr_class)] | dataset | nr_attr, nr_class | -0.1067 | -0.2461 | 0.5951 | 0.1050 | 0.2105 | yes | 0.0500 | lowers MCC |
| 3 | [log(eq_num_attr)] * [log(Model Capability)] | mixed | eq_num_attr, Model Capability | 0.1127 | 0.1892 | 0.4828 | 0.0807 | 0.2912 | yes | 0.6500 | raises MCC |
| 4 | [log(gravity)] / [log(nr_attr)] | dataset | gravity, nr_attr | -0.0482 | -0.1784 | 0.5190 | 0.0761 | 0.3673 | yes | 0.3500 | lowers MCC |
| 5 | [log(gravity)] / [Fitting Regime] | mixed | gravity, Fitting Regime | 0.0236 | 0.1760 | 0.3770 | 0.0750 | 0.4424 | yes | 0.9500 | raises MCC |
| 6 | [log(gravity)] * [log(Model Capability)] | mixed | gravity, Model Capability | 0.0084 | 0.1653 | 0.4244 | 0.0705 | 0.5128 | yes | 0.8500 | raises MCC |
| 7 | [log(gravity)] / [log(inst_to_attr)] | dataset | gravity, inst_to_attr | -0.0575 | -0.1621 | 0.2024 | 0.0692 | 0.5820 | yes | 0.0000 | lowers MCC |
| 8 | [log(eq_num_attr)] / [Input Distribution Modelling] | mixed | eq_num_attr, Input Distribution Modelling | -0.1904 | -0.1585 | 0.4343 | 0.0676 | 0.6496 | yes | 0.0500 | lowers MCC |
| 9 | [log(nr_inst)] * [log(Input Distribution Modelling)] | mixed | nr_inst, Input Distribution Modelling | -0.0220 | -0.1258 | 0.2860 | 0.0536 | 0.7032 | yes | 0.3500 | lowers MCC |
| 10 | [log(gravity)] / [Solution Stochasticity] | mixed | gravity, Solution Stochasticity | -0.0141 | -0.1185 | 0.2381 | 0.0505 | 0.7537 | yes | 0.5500 | lowers MCC |
| 11 | [log(inst_to_attr)] * [log(Model Capability)] | mixed | inst_to_attr, Model Capability | -0.0167 | -0.1129 | 0.3011 | 0.0481 | 0.8019 | yes | 0.1500 | lowers MCC |
| 12 | [log(nr_class)] / [log(Processing Units Number)] | mixed | nr_class, Processing Units Number | -0.8128 | -0.0948 | 0.2464 | 0.0404 | 0.8423 | no | 0.4000 | lowers MCC |
| 13 | [log(gravity)] / [log(Processing Units Number)] | mixed | gravity, Processing Units Number | -0.0439 | -0.0906 | 0.2288 | 0.0386 | 0.8810 | no | 0.6000 | lowers MCC |
| 14 | [nr_bin] / [Input Distribution Modelling] | mixed | nr_bin, Input Distribution Modelling | 0.0034 | 0.0892 | 0.0854 | 0.0380 | 0.9190 | no | 0.4500 | raises MCC |
| 15 | 1/Solution Stochasticity | model | Solution Stochasticity | 0.2805 | 0.0852 | 0.2104 | 0.0363 | 0.9553 | no | 0.1000 | raises MCC |
| 16 | [log(nr_attr)] * [nr_norm] | dataset | nr_attr, nr_norm | 0.0013 | 0.0609 | 0.0800 | 0.0260 | 0.9813 | no | 0.0000 | raises MCC |
| 17 | [log(inst_to_attr)] * [log(nr_inst)] | dataset | inst_to_attr, nr_inst | 0.0010 | 0.0438 | 0.1359 | 0.0187 | 1.0000 | no | 0.0000 | raises MCC |

#### The 8 largest terms, in words

(11 terms are flagged major; the leading 8 are written out, and the table above carries the rest.)

1. `[log(eq_num_attr)] * [log(nr_class)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.247 MCC, it moves predicted MCC by 0.832 across the middle 80% of its observed range, it carries 10.6% of the equation's weight mass, and it was selected in 45% of folds.
2. `[log(nr_attr)] / [log(nr_class)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.246 MCC, it moves predicted MCC by 0.595 across the middle 80% of its observed range, it carries 10.5% of the equation's weight mass, and it was selected in 5% of folds.
3. `[log(eq_num_attr)] * [log(Model Capability)]` (mixed) raises MCC: one standard deviation of this term is worth +0.189 MCC, it moves predicted MCC by 0.483 across the middle 80% of its observed range, it carries 8.1% of the equation's weight mass, and it was selected in 65% of folds.
4. `[log(gravity)] / [log(nr_attr)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.178 MCC, it moves predicted MCC by 0.519 across the middle 80% of its observed range, it carries 7.6% of the equation's weight mass, and it was selected in 35% of folds.
5. `[log(gravity)] / [Fitting Regime]` (mixed) raises MCC: one standard deviation of this term is worth +0.176 MCC, it moves predicted MCC by 0.377 across the middle 80% of its observed range, it carries 7.5% of the equation's weight mass, and it was selected in 95% of folds.
6. `[log(gravity)] * [log(Model Capability)]` (mixed) raises MCC: one standard deviation of this term is worth +0.165 MCC, it moves predicted MCC by 0.424 across the middle 80% of its observed range, it carries 7.0% of the equation's weight mass, and it was selected in 85% of folds.
7. `[log(gravity)] / [log(inst_to_attr)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.162 MCC, it moves predicted MCC by 0.202 across the middle 80% of its observed range, it carries 6.9% of the equation's weight mass, and it was selected in 0% of folds.
8. `[log(eq_num_attr)] / [Input Distribution Modelling]` (mixed) lowers MCC: one standard deviation of this term is worth -0.158 MCC, it moves predicted MCC by 0.434 across the middle 80% of its observed range, it carries 6.8% of the equation's weight mass, and it was selected in 5% of folds.

#### Large terms the folds disagreed on

7 of the major terms were selected by fewer than half of the folds. A large weight and a low selection frequency together mean the term is doing its work for *this* training set and would be replaced by something else on another, which the equation as printed does not show. **Do not build guidance on these.**

| rank | term | group | beta | share | stability |
|---|---|---|---|---|---|
| 1 | [log(eq_num_attr)] * [log(nr_class)] | dataset | -0.2475 | 0.1055 | 0.4500 |
| 2 | [log(nr_attr)] / [log(nr_class)] | dataset | -0.2461 | 0.1050 | 0.0500 |
| 4 | [log(gravity)] / [log(nr_attr)] | dataset | -0.1784 | 0.0761 | 0.3500 |
| 7 | [log(gravity)] / [log(inst_to_attr)] | dataset | -0.1621 | 0.0692 | 0.0000 |
| 8 | [log(eq_num_attr)] / [Input Distribution Modelling] | mixed | -0.1585 | 0.0676 | 0.0500 |
| 9 | [log(nr_inst)] * [log(Input Distribution Modelling)] | mixed | -0.1258 | 0.0536 | 0.3500 |
| 11 | [log(inst_to_attr)] * [log(Model Capability)] | mixed | -0.1129 | 0.0481 | 0.1500 |

#### Reading the terms in blocks

An additive form invites reading one term at a time, and that works when one or two weights dominate. When they do not, the honest unit is larger than a term and smaller than the equation: terms whose per-row contributions move together say the same thing about a row and can be read as one block. Grouping is on the contributions rather than on shared features, because two terms can share no feature and still track each other.

11 blocks over 17 terms, the largest holding 2 terms and 17% of the weight mass.

| group | n_terms | share | effect | direction | shared | terms |
|---|---|---|---|---|---|---|
| 1 | 2 | 0.1741 | 0.7287 | lowers MCC |  | [log(gravity)] / [log(inst_to_attr)] ; [log(nr_attr)] / [log(nr_class)] |
| 2 | 2 | 0.1731 | 0.8311 | lowers MCC | eq_num_attr | [log(eq_num_attr)] * [log(nr_class)] ; [log(eq_num_attr)] / [Input Distribution Modelling] |
| 3 | 3 | 0.1653 | 0.9309 | lowers MCC | gravity | [log(gravity)] / [log(nr_attr)] ; [log(gravity)] / [log(Processing Units Number)] ; [log(gravity)] / [Solution Stochasticity] |
| 4 | 2 | 0.0845 | 0.4828 | lowers MCC |  | 1/Solution Stochasticity ; [log(inst_to_attr)] * [log(Model Capability)] |
| 5 | 1 | 0.0807 | 0.4828 | raises MCC | Model Capability, eq_num_attr | [log(eq_num_attr)] * [log(Model Capability)] |
| 6 | 1 | 0.0750 | 0.3770 | raises MCC | Fitting Regime, gravity | [log(gravity)] / [Fitting Regime] |
| 7 | 1 | 0.0705 | 0.4244 | raises MCC | Model Capability, gravity | [log(gravity)] * [log(Model Capability)] |
| 8 | 2 | 0.0640 | 0.1654 | raises MCC |  | [log(nr_attr)] * [nr_norm] ; [nr_bin] / [Input Distribution Modelling] |
| 9 | 1 | 0.0536 | 0.2860 | lowers MCC | Input Distribution Modelling, nr_inst | [log(nr_inst)] * [log(Input Distribution Modelling)] |
| 10 | 1 | 0.0404 | 0.2464 | lowers MCC | Processing Units Number, nr_class | [log(nr_class)] / [log(Processing Units Number)] |
| 11 | 1 | 0.0187 | 0.1359 | raises MCC | inst_to_attr, nr_inst | [log(inst_to_attr)] * [log(nr_inst)] |

#### Which features the search reached for

**13 of 18** available meta-features appear in the equation. `share` sums the weight mass of every term a feature appears in, so a feature in two terms is credited both and the column does not sum to 1 — it answers how much of the equation touches this feature, not how much it owns. A feature the search declined to use after seeing every transform of it is itself a result.

| feature | meaning | n_terms | share | transforms | operations |
|---|---|---|---|---|---|
| gravity | gravity (separation between the majority and minority class centres) | 6 | 0.3799 | log | product, ratio |
| eq_num_attr | equivalent number of attributes (effective feature count) | 3 | 0.2538 | log | product, ratio |
| nr_class | number of classes | 3 | 0.2510 | log | product, ratio |
| nr_attr | number of attributes | 3 | 0.2070 | log | product, ratio |
| Model Capability | learner family's capability rank in the tabular-ML literature (1-10) | 3 | 0.1993 | log | product |
| Input Distribution Modelling | how much of the input distribution the learner models (1-5) | 3 | 0.1593 | id, log | product, ratio |
| inst_to_attr | instances per attribute | 3 | 0.1360 | log | product, ratio |
| Solution Stochasticity | how deep randomisation reaches into the fit (1-5) | 2 | 0.0869 | id, inv | atom, ratio |
| Processing Units Number | model capacity (number of fitted processing units) | 2 | 0.0791 | log | ratio |
| Fitting Regime | how the parameters are reached, closed form to in-context (1-5) | 1 | 0.0750 | id | ratio |
| nr_inst | number of instances in the source dataset (before sampling) | 2 | 0.0723 | log | product |
| nr_bin | number of binary attributes | 1 | 0.0380 | id | ratio |
| nr_norm | number of normally distributed attributes | 1 | 0.0260 | id | product |
| class_ent | class entropy (how evenly the labels are spread) | 0 | 0.0000 |  |  |
| nr_cor_attr | proportion of correlated attribute pairs | 0 | 0.0000 |  |  |
| nr_outliers | number of attributes containing outliers | 0 | 0.0000 |  |  |
| ns_ratio | noise-to-signal ratio | 0 | 0.0000 |  |  |
| Loss Margin Behaviour | how hard the loss penalises points far from the boundary (1-5) | 0 | 0.0000 |  |  |

#### Which operations the equation needed

The vocabulary offers five operations and five transforms and the search is free to ignore any of them, so a row that was offered and went unused is a shape this data turned out not to need. Rows marked `offered = no` were kept out of the library by the arity cap and say nothing about the data:

| kind | name | offered | n_terms | share |
|---|---|---|---|---|
| operation | atom | yes | 1 | 0.0363 |
| operation | ratio | yes | 9 | 0.5605 |
| operation | product | yes | 7 | 0.4032 |
| operation | sum_ratio | no | 0 | 0.0000 |
| operation | ratio_of_sums | no | 0 | 0.0000 |
| transform | id | yes | 5 | 0.2572 |
| transform | log | yes | 15 | 0.9256 |
| transform | sqrt | yes | 0 | 0.0000 |
| transform | inv | yes | 1 | 0.0363 |
| transform | sq | yes | 0 | 0.0000 |

#### Where the equation's variance comes from

- terms over dataset meta-features alone (how hard is this data): 6 terms, 68.4% of the equation's output variance.
- terms over model meta-features alone (how capable is this model): 1 term, -7.5% of the equation's output variance.
- terms mixing dataset and model features (which model suits which data): 10 terms, 39.1% of the equation's output variance.

| group | n_terms | share | effect_sum |
|---|---|---|---|
| dataset | 6 | 0.6839 | 0.9510 |
| model | 1 | -0.0753 | 0.2104 |
| mixed | 10 | 0.3914 | 0.7293 |

## The ceiling on model descriptors

Under LODO every model appears in every training fold, so the equation's residual can be averaged per model on the training rows and applied to the held-out dataset with no leak. That replaces the model descriptors with the best possible substitute -- the model's **identity**, fitted freely -- so what it adds is a ceiling on what any descriptor set could reach by telling these classifiers apart.

| correction | r2_LODO | mae | gain | ci_low | ci_high | sign_p | wins | verdict |
|---|---|---|---|---|---|---|---|---|
| none (E3-Valid, 17 terms) | 0.6513 | 0.1368 | 0.0000 |  |  |  | 0 | baseline |
| per-model level | 0.6708 | 0.1342 | 0.0020 | -0.0034 | 0.0077 | 0.8238 | 11 | tie |
| per-model level and slope | 0.6918 | 0.1296 | 0.0068 | -0.0013 | 0.0165 | 1.0000 | 10 | tie |

**The gap is 0.040 of LODO R²**, of which a per-model *level* recovers 0.019 and the level-plus-slope form the remaining 0.021.

**Whether that is real is a paired question**, so each rung is compared with the uncorrected equation dataset by dataset, on absolute error, over the twenty held-out folds. The two rungs come back differently, and the difference is the finding:

* a per-model **level** is a **tie** -- it wins on 11 of the twenty folds and its interval spans zero. A constant shift per model, which is what a level is, adds nothing the equation does not already have.

* neither rung survives pairing, so on this corpus perfect model identity adds nothing measurable to the published equation and the model side is as well described as free per-model numbers could make it.

Every model-side encoding the study tried and rejected was rejected for failing to recover this gap, so it is a property of the corpus rather than of the search -- and the one route to closing it that survives on the merits is measuring what a model is good at rather than asserting it.

## The dataset-only and model-only controls

E1 sees dataset meta-features only, so it can predict just one value per dataset; E2 sees model meta-features only. Together they show how much of MCC each half of the meta-data explains on its own.

**E1** (5 terms):

```
MCC = +1.5356
      -0.200185 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.2552
      -0.236638 * [log(gravity)] / [log(nr_inst)]                 # beta=-0.1932
      -0.739648 * 1/nr_class                                      # beta=-0.1095
      +0.000556802 * [log(inst_to_attr)] * [nr_bin]               # beta=+0.1012
      +7.56152 * [nr_cor_attr] / [log(nr_inst)]                   # beta=+0.0474
```

**E2** (5 terms):

```
MCC = +0.771541
      +0.0063601 * Model Capability^2                             # beta=+0.2081
      -1.09817 * [log(Model Capability)] / [log(Processing Units Number)]  # beta=-0.1434
      -0.0312464 * log(Processing Units Number)                   # beta=-0.1335
      +0.440972 * 1/Loss Margin Behaviour                         # beta=+0.1225
      -0.29446 * [log(Input Distribution Modelling)] / [Model Capability]  # beta=-0.1035
```

How much of each control survives into E3-Valid, counting terms shared *exactly*:

| | shared with E3 | which |
|---|---|---|
| E2 → E3 | **0 of 5** | — |
| E1 → E3 | **1 of 5** | `[log(eq_num_attr)] * [log(nr_class)]` |

10 of E3-Valid's 17 terms mix dataset and model features, and a mixed term is available to neither control by construction.

#### The capability variant, in full

The same feature sets under the most capable searched grammar. This variant is reported so that the published equation's accuracy can be read against what the wider additive form can do, rather than only against oracles and baselines.

It reaches **0.7271** under IS against the published equation's 0.6815, and **0.6894** under LODO against 0.6513.

**E3-MAX** (29 terms):

```
MCC = +1.18972
      -0.245443 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.3129
      +0.243557 * ([log(nr_class)] + [log(Fitting Regime)]) / [log(eq_num_attr)]  # beta=+0.2844
      -0.342488 * ([log(gravity)] + [log(Input Distribution Modelling)]) / [log(nr_inst)]  # beta=-0.2808
      -0.121396 * ([log(class_ent)] + [log(Processing Units Number)]) / [Fitting Regime]  # beta=-0.2753
      +0.889511 * ([log(Fitting Regime)] + [log(Input Distribution Modelling)]) / [Model Capability]  # beta=+0.2737
      +0.0134172 * [log(gravity)] * [log(Model Capability)]       # beta=+0.2638
      +0.0329759 * ([log(gravity)] + [log(Loss Margin Behaviour)]) / [Fitting Regime]  # beta=+0.2452
      +0.0305433 * ([log(nr_inst)] + [log(Processing Units Number)]) / [Input Distribution Modelling]  # beta=+0.2315
      -0.103105 * ([log(ns_ratio)] + [log(Loss Margin Behaviour)]) / [log(nr_class)]  # beta=-0.2295
      -0.0279901 * ([log(nr_inst)] + [log(Input Distribution Modelling)]) / [log(eq_num_attr)]  # beta=-0.2260
      -0.21849 * [log(Fitting Regime)] * [log(Model Capability)]  # beta=-0.2170
      -1.35736 * ([log(nr_class)] + [log(Solution Stochasticity)]) / [log(Processing Units Number)]  # beta=-0.1971
      -0.0133278 * [log(gravity)] * [log(nr_class)]               # beta=-0.1855
      +0.995795 * ([log(ns_ratio)] + [log(Model Capability)]) / [log(nr_inst)]  # beta=+0.1770
      +0.103583 * [log(eq_num_attr)] * [log(Model Capability)]    # beta=+0.1738
      +0.28253 * ([log(nr_class)] + [nr_cor_attr]) / [Fitting Regime]  # beta=+0.1407
      +0.653635 * ([log(nr_class)] + [log(Model Capability)]) / [log(Processing Units Number)]  # beta=+0.1323
      -0.43126 * ([log(nr_class)] + [log(ns_ratio)]) / [log(Processing Units Number)]  # beta=-0.1307
      +0.107732 * ([log(class_ent)] + [log(nr_attr)]) / [Loss Margin Behaviour]  # beta=+0.1200
      -7.79467 * [nr_cor_attr] / [log(Processing Units Number)]   # beta=-0.0933
      -0.00934462 * ([log(class_ent)] + [log(gravity)]) / [Solution Stochasticity]  # beta=-0.0765
      +1.75601 * [nr_cor_attr] / [Fitting Regime]                 # beta=+0.0760
      +0.138605 * ([nr_cor_attr] + [log(Loss Margin Behaviour)]) / [Input Distribution Modelling]  # beta=+0.0718
      -0.0415717 * ([log(class_ent)] + [log(nr_attr)]) / [log(nr_class)]  # beta=-0.0671
      -0.207154 * 1/ns_ratio                                      # beta=-0.0618
      -0.20462 * ([log(nr_class)] + [nr_cor_attr]) / [log(nr_attr)]  # beta=-0.0467
      +0.000992027 * [log(inst_to_attr)] * [nr_norm]              # beta=+0.0460
      -0.00542775 * ([log(nr_inst)] + [nr_norm]) / [Solution Stochasticity]  # beta=-0.0416
      -0.00654635 * ([log(class_ent)] + [log(Processing Units Number)]) / [log(nr_class)]  # beta=-0.0349
```

LaTeX:

```latex
\mathrm{MCC} = +1.19 -0.2454 \cdot \mathrm{[log(eq\_num\_attr)] * [log(nr\_class)]} +0.2436 \cdot \mathrm{([log(nr\_class)] + [log(Fitting Regime)]) / [log(eq\_num\_attr)]} -0.3425 \cdot \mathrm{([log(gravity)] + [log(Input Distribution Modelling)]) / [log(nr\_inst)]} -0.1214 \cdot \mathrm{([log(class\_ent)] + [log(Processing Units Number)]) / [Fitting Regime]} +0.8895 \cdot \mathrm{([log(Fitting Regime)] + [log(Input Distribution Modelling)]) / [Model Capability]} +0.01342 \cdot \mathrm{[log(gravity)] * [log(Model Capability)]} +0.03298 \cdot \mathrm{([log(gravity)] + [log(Loss Margin Behaviour)]) / [Fitting Regime]} +0.03054 \cdot \mathrm{([log(nr\_inst)] + [log(Processing Units Number)]) / [Input Distribution Modelling]} -0.1031 \cdot \mathrm{([log(ns\_ratio)] + [log(Loss Margin Behaviour)]) / [log(nr\_class)]} -0.02799 \cdot \mathrm{([log(nr\_inst)] + [log(Input Distribution Modelling)]) / [log(eq\_num\_attr)]} -0.2185 \cdot \mathrm{[log(Fitting Regime)] * [log(Model Capability)]} -1.357 \cdot \mathrm{([log(nr\_class)] + [log(Solution Stochasticity)]) / [log(Processing Units Number)]} -0.01333 \cdot \mathrm{[log(gravity)] * [log(nr\_class)]} +0.9958 \cdot \mathrm{([log(ns\_ratio)] + [log(Model Capability)]) / [log(nr\_inst)]} +0.1036 \cdot \mathrm{[log(eq\_num\_attr)] * [log(Model Capability)]} +0.2825 \cdot \mathrm{([log(nr\_class)] + [nr\_cor\_attr]) / [Fitting Regime]} +0.6536 \cdot \mathrm{([log(nr\_class)] + [log(Model Capability)]) / [log(Processing Units Number)]} -0.4313 \cdot \mathrm{([log(nr\_class)] + [log(ns\_ratio)]) / [log(Processing Units Number)]} +0.1077 \cdot \mathrm{([log(class\_ent)] + [log(nr\_attr)]) / [Loss Margin Behaviour]} -7.795 \cdot \mathrm{[nr\_cor\_attr] / [log(Processing Units Number)]} -0.009345 \cdot \mathrm{([log(class\_ent)] + [log(gravity)]) / [Solution Stochasticity]} +1.756 \cdot \mathrm{[nr\_cor\_attr] / [Fitting Regime]} +0.1386 \cdot \mathrm{([nr\_cor\_attr] + [log(Loss Margin Behaviour)]) / [Input Distribution Modelling]} -0.04157 \cdot \mathrm{([log(class\_ent)] + [log(nr\_attr)]) / [log(nr\_class)]} -0.2072 \cdot \mathrm{1/ns\_ratio} -0.2046 \cdot \mathrm{([log(nr\_class)] + [nr\_cor\_attr]) / [log(nr\_attr)]} +0.000992 \cdot \mathrm{[log(inst\_to\_attr)] * [nr\_norm]} -0.005428 \cdot \mathrm{([log(nr\_inst)] + [nr\_norm]) / [Solution Stochasticity]} -0.006546 \cdot \mathrm{([log(class\_ent)] + [log(Processing Units Number)]) / [log(nr\_class)]}
```

<!-- end generated -->

## Limitations of the equation

### Model descriptors are thin, and five of them are asserted

The five model features the corpus originally shipped captured 63% of what model identity
explains. That gap is what the asserted ordinals were added to close, and they close most of
it: E2 goes from 63% to **92%** of its ceiling.

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
about a third of that pattern under IS and a quarter out of fold — measured, not inferred
from an R² comparison. Both numbers are in the generated section above.

Both are statements that something is missing. Neither says **how much** a better set of
model descriptors would be worth, and that is a number a reader will want before deciding
whether to go and collect them. This chapter measures it.

### The measurement

Under LODO **every one of the 25 models appears in every training fold**.
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

The current pooled gap is 0.056 of LODO R²: the level recovers 0.022 and
the level-plus-slope form the remaining 0.034. **The residual is specifically the interaction
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
capture some of this interaction, but they do not recover all of the model-specific variation.

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
- **It cannot extrapolate to a new model.** Under LOMO the correction is
  empty and its predictions are bitwise E3's. Every number comes from rows where that model
  was already run.
What survives is the **measurement**: the generated per-model identity ceiling as the upper
bound on better model descriptors. It belongs to this chapter's argument about thin model
descriptors and is recomputed on every run.

`ml_meta_perf.identity` was for a long time tested and wired into no pipeline, and the
consequence was that the ceiling it measures was hand-copied into this chapter and went
stale by a factor of three. It is now run on every study:
`identity_ceiling` calls `identity.correct_out_of_fold` against the finished
`cross_validate_fixed_form` path, so the number costs no extra search and cannot drift from
the equation it is a ceiling for.
