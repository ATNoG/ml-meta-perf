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

**E2 — model features only.** There are six model features, five constant per model.
`Processing Units Number` can vary with dataset shape and configuration, so E2 is not
restricted to one constant prediction per model.

**E3 — dataset and model features.** One input, one output. E3 can express variation
*within* a dataset, which is exactly what E1 structurally cannot do, and it is the
equation the study publishes.

### One process, three feature sets

All three are fitted by the same function on the same 484 observed rows, scored under
the same four protocols, and use one configuration (`config/study.json`) and one length
rule. Each searches every feature in its permitted block.

All three are fitted on individual pairs, rather than aggregated dataset or model means.
An R² over 20 dataset averages has a different denominator from an R² over 484 pairs.
Aggregating model features would also discard the within-model variation in `Processing
Units Number`. The generated results use the common pair-level scale.

## Headline comparison

Every equation is fitted on all 484 rows and scored on all 484 rows, so there is one
scale and one table:

Each equation, the group-identity references, and the additive mean-based reference are in one generated
table — on the [index page](index.md), which carries the headline, and again with the full
metric set in [chapter 5](05-evaluation.md). The figure is the same comparison drawn:

![Equations against their ceilings](../figures/01_equation_comparison.png)

### One scale is not one ceiling

The R² values are arithmetically comparable because they use the same evaluated pairs.
The true per-dataset means bound a dataset-constant predictor such as E1. The per-model
means are a descriptive reference for E2, not a ceiling, because `Processing Units Number`
can vary within a model. E2 can therefore exceed that reference without contradiction.

## Under all four protocols

The generated tables report IS, LODO, LOMO, and DHO. `selection.protocol_spread` is the
largest minus the smallest R² across these protocols. A small spread describes coefficient
refits of the globally selected form; it does not remove selection optimism or establish
independent performance of discovery. The within-fold reselection diagnostic is reported
separately.

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

$$\hat{y}_{dm} = \bar{y}_d + \bar{y}_m - \bar{y}$$

The additive mean-based reference combines the true per-dataset and per-model means,
then clips to the target range. It uses scored targets and cannot predict unseen identities.
On the incomplete grid it is not a jointly fitted least-squares additive model and is not
a strict ceiling. E3's mixed terms may represent interactions beyond marginal group effects.
The generated table reports all these comparisons on the revised corpus.

### Ceiling 2 — what the vocabulary can reach

`analysis.grammar_ceiling` fits raw admissible columns, the best single-feature transform
per feature, and all single-feature terms. These fitted library references describe what
can be represented without cross-feature terms under their admissibility conditions.
`analysis.feature_reach` reports the per-feature comparisons. Their current values are
regenerated below rather than copied from an earlier corpus.

### Ceiling 3 — how fast interaction pays

The interaction diagnostic lays the additive-reference residual on the dataset-by-model
grid, decomposes it by SVD, and adds leading components:

$$\hat{y}^{(r)}_{dm} = \bar{y} + \alpha_d + \beta_m + \sum_{j=1}^{r}\sigma_j u_{dj}v_{mj}$$

The 16 unobserved cells are filled with zero residual for this calculation and are not
scored. This filling convention can affect the decomposition. The reference ladder uses
observed targets; its fitted gains are not deployable performance estimates.
`validate.interaction_capture` compares the equation's and target's residual interaction
patterns on observed cells. The generated alignments describe structure recovered under
this convention, rather than a causal allocation or guaranteed reachable improvement.

## Does model choice matter more than the dataset?

Dataset identity and model identity explain different amounts of observed MCC variation;
the generated decomposition reports both. This comparison describes the fixed task and
learner catalog, rather than the effect of changing a dataset or a classifier.

E1 and E2 share a fitting procedure, but only E1 is constant within its identity group.
Their contrast therefore must not be reduced to two fractions of strict ceilings. Mixed
terms in E3 additionally describe associations between the descriptor blocks.

The within-group screening table is dataset-centered: it deliberately removes between-
dataset variation to find model-side relationships. It is not a measure of which block
matters more overall. Ordinal family assignments are representation assumptions, and the
encoding sensitivity controls are needed when interpreting terms that use them.

## One configuration, two roles

* **E3-Valid** is the equation the study publishes. It uses the shared configuration and the
  shared length rule, exactly as E1 and E2 do.
* **E3-MAX** is the capability/length comparison: the same configuration at its configured
  capability arity, taking the raw maximum of the worst-protocol curve. Both current roles
  use arity 3; they differ in length-selection rule.

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

Each prediction can be decomposed into named weighted terms. This exposes transformations,
interactions, and associations under the recorded descriptor encoding; it does not make a
coefficient a causal effect or validate the numeric ordering of learner categories.
The generated coefficient, effect, and reselection tables provide complementary readings.

### The cost, stated plainly

Raw weights depend on a term's units. Standardised weights compare coefficients on a
common term scale, while contribution ranges show their variation in MCC units.
Correlated terms can offset one another, so neither quantity uniquely attributes the
prediction to an individual raw feature.

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

The generated prediction and error tables show the weak observations and groups. Zero
and negative targets remain included. Ridge shrinkage and bounded prediction can produce
positive estimates for weak classifiers; this is distinct from predicting whether a
training job will complete. The 16 eligibility exclusions have no observed targets.

<!-- generated: do not edit below -->

## The equation

E3-Valid uses **15 terms** over dataset and model meta-features, simplified and refitted after pruning, so it evaluates exactly as printed.

```
MCC = +1.86445
      +3.73913 * ([log(nr_class)] + [log(nr_cor_attr)]) / [log(nr_inst)]  # beta=+0.2750
      -1.3466 * ([log(nr_class)] + [log(Fitting Regime)]) / [log(Processing Units Number)]  # beta=-0.2399
      -0.192051 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.2318
      -1.44595 * ([log(eq_num_attr)] + [log(nr_cor_attr)]) / [log(Processing Units Number)]  # beta=-0.2145
      +0.0093532 * [log(gravity)] * [log(Model Capability)]       # beta=+0.1890
      +0.441226 * ([log(eq_num_attr)] + [log(Fitting Regime)]) / [log(nr_outliers)]  # beta=+0.1628
      -0.034492 * ([log(gravity)] + [log(inst_to_attr)]) / [log(nr_outliers)]  # beta=-0.1409
      -0.0761179 * ([log(nr_class)] + [log(Model Capability)]) / [log(eq_num_attr)]  # beta=-0.1167
      +0.490605 * ([log(eq_num_attr)] + [log(Loss Margin Behaviour)]) / [log(Processing Units Number)]  # beta=+0.1116
      -0.15248 * ([log(Input Distribution Modelling)] + [log(Loss Margin Behaviour)]) / [log(nr_class)]  # beta=-0.0967
      -0.0411603 * ([log(eq_num_attr)] + [log(gravity)]) / [log(Processing Units Number)]  # beta=-0.0859
      +0.113505 * ([log(nr_attr)] + [log(Fitting Regime)]) / [log(nr_cor_attr)]  # beta=+0.0670
      -0.0505464 * ([log(class_ent)] + [log(ns_ratio)]) / [log(nr_class)]  # beta=-0.0626
      +0.0697121 * ([log(Input Distribution Modelling)] + [log(Model Capability)]) / [log(nr_class)]  # beta=+0.0565
      +0.141709 * 1/gravity                                       # beta=+0.0276
```

LaTeX:

```latex
\mathrm{MCC} = +1.864 +3.739 \cdot \mathrm{([log(nr\_class)] + [log(nr\_cor\_attr)]) / [log(nr\_inst)]} -1.347 \cdot \mathrm{([log(nr\_class)] + [log(Fitting Regime)]) / [log(Processing Units Number)]} -0.1921 \cdot \mathrm{[log(eq\_num\_attr)] * [log(nr\_class)]} -1.446 \cdot \mathrm{([log(eq\_num\_attr)] + [log(nr\_cor\_attr)]) / [log(Processing Units Number)]} +0.009353 \cdot \mathrm{[log(gravity)] * [log(Model Capability)]} +0.4412 \cdot \mathrm{([log(eq\_num\_attr)] + [log(Fitting Regime)]) / [log(nr\_outliers)]} -0.03449 \cdot \mathrm{([log(gravity)] + [log(inst\_to\_attr)]) / [log(nr\_outliers)]} -0.07612 \cdot \mathrm{([log(nr\_class)] + [log(Model Capability)]) / [log(eq\_num\_attr)]} +0.4906 \cdot \mathrm{([log(eq\_num\_attr)] + [log(Loss Margin Behaviour)]) / [log(Processing Units Number)]} -0.1525 \cdot \mathrm{([log(Input Distribution Modelling)] + [log(Loss Margin Behaviour)]) / [log(nr\_class)]} -0.04116 \cdot \mathrm{([log(eq\_num\_attr)] + [log(gravity)]) / [log(Processing Units Number)]} +0.1135 \cdot \mathrm{([log(nr\_attr)] + [log(Fitting Regime)]) / [log(nr\_cor\_attr)]} -0.05055 \cdot \mathrm{([log(class\_ent)] + [log(ns\_ratio)]) / [log(nr\_class)]} +0.06971 \cdot \mathrm{([log(Input Distribution Modelling)] + [log(Model Capability)]) / [log(nr\_class)]} +0.1417 \cdot \mathrm{1/gravity}
```

## How far the form could reach

The following library fits and length comparison describe representational reach under the recorded corpus and settings. They are not independent performance estimates or universal ceilings.

#### How far the additive form reaches

The published equation uses arity 3. The same configuration under arity 3, at the raw maximum of its worst-protocol curve (E3-MAX), reaches 24 terms at R² 0.7247 under IS:

| | terms | IS | LODO | LOMO | DHO | form stability |
|---|---|---|---|---|---|---|
| E3-Valid (arity 3) | 15 | 0.6997 | 0.6771 | 0.6574 | 0.6478 | 0.31 |
| E3-MAX (arity 3) | 24 | 0.7247 | 0.6950 | 0.6736 | 0.6696 | 0.30 |

This is a **capability measurement, not a recommendation**. It answers the question the published equation cannot answer about itself — whether the additive form is out of room or whether this equation is short of it — and the answer is that +0.0250 of IS R² separates the equations under the configured arities and length rules. Form stability is the mean share of LODO folds that re-select an equation's own terms when selection is re-run without the held-out dataset.

#### What the vocabulary could reach, before any search

Three levels of what the vocabulary can explain, each a least-squares fit over the library and each computable before the search runs. They bound a *sum of per-feature functions*, which is a different question from the additive mean-based reference above: that one bounds a per-dataset value plus a per-model value.

| level | terms | R² |
|---|---|---|
| every raw term admitted by the grammar | 10 | 0.2879 |
| the best single-feature term per feature | 18 | 0.5604 |
| every single-feature term at once | 56 | 0.6336 |
| **the fitted equation (E3-Valid)** | **15** | **0.6997** |

No individual feature carries much: the strongest is `eq_num_attr` at R² 0.161, so any accuracy beyond that is combination rather than a single dominant driver. Taking one best admissible term per feature is worth +0.272 over the admissible raw-term fit.

E3-Valid reaches 0.6997 with 15 terms, **above** the 0.6336 that all 56 single-feature terms reach together. An equation cannot pass that level by describing features one at a time, so the excess is what the cross-feature terms buy — the same conclusion the additive mean-based reference reaches, by an independent route.

#### Where the variance is, before any equation

Variance of MCC explained by group identity alone, with no equation involved:

| knowing only | n_groups | variance_explained |
|---|---|---|
| dataset identity | 20 | 0.3866 |
| model identity | 25 | 0.2443 |

#### How fast interaction pays

The additive form cannot represent dataset-by-model interaction beyond what its mixed terms reach. The ladder below adds interaction components to an oracle that is handed the true group means, so it measures the ceiling rather than any equation:

| interaction_rank | r2 | gain |
|---|---|---|
| 0 | 0.6611 |  |
| 1 | 0.8013 | 0.1402 |
| 2 | 0.8702 | 0.0688 |
| 3 | 0.9118 | 0.0416 |
| 4 | 0.9399 | 0.0281 |
| 6 | 0.9726 | 0.0327 |
| 8 | 0.9871 | 0.0145 |

That is a ceiling, not a score. Whether the equation reaches any of it is a separate question, and the answer is that it reaches some: below, `alignment` is the squared correlation between the equation's own interaction residual and the leading components of the oracle's, over observed cells. `leading_share` is how much of the interaction variance those components carry, and `interaction_share` how much of MCC's variance is interaction at all.

| protocol | rank | alignment | leading_share | interaction_share |
|---|---|---|---|---|
| IS | 1 | 0.4016 | 0.4362 | 0.3682 |
| IS | 2 | 0.3788 | 0.6286 | 0.3682 |
| LODO | 1 | 0.3762 | 0.4362 | 0.3682 |
| LODO | 2 | 0.3259 | 0.6286 | 0.3682 |

## Equation analysis

The equation has 15 terms, of which **9** carry 81% of the standardised weight mass; the single largest carries 13.2%, and the weights behave like **11.7 equally-weighted terms** (inverse Simpson index of the shares).

That last number is the one to read for concentration, because it does not depend on where a threshold is drawn. At 78% of the term count the equation is **flat**: no single term dominates. That is a statement about the *unit of explanation*, not about the quality of the equation — MCC here is inferred by a set of terms acting together rather than by one or two that could be quoted on their own. Three readings follow, and the sections below give each one: read the terms in the blocks that move together, read which features the search reached for, and read which operations it needed to apply to them.

`beta` is the standardised weight — the MCC contributed per standard deviation of the term, which is what makes terms in unrelated units comparable. `effect` is the swing in predicted MCC across the middle 80% of the term's observed range. `stability` is the fraction of LODO folds that selected the term.

| rank | term | group | features | weight | beta | effect | share | cumulative | major | stability | direction |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | ([log(nr_class)] + [log(nr_cor_attr)]) / [log(nr_inst)] | dataset | nr_class, nr_cor_attr, nr_inst | 3.7391 | 0.2750 | 0.7320 | 0.1323 | 0.1323 | yes | 0.0500 | raises MCC |
| 2 | ([log(nr_class)] + [log(Fitting Regime)]) / [log(Processing Units Number)] | mixed | nr_class, Fitting Regime, Processing Units Number | -1.3466 | -0.2399 | 0.6181 | 0.1154 | 0.2477 | yes | 1.0000 | lowers MCC |
| 3 | [log(eq_num_attr)] * [log(nr_class)] | dataset | eq_num_attr, nr_class | -0.1921 | -0.2318 | 0.7545 | 0.1115 | 0.3592 | yes | 0.2500 | lowers MCC |
| 4 | ([log(eq_num_attr)] + [log(nr_cor_attr)]) / [log(Processing Units Number)] | mixed | eq_num_attr, nr_cor_attr, Processing Units Number | -1.4459 | -0.2145 | 0.5240 | 0.1032 | 0.4624 | yes | 1.0000 | lowers MCC |
| 5 | [log(gravity)] * [log(Model Capability)] | mixed | gravity, Model Capability | 0.0094 | 0.1890 | 0.4736 | 0.0910 | 0.5534 | yes | 0.5500 | raises MCC |
| 6 | ([log(eq_num_attr)] + [log(Fitting Regime)]) / [log(nr_outliers)] | mixed | eq_num_attr, Fitting Regime, nr_outliers | 0.4412 | 0.1628 | 0.4218 | 0.0783 | 0.6317 | yes | 0.0000 | raises MCC |
| 7 | ([log(gravity)] + [log(inst_to_attr)]) / [log(nr_outliers)] | dataset | gravity, inst_to_attr, nr_outliers | -0.0345 | -0.1409 | 0.4621 | 0.0678 | 0.6995 | yes | 0.0000 | lowers MCC |
| 8 | ([log(nr_class)] + [log(Model Capability)]) / [log(eq_num_attr)] | mixed | nr_class, Model Capability, eq_num_attr | -0.0761 | -0.1167 | 0.2864 | 0.0561 | 0.7557 | yes | 0.0000 | lowers MCC |
| 9 | ([log(eq_num_attr)] + [log(Loss Margin Behaviour)]) / [log(Processing Units Number)] | mixed | eq_num_attr, Loss Margin Behaviour, Processing Units Number | 0.4906 | 0.1116 | 0.2865 | 0.0537 | 0.8093 | yes | 0.1000 | raises MCC |
| 10 | ([log(Input Distribution Modelling)] + [log(Loss Margin Behaviour)]) / [log(nr_class)] | mixed | Input Distribution Modelling, Loss Margin Behaviour, nr_class | -0.1525 | -0.0967 | 0.2487 | 0.0465 | 0.8559 | no | 0.4000 | lowers MCC |
| 11 | ([log(eq_num_attr)] + [log(gravity)]) / [log(Processing Units Number)] | mixed | eq_num_attr, gravity, Processing Units Number | -0.0412 | -0.0859 | 0.2128 | 0.0413 | 0.8972 | no | 0.9000 | lowers MCC |
| 12 | ([log(nr_attr)] + [log(Fitting Regime)]) / [log(nr_cor_attr)] | mixed | nr_attr, Fitting Regime, nr_cor_attr | 0.1135 | 0.0670 | 0.1895 | 0.0322 | 0.9294 | no | 0.0000 | raises MCC |
| 13 | ([log(class_ent)] + [log(ns_ratio)]) / [log(nr_class)] | dataset | class_ent, ns_ratio, nr_class | -0.0505 | -0.0626 | 0.1860 | 0.0301 | 0.9596 | no | 0.0000 | lowers MCC |
| 14 | ([log(Input Distribution Modelling)] + [log(Model Capability)]) / [log(nr_class)] | mixed | Input Distribution Modelling, Model Capability, nr_class | 0.0697 | 0.0565 | 0.1616 | 0.0272 | 0.9867 | no | 0.0500 | raises MCC |
| 15 | 1/gravity | dataset | gravity | 0.1417 | 0.0276 | 0.0883 | 0.0133 | 1.0000 | no | 0.3000 | raises MCC |

#### The 8 largest terms, in words

(9 terms are flagged major; the leading 8 are written out, and the table above carries the rest.)

1. `([log(nr_class)] + [log(nr_cor_attr)]) / [log(nr_inst)]` (dataset) raises MCC: one standard deviation of this term is worth +0.275 MCC, it moves predicted MCC by 0.732 across the middle 80% of its observed range, it carries 13.2% of the equation's weight mass, and it was selected in 5% of folds.
2. `([log(nr_class)] + [log(Fitting Regime)]) / [log(Processing Units Number)]` (mixed) lowers MCC: one standard deviation of this term is worth -0.240 MCC, it moves predicted MCC by 0.618 across the middle 80% of its observed range, it carries 11.5% of the equation's weight mass, and it was selected in every fold.
3. `[log(eq_num_attr)] * [log(nr_class)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.232 MCC, it moves predicted MCC by 0.754 across the middle 80% of its observed range, it carries 11.2% of the equation's weight mass, and it was selected in 25% of folds.
4. `([log(eq_num_attr)] + [log(nr_cor_attr)]) / [log(Processing Units Number)]` (mixed) lowers MCC: one standard deviation of this term is worth -0.214 MCC, it moves predicted MCC by 0.524 across the middle 80% of its observed range, it carries 10.3% of the equation's weight mass, and it was selected in every fold.
5. `[log(gravity)] * [log(Model Capability)]` (mixed) raises MCC: one standard deviation of this term is worth +0.189 MCC, it moves predicted MCC by 0.474 across the middle 80% of its observed range, it carries 9.1% of the equation's weight mass, and it was selected in 55% of folds.
6. `([log(eq_num_attr)] + [log(Fitting Regime)]) / [log(nr_outliers)]` (mixed) raises MCC: one standard deviation of this term is worth +0.163 MCC, it moves predicted MCC by 0.422 across the middle 80% of its observed range, it carries 7.8% of the equation's weight mass, and it was selected in 0% of folds.
7. `([log(gravity)] + [log(inst_to_attr)]) / [log(nr_outliers)]` (dataset) lowers MCC: one standard deviation of this term is worth -0.141 MCC, it moves predicted MCC by 0.462 across the middle 80% of its observed range, it carries 6.8% of the equation's weight mass, and it was selected in 0% of folds.
8. `([log(nr_class)] + [log(Model Capability)]) / [log(eq_num_attr)]` (mixed) lowers MCC: one standard deviation of this term is worth -0.117 MCC, it moves predicted MCC by 0.286 across the middle 80% of its observed range, it carries 5.6% of the equation's weight mass, and it was selected in 0% of folds.

#### Large terms the folds disagreed on

6 of the major terms were selected by fewer than half of the folds. A large weight and a low selection frequency together mean the term is doing its work for *this* training set and would be replaced by something else on another, which the equation as printed does not show. **Do not build guidance on these.**

| rank | term | group | beta | share | stability |
|---|---|---|---|---|---|
| 1 | ([log(nr_class)] + [log(nr_cor_attr)]) / [log(nr_inst)] | dataset | 0.2750 | 0.1323 | 0.0500 |
| 3 | [log(eq_num_attr)] * [log(nr_class)] | dataset | -0.2318 | 0.1115 | 0.2500 |
| 6 | ([log(eq_num_attr)] + [log(Fitting Regime)]) / [log(nr_outliers)] | mixed | 0.1628 | 0.0783 | 0.0000 |
| 7 | ([log(gravity)] + [log(inst_to_attr)]) / [log(nr_outliers)] | dataset | -0.1409 | 0.0678 | 0.0000 |
| 8 | ([log(nr_class)] + [log(Model Capability)]) / [log(eq_num_attr)] | mixed | -0.1167 | 0.0561 | 0.0000 |
| 9 | ([log(eq_num_attr)] + [log(Loss Margin Behaviour)]) / [log(Processing Units Number)] | mixed | 0.1116 | 0.0537 | 0.1000 |

#### Reading the terms in blocks

An additive form invites reading one term at a time, and that works when one or two weights dominate. When they do not, the honest unit is larger than a term and smaller than the equation: terms whose per-row contributions move together say the same thing about a row and can be read as one block. Grouping is on the contributions rather than on shared features, because two terms can share no feature and still track each other.

10 blocks over 15 terms, the largest holding 2 terms and 21% of the weight mass.

| group | n_terms | share | effect | direction | shared | terms |
|---|---|---|---|---|---|---|
| 1 | 2 | 0.2147 | 1.2132 | lowers MCC | eq_num_attr | [log(eq_num_attr)] * [log(nr_class)] ; ([log(eq_num_attr)] + [log(nr_cor_attr)]) / [log(Processing Units Number)] |
| 2 | 3 | 0.1653 | 0.8107 | lowers MCC | eq_num_attr, gravity | ([log(eq_num_attr)] + [log(gravity)]) / [log(Processing Units Number)] ; ([log(gravity)] + [log(inst_to_attr)]) / [log(nr_outliers)] ; ([log(nr_class)] + [log(Model Capability)]) / [log(eq_num_attr)] |
| 3 | 2 | 0.1624 | 0.9328 | raises MCC | nr_class | ([log(class_ent)] + [log(ns_ratio)]) / [log(nr_class)] ; ([log(nr_class)] + [log(nr_cor_attr)]) / [log(nr_inst)] |
| 4 | 2 | 0.1426 | 0.7483 | lowers MCC | nr_class | ([log(nr_class)] + [log(Fitting Regime)]) / [log(Processing Units Number)] ; ([log(Input Distribution Modelling)] + [log(Model Capability)]) / [log(nr_class)] |
| 5 | 1 | 0.0910 | 0.4736 | raises MCC | Model Capability, gravity | [log(gravity)] * [log(Model Capability)] |
| 6 | 1 | 0.0783 | 0.4218 | raises MCC | Fitting Regime, eq_num_attr, nr_outliers | ([log(eq_num_attr)] + [log(Fitting Regime)]) / [log(nr_outliers)] |
| 7 | 1 | 0.0537 | 0.2865 | raises MCC | Loss Margin Behaviour, Processing Units Number, eq_num_attr | ([log(eq_num_attr)] + [log(Loss Margin Behaviour)]) / [log(Processing Units Number)] |
| 8 | 1 | 0.0465 | 0.2487 | lowers MCC | Input Distribution Modelling, Loss Margin Behaviour, nr_class | ([log(Input Distribution Modelling)] + [log(Loss Margin Behaviour)]) / [log(nr_class)] |
| 9 | 1 | 0.0322 | 0.1895 | raises MCC | Fitting Regime, nr_attr, nr_cor_attr | ([log(nr_attr)] + [log(Fitting Regime)]) / [log(nr_cor_attr)] |
| 10 | 1 | 0.0133 | 0.0883 | raises MCC | gravity | 1/gravity |

#### Which features the search reached for

**15 of 18** available meta-features appear in the equation. `share` sums the weight mass of every term a feature appears in, so a feature in two terms is credited both and the column does not sum to 1 — it answers how much of the equation touches this feature, not how much it owns. A feature the search declined to use after seeing every transform of it is itself a result.

| feature | meaning | n_terms | share | transforms | operations |
|---|---|---|---|---|---|
| nr_class | number of classes | 7 | 0.5192 | log | product, sum_ratio |
| eq_num_attr | equivalent number of attributes (effective feature count) | 6 | 0.4442 | log | product, sum_ratio |
| Processing Units Number | family-specific structural-size proxy | 4 | 0.3137 | log | sum_ratio |
| nr_cor_attr | proportion of correlated attribute pairs | 3 | 0.2677 | log | sum_ratio |
| Fitting Regime | assigned fitting-regime category (1-5) | 3 | 0.2260 | log | sum_ratio |
| gravity | gravity (separation between the majority and minority class centres) | 4 | 0.2134 | inv, log | atom, product, sum_ratio |
| Model Capability | heuristic study-specific learner-family code (1-10) | 3 | 0.1743 | log | product, sum_ratio |
| nr_outliers | number of attributes containing outliers | 2 | 0.1461 | log | sum_ratio |
| nr_inst | number of instances in the source dataset (before sampling) | 1 | 0.1323 | log | sum_ratio |
| Loss Margin Behaviour | assigned reference loss or decision-criterion category (1-5) | 2 | 0.1002 | log | sum_ratio |
| Input Distribution Modelling | assigned distribution-modelling category (1-5) | 2 | 0.0737 | log | sum_ratio |
| inst_to_attr | instances per attribute | 1 | 0.0678 | log | sum_ratio |
| nr_attr | number of attributes | 1 | 0.0322 | log | sum_ratio |
| class_ent | class entropy (how evenly the labels are spread) | 1 | 0.0301 | log | sum_ratio |
| ns_ratio | noise-to-signal ratio | 1 | 0.0301 | log | sum_ratio |
| nr_bin | number of binary attributes | 0 | 0.0000 |  |  |
| nr_norm | number of normally distributed attributes | 0 | 0.0000 |  |  |
| Solution Stochasticity | assigned stochasticity reference category (1-5) | 0 | 0.0000 |  |  |

#### Which operations the equation needed

The vocabulary offers five operations and five transforms and the search is free to ignore any of them, so a row that was offered and went unused is a shape this data turned out not to need. Rows marked `offered = no` were kept out of the library by the arity cap and say nothing about the data:

| kind | name | offered | n_terms | share |
|---|---|---|---|---|
| operation | atom | yes | 1 | 0.0133 |
| operation | ratio | yes | 0 | 0.0000 |
| operation | product | yes | 2 | 0.2025 |
| operation | sum_ratio | yes | 12 | 0.7843 |
| operation | ratio_of_sums | no | 0 | 0.0000 |
| transform | id | yes | 0 | 0.0000 |
| transform | log | yes | 14 | 0.9867 |
| transform | sqrt | yes | 0 | 0.0000 |
| transform | inv | yes | 1 | 0.0133 |
| transform | sq | yes | 0 | 0.0000 |

#### Where the equation's variance comes from

- terms over dataset meta-features alone (how hard is this data): 5 terms, 39.9% of the equation's output variance.
- terms over model meta-features alone (how capable is this model): 0 terms, 0.0% of the equation's output variance.
- terms mixing dataset and model features (which model suits which data): 10 terms, 60.1% of the equation's output variance.

| group | n_terms | share | effect_sum |
|---|---|---|---|
| dataset | 5 | 0.3988 | 0.9607 |
| model | 0 | 0.0000 | 0.0000 |
| mixed | 10 | 0.6012 | 0.8617 |

## The ceiling on model descriptors

Under LODO every model appears in every training fold, so the equation's residual can be averaged per model on the training rows and applied to the held-out dataset using fitting histories. These model-identity corrections are conditional diagnostic references, not universal ceilings on better descriptors. They are unavailable for a learner whose entire history is excluded.

| correction | r2_LODO | mae | gain | ci_low | ci_high | sign_p | wins | verdict |
|---|---|---|---|---|---|---|---|---|
| none (E3-Valid, 15 terms) | 0.6771 | 0.1231 | 0.0000 |  |  |  | 0 | baseline |
| per-model level | 0.6958 | 0.1203 | 0.0023 | -0.0029 | 0.0078 | 0.8238 | 9 | tie |
| per-model level and slope | 0.7338 | 0.1078 | 0.0147 | 0.0058 | 0.0250 | 0.0414 | 15 | real |

**The gap is 0.057 of LODO R²**, of which a per-model *level* recovers 0.019 and the level-plus-slope form the remaining 0.038.

**Whether that is real is a paired question**, so each rung is compared with the uncorrected equation dataset by dataset, on absolute error, over the twenty held-out folds. The two rungs come back differently, and the difference is the finding:

* a per-model **level** is a **tie** -- it wins on 9 of the twenty folds and its interval spans zero. A constant shift per model, which is what a level is, adds nothing the equation does not already have.

* a per-model **slope** is **not** a tie: the bootstrap interval [+0.0058, +0.0250] lies entirely above zero, on 15 winning folds of twenty. Read it with the sign test beside it, which at p = 0.041 does **not** reach significance -- so the gain is carried by its size on the folds it wins rather than by winning nearly all of them. That is weaker evidence than the interval alone suggests, and stronger than a tie.

**So the question this chapter was written to close is not closed.** The half of the correction that survives is the one that lets a model's advantage depend on the data -- exactly what a *capability* descriptor would have to do, and exactly what none of the descriptors this corpus records does. The mixed terms were supposed to absorb that interaction and have absorbed only part of it.

This correction tests information carried by model histories. Its outcome does not establish which alternative descriptors or categorical encodings would transfer to unseen learners. Encoding comparisons remain conditional on form selection.

## The dataset-only and model-only controls

E1 sees dataset meta-features only, so it can predict just one value per dataset; E2 sees model meta-features only. Together they show how much of MCC each half of the meta-data explains on its own.

**E1** (5 terms):

```
MCC = +1.40424
      -0.170479 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.2057
      -0.0382919 * ([log(gravity)] + [log(ns_ratio)]) / [log(nr_outliers)]  # beta=-0.1364
      -0.303654 * ([log(nr_attr)] + [log(nr_outliers)]) / [log(nr_inst)]  # beta=-0.0818
      +0.0419384 * [log(class_ent)] / [log(nr_class)]             # beta=+0.0456
      -0.0250595 * [log(nr_class)] * [log(nr_cor_attr)]           # beta=-0.0340
```

**E2** (7 terms):

```
MCC = +0.0892002
      +0.0102989 * Model Capability^2                             # beta=+0.3377
      +1.08395 * 1/Loss Margin Behaviour                          # beta=+0.2989
      +1.56368 * ([log(Input Distribution Modelling)] + [log(Loss Margin Behaviour)]) / [log(Processing Units Number)]  # beta=+0.1873
      -0.353972 * log(Input Distribution Modelling)               # beta=-0.1759
      -0.954111 * ([log(Model Capability)] + [log(Solution Stochasticity)]) / [log(Processing Units Number)]  # beta=-0.1721
      -0.0387516 * log(Processing Units Number)                   # beta=-0.1628
      +0.311189 * ([log(Fitting Regime)] + [log(Solution Stochasticity)]) / [Model Capability]  # beta=+0.0617
```

How much of each control survives into E3-Valid, counting terms shared *exactly*:

| | shared with E3 | which |
|---|---|---|
| E2 → E3 | **0 of 7** | — |
| E1 → E3 | **1 of 5** | `[log(eq_num_attr)] * [log(nr_class)]` |

10 of E3-Valid's 15 terms mix dataset and model features, and a mixed term is available to neither control by construction.

#### The capability variant, in full

The same feature sets under the most capable searched grammar. This variant is reported so that the published equation's accuracy can be read against what the wider additive form can do, rather than only against oracles and baselines.

It reaches **0.7247** under IS against the published equation's 0.6997, and **0.6950** under LODO against 0.6771.

**E3-MAX** (24 terms):

```
MCC = +2.17249
      -1.46132 * ([log(nr_class)] + [log(Fitting Regime)]) / [log(Processing Units Number)]  # beta=-0.2604
      -0.0534473 * ([log(gravity)] + [log(inst_to_attr)]) / [log(nr_attr)]  # beta=-0.2008
      -0.165545 * [log(eq_num_attr)] * [log(nr_class)]            # beta=-0.1998
      -1.28182 * ([log(eq_num_attr)] + [log(nr_cor_attr)]) / [log(Processing Units Number)]  # beta=-0.1901
      +2.37591 * ([log(nr_class)] + [log(nr_cor_attr)]) / [log(nr_inst)]  # beta=+0.1747
      +0.00797783 * [log(gravity)] * [log(Model Capability)]      # beta=+0.1612
      +0.167874 * ([log(nr_cor_attr)] + [log(Model Capability)]) / [Solution Stochasticity]  # beta=+0.1421
      +0.224714 * ([log(nr_attr)] + [log(Fitting Regime)]) / [log(nr_cor_attr)]  # beta=+0.1326
      +0.362621 * ([log(Fitting Regime)] + [log(Input Distribution Modelling)]) / [Model Capability]  # beta=+0.1109
      -0.052401 * ([log(eq_num_attr)] + [log(gravity)]) / [log(Processing Units Number)]  # beta=-0.1094
      +0.0123784 * ([log(gravity)] + [log(ns_ratio)]) / [Loss Margin Behaviour]  # beta=+0.1069
      +0.153526 * ([log(nr_cor_attr)] + [log(Loss Margin Behaviour)]) / [log(eq_num_attr)]  # beta=+0.1069
      +0.285647 * ([log(eq_num_attr)] + [log(Fitting Regime)]) / [log(nr_outliers)]  # beta=+0.1054
      +0.44378 * ([log(eq_num_attr)] + [log(Loss Margin Behaviour)]) / [log(Processing Units Number)]  # beta=+0.1009
      -0.0381458 * ([log(class_ent)] + [log(Processing Units Number)]) / [Fitting Regime]  # beta=-0.0856
      -0.131408 * ([log(Input Distribution Modelling)] + [log(Loss Margin Behaviour)]) / [log(nr_class)]  # beta=-0.0833
      +0.0827544 * ([log(class_ent)] + [log(nr_class)]) / [Loss Margin Behaviour]  # beta=+0.0782
      -0.0536863 * ([log(class_ent)] + [log(ns_ratio)]) / [log(nr_class)]  # beta=-0.0665
      +0.045698 * ([log(Model Capability)] + [log(Solution Stochasticity)]) / [log(nr_class)]  # beta=+0.0626
      +0.0152647 * ([log(nr_inst)] + [log(ns_ratio)]) / [Fitting Regime]  # beta=+0.0623
      -0.0907407 * ([log(eq_num_attr)] + [log(nr_cor_attr)]) / [Solution Stochasticity]  # beta=-0.0533
      -0.226034 * ([log(Input Distribution Modelling)] + [log(Solution Stochasticity)]) / [log(nr_attr)]  # beta=-0.0370
      +0.157742 * 1/gravity                                       # beta=+0.0307
      +0.178951 * ([log(eq_num_attr)] + [log(nr_cor_attr)]) / [log(inst_to_attr)]  # beta=+0.0247
```

LaTeX:

```latex
\mathrm{MCC} = +2.172 -1.461 \cdot \mathrm{([log(nr\_class)] + [log(Fitting Regime)]) / [log(Processing Units Number)]} -0.05345 \cdot \mathrm{([log(gravity)] + [log(inst\_to\_attr)]) / [log(nr\_attr)]} -0.1655 \cdot \mathrm{[log(eq\_num\_attr)] * [log(nr\_class)]} -1.282 \cdot \mathrm{([log(eq\_num\_attr)] + [log(nr\_cor\_attr)]) / [log(Processing Units Number)]} +2.376 \cdot \mathrm{([log(nr\_class)] + [log(nr\_cor\_attr)]) / [log(nr\_inst)]} +0.007978 \cdot \mathrm{[log(gravity)] * [log(Model Capability)]} +0.1679 \cdot \mathrm{([log(nr\_cor\_attr)] + [log(Model Capability)]) / [Solution Stochasticity]} +0.2247 \cdot \mathrm{([log(nr\_attr)] + [log(Fitting Regime)]) / [log(nr\_cor\_attr)]} +0.3626 \cdot \mathrm{([log(Fitting Regime)] + [log(Input Distribution Modelling)]) / [Model Capability]} -0.0524 \cdot \mathrm{([log(eq\_num\_attr)] + [log(gravity)]) / [log(Processing Units Number)]} +0.01238 \cdot \mathrm{([log(gravity)] + [log(ns\_ratio)]) / [Loss Margin Behaviour]} +0.1535 \cdot \mathrm{([log(nr\_cor\_attr)] + [log(Loss Margin Behaviour)]) / [log(eq\_num\_attr)]} +0.2856 \cdot \mathrm{([log(eq\_num\_attr)] + [log(Fitting Regime)]) / [log(nr\_outliers)]} +0.4438 \cdot \mathrm{([log(eq\_num\_attr)] + [log(Loss Margin Behaviour)]) / [log(Processing Units Number)]} -0.03815 \cdot \mathrm{([log(class\_ent)] + [log(Processing Units Number)]) / [Fitting Regime]} -0.1314 \cdot \mathrm{([log(Input Distribution Modelling)] + [log(Loss Margin Behaviour)]) / [log(nr\_class)]} +0.08275 \cdot \mathrm{([log(class\_ent)] + [log(nr\_class)]) / [Loss Margin Behaviour]} -0.05369 \cdot \mathrm{([log(class\_ent)] + [log(ns\_ratio)]) / [log(nr\_class)]} +0.0457 \cdot \mathrm{([log(Model Capability)] + [log(Solution Stochasticity)]) / [log(nr\_class)]} +0.01526 \cdot \mathrm{([log(nr\_inst)] + [log(ns\_ratio)]) / [Fitting Regime]} -0.09074 \cdot \mathrm{([log(eq\_num\_attr)] + [log(nr\_cor\_attr)]) / [Solution Stochasticity]} -0.226 \cdot \mathrm{([log(Input Distribution Modelling)] + [log(Solution Stochasticity)]) / [log(nr\_attr)]} +0.1577 \cdot \mathrm{1/gravity} +0.179 \cdot \mathrm{([log(eq\_num\_attr)] + [log(nr\_cor\_attr)]) / [log(inst\_to\_attr)]}
```

<!-- end generated -->

## Limitations of the equation

### Model descriptors are thin, and five of them are asserted

The five fixed ordinal columns encode study-specific distinctions between learners;
they are not calibrated capacity, robustness, or performance scales. `Model Capability`
is a bijection with ten families, and its ordering and distances are assumptions.
The revised pooled association with MCC is approximately Spearman 0.33; this descriptive
association does not independently validate that ordering.

`Processing Units Number` is a family-specific structural proxy derived from the selected
configuration and dataset shape, not a measured common-unit compute cost. Assigning
descriptors to a new learner requires an explicit mapping. Code-permutation and categorical
controls assess representation sensitivity of the retained form, conditional on selection.

### The bound on what better model descriptors could buy

The generated model-identity correction diagnostics measure whether model history can
improve a fixed equation under LODO. They are conditional comparisons, not universal
bounds on better descriptors or evidence about an unseen learner whose history is absent.

### The measurement

The diagnostic adds per-model levels, then per-model levels and feature slopes, using
only fitting observations in each LODO fold. Per-dataset errors and paired comparisons
assess the corrections. The reference uses model identities and their historical targets;
it is not available under DHO, where the target model's history is excluded.

### Why the model side, specifically

A model-identity correction can capture distinctions absent from the recorded descriptor
mapping. Its benefit on this catalog does not establish which measured replacement
features would generalise to new classifiers. Richer descriptor collection and independent
assessment remain separate experiments.

### Why this is a measurement and not a fourth equation

The correction is a diagnostic reference rather than the study's published equation.
It uses model history, while the primary representation aims to expose relationships
through recorded descriptors. Comparing them requires matching the information available
under each protocol.

