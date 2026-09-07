# 6. Best practices against the equation

*Implemented in `ml_meta_perf.attribution`, `ml_meta_perf.practices`, `ml_meta_perf.report` and
`ml_meta_perf.guidance`.*

This is what the accuracy was traded for. An equation nobody can turn into guidance has
bought nothing over a black box.

But the step from equation to guidance is two steps, and collapsing them was a mistake this
chapter used to make.

**A best practice is not a property of an equation.** It is general, transferable advice —
short enough to remember, cheap enough to apply — that already circulates in the field and
that evidence can support, qualify or challenge. "Higher `nr_norm` went with lower MCC on
these twenty datasets" is not that. It is a *measurement*: it names a column rather than an
action, it is true only of this corpus, and [chapter 4](04-equation.md) shows that its
sign can change when the equation changes.

So the chapter runs in two layers:

| layer | what it produces | module |
|---|---|---|
| **evidence** | what the fitted equation does as each feature and each term moves | `ml_meta_perf.practices`, `ml_meta_perf.report` |
| **practice** | recommendations taken from the literature, each weighed against that evidence | `ml_meta_perf.guidance` |

The second layer is where the study earns its keep. Twenty datasets from one domain is a
narrow base from which to *invent* advice and a perfectly reasonable base from which to
*test* it, so `ml_meta_perf.guidance` starts from ten practices the tabular machine-learning
literature already recommends and asks what this corpus says about each. The statements and
citations are written by hand — a fitting procedure does not produce a citation — and every
verdict and every number inside it is computed, against a stated threshold, so other data
can overturn any of them.

Verdicts are **supported**, **qualified**, **challenged** or **not tested**. The tally is
deliberately not written here: it moves whenever the equation does, and
the generated section below opens the section with the current one. What is worth stating is
the *shape* of it — most practices are supported, two or three sit at `not tested`, and
that last group is the one to read first.

`not tested` is a real verdict rather than a gap. Two kinds of thing land there. The
balanced-metric practice is untested because the study adopts MCC and never compares it to
accuracy or F1, so it cannot be evidence about the choice. The outlier-robustness practice
is untested because the column that recorded a *learner's* robustness was retired from the
corpus, and the nearest surviving column, `nr_outliers`, describes the data instead —
answering it with that number would produce a verdict that read as though the practice had
been tested. Marking both rather than quietly counting them as wins is the point of having
the verdict at all.

**A high supported count is a weak-looking result and should be read carefully.** It is not
that the catalogue was chosen to pass: each check has a stated threshold and returns
`qualified` or `challenged` when the numbers say so, and some do.
It is that the practices selected are ones a corpus of this shape *can* speak to. The
honest reading is that this study corroborates established tabular-ML guidance on an
independent corpus, not that it discovered anything the field disagreed about.

One practice was **withdrawn** rather than reported. *"Neural architectures catch up once
the dataset is large enough"* looked testable and is not: every model here was trained on a
stratified sample capped at 100,000 rows, so splitting the corpus by `nr_inst` splits it by
*source* size while every training set above the cap is identical in size. The withdrawal is
recorded in [chapter 1](01-dataset.md#training-set-size-is-not-a-variable-here) because
the near-miss is instructive — the split produced a clean-looking number, and the number
meant nothing.

the generated section below carries the full assessment, regenerated on every run.

## Making terms comparable

Raw weights cannot be compared — they are in the units of whatever the term computes. Two
comparable quantities are derived instead.

**Effect** is the swing in predicted MCC produced by moving a term from its 10th to its
90th percentile. Stated in MCC units, it is comparable across terms and against the target
itself. Percentiles rather than min-to-max, so a single extreme row cannot define the
headline.

**Share** attributes the variance of the equation's output to groups of terms, labelled by
the features they use — dataset-only, model-only, or mixed. Shares are computed as

$$\text{share}_g = \frac{\operatorname{cov}(\textstyle\sum_{i \in g} c_i,\; \sum_i c_i)}{\operatorname{var}(\sum_i c_i)}$$

which sums to exactly 1 even when groups are correlated — and they are, since three model
features are functions of dataset size. A naive $\operatorname{var}(g)/\operatorname{var}(\text{total})$
does not. Population covariance is used throughout to match `np.var`; mixing `np.cov`'s
default `ddof=1` with `np.var`'s `ddof=0` inflates every share by $n/(n-1)$.

## Deriving direction empirically

Reading a sign off a weight is wrong as soon as a feature appears in more than one term,
appears inside a denominator, or appears under a transform that flips its monotonicity.

So the derivation is empirical: every term's contribution is evaluated over the real data,
contributions are summed **per raw feature**, and

- **direction** is the rank correlation between the feature and the total it drives;
- **effect** is the mean contribution in the feature's top decile minus that in its bottom
  decile — signed, and ordered *by the feature*, which is what makes it a response rather
  than a spread.

A feature sharing a term with another is credited the whole term, since the term cannot be
split between them. `effect` therefore reads as "how much MCC moves across this feature's
range", not "how much this feature owns".

## Filters

A practice is published only if all three hold:

| filter | threshold | why |
|---|---|---|
| effect | ≥ 0.02 MCC | below this it is not worth writing down |
| monotonicity | \|ρ\| ≥ 0.15 | a non-monotone feature has no true directional sentence |
| fold stability | ≥ 0.50 | a term chosen in 4 of 20 folds is an artefact of the split |

Features failing any test are **dropped, not hedged**. A practice nobody should act on is
better left unwritten.

`confidence` combines effect size with fold stability: *strong* is ≥0.85 stability and
≥0.10 effect; *moderate* is ≥0.50 and ≥0.05; anything else is *weak*.

## The measured associations

The evidence layer. The table is regenerated with the equation on every run and appears in
the generated section at the end of this chapter, so it always describes the current
equation.

What belongs here is the caveat that has to travel with one of its rows wherever it is
quoted. `Model Capability` is the one feature in the set that was *asserted* rather than
measured, so "higher capability rank went with higher MCC" is partly the ladder being read
back out.

What is not built in is the conditional part — how family capability interacts with dataset properties — and
[chapter 4](04-equation.md) sets out the three costs in full.

![Feature effects](../figures/practice_effects.png)

Read together these point somewhere — *prefer a capable, high-capacity model; expect
trouble on noisy data and on data spread across many classes* — but the pointing
is the reader's inference, not the table's content. Turning it into advice, and checking
that advice against something other than this one equation, is the next section's job.

## What these are not

**They are not best practices.** Each names a meta-feature column, which is a thing to
measure rather than a thing to do, and each is true of one equation on one corpus. The
practices are in the generated section below, taken from the literature and weighed against
these numbers among others.

The defensible claim is directional, and the tables in this chapter should be read as a
set of signed statements rather than as a ranking. Any write-up that orders these
associations by effect size is claiming more than the evidence supports.

**Associations measured across 20 datasets, not causal claims.** "Higher training cost
went with lower MCC" does not mean that cheaper models are better; it means that among the
models run here, the expensive ones were not the ones that scored well on the datasets
where they were expensive — and training cost is partly a function of dataset size, so it
is carrying data difficulty as well as model capacity.

`Training Operations` flipped sign between two equations fitted on the same data, and was
one of four columns retired from the corpus on 2026-09-05 — so the example is now
historical, and none of the associations in chapter 10 rests on it. It is written up as a limitation in
the limitations at the end of this chapter, because the
caveat it raises applies to the whole extraction rather than to that one row.

### These are conditional statements, not marginal ones

A practice states what the *equation* does as a feature rises, with every other term
present. That is not the same as what the feature does on its own, and on this data the two
mostly disagree. Rank correlation of each raw feature against MCC, against the direction
its practice states:

| feature | marginal ρ with MCC | practice says | agree |
|---|---|---|---|
| built-in outlier robustness | +0.298 | higher | ✓ |
| noise-to-signal ratio | -0.395 | lower | ✓ |
| training cost | +0.205 | lower | ✗ |
| class entropy | +0.089 | lower | ✗ |
| normally distributed attributes | +0.054 | lower | ✗ |
| correlated attribute pairs | -0.197 | higher | ✗ |
| number of instances | +0.056 | lower | ✗ |

**Two of seven agree.** This is not a contradiction and not a bug — it is what conditioning
does. A feature's marginal correlation mixes its own effect with everything it travels
with; inside a fitted equation, the terms that carry those companions are already present,
so what is left for this feature is what it adds beyond them. Class entropy is the clearest
case: on its own it is faintly positive, but in the equation it appears in ratios against
`log(Processing Units Number)`, so what its practice describes is entropy *relative to the
capacity thrown at it*, and that is negative.

The consequence for a reader is concrete. **These statements are advice about what to
expect once the other factors are accounted for, not about what a scatter plot of that one
feature will show.** Where the two disagree, the marginal view is the one a practitioner
will accidentally verify against, and be confused by. Both numbers are printed above for
exactly that reason.

Twenty datasets from one domain is a narrow evidential base. Every statement above should
be read as a hypothesis this data is consistent with, not a finding established by it.

## Three ways to read a flat equation

The published equation's sixteen weights behave like **13.2 equally-weighted terms**
(inverse Simpson index of the standardised-weight shares, $1/\sum_i s_i^2$); the largest
single term carries 10.8% of the mass, and ten terms are needed to reach 80%. Those figures
move with the equation — the generated section below is the regenerated version — but the
*shape* is stable across every configuration tried: no term dominates, and no small handful
of them accounts for the result.

The additive form $f(X) = w_0t_0 + w_1t_1 + \dots$ invites reading one term at a time, and
a flat equation refuses that. **This is a statement about the unit of explanation, not
about the quality of the equation.** MCC here is inferred by a *set* of terms acting
together, and the right response is to change the unit rather than to conclude that the
equation cannot be read. Three units work, and `ml_meta_perf.report` generates all three into
the generated section below.

### 1. Blocks of terms that move together

Terms whose per-row contributions correlate say the same thing about a row and can be read
as one. Grouping is on the *contributions* rather than on shared features — two terms can
share no feature and still track each other, and two terms over the same feature can move
independently once their transforms differ.

`shared` names the features a strict majority of a block's terms contain — a genuine
question, since the grouping never looked at what the terms contained.

**The blocks are recomputed from the fitted equation on every run** and are listed in the
generated section at the end of this chapter rather than written out here, so what a reader
sees always describes the equation the code currently produces.

What survives restating is why the unit is worth having. A block can share no feature at
all and still be one idea: the grouping is on per-row *contributions*, so it finds terms
that move together for reasons no feature-based grouping would reach — dataset difficulty
expressed through four different columns, say. That is the case that justifies the method,
and it is the one a reader should look for in the generated table.

That structure is invisible in the term-by-term table and is not recoverable by reading the
printed equation.

### 2. Which features the search reached for

Asked of the vocabulary rather than of the weights, so a spread of weights does not blunt
it. **12 of the 18 available meta-features appear in the equation.** Six do not:
`class_ent`, `inst_to_attr`, `nr_norm` and `ns_ratio` on the dataset side, and
`Solution Stochasticity` and `Loss Margin Behaviour` on the model side. All were offered
under every transform and none earned a place, which is a result about the meta-data rather
than about the search — and, for the two model ordinals, a live tension: they are the columns
that let the feature set tell individual learners apart, and the fit does not want them.

`Processing Units Number` appears in 6 of 15 terms and `gravity` in 5; everything else is
spread thinner, 1–3 terms each. The full table, with the transforms
and operations each feature was used under, is in the generated section below.

### 3. Which operations the equation needed

| operation | terms | share |
|---|---|---|
| `sum_ratio` — $(f_1+f_2)/f_3$ | 11 | 50% |
| `product` — $f_1 \cdot f_2$ | 8 | 31% |
| `atom` — $f$ | 4 | 16% |
| `ratio` — $f_1/f_2$ | 1 | 3% |

| transform | terms | share |
|---|---|---|
| `log` | 19 | 79% |
| `id` | 12 | 41% |
| `sqrt` | 1 | 6% |
| `inv`, `sq` | **0** | **0%** |

Two things worth stating. **The three-feature operation carries half the equation**, which
is the direct evidence for the arity design point argued in
[chapter 2](02-additive-model.md) — a two-feature grammar would have had to express that
half some other way. And **`1/f` and `f²` were offered and never used**: the search
preferred `log` for compression and had no use for inversion or squaring at all.

`ratio_of_sums` is absent because `max_arity = 3` kept it out of the library, not because
the data declined it. The generated table marks that distinction with an `offered` column,
since reading a configuration choice as a finding is exactly the mistake this section is
built to avoid.

### And one thing that does temper the guidance

**Four of the sixteen major terms were selected by fewer than half of the folds**,
including the very largest. A large standardised weight with a low selection frequency
means the term is doing its work for *this* training set and would be replaced on another;
printed as a coefficient it looks exactly like a stable one. `report.unstable_majors`
extracts them so they cannot be quietly read as findings.

On the current equation the check passes rather than fires: the rank-1 term,
`[log(gravity)] * [log(Model Capability)]` at 10.6% of the mass, appears in **95% of folds**.
An earlier configuration's rank-1 term appeared in 20%, which is what the check exists for.
That the same test now comes back clean is a property of the fixed-form protocol — the
equation's terms are chosen once, so fold-to-fold reselection measures whether the *form*
survives resampling rather than which of twenty different equations happened to be fitted.

## Reading a single prediction

`ml_meta_perf.attribution.contributions` gives the per-term breakdown for one row, which is the
interpretability payoff in its most direct form:

```
dataset : 5G_Slicing
model   : AdaBoost
actual  : +1.0000
predicted: +1.0000

contribution breakdown:
     +1.4787   intercept
     -0.2843   [log(eq_num_attr)] / [log(Processing Units Number)]
     -0.2214   [log(eq_num_attr)] * [log(nr_class)]
     -0.1426   [log(Processing Units Number)] / [log(nr_class)]
     +0.1417   1/Fitting Regime
     ...
```

## The analysis is generated, not authored

There is a failure mode this chapter has to avoid. If the fitted equation is printed and
somebody then sits down to explain which terms matter and what they imply, the explanation
is the product and the equation is only its raw material — and the claim "this model is
interpretable" quietly becomes "this model was interpreted by an expert, once".

So `ml_meta_perf.report` derives the written analysis arithmetically, and
the generated section below is its output. The same equation always produces the same
sentences, and every sentence maps onto a row of a table printed next to it.

**Terms are ranked by standardised weight.** The target is centred but never scaled during
fitting, so a standardised weight `beta` is already in MCC units: how far predicted MCC
moves when that term moves by one standard deviation of itself. That is what makes a term
over instance counts comparable with a term over class entropy. Ranking on the raw weights
instead would rank the terms by the size of their units.

**Major terms are the leading ones carrying 80% of the weight mass.** `term_importance`
reports each term's `|beta|` as a share of the total and the running sum down the ranking,
so the threshold is visible rather than buried; a reader who wants a different cut can read
one off the `cumulative` column. The term that crosses the line is kept, so the flagged
group always accounts for at least the stated mass.

**Effect is reported next to every weight**, because a large weight on a term that barely
varies is not important. The two disagree often enough to be worth printing together: a
term can rank third by `beta` and first by `effect`.

**The sentences make no claim about the features inside a term**, only about the term. A
feature's direction depends on every term it appears in, and is `best_practices`' job —
derived empirically, as the section above describes, rather than read off a sign.

<!-- generated: do not edit below -->

## Best practices

A best practice is general, transferable advice that already circulates in the field — not a property of this equation. So the practices below are taken from the literature and this study is used to *weigh* them: each verdict, and the numbers inside it, are computed from this run by `ml_meta_perf.guidance`, against a stated threshold, so other data can overturn any of them.

10 practices assessed: 3 not tested, 2 qualified, 5 supported.

#### 1. Characterise the dataset before choosing a model. What the data is like bounds what any model can reach, and that bound is usually the larger effect.

**Verdict: supported.** Knowing only which dataset a row came from explains 35.4% of MCC variance; knowing only which model, 28.2%. The dataset side is also the better described: 12 dataset meta-features reach 98% of what dataset identity explains, while 6 model meta-features reach 88% of theirs. Both the effect and our ability to measure it favour the data.

*Practice from:* Zha et al., 'Data-centric AI: A Survey', arXiv:2303.10158 (2023). The data-centric position holds that returns from improving data exceed returns from swapping architectures. It is an argument about where to spend effort.

#### 2. On tabular data, start from tree ensembles. Reach for a neural architecture only when a tree ensemble has been tried and found wanting.

**Verdict: supported.** On the 17 datasets where every model ran, tree-based families average MCC 0.927 against 0.660 for neural ones. The neural side splits sharply: architectures built for tabular data reach 0.797 while a plain MLP or DNN reaches 0.454, last of the ten families. The advice holds, and it holds most strongly against exactly the architectures that are not designed for this kind of data.

*Practice from:* Grinsztajn et al., arXiv:2207.08815 (2022); Shwartz-Ziv & Armon, 'Deep Learning is Not All You Need', arXiv:2106.03253 (2021). Trees handle irregular, non-smooth target functions and uninformative features, which is what tabular data usually contains.

#### 3. Include a pretrained tabular model (TabPFN, TabICL) in the first round of candidates: it costs one fit and is frequently competitive with a tuned ensemble.

**Verdict: supported.** Pretrained tabular models average MCC 0.953, against 0.951 for the best tree family, and place 1 and 3 of 25 models. They match the strongest tree ensembles here without a tuning budget, which is the whole of the claim.

*Practice from:* Hollmann et al., TabPFN, arXiv:2207.01848 (2022); TabICL, arXiv:2502.05564 (2025). In-context learning on tabular data removes the tuning budget that usually separates a quick baseline from a competitive one.

#### 4. When rows share a group -- a subject, a site, a dataset -- validate by holding out whole groups. A random split reports a number that will not survive deployment.

**Verdict: qualified.** The same equation scores R² 0.639 under a random 10-fold split and 0.638 when whole datasets are held out -- 0.001 of pure protocol. Dataset meta-features are constant within a dataset, so a random fold shows the equation rows from a dataset it is being scored on.

*Practice from:* Walsh et al., 'Machine learning reporting standards', Nature Methods 18 (2021). Any feature constant within a group lets the model recognise the group rather than generalise to it, and a random split puts the group on both sides.

#### 5. Spend the first effort on reducing noise in the data, not on a larger model. Noise sets a ceiling that capacity cannot lift.

**Verdict: not tested.** No noise-to-signal practice survived the extraction filters in this run.

*Practice from:* Zha et al., 'Data-centric AI: Perspectives and Challenges', arXiv:2301.04819 (2023). Irreducible error from noisy features or labels bounds every model on that data, so capacity spent against it buys nothing.

#### 6. On real-world data that has not been carefully curated, prefer a learner with built-in robustness to outliers.

**Verdict: not tested.** This corpus cannot weigh it. The practice is about a property of the learner, and the column that recorded one -- `Robust to Outliers` -- was retired because it varied within a model and was undefined under every transform in the grammar but two. `nr_outliers` counts outliers in the data, not resistance to them in the model, so it answers a different question. Reported as untested rather than answered with the nearest available number.

*Practice from:* Grinsztajn et al., arXiv:2207.08815 (2022), on non-smooth targets and outliers. Real tabular data carries outliers that a squared-error learner chases and a split-based or margin-based one largely ignores.

#### 7. Match capacity to the problem. A larger, more expensive model is not a safer default; on small tabular problems it is usually a worse one.

**Verdict: supported.** The highest-capacity family here is also the worst: generic neural networks average MCC 0.454 against 0.927 for tree ensembles. The equation says it conditionally rather than flatly: `Processing Units Number` carries 5 of its terms, mostly against a dataset property, so what raises MCC is capacity *matched to* the problem rather than capacity itself.

*Practice from:* Shwartz-Ziv & Armon, arXiv:2106.03253 (2021). Capacity beyond what the sample supports fits noise, and the cost is paid twice: in accuracy and in the tuning budget needed to recover it.

#### 8. Before adopting a meta-learner to choose models, check it against 'use whatever usually works'. Ranking is an easier problem than prediction and often needs less.

**Verdict: qualified.** Tested against this study's own equation and the two cannot be separated -- which is the practice being right, since it claims the trivial baseline is competitive rather than that it wins. Ranking models within a held-out dataset, against the per-model-mean baseline -- ap 0.831 against 0.798, equation better on 7 of 18 datasets that differ, 95% CI [-0.066, +0.159]; mrr 0.882 against 0.835, equation better on 3 of 6 datasets that differ, 95% CI [-0.076, +0.185]; hit_at_1 0.800 against 0.750, equation better on 3 of 5 datasets that differ, 95% CI [-0.150, +0.250]; regret 0.015 against 0.011, equation better on 12 of 17 datasets that differ, 95% CI [-0.021, +0.010]. Every interval is a paired bootstrap over the twenty held-out datasets, because a difference of two means over twenty folds is not yet a measurement.

*Practice from:* Rice, 'The Algorithm Selection Problem' (1976); standard meta-learning practice. A per-model mean over previous datasets carries most of the ranking signal at zero modelling cost, and is the baseline any selection method has to clear.

#### 9. Report which (dataset, model) runs were excluded and why. Aggregate comparisons over an incomplete grid compare different models on different problems.

**Verdict: supported.** 24 of 500 (dataset, model) cells are absent -- 5% -- and they are not absent at random: eight models are missing from the same three datasets. Measuring the bias rather than assuming it is small: restricting to the 17 complete datasets moves the model ranking by Spearman 0.975, so the ordering survives, but a mean over all rows still compares eight of the models on a different set of problems from the rest. Every family figure quoted here uses the complete subset for that reason.

*Practice from:* Walsh et al., Nature Methods 18 (2021); benchmarking reporting standards. Runs usually go missing where a model struggles or will not fit, so exclusions are correlated with the outcome being measured.

#### 10. Score imbalanced classification with a metric that accounts for all four confusion-matrix cells -- MCC rather than accuracy or F1.

**Verdict: not tested.** This study adopts MCC as its target and never measures an alternative, so it is not evidence for the practice. What it does show is the shape MCC has: 15 of 476 rows sit at exactly 0, which is a classifier that has learned nothing being scored as having learned nothing. Accuracy would not have said that.

*Practice from:* Chicco & Jurman, BMC Genomics 21:6 (2020). Accuracy and F1 can both look strong on a classifier that has learned only the majority class; MCC cannot.


At a glance:

| practice | verdict | magnitude |
|---|---|---|
| Characterise the dataset before choosing a model. What the data is like bounds what any model can reach, and that bound is usually the larger effect. | supported | 0.0718 |
| On tabular data, start from tree ensembles. Reach for a neural architecture only when a tree ensemble has been tried and found wanting. | supported | 0.2675 |
| Include a pretrained tabular model (TabPFN, TabICL) in the first round of candidates: it costs one fit and is frequently competitive with a tuned ensemble. | supported | 0.0028 |
| When rows share a group -- a subject, a site, a dataset -- validate by holding out whole groups. A random split reports a number that will not survive deployment. | qualified | 0.0007 |
| Spend the first effort on reducing noise in the data, not on a larger model. Noise sets a ceiling that capacity cannot lift. | not tested |  |
| On real-world data that has not been carefully curated, prefer a learner with built-in robustness to outliers. | not tested |  |
| Match capacity to the problem. A larger, more expensive model is not a safer default; on small tabular problems it is usually a worse one. | supported | -0.4727 |
| Before adopting a meta-learner to choose models, check it against 'use whatever usually works'. Ranking is an easier problem than prediction and often needs less. | qualified | 0.0333 |
| Report which (dataset, model) runs were excluded and why. Aggregate comparisons over an incomplete grid compare different models on different problems. | supported | 0.0480 |
| Score imbalanced classification with a metric that accounts for all four confusion-matrix cells -- MCC rather than accuracy or F1. | not tested | 0.0315 |

## The measurements underneath

What the fitted equation says about each raw feature it uses, kept only when the feature moves predicted MCC enough to matter, does so monotonically enough for a sentence to be true of it, and does so through terms that survived most folds. Directions are measured on the data rather than read off weight signs, because a feature can appear in several terms and inside denominators.

**These are associations across 20 datasets, not causal claims, and not practices on their own** — a statement about a meta-feature column is a measurement. Section 5 is where they become advice, by supporting or failing to support something a practitioner could already have been told.

 1. [moderate] Higher equivalent number of attributes (effective feature count) went with lower MCC (about 0.62 MCC between its lowest and highest decile).
 2. [moderate] Higher gravity (separation between the majority and minority class centres) went with lower MCC (about 0.36 MCC between its lowest and highest decile).
 3. [moderate] Higher model capacity (log processing units) went with higher MCC (about 0.34 MCC between its lowest and highest decile).
 4. [strong  ] Higher learner family's capability rank in the tabular-ML literature (1-10) went with higher MCC (about 0.31 MCC between its lowest and highest decile).
 5. [moderate] Higher how the parameters are reached, closed form to in-context (1-5) went with lower MCC (about 0.26 MCC between its lowest and highest decile).
 6. [moderate] Higher proportion of correlated attribute pairs went with lower MCC (about 0.20 MCC between its lowest and highest decile).
 7. [moderate] Higher how much of the input distribution the learner models (1-5) went with lower MCC (about 0.19 MCC between its lowest and highest decile).
 8. [strong  ] Higher number of binary attributes went with higher MCC (about 0.18 MCC between its lowest and highest decile).

Evidence:

| feature | meaning | n_terms | direction | effect | stability | confidence |
|---|---|---|---|---|---|---|
| eq_num_attr | equivalent number of attributes (effective feature count) | 2 | -0.8560 | -0.6230 | 0.7750 | moderate |
| gravity | gravity (separation between the majority and minority class centres) | 5 | -0.5511 | -0.3641 | 0.8300 | moderate |
| Processing Units Number | model capacity (log processing units) | 5 | 0.4387 | 0.3396 | 0.7100 | moderate |
| Model Capability | learner family's capability rank in the tabular-ML literature (1-10) | 2 | 0.5623 | 0.3104 | 0.9250 | strong |
| Fitting Regime | how the parameters are reached, closed form to in-context (1-5) | 3 | -0.7870 | -0.2639 | 0.7167 | moderate |
| nr_cor_attr | proportion of correlated attribute pairs | 1 | -0.9284 | -0.1961 | 0.5000 | moderate |
| Input Distribution Modelling | how much of the input distribution the learner models (1-5) | 1 | -0.9978 | -0.1937 | 0.8000 | moderate |
| nr_bin | number of binary attributes | 1 | 0.8439 | 0.1845 | 0.8500 | strong |

#### These are conditional statements, not marginal ones

A practice states what the *equation* does as a feature rises, with every other term present. A marginal correlation states what the feature does alone. They are different quantities, and here **7 of 8 agree** on the sign:

| feature | meaning | marginal | conditional | practice_says | agrees |
|---|---|---|---|---|---|
| eq_num_attr | equivalent number of attributes (effective feature count) | -0.1781 | -0.8560 | lower | yes |
| gravity | gravity (separation between the majority and minority class centres) | -0.2863 | -0.5511 | lower | yes |
| Processing Units Number | model capacity (log processing units) | 0.3080 | 0.4387 | higher | yes |
| Model Capability | learner family's capability rank in the tabular-ML literature (1-10) | 0.3908 | 0.5623 | higher | yes |
| Fitting Regime | how the parameters are reached, closed form to in-context (1-5) | 0.0502 | -0.7870 | lower | no |
| nr_cor_attr | proportion of correlated attribute pairs | -0.1966 | -0.9284 | lower | yes |
| Input Distribution Modelling | how much of the input distribution the learner models (1-5) | -0.1889 | -0.9978 | lower | yes |
| nr_bin | number of binary attributes | 0.0368 | 0.8439 | higher | yes |

Disagreement is what conditioning does, not a defect. A marginal correlation mixes a feature's effect with everything it travels with; inside the equation the terms carrying those companions are already present, so what is left for this feature is what it adds beyond them. The practical consequence: **these statements describe what to expect once the other factors are accounted for, not what a scatter plot of that one feature will show** — and the scatter plot is what a reader will accidentally check against.

<!-- end generated -->

## Limitations of the practice comparison

### A per-feature association can flip sign between equations

`Training Operations` is the worked example. It is the reason the study treats per-feature
associations as evidence rather than as advice — and, in the end, the reason the column was
removed from the corpus altogether. The measurement below is retained because the retirement
does not make the lesson go away; it is what the lesson cost.

| equation | association | effect | confidence |
|---|---|---|---|
| earlier default (14 terms, arity 2, λ=20) | higher training cost → **higher** MCC | 0.16 | moderate |
| a later default (24 terms, arity 3, λ=5) | higher training cost → **lower** MCC | 0.38 | moderate |

Same data, same extraction procedure, same confidence rating, opposite sign — and the later
equation is the better one on every metric, so this is not a case of a bad equation being
corrected by a good one.

**Neither reading is an error.** Each correctly describes the equation it was extracted
from. The problem is the feature: `Training Operations` moves with two things at once. It
rises with model capacity, which raises MCC, and it rises with dataset size, which is where
the hard datasets are. Which of the two an equation ends up expressing depends on what
*other* terms it has available to absorb the other half — and that depends on the grammar,
the penalty and the length, none of which the practitioner reading the practice can see.

**The feature was eventually retired for exactly this.** A column that moves with model
capacity and dataset size at once is not a model descriptor, and two of the other three
retired columns had the same defect. That is a resolution of this particular case, not of the
general problem: nothing guarantees the six current features are free of it, and the mitigations
below are still what stands between a fitted weight and a stated practice.

This generalises past this one feature, and it is the reason the study does not present
per-feature associations as advice. A measurement like "higher training cost went with
lower MCC" is **evidence**, and evidence whose sign depends on the equation it came from is
weak evidence. It is not, on its own, a best practice: a practice is a general
recommendation that a body of evidence can support, and no single fitted equation is that
body.

That is why the generated section below states its practices at the level of received
guidance from the literature and uses this study to weigh each one, rather than reading new
guidance out of the weights. A recommendation that survives being weighed against several
independent measurements is worth something; one extracted from a single equation inherits
that equation'"'"'s instabilities, of which this is a worked example.

Three mitigations are in place and none of them is sufficient:

- terms selected in fewer than half the folds produce no practice, which catches
  instability *within* one configuration but says nothing about instability *across*
  configurations — `Training Operations` sat at 0.66 fold stability in the equation
  published at the time, which was not low enough to suppress it;
- the marginal correlation is printed beside the conditional direction
  (the generated section below), so a reader can at least see when the two disagree, as they
  do here;
- features whose rank direction and decile effect disagree in sign are dropped outright.

What would actually settle it is refitting across a grid of configurations and reporting
only the practices whose sign is stable across all of them. That is a sign-stability
analogue of the fold-stability filter, it is affordable, and it has not been done.
