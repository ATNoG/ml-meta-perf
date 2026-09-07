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
action, it is true only of this corpus, and [chapter 7](07-limitations.md) shows that its
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
[chapter 10](10-report.md) opens the section with the current one. What is worth stating is
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
recorded in [chapter 7](07-limitations.md#training-set-size-is-not-a-variable-here) because
the near-miss is instructive — the split produced a clean-looking number, and the number
meant nothing.

[Chapter 10](10-report.md) carries the full assessment, regenerated on every run.

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

The evidence layer. **The table itself is not reproduced here.** It was, and it went on
describing a 20-term equation over features the corpus no longer carries — the same failure
the block table below was removed for. It lives in [chapter 10](10-report.md), which is
regenerated with the equation on every run, and that is the version to quote.

What belongs here is the caveat that has to travel with one of its rows wherever it is
quoted. `Model Capability` is the one feature in the set that was *asserted* rather than
measured, so "higher capability rank went with higher MCC" is partly the ladder being read
back out.

What is not built in is the conditional part — how family capability interacts with dataset properties — and
[chapter 7](07-limitations.md) sets out the three costs in full.

![Feature effects](../figures/practice_effects.png)

Read together these point somewhere — *prefer a capable, high-capacity model; expect
trouble on noisy data and on data spread across many classes* — but the pointing
is the reader's inference, not the table's content. Turning it into advice, and checking
that advice against something other than this one equation, is the next section's job.

## What these are not

**They are not best practices.** Each names a meta-feature column, which is a thing to
measure rather than a thing to do, and each is true of one equation on one corpus. The
practices are in [chapter 10](10-report.md), taken from the literature and weighed against
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
[chapter 7](07-limitations.md#a-per-feature-association-can-flip-sign-between-equations), because the
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
move with the equation — [chapter 10](10-report.md) is the regenerated version — but the
*shape* is stable across every configuration tried: no term dominates, and no small handful
of them accounts for the result.

The additive form $f(X) = w_0t_0 + w_1t_1 + \dots$ invites reading one term at a time, and
a flat equation refuses that. **This is a statement about the unit of explanation, not
about the quality of the equation.** MCC here is inferred by a *set* of terms acting
together, and the right response is to change the unit rather than to conclude that the
equation cannot be read. Three units work, and `ml_meta_perf.report` generates all three into
[chapter 10](10-report.md).

### 1. Blocks of terms that move together

Terms whose per-row contributions correlate say the same thing about a row and can be read
as one. Grouping is on the *contributions* rather than on shared features — two terms can
share no feature and still track each other, and two terms over the same feature can move
independently once their transforms differ.

`shared` names the features a strict majority of a block's terms contain — a genuine
question, since the grouping never looked at what the terms contained.

**The block table is regenerated on every run and lives in [chapter 10](10-report.md).** An
earlier draft copied it into this chapter, where it went on describing a 24-term equation over
features the corpus no longer carries; the listing now stays on the generated side for the
same reason E3 itself does.

The blocks themselves are **not listed here**, for the reason the table above is not: they
are recomputed from the fitted equation on every run, and a copy in this chapter goes stale
silently. An earlier version of this section described three blocks, one of which was said
to pair capacity with inference cost — a reading of a 24-term equation over columns the
corpus dropped in 2026-09-05.

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

`Processing Units Number` appears in 7 of 16 terms and `gravity` in 5; everything else is
spread thinner, 1–3 terms each. The full table, with the transforms
and operations each feature was used under, is in [chapter 10](10-report.md).

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
[chapter 10](10-report.md) is its output. The same equation always produces the same
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
