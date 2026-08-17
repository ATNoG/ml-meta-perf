# 7. From equation to practice

*Implemented in `metafit.attribution`, `metafit.practices` and `metafit.report`.*

This is what the accuracy was traded for. An equation nobody can turn into guidance has
bought nothing over a black box.

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

## The extracted practices

Reproduced from the published 24-term E2; [chapter 9](09-report.md) is the generated
version and is regenerated with the equation, so it is the one to trust if the two ever
disagree.

| # | practice | effect (MCC) | confidence |
|---|---|---|---|
| 1 | Higher **built-in outlier robustness** → **higher** MCC | 0.42 | moderate |
| 2 | Higher **training cost** (log operations) → **lower** MCC | 0.38 | moderate |
| 3 | Higher **number of normally distributed attributes** → **lower** MCC | 0.33 | moderate |
| 4 | Higher **class entropy** → **lower** MCC | 0.24 | moderate |
| 5 | Higher **noise-to-signal ratio** → **lower** MCC | 0.17 | moderate |
| 6 | Higher **proportion of correlated attributes** → **higher** MCC | 0.02 | weak |
| 7 | Higher **number of instances** → **lower** MCC | 0.02 | weak |

![Feature effects](../figures/practice_effects.png)

Read together: *prefer an outlier-robust model; expect trouble on noisy data, on data
whose labels are evenly spread across many classes, and on data whose attributes are
mostly well-behaved normal ones.*

## What these are not

The defensible claim is directional, and the tables in this chapter should be read as a
set of signed statements rather than as a ranking. Any write-up that orders these
practices by effect size is claiming more than the evidence supports.

**Associations measured across 20 datasets, not causal claims.** "Higher training cost
went with lower MCC" does not mean that cheaper models are better; it means that among the
models run here, the expensive ones were not the ones that scored well on the datasets
where they were expensive — and training cost is partly a function of dataset size, so it
is carrying data difficulty as well as model capacity.

### A direction that flipped, and what it means

`Training Operations` is worth dwelling on. An earlier, shorter equation (14 terms, R²
0.558) put it at *higher* MCC with moderate confidence; the published 24-term equation puts
it at *lower* MCC, also with moderate confidence, and with more than twice the effect. Same
data, same method, different equation length.

This is not a defect in the extraction — both readings are correct descriptions of their
own equation. It is a statement about the feature: `Training Operations` is not a clean
signal in this meta-data. It rises with model capacity, which helps, and it also rises with
dataset size, which is where the hard datasets are. Which of the two an equation ends up
expressing depends on what else it has available to soak up the other. **A practice whose
sign depends on the equation it was extracted from is not a practice**, and this one is
reported here mainly as the worked example of why the confidence column is not enough on
its own.

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

## What the weights say about the equation itself

Two properties of the published equation are visible only in the importance table, and
both temper what the practices above are worth.

**The equation is flat.** Its 24 weights behave like **20.2 equally-weighted terms**
(inverse Simpson index of the standardised-weight shares, $1/\sum_i s_i^2$), and the
largest single term carries 8.9% of the mass. Sixteen terms are needed to reach 80%. This
is a weaker kind of explanation than a short equation with one dominant term: the guidance
rests on many small contributions, none of which is individually load-bearing. It is
directly what the accuracy at this length costs, and it is the honest answer to "which term
should I pay attention to" — mostly, none of them alone.

**Four of the sixteen major terms were selected by fewer than half of the folds**,
including the very largest. A large standardised weight with a low selection frequency
means the term is doing its work for *this* training set and would be replaced on another;
printed as a coefficient it looks exactly like a stable one. `report.unstable_majors`
extracts them so they cannot be quietly read as findings. On the published equation the
rank-1 term — `(log(gravity) + log(ns_ratio)) / log(Training Operations)`, 8.9% of the mass
— appears in only 20% of folds, which is why no practice in the table above rests on
`gravity`.

## Reading a single prediction

`examples/predict_new_dataset.py` prints the per-term contribution breakdown for one row,
which is the interpretability payoff in its most direct form:

```
dataset : 5G_Slicing
model   : AdaBoost
actual  : +1.0000
predicted: +0.9602

contribution breakdown:
     +0.8295   intercept
     +0.3296   Training Operations
     -0.3094   [log(eq_num_attr)] * [log(nr_class)]
     +0.2032   sqrt(Prediction Operations)
     ...
```

## The analysis is generated, not authored

There is a failure mode this chapter has to avoid. If the fitted equation is printed and
somebody then sits down to explain which terms matter and what they imply, the explanation
is the product and the equation is only its raw material — and the claim "this model is
interpretable" quietly becomes "this model was interpreted by an expert, once".

So `metafit.report` derives the written analysis arithmetically, and
[chapter 9](09-report.md) is its output. The same equation always produces the same
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
