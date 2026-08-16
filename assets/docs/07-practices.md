# 7. From equation to practice

*Implemented in `metafit.attribution` and `metafit.practices`.*

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

| # | practice | effect (MCC) | confidence |
|---|---|---|---|
| 1 | Higher **effective feature count** → **lower** MCC | 0.53 | moderate |
| 2 | Higher **gravity** (class-centre separation) → **lower** MCC | 0.42 | strong |
| 3 | Higher **model capacity** (log processing units) → **higher** MCC | 0.27 | moderate |
| 4 | Higher **number of attributes** → **higher** MCC | 0.23 | moderate |
| 5 | Higher **number of binary attributes** → **higher** MCC | 0.22 | moderate |
| 6 | Higher **inference cost** (log operations) → **higher** MCC | 0.20 | strong |
| 7 | Higher **training cost** (log operations) → **higher** MCC | 0.16 | moderate |
| 8 | Higher **built-in outlier robustness** → **higher** MCC | 0.13 | strong |
| 9 | Higher **proportion of correlated attributes** → **higher** MCC | 0.04 | weak |

![Feature effects](../figures/practice_effects.png)

Read together: *pick a bigger, more expensive, outlier-robust model; expect trouble on
data with a high effective feature count and well-separated class centres.*

## Are these practices a property of the data, or of the search?

The sharpest test available. Fit equations by methods that share no machinery, then
compare the *guidance* rather than the R2.

**Admissibility must be held constant for this to mean anything.** Construction and
enumeration each apply a `max_abs_zscore` cap, and comparing a library built under a loose
cap with one built under a strict cap measures the caps, not the methods. Every row below
uses the same cap on both sides.

At `max_abs_zscore = 3`, against the enumerated 14-term equation (in-sample R2 0.5582):

| method | terms | in-sample R2 | shared | direction agreement | effect-rank rho |
|---|---|---|---|---|---|
| hybrid — dendrogram shortlist, then selection | 14 | **0.5344** | 12 | **91.7%** | **+0.699** |
| parameter-free — merge until linear enough | 12 | 0.4600 | 13 | 61.5% | +0.440 |
| dendrogram cut, no selection | 6 | 0.4251 | 14 | 64.3% | +0.565 |

At `max_abs_zscore = 8` the ordering holds and agreement runs 58–69%.

### What this does and does not support

**Supported:** a method that fits comparably agrees substantially. The hybrid reaches
0.5344 against the enumerated 0.5582 — from a pool of **23 terms rather than 172** — and
agrees on the direction of 11 of 12 shared features, with effect magnitudes correlating at
+0.70. Two genuinely different searches, nearly equal accuracy, nearly the same advice.

**Not supported:** that the practices are independent of the search *in general*.
Concordance tracks accuracy. The methods that fit substantially worse (0.43–0.46) agree on
only 58–69% of directions — better than the 50% a coin would give, but not the robustness
a strong claim would need.

The honest statement is conditional: **among equations that fit about as well, the
extracted directions are largely stable; among equations that fit materially worse, they
are not.** That is weaker than method-independence, and it is what the data shows.

> **Correction.** An earlier version of this chapter reported 100% agreement (9 of 9)
> between the enumerated equation and a 6-term dendrogram cut. That comparison was invalid:
> the cut was constructed under the default cap of 8 and then fitted with admissibility
> effectively disabled, while the enumerated library used a cap of 3. Under matched caps
> the same comparison gives 64.3%. The construction functions now take `max_abs_zscore`
> explicitly so the mismatch cannot recur silently.

## What these are not

The defensible claim is directional, and the tables in this chapter should be read as a
set of signed statements rather than as a ranking. Any write-up that orders these
practices by effect size is claiming more than the evidence supports.

**Associations measured across 20 datasets, not causal claims.** "Higher training cost
went with higher MCC" does not mean padding a model with FLOPs raises MCC; it means the
models that scored well here were the expensive ones. Training cost is a proxy for model
capacity, and capacity is what is doing the work.

The `gravity` result deserves the most caution: the direction is stable and the effect
large, but a single term carries it.

Twenty datasets from one domain is a narrow evidential base. Every statement above should
be read as a hypothesis this data is consistent with, not a finding established by it.

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
