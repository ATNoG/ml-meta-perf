# 3. Search and fitting

*Implemented in `metafit.fit`, `metafit.analysis` and `metafit.selection`.*

Two problems are involved and only one of them is hard.

| problem | nature | method |
|---|---|---|
| Given a set of terms, what are the best weights? | linear | `numpy.linalg.solve` on the ridge normal equations — **exact, no iteration** |
| Which $k$ of ~360 candidate terms? | combinatorial | beam search with local refinement |

Handing every term to `lstsq` at once gives in-sample R² 0.656 and leave-one-dataset-out
R² of **-5.94**; the 14-term selected equation gets 0.443. The solver is not the hard
part.

## Stage 1 — correlation screening

Every candidate term is scored by **Pearson and Spearman** correlation against MCC. The
two disagree informatively:

- when they agree, the relation is linear and a plain `f` is the right term;
- when Spearman is clearly the larger, the relation is monotone but curved, which is the
  signal that a log, an inverse or a ratio will pay for itself.

`metafit.fit.transform_gap` is exactly $|\rho_s| - |\rho_p|$, and `guided_screen` ranks by
the stronger of the two while applying a small penalty to terms that are only monotone,
because linear terms read more simply. Near-duplicates of an already-kept term are
dropped so the beam does not spend its width on variations of one idea.

### Within-group screening

`metafit.analysis.screen` additionally centres **both the term and the target inside each
group** before correlating. A term is then credited only for variance that group identity
does not already explain.

This column is a diagnostic, not a ranking of importance, and misreading it is easy: it
removes dataset variance *by construction*, so only model-feature terms can score well
there. It answers "does this term explain anything *within* a dataset", which is the right
question for model effects and the wrong one for comparing dataset against model
importance.

## Stage 2 — beam search over subsets

Greedy forward selection commits to its first term permanently, and on a collinear
library that first commitment is often wrong. A beam keeps several partial equations
alive.

At each size the search:

1. scores every pooled candidate by its correlation with the current residual;
2. rejects candidates whose absolute correlation with an already-chosen term exceeds
   **0.95** — the collinearity guard, and what keeps the printed equation readable;
3. evaluates the best `candidates` extensions of each beam member with an exact ridge
   solve;
4. keeps the `beam_width` best by residual sum of squares;
5. refines the leader by **single-term swaps** until no swap improves it.

Because the beam records its best subset at **every** size, one search yields the whole
accuracy-versus-length curve, and a cross-validated sweep over lengths costs one search
per fold rather than one per length.

## The ridge solve

For a centred, standardised design, centring both sides removes the intercept from the
system, so no penalty is ever applied to it — shrinking an intercept toward zero would
bias every prediction toward MCC = 0, a value this data goes nowhere near.

$$(G + \lambda I)\,w = b, \qquad G = Z^\top Z, \quad b = Z^\top (y - \bar{y})$$

The residual sum of squares uses an identity rather than a second quadratic form. Since
$(G + \lambda I) w = b$ implies $w^\top G w = w^\top b - \lambda w^\top w$,

$$\mathrm{RSS} = \|y - \bar y\|^2 - 2 w^\top b + w^\top G w = \|y - \bar y\|^2 - w^\top b - \lambda w^\top w$$

which removes a matrix-vector product from the inner loop. The test suite checks the
identity against the literal definition across penalties and subset sizes.

### Cost

Every candidate refit is a $k \times k$ solve against a precomputed Gram matrix rather
than a least-squares call against the full design. A full 32-term leave-one-dataset-out
sweep runs in **15 seconds**, down from 39 after three optimisations that changed no
result (`np.ix_` called once rather than twice, the penalty added along a diagonal rather
than via `penalty * np.eye(k)`, and the RSS identity above), plus memoisation of subsets
and a vectorised collinearity mask.

## Negative result: the search is already converged

Accuracy is **not** limited by search strength. Sweeping beam width, candidate pool and
refinement rounds over an order of magnitude each:

| setting | beam | candidates | rounds | in-sample (k=14) | in-sample (k=20) |
|---|---|---|---|---|---|
| baseline | 6 | 24 | 2 | 0.5582 | 0.5662 |
| wide beam | 16 | 24 | 2 | 0.5582 | 0.5663 |
| more candidates | 6 | 64 | 2 | 0.5582 | 0.5661 |
| more rounds | 6 | 24 | 6 | 0.5582 | 0.5662 |
| maximal | 48 | 96 | 10 | 0.5582 | 0.5663 |

(default library; the wider library gains ~0.01 non-monotonically, which is search noise
rather than systematic improvement.)

Spending eight times the compute changes the fourth decimal place. Together with the
negative result on a richer vocabulary ([chapter 2](02-equation-form.md)), this locates
the limit in the model *form*, not in the optimiser.

**Per-length penalty tuning** was also tested: choosing $\lambda$ separately for each $k$
lifts leave-one-dataset-out R² at $k=14$ from 0.4429 to 0.4456. A gain of 0.003 does not
justify the extra configuration surface, and one global penalty is retained.

## An alternative that was built and measured: agglomerative construction

`metafit.construct` implements a different way of finding terms, structurally identical to
**hierarchical clustering**. Every feature starts as a singleton, straightened by whichever
transform makes it most linear in MCC. At each step the pair whose merge — under one of the
grammar's operations — becomes most linear in MCC is joined; features no merge improves are
left isolated; the process stops when no merge helps, so depth is discovered rather than
imposed.

The linkage criterion is **linearity**, because linearity is what the downstream model can
actually use: an equation is a weighted sum, so a term linear in MCC contributes perfectly
with weight 1, while a term that is monotone but curved contributes badly at any weight.

Three variants were measured against the enumerated library, with construction rebuilt
inside every fold (it consumes the target, so anything else would rig the comparison).

### It finds terms enumeration misses

| library | in-sample k=4 | k=14 | k=20 |
|---|---|---|---|
| enumerated (172 terms) | 0.4666 | 0.5582 | 0.5662 |
| **enumerated ∪ agglomerated** | **0.4935** | **0.5606** | **0.5690** |

The gain is largest at **short** equations — +0.027 R² at four terms — which is the
interpretability regime that matters most. The reason is specific: enumeration fixes
composite operands to `log` or `id`, while agglomeration picks the transform per operand by
linearity, so it can build `[sqrt(ns_ratio)] / [log(Processing Units Number)]` where
enumeration cannot.

### Target-free pairing adds nothing

Pairing features by their correlation with *each other* rather than with the target
(`structural_terms`) would avoid the per-fold cost entirely. It contributes nothing: at a
0.7 correlation threshold the union library is **172 terms, unchanged**. Every term it
proposes is already in the enumerated library.

That is a fact about the size of this problem rather than about the idea. With 17 features,
exhaustive depth-2 enumeration is *complete*, so any pair-selection heuristic can only
return a subset of it. Selection heuristics start to pay when the space is too large to
enumerate, which at 17 features it is not.

### The dendrogram cut: clusters as the equation

Taken to its conclusion, the clustering analogy removes the selection step entirely.
`cluster_terms` merges until exactly *k* clusters remain and returns them as the equation's
*k* terms. The cut level **is** the equation length, and least squares is left with nothing
to do but the weights. One mechanism instead of two.

| cut at | in-sample R² | max term depth |
|---|---|---|
| 2 | 0.267 | 4 |
| 4 | 0.370 | 4 |
| 6 | 0.408 | 3 |
| 10 | 0.443 | 3 |
| 14 | 0.471 | 2 |

It fits worse than beam selection at the same length, which is expected: it has no freedom
to choose *which* terms, only how to group all of them.

**The hybrid is the version that works.** `dendrogram_terms` runs the merge all the way to
a single cluster and keeps every intermediate node as a candidate — `2n - 1` nodes, so a
shortlist of about 23 admissible terms against the enumerated library's 172. Beam search
then selects from that shortlist:

| library | terms available | in-sample R² (k=14) | LOO-dataset |
|---|---|---|---|
| enumerated | 172 | 0.5582 | +0.4429 |
| **dendrogram shortlist** | **23** | **0.5344** | +0.3659 |

A pool an eighth the size reaches within 0.024 R² of it. Construction proposes, selection
disposes, and the shortlist has a reason behind every entry.

**Match the admissibility cap when comparing.** Construction applies `max_abs_zscore` while
building, and enumeration applies it while filtering; comparing a library built under one
cap against a library built under another measures the caps rather than the methods. All
construction functions take the parameter explicitly for this reason — an earlier version
of this chapter compared a cap-8 construction against a cap-3 enumeration and drew a
conclusion from it that did not survive matching them (see the correction in
[chapter 7](07-practices.md)).

Deep cuts also show the readability cost directly. The four-term cut contains

```
[[log(gravity)] / [log(inst_to_attr)]] * [[[nr_cor_attr] * [nr_norm^2]] *
    [[[1/class_ent] * [nr_bin^2]] * [[nr_outliers^2] / [log(nr_attr)]]]]
```

which is one term by the equation's accounting and unreadable by any other. Depth is
capped for this reason, and the cap is the parameter that decides whether "short equation"
means anything.

### A guided merge: cheaper, and worse

If a merge step is itself a search, it has not reduced the work the beam search has to do
— it has moved it. `guided_merge` is the version that decides in advance rather than
searching: **which** terms are worth merging (Spearman with MCC above a floor, since a
monotone relationship is one a merge can straighten), **which pairs** (ranked by
`co_movement`, the correlation between the two terms in log space), and **which operation**
— one per pair, ratio when the pair shares a growth component and a division has something
to cancel, product when it does not.

It works as designed and it is cheaper:

| pool builder | candidate evaluations | pool size | in-sample (k=14) | LOO-dataset |
|---|---|---|---|---|
| brute-force dendrogram | 1328 | 23 | 0.534 | +0.366 |
| **guided** | **200** (6.6× fewer) | 17 | 0.486 | +0.167 |
| guided + feature reuse | 536 | 19 | 0.486 | +0.170 |
| enumeration | 0 (one pass over the grammar) | 172 | **0.558** | **+0.443** |

**It is cheaper and worse, and the reason is that the search was never the bottleneck.**
Picking one operation per pair evaluates a third as many candidates, but when the chosen
operation is the wrong one the merge is lost rather than merely delayed. The pool shrinks
from 23 to 17, and pool size is the binding constraint here — enumeration wins with 172.

Allowing features to be reused across merges (`reuse_features=True`) grows the pool from 17
to 19 and changes the results by less than fold noise. The ceiling is arithmetic: *n*
features give O(*n*) merge products, against O(*n*² · transforms) for enumeration. At
*n* = 17 that is not a close contest.

Guiding a merge would pay where enumeration is infeasible — many more features, or a deeper
grammar. It does not pay here, and the honest reading is that this dataset is too small in
its *feature* dimension for construction to beat exhaustion.

### Nesting does not pay on this data

Allowing merged terms to merge again was implemented and measured. Even with the merge
threshold set to zero — accepting *any* improvement in linearity — only 2 of 14 constructed
terms exceed depth 1, and in-sample R² at 8 terms moves from 0.4538 to 0.4616.

The mechanism works; the data does not reward it. Composing an already-composed term
rarely makes it more linear in MCC, which is consistent with everything else here: the
missing structure is a rank-1 interaction ([chapter 5](05-oracles.md)), and no amount of
nesting products and ratios of single features reproduces it.

### Every increase in expressiveness costs transfer

| library | in-sample (k=14) | LOO-dataset (k=14) |
|---|---|---|
| enumerated (172) | 0.5582 | **+0.4429** |
| ∪ agglomerated | 0.5606 | +0.3759 |
| ∪ all-transform composites (427) | **0.5768** | +0.2942 |
| richer unary vocabulary | 0.5582 | +0.4196 |

Enumerating composites over *every* transform-atom rather than just the log/id compression
— 2336 additional terms — buys the best in-sample figures anywhere in this study (0.5893 at
20 terms) and drops leave-one-dataset-out from 0.443 to 0.294.

Four independent expansions of the library, all pointing the same way. Taken with the
negative results on search strength and on unary vocabulary, the conclusion is that
**library expressiveness on 20 datasets is already at or past its useful limit**. The
binding constraint is the sample, not the search.

The module ships, tested, because the alternative is worth documenting and the finding is
worth reproducing. It is **not** wired into the default pipeline.

## Stage 3 — choosing the number of terms

Picking the bend of the curve by eye is the kind of judgement this project exists to
remove from its results, so it is delegated to a detector.
[kneeliverse](https://github.com/mariolpantunes/knee)'s `autoelbow` takes no threshold,
sensitivity or smoothing window, so the chosen length is a property of the curve rather
than of a parameter chosen to produce a preferred answer.

Three rules are reported rather than one:

| rule | terms | in-sample R² | LOO-dataset R² |
|---|---|---|---|
| knee of the in-sample curve | 8 | 0.533 | 0.303 |
| knee of the cross-validated curve | 12 | 0.556 | 0.371 |
| **best cross-validated** | **14** | **0.558** | **0.443** |

Fourteen is the headline, and two independent metrics agree on it: it is both the maximum
of leave-one-dataset-out R² and the minimum of leave-one-dataset-out MAE (0.1884).

Both **Pareto fronts** are also reported. Over (length, LOO-dataset R²) the front is 2, 4,
8, 10, 12, 14 — nothing longer than 14 terms earns its length on transfer. Over (length,
in-sample R²) *every* length is on the front, because fit is monotone in terms and so
nothing is ever dominated. That is precisely why the in-sample curve cannot choose a
length by itself and the knee detector exists for it.

![Accuracy versus equation length](../figures/term_count_curve.png)
