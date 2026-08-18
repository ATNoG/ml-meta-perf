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
than a least-squares call against the full design. The whole study — seven cross-validated
sweeps, three equations, both protocols — runs in **27 seconds**, and every optimisation
that got it there was verified to leave every output file byte-identical.

| change | what it does | effect |
|---|---|---|
| Gram-matrix arithmetic | a $k\times k$ solve instead of a least-squares call | the design |
| the RSS identity above | removes a matrix-vector product from the inner loop | |
| `np.ix_` replaced by a direct broadcast reshape | skips dtype checks already known to hold | 1.8× on the indexing |
| penalty folded into the full Gram diagonal once | removes a copy and an index array per solve | |
| memoised subsets, vectorised collinearity mask | | 39s → 15s per sweep |
| **stacked solves** | a beam step's candidates go to LAPACK as one `(n, k, k)` array | **1.16M solve calls → 87k** |
| **target ranked once per screen** | Spearman's expensive half hoisted out of the candidate loop | 52k rank calls → 29k |

The last two are the current ones. At $k \le 32$ numpy's per-call wrapper — dtype
promotion, array coercion, the errstate context manager — costs several times the LAPACK
call it guards, so batching the candidates of one beam step into a single stacked solve is
worth 2–10× depending on $k$ while running the identical routine per slice. What remains
in the profile is the arithmetic itself: gathering the $(n, k, k)$ submatrices and the
solves, in that order. There is no Python-level hotspot left above 20%.

### One BLAS thread, deliberately

The solver underneath is whatever LAPACK numpy was built against — for the wheels used
here, the OpenBLAS build that numpy vendors (`numpy.libs/libscipy_openblas64_*.so`, built
`MAX_THREADS=64`). **That is a bundled shared library, not the scipy package**, which this
project does not depend on ([chapter 4](04-evaluation.md) covers the statistics written out
by hand for the same reason).

OpenBLAS threads by default, and on this workload the threads do nothing but spin:

| `OPENBLAS_NUM_THREADS` | wall | CPU |
|---|---|---|
| **1** | **24.7 s** | **24.7 s** |
| 2 | 24.8 s | 33.9 s |
| 4 | 24.7 s | 52.1 s |
| 8 | 25.3 s | 87.7 s |

A 32×32 solve is far below the size where BLAS parallelism pays, so eight threads buy no
wall time and burn 3.5× the CPU, so `OPENBLAS_NUM_THREADS=1` is worth setting in the environment before a run. It is left to the caller: a library that silently pins a global thread count is a library that surprises whoever imports it.

It has to be pinned on the command line rather than inside the package: `python -m metafit`
imports `metafit`, and therefore numpy, and therefore OpenBLAS, *before* `__main__` runs,
and OpenBLAS reads the variable when it loads. Anyone timing this study by calling
`python -m metafit` directly will see the same 25 seconds against five times the CPU.

### Using the host's BLAS instead of the wheel's

`pip install numpy` installs a wheel that **vendors its own OpenBLAS** — a 25 MB
`numpy.libs/libscipy_openblas64_*.so`, built ILP64 with prefixed symbols. It is linked at
build time and there is no runtime switch, so a different BLAS means rebuilding numpy:

```bash
venv/bin/pip install --no-binary numpy --force-reinstall numpy \
  -Csetup-args=-Dblas=openblas -Csetup-args=-Dlapack=openblas
```

That needs the BLAS development files (an `openblas.pc` for pkg-config, plus the headers),
a C compiler and `ninja`. It takes about two minutes the first time on 16 cores; pip caches
the built wheel, so recreating the venv afterwards reuses it and costs seconds.
`venv/bin/pip install --force-reinstall numpy` goes back to the wheel. **Any later
`pip install` that resolves numpy will silently replace a source build with the wheel
again**, since `pyproject.toml` asks only for `numpy>=2.0.0`. The rebuild is deliberately
optional: requiring a compiler and BLAS headers is a heavier ask than the rest of this
project makes, and it changes speed rather than results.

Why do it at all: the wheel's build is chosen for portability rather than for the host. On
the machine these numbers were taken it selects the `Haswell` kernel on a Zen 5 CPU, while
the system build selects `Zen`. Whether that matters is a property of the host, and here it
is not measurable:

| | wall | CPU | results |
|---|---|---|---|
| vendored OpenBLAS (Haswell kernel) | 24.4 s | 27.9 s | — |
| system OpenBLAS (Zen kernel) | 24.6 s | 28.0 s | **byte-identical** |

Every one of the study's twenty output tables is unchanged across the swap, which is the
check that matters more than the timing: a different BLAS can round differently, and
different rounding could in principle change which term the beam selects. It does not here.

Head to head on `dgesv` at the sizes this study actually uses, timed with
[exectimeit](https://github.com/mariolpantunes/exectimeit) — which fits
$T_k = k \cdot t_{\text{exec}} + t_{\text{overhead}}$ and takes the slope, so the timer's
own overhead is removed rather than averaged over:

| n | vendored (Haswell) | system (Zen) |
|---|---|---|
| 8 | 7.88 ± 0.74 µs | 7.92 ± 0.46 µs |
| 16 | 9.72 ± 0.89 µs | 9.46 ± 1.05 µs |
| 24 | 12.52 ± 1.50 µs | 12.58 ± 1.51 µs |
| 32 | 16.68 ± 3.19 µs | 16.90 ± 2.22 µs |

Every difference is inside two standard errors. A plain timing loop had reported a
consistent 5–6% gap at n = 24 and 32 which **vanished** once the measurement was done
properly — which is the argument for measuring it this way rather than with a stopwatch
around a loop. The mechanism agrees: a 32×32 system is 8 KB and sits in L1, so there is no
cache behaviour for a tuned kernel to improve. A better-tuned BLAS pays on large matrix
products, and this study never forms one.

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

A different way of finding terms was built and measured, structurally identical to
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

**It is cheaper and worse, and the reason is not the search at all — it is what each method
generates.**

| | terms *generated* | admissible | how it scales |
|---|---|---|---|
| enumeration | **564** | 172 | O(*n*² × operations) |
| guided merge | **22** (17 leaves + 5 merges) | 22 | O(*n*) |

Enumeration writes down *every* expression the grammar allows and then filters. A merge
only ever produces terms along its merge path, and a path over *n* items has at most
*n − 1* joins before everything is one cluster. **Seventeen features can yield at most
sixteen merges, whatever the linkage rule, however it is tuned.** The two methods are not
searching the same space differently; one enumerates the space and the other walks a path
through it.

Three tuning attempts confirm the ceiling is structural rather than a parameter choice:

| variant | evaluations | pool | in-sample (k=14) |
|---|---|---|---|
| greedy, one merge per round | 200 | 17 | 0.4865 |
| **all improving merges per round** | **100** | 17 | 0.4865 |
| greedy + feature reuse | 536 | 19 | 0.4865 |
| all improving + feature reuse | 180 | 19 | 0.4865 |

Accepting every improving merge rather than only the best **halves the evaluations and
changes nothing else** — it reaches the same terms sooner. Retaining parents so a feature
can join several merges adds exactly two terms, and *neither is ever selected*: the fitted
equation is byte-identical. Being less greedy and reusing features both help in principle,
and neither can lift an O(*n*) generator to an O(*n*²) one.

Guiding a merge would pay where enumeration is infeasible — hundreds of features, or a
grammar deep enough that writing every expression down is impossible. At *n* = 17,
enumeration costs one pass and yields eight times the terms. The honest reading is that
this meta-dataset is too small **in its feature dimension** for construction to beat
exhaustion.

**Status: measured, rejected, and no longer in the tree.** It lived at
`src/metafit/construct.py` with its own tests until the repository was cut back to the code
the study actually runs; `git log -- src/metafit/construct.py` recovers it. The measurement
is kept here because a documented negative result is worth more than a deleted one — "why
not build terms by clustering instead of enumerating them?" is the first question a reader
will ask, and the answer is measured rather than asserted. Its ideas were
also tried as a *filter* on the enumerated library rather than a replacement for it, and
that fails too, for the reason given above: any filter over marginal impact discards the
weak-but-complementary terms the equation depends on.

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
| knee of the in-sample curve | 4 | 0.488 | 0.215 |
| knee of the cross-validated curve | 8 | 0.563 | 0.401 |
| **best cross-validated** | **20** | **0.614** | **0.478** |

**Twenty is the headline, and the three rules still disagree.** Both knees land short of it,
the in-sample knee badly so: it costs 0.26 of transfer to save sixteen terms. The knee is
the right question for the in-sample curve, which is monotone and flattens; it is the wrong
question for the cross-validated curve, which is not monotone at 20 groups.

### Sixteen terms is the shorter equation worth knowing about

| terms | in-sample R² | LOO-dataset R² | LOO-dataset MAE |
|---|---|---|---|
| 16 | 0.6033 | 0.4744 | **0.1776** |
| **20** | **0.6141** | **0.4779** | 0.1788 |

Four more terms buy **+0.0035 of transfer** and +0.011 of fit, and *cost* a thousandth of
MAE. On this configuration the two lengths are all but indistinguishable on transfer, so the
choice is almost purely about readability — which is a better position to be in than the
previous grammar's, where the same comparison was worth +0.051 and the trade was real.

The study publishes 20 because the stated rule selects it. Anyone reproducing this with a
stricter readability budget should take 16 and lose essentially nothing. The rule was fixed
before the numbers were in, and the fact that it now selects a length whose MAE is very
slightly worse than its neighbour's is exactly the kind of thing a pre-stated rule is
supposed to survive.

Both **Pareto fronts** are also reported. Over (length, LOO-dataset R²) the front is 2, 8,
12, 16 and 20 — nothing longer than 20 terms earns its length on transfer. Over
(length, in-sample R²) *every* length is on the front, because fit is monotone in terms and
so nothing is ever dominated. That is precisely why the in-sample curve cannot choose a
length by itself and the knee detector exists for it.

![Accuracy versus equation length](../figures/term_count_curve.png)
