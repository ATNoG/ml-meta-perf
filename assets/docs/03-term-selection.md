# 3. Term generation and selection

*Implemented in `ml_meta_perf.search`, `ml_meta_perf.fit`, `ml_meta_perf.analysis` and `ml_meta_perf.selection`.*

Two problems are involved and only one of them is hard.

| problem | nature | method |
|---|---|---|
| Given a set of terms, what are the best weights? | linear | `numpy.linalg.solve` on the ridge normal equations — **exact, no iteration** |
| Which $k$ of a few hundred candidate terms? | combinatorial | beam search with local refinement |

Handing every term to `lstsq` at once fits *better* in-sample than the published equation and
transfers catastrophically — the generated section at the foot of this chapter measures both,
against the published equation, on this run. The solver is not the hard part; the sample size
is, and everything below is about spending it well.

## Stage 1 — correlation screening

Every candidate term is scored by **Pearson and Spearman** correlation against MCC. The
two disagree informatively:

- when they agree, the relation is linear and a plain `f` is the right term;
- when Spearman is clearly the larger, the relation is monotone but curved, which is the
  signal that a log, an inverse or a ratio will pay for itself.

`ml_meta_perf.search.transform_gap` is exactly $|\rho_s| - |\rho_p|$, and `guided_screen` ranks by
the stronger of the two while applying a small penalty to terms that are only monotone,
because linear terms read more simply. Near-duplicates of an already-kept term are
dropped so the beam does not spend its width on variations of one idea.

### Within-group screening

`ml_meta_perf.analysis.screen` additionally centres **both the term and the target inside each
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
than a least-squares call against the full design. Everything this study fits — the term
sweeps, four equations across two grammars, and all four protocols including the
doubly-held-out cell — runs in **24 seconds**, and every optimisation that got it there was
verified to leave every output file byte-identical.

A full `python -m ml_meta_perf` takes about four minutes, and the other 215 seconds are the
opaque comparison in [chapter 5](05-evaluation.md#what-an-opaque-model-reaches-and-does-not):
three scikit-learn regressors refitted once per observed cell, 476 times each. That the
priced *alternative* to a readable equation costs an order of magnitude more than the
equation is not the point of this section, but it is not nothing either.

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
project does not depend on ([chapter 5](05-evaluation.md) covers the statistics written out
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

It has to be pinned on the command line rather than inside the package: `python -m ml_meta_perf`
imports `ml-meta-perf`, and therefore numpy, and therefore OpenBLAS, *before* `__main__` runs,
and OpenBLAS reads the variable when it loads. Anyone timing this study by calling
`python -m ml_meta_perf` directly will see the same 25 seconds against five times the CPU.

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

## Is the search the limitation? No

Accuracy is **not** limited by search strength. Sweeping beam width, candidate pool and
refinement rounds over an order of magnitude each:

| setting | beam | candidates | rounds | in-sample (k=14) | in-sample (k=20) |
|---|---|---|---|---|---|
| baseline | 6 | 24 | 2 | 0.5582 | 0.5662 |
| wide beam | 16 | 24 | 2 | 0.5582 | 0.5663 |
| more candidates | 6 | 64 | 2 | 0.5582 | 0.5661 |
| more rounds | 6 | 24 | 6 | 0.5582 | 0.5662 |
| maximal | 48 | 96 | 10 | 0.5582 | 0.5663 |

<sub>Default library. The wider library gains ~0.01 non-monotonically, which is search noise
rather than systematic improvement. **Measured before the 2026-09-05 protocol change**, so
these are re-selecting numbers and are not comparable with any current figure — the
in-sample column here is not the in-sample column of chapter 5. What the table supports is a
statement about *differences between search settings*, and those are internally consistent
because every row was measured the same way.</sub>

Spending eight times the compute changes the fourth decimal place. Together with the
negative result on a richer vocabulary ([chapter 2](02-additive-model.md)), this locates
the limit in the model *form*, not in the optimiser.

**Per-length penalty tuning** was also tested: choosing $\lambda$ separately for each $k$
lifted leave-one-dataset-out R² at $k=14$ from 0.4429 to 0.4456 — again on the pre-2026-09-05
re-selecting protocol, so read the *gain* and not the level. A gain of 0.003 does not justify
the extra configuration surface, and one global penalty is retained.

## Stage 3 — choosing the number of terms

More terms fit better and read worse. Picking the bend of that curve by eye is the kind of
judgement this project exists to remove from its results, so the length comes from a stated
rule with no threshold, no smoothing window and no sensitivity parameter — making the chosen
length a property of the curve rather than of a parameter picked to produce a preferred
answer, and one that re-derives itself when the corpus changes.

Two decisions have to be made before the detector is run, and both were previously made
badly enough to invalidate the answer.

### Which curve the detector runs on

**Not in-sample R², which is what `knee_terms` used to default to.** In-sample is monotone in
the number of terms — adding a term cannot reduce the fit — so it can only ever say "more".
A length chosen on it is chosen on the one curve that cannot express the trade the choice is
about.

**And not one cross-validated curve either.** On twenty groups they wander, and
leave-one-dataset-out on this corpus has genuine craters — lengths where the held-out number
collapses by a fifth of the scale while its two neighbours are untouched. That is not noise,
and it is not a property of the length either: at such a length one held-out dataset sits
outside the convex hull of the other nineteen in term space, where a linear equation
extrapolates without limit and `validate._clip_to_training` pins the fold to its training
floor. `ASNM-CDX-2009` is the fold this happens to. One fold's extrapolation should not
choose the published length.

So the rule reads **every protocol at once**, and it takes the **minimum** —
`selection.floor_curve`, the worst of the four at each length, including the doubly-held-out
cell. A length is judged by its weakest showing, so a length that is strong in-sample and
craters when both groups are held out cannot be selected on the strength of the first.

A second reading is computed and reported beside it: `selection.consensus_curve`, the
per-length **median** of the three single-group protocols. The median is robust in a different
way — it discards a crater where a mean would be dragged down by it — and it is what the
chapters plot. The two agree on the current corpus, both selecting 15 terms under the
parsimonious grammar and 23 under the full one, and they are kept apart because agreement is a
result rather than a guarantee. **Which length craters moves with the configuration**, so the
worked example is generated rather than written here — the section below names the deepest one
on the current curve and gives the median and the mean side by side.

### Which lengths the curve is reported at

**Every length from 1 to `max_terms`**, on a uniform grid, and this matters more than it
sounds. A non-uniform grid — `(2, 4, 8, 12, 16, 20, 24, 26, 28, 32)`, say — skips 13, 15, 17
and 31, which are exactly the four lengths where the transfer curve craters. Any rule reading
such a curve is partly reading the grid: the same detector returns 4 terms on that grid and 6
on the dense one. Reporting every length costs nothing, because the beam search already builds
the whole path.

![Accuracy versus equation length](../figures/02_term_count_curve.png)

The craters are visible in that figure. They are a real property of leave-one-dataset-out on
twenty groups and they belong on the plot.

### The rule, and everything it beats

**The rule is the argmax of the worst protocol at each length**, implemented as
`selection.floor_argmax`. Stated in full, so that nothing about it has to be taken on trust:

1. Fit the equation at every length from 1 to `max_terms`, and score each length under all
   four protocols — in-sample, leave-one-dataset-out, leave-one-model-out, and the
   doubly-held-out cell ([chapter 5](05-evaluation.md#four-protocols)).
2. For each length, take the **minimum** of those four R² values. That is the floor curve.
3. Publish the length where the floor is highest.

It has no threshold, no smoothing window and no sensitivity parameter, so the chosen length
is a property of the curve rather than of a value picked to produce a preferred answer, and
it re-derives itself when the corpus changes rather than needing to be re-tuned by hand.
**Nothing in the rule mentions how many rows the corpus has** — an earlier version priced a
term against `n` and would have selected a different length on a corpus of a different size,
which is the defect `tests/test_selection.py` now pins as a signature check.

Applied under the two grammars the study reports, it selects **15 terms** under arity 2 and
**23** under arity 3. Neither number appears anywhere in the code.

The same floor is what decides between the two grammars, one step up: `floor_argmax` gives one
length per grammar and `selection.best_configuration` chooses among those candidates. That
comparison is in [chapter 2](02-additive-model.md#the-arity-is-searched-not-set).

Every alternative that was computed is reported beside it, because a selection rule is only
defensible if what it beats is on the page. The table is generated, in the section below;
`selection.recommend` builds it and it carries one row per stated rule, the two readings of
the curve among them.

**The geometric rules and the paired test disagree, and the disagreement is the finding.**
Every geometric reading of this curve — three knee detectors on four curves, raw and
gRDP-smoothed at seven tolerances, plus the Pareto-front knee by all three standard forms —
lands between 4 and 8 terms. Every one of those lengths is **significantly worse** than the
selected one when the two are paired fold by fold over the twenty held-out datasets, and the
generated section counts how many of the searched lengths are. A knee finds where the
*marginal* return per term collapses, which on a saturating curve is early. It does not ask
whether the accuracy still being added is real, and here it is, for several terms past the
bend.

**Knee detection has therefore been removed rather than reported.** It was tried properly
first — the gRDP simplification works exactly as intended, taking three detectors that split
6/8/4 on the raw curve to unanimous agreement at 8 — and `kneeliverse` left the dependency
list with it. What replaced it is not a different detector but a different question.

**The parsimony alternative is reported and not adopted.** Ten terms is the shortest length
whose paired interval against 15 spans zero, and it is the right answer for a reader whose
readability budget is tighter than this study's. It gives up 0.023 of leave-one-dataset-out
R², which is measurable even where it is not significant, so the study takes the accuracy;
`results/length_choice.csv` carries the whole table so that choice can be remade.

Both **Pareto fronts** are reported, in `results/pareto.csv`. The two behave differently and
the difference is the point.

Over (length, in-sample R²) **almost every length is on the front**, because a longer equation
contains a longer search and fit does not fall as terms are added — so nothing is dominated
and the front says nothing. That is precisely why the in-sample curve cannot choose a length
by itself. *Almost*: one length is off it, and only because the beam is a heuristic rather
than an exhaustive search, so its best-at-24 can be a hair below its best-at-23. A front that
is nearly everything is not a selection device either way.

Over (length, LOO-dataset R²) the front is much shorter — it runs out well before the search
horizon does, which is the transfer curve flattening and then wandering. The membership moves
with the configuration, so it is in the file rather than written here; **this paragraph used
to list it, and listed a front from a configuration two changes ago.**

<!-- generated: do not edit below -->

## Why a subset rather than every term

The control for the whole selection stage. If handing every candidate term to unpenalised least squares in one go transferred well, the beam search and the length rule would be machinery in search of a problem.

| terms | r2_in_sample | r2_loo_dataset_clipped | r2_loo_dataset_unclipped |
|---|---|---|---|
| 220.0000 | 0.7860 | -2.1696 | -1536.5000 |

**The solver is not the hard part; the sample size is.** All 220 terms at once fit better in-sample than the published equation (0.7860 against 0.6578) and transfer at -2.1696 leave-one-dataset-out, against the published equation's 0.6381. The unclipped figure — -1536.5 — is what the fit does when a held-out dataset falls outside the convex hull of the other nineteen and nothing bounds the extrapolation. A design this much wider than 20 held-out groups can support has nothing to constrain it, which is what selection is for.

## Equation length

The length is chosen by one rule with no threshold and no smoothing: **the argmax of the worst protocol at each length** (`selection.floor_argmax`), which here selects **15 terms**. Nothing about that number is written down — it falls out of the curve, and it re-derives itself if the corpus changes. The three-protocol median reading of the same curve (`selection.best_length`) is reported beside it in the table below and agrees here.

**Why the consensus is a median and not a mean.** The deepest crater on this curve is at **24 terms**, where the three protocols read 0.671 / 0.387 / 0.626. The median takes 0.626 and ignores it; a mean would be dragged to 0.562. The crater is 0.244 below the neighbouring lengths and is not a property of the length at all -- it is one held-out dataset sitting outside the convex hull of the other nineteen in term space, where a linear equation extrapolates without limit and `validate._clip_to_training` pins the fold to its training floor. One fold's extrapolation should not choose the published length.

Every alternative rule is reported beside it, because a selection rule is only defensible if what it beats is on the page:

| rule | n_terms | r2_in_sample | r2_loo_dataset |
|---|---|---|---|
| pareto front, closest to ideal | 4 | 0.5390 | 0.5053 |
| pareto front, furthest from nadir | 4 | 0.5390 | 0.5053 |
| pareto front, furthest from chord | 4 | 0.5390 | 0.5053 |
| best loo-dataset | 15 | 0.6578 | 0.6381 |
| best consensus (median of three) | 15 | 0.6578 | 0.6381 |
| best floor over four protocols (the rule) | 15 | 0.6578 | 0.6381 |
| published | 15 | 0.6578 | 0.6381 |

The geometric rules — the Pareto-front knee by its three standard forms — choose far shorter equations, and **9 of the 25 lengths searched are significantly worse** than the selected one when paired fold by fold over the held-out datasets. A knee finds where the *marginal* return per term collapses, which on a saturating curve is early; it does not ask whether the accuracy still being added is real.

The parsimony alternative is **10 terms** — the shortest length whose paired interval against the selected one spans zero. It is reported and not adopted: the accuracy it gives up is measurable (0.6151 against 0.6381 leave-one-dataset-out) even where it is not significant.

The full curve the rule reads, at every length under all three protocols:

| n_terms | r2_in_sample | mae_in_sample | smape_in_sample | r2_loo_dataset | mae_loo_dataset | smape_loo_dataset | r2_loo_model | mae_loo_model | smape_loo_model | r2_loo_cell | mae_loo_cell | smape_loo_cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.2173 | 0.2440 | 46.8473 | 0.1654 | 0.2517 | 47.6765 | 0.1994 | 0.2472 | 47.1900 | 0.1513 | 0.2543 | 47.9477 |
| 2 | 0.3679 | 0.2048 | 42.9474 | 0.3191 | 0.2122 | 43.9902 | 0.3553 | 0.2070 | 43.2309 | 0.3110 | 0.2137 | 44.1368 |
| 3 | 0.4752 | 0.1843 | 40.9152 | 0.4277 | 0.1913 | 42.1394 | 0.4534 | 0.1881 | 41.3969 | 0.4106 | 0.1945 | 42.5125 |
| 4 | 0.5390 | 0.1676 | 38.8722 | 0.5053 | 0.1733 | 39.8140 | 0.5173 | 0.1716 | 39.4355 | 0.4900 | 0.1764 | 40.2621 |
| 5 | 0.5581 | 0.1635 | 38.2407 | 0.5315 | 0.1678 | 39.0066 | 0.5335 | 0.1682 | 38.9153 | 0.5139 | 0.1716 | 39.6352 |
| 6 | 0.5903 | 0.1560 | 37.7603 | 0.5659 | 0.1617 | 38.1458 | 0.5652 | 0.1607 | 38.2592 | 0.5489 | 0.1653 | 38.9708 |
| 7 | 0.6125 | 0.1499 | 36.0807 | 0.5876 | 0.1560 | 37.0900 | 0.5873 | 0.1548 | 36.9766 | 0.5704 | 0.1595 | 37.7298 |
| 8 | 0.6243 | 0.1486 | 36.1415 | 0.5965 | 0.1545 | 36.4395 | 0.6014 | 0.1532 | 36.5711 | 0.5799 | 0.1581 | 37.0271 |
| 9 | 0.6275 | 0.1482 | 36.0784 | 0.6059 | 0.1530 | 36.7461 | 0.6029 | 0.1534 | 36.7278 | 0.5889 | 0.1574 | 37.4025 |
| 10 | 0.6344 | 0.1448 | 35.8198 | 0.6151 | 0.1492 | 36.5028 | 0.6068 | 0.1502 | 36.6987 | 0.5958 | 0.1534 | 37.4385 |
| 11 | 0.6408 | 0.1435 | 35.7871 | 0.6187 | 0.1497 | 36.6836 | 0.6124 | 0.1493 | 36.6601 | 0.5991 | 0.1536 | 36.9735 |
| 12 | 0.6452 | 0.1421 | 35.4886 | 0.6231 | 0.1476 | 36.1081 | 0.6080 | 0.1491 | 36.4758 | 0.5963 | 0.1527 | 36.8258 |
| 13 | 0.6471 | 0.1416 | 35.4546 | 0.4829 | 0.1734 | 40.2100 | 0.6099 | 0.1488 | 36.2947 | 0.4424 | 0.1794 | 40.9112 |
| 14 | 0.6497 | 0.1410 | 34.9209 | 0.5003 | 0.1715 | 39.5174 | 0.6102 | 0.1487 | 36.1729 | 0.4582 | 0.1782 | 40.8685 |
| 15 | 0.6578 | 0.1371 | 34.4737 | 0.6381 | 0.1417 | 34.8803 | 0.6218 | 0.1449 | 35.4201 | 0.6162 | 0.1481 | 36.0783 |
| 16 | 0.6619 | 0.1357 | 33.8603 | 0.6317 | 0.1451 | 34.8130 | 0.6263 | 0.1431 | 34.7298 | 0.6056 | 0.1510 | 35.8551 |
| 17 | 0.6633 | 0.1357 | 34.0624 | 0.6299 | 0.1446 | 34.9541 | 0.6254 | 0.1435 | 34.8493 | 0.6044 | 0.1508 | 36.0054 |
| 18 | 0.6639 | 0.1360 | 34.0860 | 0.6343 | 0.1433 | 34.7439 | 0.6254 | 0.1438 | 34.8883 | 0.6102 | 0.1493 | 35.8261 |
| 19 | 0.6663 | 0.1350 | 33.8645 | 0.6289 | 0.1425 | 34.9076 | 0.6266 | 0.1430 | 34.9297 | 0.6078 | 0.1481 | 35.8201 |
| 20 | 0.6679 | 0.1343 | 33.8598 | 0.6379 | 0.1433 | 34.5722 | 0.6282 | 0.1423 | 34.5920 | 0.6110 | 0.1493 | 35.4416 |
| 21 | 0.6693 | 0.1342 | 33.9862 | 0.6337 | 0.1431 | 34.8863 | 0.6288 | 0.1422 | 34.7660 | 0.6086 | 0.1488 | 35.7303 |
| 22 | 0.6701 | 0.1338 | 33.8926 | 0.6255 | 0.1461 | 35.0555 | 0.6284 | 0.1421 | 34.6898 | 0.5995 | 0.1515 | 35.7719 |
| 23 | 0.6707 | 0.1335 | 33.8135 | 0.6306 | 0.1438 | 34.4890 | 0.6283 | 0.1419 | 34.8448 | 0.6001 | 0.1502 | 35.6028 |
| 24 | 0.6707 | 0.1337 | 33.8867 | 0.3874 | 0.1749 | 42.0258 | 0.6265 | 0.1424 | 35.1649 | 0.3637 | 0.1782 | 42.6830 |
| 25 | 0.6718 | 0.1341 | 33.8208 | 0.6331 | 0.1452 | 35.6574 | 0.6278 | 0.1429 | 34.9396 | 0.6028 | 0.1517 | 36.1071 |

<!-- end generated -->

## Limitations of the selection procedure

### Hyperparameter selection is not nested

The stability cap, penalty and equation length were tuned by inspecting
leave-one-dataset-out scores. Those scores are therefore **mildly optimistic** as estimates
of performance on genuinely new data. A fully nested protocol would cost another factor of
20 in compute and, at this sample size, would mostly measure noise; the honest reading is
that the reported transfer numbers are an upper estimate rather than an unbiased one.

The **in-sample** numbers are unaffected by this.
