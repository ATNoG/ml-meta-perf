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
chapters plot. The two agree at 25 terms under the retained arity-3 grammar. Under the rejected
arity-2 grammar, the four-protocol floor selects 17 terms while the three-protocol median
selects 19, which is why the readings remain separate. **Which length craters moves with the configuration**, so the
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

Applied under the two default grammars, the four-protocol rule selects **17 terms** under
arity 2 and **25** under arity 3. Neither number appears anywhere in the code.

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

**The parsimony alternative is reported and not adopted.** Eight terms is the shortest length
whose paired interval against 12 spans zero, and it is the right answer for a reader whose
readability budget is tighter than this study's. It gives up 0.047 of leave-one-dataset-out
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
| 901.0000 | 0.8080 | -0.5582 | -1491.4784 |

**The solver is not the hard part; the sample size is.** All 901 terms at once fit better in-sample than the published equation (0.8080 against 0.7194) and transfer at -0.5582 leave-one-dataset-out, against the published equation's 0.6911. The unclipped figure — -1491.5 — is what the fit does when a held-out dataset falls outside the convex hull of the other nineteen and nothing bounds the extrapolation. A design this much wider than 20 held-out groups can support has nothing to constrain it, which is what selection is for.

## Equation length

The length is chosen by one rule with no threshold and no smoothing: **the argmax of the worst protocol at each length** (`selection.floor_argmax`), which here selects **25 terms**. Nothing about that number is written down — it falls out of the curve, and it re-derives itself if the corpus changes. The three-protocol median reading of the same curve (`selection.best_length`) is reported beside it in the table below and agrees here.

**Why the consensus is a median and not a mean.** The deepest crater on this curve is at **3 terms**, where the three protocols read 0.456 / 0.409 / 0.438. The median takes 0.438 and ignores it; a mean would be dragged to 0.434. The crater is 0.038 below the neighbouring lengths and is not a property of the length at all -- it is one held-out dataset sitting outside the convex hull of the other nineteen in term space, where a linear equation extrapolates without limit and `validate._clip_to_training` pins the fold to its training floor. One fold's extrapolation should not choose the published length.

Every alternative rule is reported beside it, because a selection rule is only defensible if what it beats is on the page:

| rule | n_terms | r2_in_sample | r2_loo_dataset |
|---|---|---|---|
| pareto front, closest to ideal | 5 | 0.5831 | 0.5636 |
| pareto front, furthest from nadir | 5 | 0.5831 | 0.5636 |
| pareto front, furthest from chord | 5 | 0.5831 | 0.5636 |
| best loo-dataset | 25 | 0.7194 | 0.6911 |
| best consensus (median of three) | 25 | 0.7194 | 0.6911 |
| best floor over four protocols (the rule) | 25 | 0.7194 | 0.6911 |
| published | 25 | 0.7194 | 0.6911 |

The geometric rules — the Pareto-front knee by its three standard forms — choose far shorter equations, and **23 of the 25 lengths searched are significantly worse** than the selected one when paired fold by fold over the held-out datasets. A knee finds where the *marginal* return per term collapses, which on a saturating curve is early; it does not ask whether the accuracy still being added is real.

The parsimony alternative is **23 terms** — the shortest length whose paired interval against the selected one spans zero. It is reported and not adopted: the accuracy it gives up is measurable (0.6729 against 0.6911 leave-one-dataset-out) even where it is not significant.

The full curve the rule reads, at every length under all three protocols:

| n_terms | r2_in_sample | mae_in_sample | smape_in_sample | r2_loo_dataset | mae_loo_dataset | smape_loo_dataset | r2_loo_model | mae_loo_model | smape_loo_model | r2_loo_cell | mae_loo_cell | smape_loo_cell |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.3236 | 0.2112 | 43.6222 | 0.2884 | 0.2173 | 45.0377 | 0.3100 | 0.2135 | 43.9877 | 0.2785 | 0.2190 | 44.5788 |
| 2 | 0.4074 | 0.1955 | 41.7340 | 0.3726 | 0.2022 | 42.9325 | 0.3952 | 0.1977 | 42.0634 | 0.3648 | 0.2036 | 43.1092 |
| 3 | 0.4563 | 0.1858 | 40.7592 | 0.4091 | 0.1946 | 42.3107 | 0.4377 | 0.1890 | 41.2251 | 0.3962 | 0.1970 | 42.6226 |
| 4 | 0.5510 | 0.1633 | 37.8881 | 0.5208 | 0.1692 | 38.8035 | 0.5256 | 0.1677 | 38.5687 | 0.5022 | 0.1723 | 39.2456 |
| 5 | 0.5831 | 0.1526 | 36.1027 | 0.5636 | 0.1571 | 37.0633 | 0.5605 | 0.1570 | 36.9517 | 0.5495 | 0.1601 | 37.5502 |
| 6 | 0.5974 | 0.1465 | 36.2564 | 0.5728 | 0.1514 | 36.6320 | 0.5713 | 0.1515 | 37.0675 | 0.5555 | 0.1550 | 37.5101 |
| 7 | 0.6059 | 0.1462 | 36.1093 | 0.5743 | 0.1523 | 36.6504 | 0.5783 | 0.1521 | 36.8637 | 0.5569 | 0.1566 | 37.5712 |
| 8 | 0.6296 | 0.1400 | 34.2213 | 0.5933 | 0.1509 | 39.1762 | 0.5975 | 0.1459 | 35.1298 | 0.5716 | 0.1545 | 39.5514 |
| 9 | 0.6334 | 0.1390 | 34.1653 | 0.5926 | 0.1511 | 39.7821 | 0.5992 | 0.1451 | 35.1759 | 0.5706 | 0.1547 | 39.9051 |
| 10 | 0.6405 | 0.1377 | 33.7930 | 0.6075 | 0.1462 | 36.5566 | 0.6051 | 0.1441 | 34.7956 | 0.5844 | 0.1502 | 36.3636 |
| 11 | 0.6437 | 0.1374 | 34.0481 | 0.6180 | 0.1448 | 36.7953 | 0.6072 | 0.1441 | 35.0436 | 0.5960 | 0.1491 | 37.1660 |
| 12 | 0.6482 | 0.1356 | 34.1204 | 0.6238 | 0.1410 | 35.5046 | 0.6132 | 0.1425 | 35.1143 | 0.6049 | 0.1450 | 35.6973 |
| 13 | 0.6529 | 0.1340 | 33.4882 | 0.6253 | 0.1407 | 35.7682 | 0.6162 | 0.1416 | 34.8633 | 0.6062 | 0.1449 | 36.2753 |
| 14 | 0.6592 | 0.1317 | 33.3312 | 0.6291 | 0.1371 | 34.6015 | 0.6194 | 0.1390 | 34.4887 | 0.6090 | 0.1417 | 35.7710 |
| 15 | 0.6632 | 0.1310 | 33.2720 | 0.6285 | 0.1363 | 34.4321 | 0.6198 | 0.1391 | 34.6495 | 0.6070 | 0.1414 | 35.8366 |
| 16 | 0.6697 | 0.1290 | 33.5386 | 0.6321 | 0.1365 | 35.0390 | 0.6194 | 0.1381 | 35.1852 | 0.6014 | 0.1427 | 36.8961 |
| 17 | 0.6729 | 0.1280 | 33.5598 | 0.6307 | 0.1359 | 35.5790 | 0.6206 | 0.1375 | 34.8850 | 0.5999 | 0.1425 | 37.2675 |
| 18 | 0.6766 | 0.1275 | 33.6476 | 0.6312 | 0.1367 | 35.6108 | 0.6233 | 0.1370 | 34.9672 | 0.6008 | 0.1425 | 37.2564 |
| 19 | 0.6822 | 0.1264 | 33.3590 | 0.6330 | 0.1371 | 37.1092 | 0.6292 | 0.1366 | 34.7440 | 0.5991 | 0.1432 | 38.1021 |
| 20 | 0.6865 | 0.1257 | 32.8666 | 0.6410 | 0.1362 | 36.6852 | 0.6328 | 0.1357 | 34.4644 | 0.6092 | 0.1418 | 37.4837 |
| 21 | 0.6890 | 0.1259 | 33.6182 | 0.6260 | 0.1416 | 38.0107 | 0.6277 | 0.1367 | 35.2025 | 0.5890 | 0.1482 | 39.7591 |
| 22 | 0.7052 | 0.1218 | 33.2901 | 0.6648 | 0.1327 | 36.7837 | 0.6433 | 0.1325 | 34.2909 | 0.6259 | 0.1387 | 37.1073 |
| 23 | 0.7099 | 0.1195 | 32.3709 | 0.6729 | 0.1284 | 34.1936 | 0.6491 | 0.1304 | 33.8556 | 0.6332 | 0.1355 | 35.3868 |
| 24 | 0.7114 | 0.1188 | 32.4690 | 0.6684 | 0.1325 | 36.5710 | 0.6471 | 0.1300 | 34.0374 | 0.6274 | 0.1387 | 36.5651 |
| 25 | 0.7194 | 0.1161 | 30.9675 | 0.6911 | 0.1231 | 32.0835 | 0.6554 | 0.1272 | 33.0012 | 0.6554 | 0.1290 | 32.9661 |

<!-- end generated -->

## Limitations of the selection procedure

### Hyperparameter selection is not nested

The stability cap, penalty and equation length were tuned by inspecting
leave-one-dataset-out scores. Those scores are therefore **mildly optimistic** as estimates
of performance on genuinely new data. A fully nested protocol would cost another factor of
20 in compute and, at this sample size, would mostly measure noise; the honest reading is
that the reported transfer numbers are an upper estimate rather than an unbiased one.

The **in-sample** numbers are unaffected by this.
