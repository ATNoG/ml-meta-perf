# 3. Term generation and selection

*Implemented in `ml_meta_perf.search`, `ml_meta_perf.fit`, `ml_meta_perf.analysis` and `ml_meta_perf.selection`.*

The abbreviations used in this chapter are Matthews Correlation Coefficient (**MCC**),
residual sum of squares (**RSS**), Linear Algebra PACKage (**LAPACK**), Basic Linear Algebra
Subprograms (**BLAS**), central processing unit (**CPU**), coefficient of determination
(**R²**), in-sample (**IS**), leave-one-dataset-out (**LODO**), leave-one-model-out
(**LOMO**), and doubly held out (**DHO**). Memory and interface abbreviations are megabyte (**MB**), kilobyte (**KB**),
level-1 cache (**L1**), and the 64-bit-integer BLAS interface (**ILP64**).
The equation labels are **E3-Valid** (the plateau-selected dataset-and-model equation) and
**E3-MAX** (the maximum-capability dataset-and-model equation).

Two problems are involved and only one of them is hard.

| problem | nature | method |
|---|---|---|
| Given a set of terms, what are the best weights? | linear | `numpy.linalg.solve` on the ridge normal equations — **exact, no iteration** |
| Which $k$ of a few hundred candidate terms? | combinatorial | beam search with local refinement |

Handing every term to `lstsq` at once fits *better* under IS than the published equation and
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
DHO cell — runs in tens of seconds on the reference environment, and every
optimisation that got it there was verified to leave the outputs unchanged.

A full `python -m ml_meta_perf` takes about six minutes on the reference Windows environment;
most of that time is the opaque comparison in
[chapter 5](05-evaluation.md#what-an-opaque-model-reaches-and-does-not):
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

The last two are the current ones. At $k \le 32$ NumPy's per-call wrapper — dtype
promotion, array coercion, the errstate context manager — costs several times the LAPACK
call it guards, so batching the candidates of one beam step into a single stacked solve is
worth 2–10× depending on $k$ while running the identical routine per slice. What remains
in the profile is the arithmetic itself: gathering the $(n, k, k)$ submatrices and the
solves, in that order. There is no Python-level hotspot left above 20%.

### One BLAS thread, deliberately

The solver underneath is whatever LAPACK NumPy was built against — for the wheels used
here, the OpenBLAS build that NumPy vendors (`numpy.libs/libscipy_openblas64_*.so`, built
`MAX_THREADS=64`). **That is a bundled shared library, not the installed SciPy package.**
The project receives SciPy transitively through scikit-learn, while its small internal
statistics remain implemented in `ml_meta_perf.stats`.

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
imports `ml-meta-perf`, and therefore NumPy, and therefore OpenBLAS, *before* `__main__` runs,
and OpenBLAS reads the variable when it loads. Anyone timing this study by calling
`python -m ml_meta_perf` directly will see the same 25 seconds against five times the CPU.

### Using the host's BLAS instead of the wheel's

`pip install numpy` installs a wheel that **vendors its own OpenBLAS** — a 25 MB
`numpy.libs/libscipy_openblas64_*.so`, built ILP64 with prefixed symbols. It is linked at
build time and there is no runtime switch, so a different BLAS means rebuilding NumPy:

```bash
venv/bin/pip install --no-binary numpy --force-reinstall numpy \
  -Csetup-args=-Dblas=openblas -Csetup-args=-Dlapack=openblas
```

That needs the BLAS development files (an `openblas.pc` for pkg-config, plus the headers),
a C compiler and `ninja`. It takes about two minutes the first time on 16 cores; pip caches
the built wheel, so recreating the venv afterwards reuses it and costs seconds.
`venv/bin/pip install --force-reinstall numpy` goes back to the wheel. **Any later
`pip install` that resolves NumPy will silently replace a source build with the wheel
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

Every output table produced in that benchmark was unchanged across the swap, which is the
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

| setting | beam | candidates | rounds | IS (k=14) | IS (k=20) |
|---|---|---|---|---|---|
| baseline | 6 | 24 | 2 | 0.5582 | 0.5662 |
| wide beam | 16 | 24 | 2 | 0.5582 | 0.5663 |
| more candidates | 6 | 64 | 2 | 0.5582 | 0.5661 |
| more rounds | 6 | 24 | 6 | 0.5582 | 0.5662 |
| maximal | 48 | 96 | 10 | 0.5582 | 0.5663 |

<sub>Default library. The wider library gains ~0.01 non-monotonically, which is search noise
rather than systematic improvement. **Measured before the 2026-09-05 protocol change**, so
these are re-selecting numbers and are not comparable with any current figure — the
IS column here is not the IS column of chapter 5. What the table supports is a
statement about *differences between search settings*, and those are internally consistent
because every row was measured the same way.</sub>

Spending eight times the compute changes the fourth decimal place. Together with the
negative result on a richer vocabulary ([chapter 2](02-additive-model.md)), this locates
the limit in the model *form*, not in the optimiser.

**Per-length penalty tuning** was also tested: choosing $\lambda$ separately for each $k$
lifted LODO R² at $k=14$ from 0.4429 to 0.4456 — again on the pre-2026-09-05
re-selecting protocol, so read the *gain* and not the level. A gain of 0.003 does not justify
the extra configuration surface, and one global penalty is retained.

## Stage 3 — choosing the number of terms

More terms fit better and read worse. E3-Valid chooses the equation length with a stated,
reproducible sustained-plateau rule. The rule follows the best-so-far Combined R² curve and
uses a forward window of three evaluated lengths with a tolerance of 0.001.

### Which evidence the rule combines

IS R² is monotone in the number of terms, so it cannot express the trade between fit
and equation length by itself. A single cross-validated curve is also insufficient. On twenty groups they wander, and
LODO on this corpus has genuine craters — lengths where the held-out number
collapses by a fifth of the scale while its two neighbours are untouched. That is not noise,
and it is not a property of the length either: at such a length one held-out dataset sits
outside the convex hull of the other nineteen in term space, where a linear equation
extrapolates without limit and `validate._clip_to_training` pins the fold to its training
floor. `ASNM-CDX-2009` is the fold this happens to. One fold's extrapolation should not
choose the published length.

The rule therefore uses `selection.consensus_curve`: the per-length **median** of IS,
LODO, and LOMO R². The median keeps one unstable validation
protocol from determining the equation length. Which length contains the deepest crater moves
with the configuration, so the generated section below identifies it from the current curve
and reports the median and mean side by side.

### Which lengths the curve is reported at

**Every length from 1 to `max_terms`**, on a uniform grid, and this matters more than it
sounds. A non-uniform grid — `(2, 4, 8, 12, 16, 20, 24)`, say — can skip the beginning or end
of a plateau and change the selected point. The current curve's deepest transfer crater is at
eight terms. Any rule reading a sparse curve is partly reading the grid. Reporting every
length costs nothing, because the beam search already builds the whole path.

![Accuracy versus equation length](../figures/02_term_count_curve.png)

The craters are visible in that figure. They are a real property of LODO on
twenty groups and they belong on the plot.

### The retained E3-Valid rule

E3-Valid is selected from the complete curves for both arities. For every evaluated term
count, the implementation retains the arity with the highest **Combined R²**:

$$
R^2_{\mathrm{combined}} = \operatorname{median}\left(
R^2_{\mathrm{in\text{-}sample}},
R^2_{\mathrm{LODO}},
R^2_{\mathrm{LOMO}}
\right).
$$

It then follows the best Combined R² observed so far and selects the equation immediately
before the first sustained plateau. A plateau is reached when the best-so-far gain over the
next three evaluated term counts is at most 0.001. The two retained values are explicit
constants and configuration-search arguments: `plateau_window = 3` and
`plateau_tolerance = 0.001`.

On the corrected corpus and the retained 1-to-25-term search, this rule selects **18 terms at
arity 2**. The selection is derived from the curve by `selection.plateau_configuration`; the
chosen term count and arity are not separately hard-coded.

E3-MAX answers a separate capability question. It chooses the arity and length that maximise
the minimum R² over all four protocols, including DHO evaluation. It selects
**25 terms at arity 3**. E3-MAX is reported as a bound and is not used for the downstream
prediction and model-ranking results.

The paired per-dataset table in the generated section compares other lengths with E3-Valid as
a sensitivity analysis. It does not define additional E3-Valid equations.
<!-- generated: do not edit below -->

## Why a subset rather than every term

The control for the whole selection stage. If handing every candidate term to unpenalised least squares in one go transferred well, the beam search and the length rule would be machinery in search of a problem.

| terms | r2_IS | r2_LODO_clipped | r2_LODO_unclipped |
|---|---|---|---|
| 229.0000 | 0.7805 | -0.0972 | -1281.2773 |

**The solver is not the hard part; the sample size is.** All 229 terms at once fit better under IS than the published equation (0.7805 against 0.6787) and transfer at -0.0972 under LODO, against the published equation's 0.6517. The unclipped figure — -1281.3 — is what the fit does when a held-out dataset falls outside the convex hull of the other nineteen and nothing bounds the extrapolation. A design this much wider than 20 held-out groups can support has nothing to constrain it, which is what selection is for.

## Equation length

E3-Valid selects **18 terms** immediately before the first sustained plateau in Combined R², the median of IS, LODO, and LOMO R². The retained rule uses a forward window of three evaluated lengths and a maximum best-so-far gain of 0.001. It compares both searched arities before selecting the equation.

**Why the consensus is a median and not a mean.** The deepest crater on this curve is at **8 terms**, where the three protocols read 0.611 / 0.463 / 0.581. The median takes 0.581 and ignores it; a mean would be dragged to 0.552. The crater is 0.117 below the neighbouring lengths and is not a property of the length at all -- it is one held-out dataset sitting outside the convex hull of the other nineteen in term space, where a linear equation extrapolates without limit and `validate._clip_to_training` pins the fold to its training floor. One fold's extrapolation should not choose the published length.

The following paired analysis compares every length on the selected arity against E3-Valid; it is a sensitivity analysis rather than an additional selector:

| n_terms | r2_LODO | mae_LODO | mean_difference | p_value | ci_low | ci_high | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.1952 | 0.2393 | 0.1026 | 0.0000 | 0.0758 | 0.1302 | worse |
| 2 | 0.3112 | 0.2120 | 0.0754 | 0.0004 | 0.0493 | 0.1015 | worse |
| 3 | 0.4557 | 0.1826 | 0.0459 | 0.0118 | 0.0229 | 0.0707 | worse |
| 4 | 0.5035 | 0.1688 | 0.0322 | 0.1153 | 0.0140 | 0.0515 | worse |
| 5 | 0.5499 | 0.1601 | 0.0234 | 0.2632 | 0.0077 | 0.0408 | worse |
| 6 | 0.5547 | 0.1592 | 0.0225 | 0.1153 | 0.0081 | 0.0386 | worse |
| 7 | 0.5513 | 0.1594 | 0.0228 | 0.2632 | 0.0080 | 0.0400 | worse |
| 8 | 0.4632 | 0.1669 | 0.0302 | 0.5034 | 0.0053 | 0.0678 | worse |
| 9 | 0.6090 | 0.1436 | 0.0070 | 0.1153 | -0.0015 | 0.0152 | tie |
| 10 | 0.6078 | 0.1456 | 0.0090 | 0.1153 | -0.0001 | 0.0189 | tie |
| 11 | 0.6130 | 0.1434 | 0.0067 | 0.2632 | -0.0015 | 0.0158 | tie |
| 12 | 0.6194 | 0.1404 | 0.0038 | 0.5034 | -0.0030 | 0.0114 | tie |
| 13 | 0.6288 | 0.1428 | 0.0062 | 0.0414 | -0.0004 | 0.0132 | tie |
| 14 | 0.6364 | 0.1392 | 0.0026 | 0.1153 | -0.0034 | 0.0082 | tie |
| 15 | 0.6425 | 0.1383 | 0.0017 | 0.2632 | -0.0038 | 0.0067 | tie |
| 16 | 0.6394 | 0.1401 | 0.0035 | 0.1153 | -0.0015 | 0.0080 | tie |
| 17 | 0.6391 | 0.1389 | 0.0023 | 0.0414 | -0.0022 | 0.0064 | tie |
| 18 | 0.6517 | 0.1366 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | selected |
| 19 | 0.6526 | 0.1343 | -0.0024 | 0.5034 | -0.0049 | -0.0005 | better |
| 20 | 0.6422 | 0.1380 | 0.0014 | 1.0000 | -0.0018 | 0.0048 | tie |
| 21 | 0.6435 | 0.1357 | -0.0009 | 0.5034 | -0.0047 | 0.0031 | tie |
| 22 | 0.6446 | 0.1388 | 0.0022 | 0.5034 | -0.0035 | 0.0081 | tie |
| 23 | 0.6451 | 0.1372 | 0.0006 | 0.8238 | -0.0061 | 0.0088 | tie |
| 24 | 0.6422 | 0.1368 | 0.0002 | 0.5034 | -0.0073 | 0.0091 | tie |
| 25 | 0.6445 | 0.1367 | 0.0001 | 0.8238 | -0.0078 | 0.0091 | tie |

The full curve reports all four protocols at every length. E3-Valid reads IS, LODO, and LOMO through Combined R²; DHO is reported alongside them but is not an input to that plateau rule:

| n_terms | r2_IS | mae_IS | smape_IS | r2_LODO | mae_LODO | smape_LODO | r2_LOMO | mae_LOMO | smape_LOMO | r2_DHO | mae_DHO | smape_DHO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.2427 | 0.2358 | 46.1034 | 0.1952 | 0.2432 | 47.0548 | 0.2268 | 0.2388 | 46.4802 | 0.1834 | 0.2455 | 47.3340 |
| 2 | 0.3591 | 0.2062 | 43.2745 | 0.3112 | 0.2135 | 44.3767 | 0.3466 | 0.2084 | 43.5564 | 0.3037 | 0.2150 | 44.5448 |
| 3 | 0.4911 | 0.1755 | 39.6329 | 0.4557 | 0.1808 | 40.4928 | 0.4678 | 0.1794 | 40.2263 | 0.4381 | 0.1839 | 40.9501 |
| 4 | 0.5436 | 0.1626 | 37.8369 | 0.5035 | 0.1691 | 38.4502 | 0.5171 | 0.1674 | 38.4511 | 0.4834 | 0.1726 | 38.9836 |
| 5 | 0.5784 | 0.1544 | 37.0102 | 0.5499 | 0.1594 | 37.5614 | 0.5493 | 0.1597 | 37.7126 | 0.5288 | 0.1633 | 37.7581 |
| 6 | 0.5890 | 0.1524 | 36.9959 | 0.5547 | 0.1587 | 37.4046 | 0.5591 | 0.1579 | 37.7374 | 0.5348 | 0.1628 | 38.1913 |
| 7 | 0.5963 | 0.1495 | 36.4107 | 0.5513 | 0.1576 | 37.6541 | 0.5637 | 0.1556 | 37.2876 | 0.5295 | 0.1616 | 38.1192 |
| 8 | 0.6107 | 0.1440 | 34.8157 | 0.4632 | 0.1678 | 42.1530 | 0.5808 | 0.1498 | 35.4811 | 0.4404 | 0.1705 | 42.2555 |
| 9 | 0.6298 | 0.1385 | 35.5275 | 0.6090 | 0.1427 | 36.0247 | 0.5903 | 0.1459 | 36.6082 | 0.5825 | 0.1480 | 36.8934 |
| 10 | 0.6339 | 0.1393 | 35.6382 | 0.6078 | 0.1442 | 36.7001 | 0.5921 | 0.1468 | 36.6733 | 0.5828 | 0.1491 | 37.3524 |
| 11 | 0.6414 | 0.1353 | 34.5140 | 0.6130 | 0.1421 | 36.0180 | 0.6007 | 0.1428 | 35.7904 | 0.5864 | 0.1471 | 36.4225 |
| 12 | 0.6467 | 0.1344 | 33.9501 | 0.6194 | 0.1399 | 34.8157 | 0.5978 | 0.1427 | 35.4014 | 0.5848 | 0.1460 | 36.0287 |
| 13 | 0.6562 | 0.1348 | 34.5278 | 0.6288 | 0.1421 | 35.7506 | 0.6099 | 0.1433 | 35.8003 | 0.5972 | 0.1479 | 36.3182 |
| 14 | 0.6646 | 0.1314 | 33.7281 | 0.6364 | 0.1385 | 36.1097 | 0.6193 | 0.1398 | 35.2132 | 0.6084 | 0.1437 | 36.1943 |
| 15 | 0.6702 | 0.1307 | 33.7804 | 0.6425 | 0.1379 | 35.5392 | 0.6197 | 0.1396 | 35.2327 | 0.6101 | 0.1435 | 36.2617 |
| 16 | 0.6723 | 0.1305 | 33.9821 | 0.6394 | 0.1392 | 36.0971 | 0.6223 | 0.1391 | 35.1184 | 0.6096 | 0.1439 | 36.2481 |
| 17 | 0.6738 | 0.1299 | 34.5275 | 0.6391 | 0.1386 | 36.7567 | 0.6219 | 0.1391 | 35.9445 | 0.6109 | 0.1432 | 36.8801 |
| 18 | 0.6787 | 0.1286 | 34.3920 | 0.6517 | 0.1361 | 35.8120 | 0.6149 | 0.1398 | 36.1522 | 0.6103 | 0.1431 | 36.8174 |
| 19 | 0.6791 | 0.1277 | 34.0798 | 0.6526 | 0.1340 | 35.4156 | 0.6110 | 0.1398 | 35.8391 | 0.6070 | 0.1421 | 36.6903 |
| 20 | 0.6823 | 0.1277 | 34.4118 | 0.6422 | 0.1371 | 36.3848 | 0.6136 | 0.1396 | 36.0928 | 0.6006 | 0.1438 | 37.1948 |
| 21 | 0.6838 | 0.1271 | 34.4413 | 0.6435 | 0.1352 | 35.6291 | 0.6155 | 0.1390 | 36.2435 | 0.6031 | 0.1418 | 36.9931 |
| 22 | 0.6887 | 0.1272 | 34.7688 | 0.6446 | 0.1380 | 36.6454 | 0.6184 | 0.1396 | 36.5316 | 0.6036 | 0.1449 | 37.8717 |
| 23 | 0.6896 | 0.1268 | 34.8228 | 0.6451 | 0.1366 | 36.5903 | 0.6175 | 0.1394 | 36.6506 | 0.6041 | 0.1434 | 37.8209 |
| 24 | 0.6928 | 0.1259 | 34.6242 | 0.6422 | 0.1363 | 36.1953 | 0.6198 | 0.1390 | 36.4145 | 0.6026 | 0.1435 | 37.7329 |
| 25 | 0.6957 | 0.1254 | 34.6698 | 0.6445 | 0.1361 | 36.2106 | 0.6231 | 0.1385 | 36.5239 | 0.6060 | 0.1431 | 37.6736 |

<!-- end generated -->

## Limitations of the selection procedure

### Hyperparameter selection is not nested

The stability cap, penalty and equation length were tuned by inspecting
LODO scores. Those scores are therefore **mildly optimistic** as estimates
of performance on genuinely new data. A fully nested protocol would cost another factor of
20 in compute and, at this sample size, would mostly measure noise; the honest reading is
that the reported transfer numbers are an upper estimate rather than an unbiased one.

The **IS** numbers are unaffected by this.
