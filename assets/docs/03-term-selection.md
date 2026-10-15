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
bias every prediction toward MCC = 0 rather than the fitting partition’s target mean.

$$(G + \lambda I)\,w = b, \qquad G = Z^\top Z, \quad b = Z^\top (y - \bar{y})$$

The residual sum of squares uses an identity rather than a second quadratic form. Since
$(G + \lambda I) w = b$ implies $w^\top G w = w^\top b - \lambda w^\top w$,

$$\mathrm{RSS} = \|y - \bar y\|^2 - 2 w^\top b + w^\top G w = \|y - \bar y\|^2 - w^\top b - \lambda w^\top w$$

which removes a matrix-vector product from the inner loop. The test suite checks the
identity against the literal definition across penalties and subset sizes.

### Cost

Every candidate refit is a $k \times k$ solve against a precomputed Gram matrix rather
than a least-squares call against the full design. The selected run evaluates four
fixed forms and all four protocols; DHO requires one coefficient refit per observed
pair, currently 484. The opaque references use the same partition structure.

The timing and profiling measurements below are historical implementation benchmarks,
not timings of the updated corpus or the complete 70-configuration search. They describe
the numerical optimizations without replacing the current generated results.

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

## Search budget

The beam width, candidate budget, and refinement settings remain fixed while the
configuration sweep varies the stability cap, ridge penalty, and maximum arity.
Historical experiments with larger beams or per-length penalties are not current-corpus
validation; one penalty is retained for each configuration, as specified by the method.

## Stage 3 — choosing the number of terms

More terms fit better and read worse. **One rule chooses the length of every published
equation** — E1, E2 and E3-Valid alike — so that, like the features and the configuration, the
way the length was picked is not something that differs between them. It is
`selection.plateau_knee`, and it reads the complete cross-validated curve in three steps.

### Which evidence the rule reads

**The worst protocol at each length.** IS fit alone cannot express the trade between
fit and length; any single held-out curve can be flattered by the protocol it happens to be. The rule reads the **floor** — the minimum R² over
IS, LODO, LOMO and DHO at each length (`selection.floor_curve`) — so a length is only as good
as the protocol it does worst on, which is almost always DHO: the cell where neither the
dataset nor the model has been seen, the question the study is for.

**Smoothed, because single lengths are noisy.** Adjacent lengths can fluctuate beyond
a smooth trend in model size. LODO has genuine craters — lengths where the held-out
number collapses while its neighbours are untouched, because one held-out dataset sits outside
the convex hull of the other nineteen in term space, where a linear equation extrapolates
without limit and `validate._clip_to_training` pins the fold to its training floor. The mirror
image, a single lucky length, is just as common. Either would choose the length if the rule
read the raw curve, so it reads a **running median over three lengths**
(`selection.smoothed`): a median ignores one outlying length where a mean would spread it onto
its neighbours. The generated section below identifies the current curve's deepest crater and
what the smoothing does to it.

### Where the returns stop

**Knees propose, the plateau decides.** multi-Kneedle — from the
[`kneeliverse`](https://github.com/mariolpantunes/knee) library — is run on the smoothed curve's
Pareto front (the lengths that improve on every shorter one) and proposes the lengths where its
rate of improvement bends (`selection.knee_lengths`). A knee alone is not enough: an early
bend can still be followed by sustained improvement. The rule therefore takes **the first proposed
length after which the smoothed floor gains at most `delta` over the next `window` lengths** —
the start of the first sustained plateau, the shortest equation that has stopped improving.
The smoothed maximum is always a candidate too, since nothing after it gains; when every knee
is still followed by real gains, the plateau starts there. `delta`, `window` and the smoothing
width are hyperparameters in `config/study.json`, stated rather than implied — the generated
configuration table below lists them — and the generated section reports which knees this run
proposed and which one it took.

A rule that looked only at a threshold on raw steps — the "first gain below 0.001" rule this
replaced — reads noise: a threshold far below the length-to-length variation stops wherever
the noise happens to dip. A rule that looked only at knees stops too early. Together they find
where the curve actually levels off.

### Which lengths the curve is reported at

**Every length from 1 to `max_terms`**, on a uniform grid. A non-uniform grid can skip the
beginning or end of a plateau and change the selected point, and any rule reading a sparse
curve is partly reading the grid. Reporting every length costs nothing, because the beam
search already builds the whole path. The horizon, `search.max_terms`, lies past the readable
range, so that a plateau starting near the readable limit still has a full window ahead of it.

![Accuracy versus equation length](../figures/02_term_count_curve.png)

The craters are visible in that figure. They are a real property of LODO on twenty groups and
they belong on the plot.

### E3-MAX, the bound

E3-MAX answers a separate capability question: how far the additive form reaches if it is
not asked to stay readable. It uses the configured capability arity
(`selection.capability_arity`), at the **raw** maximum of its floor
(`selection.floor_argmax`) — no smoothing and no complexity penalty, because a bound should not
be discounted for being long. A maximum at the horizon means the bound belongs to the horizon,
and the generated summary flags it. E3-MAX is reported as a bound and is not used for the
downstream prediction and model-ranking results. In the current run, primary and
capability arities both equal three: E3-Valid selects 15 terms by the smoothed plateau
rule, while E3-MAX selects 24 terms at the raw four-protocol floor maximum. It does
not use a wider grammar than E3-Valid in this configuration.

### How the configuration itself was chosen

The rule chooses a length *given* a configuration; the configuration is chosen by the same
criterion. `ml-meta-perf-search` fits E3 once for every point of the grid in
`config/study.json`'s `sweep` section (stability cap × ridge penalty × arity), applies the rule
to each, and keeps the configurations whose chosen equation is readable (`sweep.readable_terms`
terms at most). Among those, configurations whose smoothed floor is within `sweep.band` of the
best remain eligible under this numerical tolerance, and the one whose
doubly-held-out predictions **rank the models best** (mean average precision over the twenty
datasets) is proposed. Ranking decides because it is what the equation is for, and because it
separates configurations R² cannot: at the same floor, ranking precision varies widely from
one configuration to the next. The sweep writes every candidate's scores and a proposal;
adopting it is a reviewed edit to the file. The unchanged grid contains seven stability
caps, five penalties, and arities two and three: 70 configurations. Repeating that full
procedure on the updated corpus retained a stability cap of 3.5, penalty 0.1, and arity
three; the selection criteria and length rule were not changed.

The paired per-dataset table in the generated section compares other lengths with E3-Valid as
a sensitivity analysis. It does not define additional E3-Valid equations.
<!-- generated: do not edit below -->

## The configuration this run used

Meta-dataset: `dataset/meta_dataset.csv`

Every hyperparameter below is read from `config/study.json`; E1, E2 and E3 share all of them.

| setting | value |
|---|---|
| search.max_abs_zscore | 3.5 |
| search.penalty | 0.1 |
| search.pool_size | 600 |
| search.max_terms | 30 |
| search.beam_width | 6 |
| search.max_arity | 3 |
| selection.delta | 0.01 |
| selection.window | 4 |
| selection.smoothing | 3 |
| selection.capability_arity | 3 |

## Why a subset rather than every term

The control for the whole selection stage. If handing every candidate term to unpenalised least squares in one go transferred well, the beam search and the length rule would be machinery in search of a problem.

| terms | r2_IS | r2_LODO_clipped | r2_LODO_unclipped |
|---|---|---|---|
| 1080.0000 | 0.8110 | -0.6759 | -3.9684 |

**The solver is not the hard part; the sample size is.** All 1080 terms at once fit better under IS than the published equation (0.8110 against 0.6997) and transfer at -0.6759 under LODO, against the published equation's 0.6771. The unclipped figure — -4.0 — is what the fit does when a held-out dataset falls outside the convex hull of the other nineteen and nothing bounds the extrapolation. A design this much wider than 20 held-out groups can support has nothing to constrain it, which is what selection is for.

## Equation length

E3-Valid has **15 terms**. The rule, `selection.plateau_knee`, is the same for E1, E2 and E3-Valid. It reads the worst R² over IS, LODO, LOMO and DHO at every length, smoothed by a running median of 3 lengths; multi-Kneedle (`kneeliverse`) proposes the lengths where that curve's Pareto front bends -- here **5, 8, 10, 13, 15, 16** -- and the rule takes the first of them, or the smoothed maximum, after which the smoothed curve gains at most 0.01 over the next 4 lengths: the shortest equation that has stopped improving. It is a knee: the curve stops improving right after it.

**Why the curve is smoothed before a length is chosen.** The deepest crater on this curve is at **7 terms**, where the worst protocol reads 0.545, 0.013 below the neighbouring lengths. The running median of 3 lengths reads 0.545 there. A crater like this is not a property of the length -- it is one held-out dataset sitting outside the convex hull of the other nineteen in term space, where a linear equation extrapolates without limit and `validate._clip_to_training` pins the fold to its training floor -- and the same holds for a single lucky length. Neither should choose the published length.

The following paired analysis compares every length against E3-Valid, dataset by dataset; it is a sensitivity analysis rather than an additional selector:

| n_terms | r2_LODO | mae_LODO | mean_difference | p_value | ci_low | ci_high | verdict |
|---|---|---|---|---|---|---|---|
| 1 | 0.2148 | 0.2219 | 0.0990 | 0.0004 | 0.0724 | 0.1253 | worse |
| 2 | 0.4005 | 0.1838 | 0.0609 | 0.0000 | 0.0377 | 0.0859 | worse |
| 3 | 0.4633 | 0.1724 | 0.0495 | 0.0026 | 0.0297 | 0.0718 | worse |
| 4 | 0.5311 | 0.1585 | 0.0356 | 0.0026 | 0.0173 | 0.0559 | worse |
| 5 | 0.5658 | 0.1527 | 0.0298 | 0.0026 | 0.0140 | 0.0473 | worse |
| 6 | 0.5629 | 0.1525 | 0.0296 | 0.0118 | 0.0101 | 0.0499 | worse |
| 7 | 0.5695 | 0.1479 | 0.0250 | 0.0414 | 0.0096 | 0.0424 | worse |
| 8 | 0.5997 | 0.1442 | 0.0213 | 0.1153 | 0.0071 | 0.0364 | worse |
| 9 | 0.6198 | 0.1386 | 0.0156 | 0.1153 | 0.0050 | 0.0284 | worse |
| 10 | 0.6416 | 0.1344 | 0.0115 | 0.2632 | 0.0013 | 0.0242 | worse |
| 11 | 0.6487 | 0.1326 | 0.0097 | 0.5034 | -0.0005 | 0.0206 | tie |
| 12 | 0.6552 | 0.1317 | 0.0088 | 1.0000 | 0.0008 | 0.0173 | worse |
| 13 | 0.6596 | 0.1291 | 0.0062 | 0.8238 | -0.0011 | 0.0140 | tie |
| 14 | 0.6706 | 0.1258 | 0.0029 | 0.8238 | -0.0035 | 0.0110 | tie |
| 15 | 0.6771 | 0.1229 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | selected |
| 16 | 0.6702 | 0.1239 | 0.0010 | 0.5034 | -0.0036 | 0.0056 | tie |
| 17 | 0.6733 | 0.1218 | -0.0011 | 0.8238 | -0.0050 | 0.0030 | tie |
| 18 | 0.6812 | 0.1217 | -0.0013 | 0.1153 | -0.0060 | 0.0035 | tie |
| 19 | 0.6798 | 0.1222 | -0.0008 | 1.0000 | -0.0051 | 0.0035 | tie |
| 20 | 0.6782 | 0.1235 | 0.0006 | 0.8238 | -0.0050 | 0.0065 | tie |
| 21 | 0.6798 | 0.1239 | 0.0010 | 0.8238 | -0.0072 | 0.0099 | tie |
| 22 | 0.6804 | 0.1241 | 0.0012 | 0.8238 | -0.0069 | 0.0099 | tie |
| 23 | 0.6901 | 0.1209 | -0.0020 | 0.2632 | -0.0073 | 0.0031 | tie |
| 24 | 0.6950 | 0.1210 | -0.0019 | 1.0000 | -0.0075 | 0.0040 | tie |
| 25 | 0.6925 | 0.1217 | -0.0012 | 1.0000 | -0.0070 | 0.0047 | tie |
| 26 | 0.6837 | 0.1228 | -0.0002 | 0.8238 | -0.0071 | 0.0068 | tie |
| 27 | 0.6855 | 0.1227 | -0.0003 | 1.0000 | -0.0075 | 0.0071 | tie |
| 28 | 0.6938 | 0.1201 | -0.0029 | 1.0000 | -0.0125 | 0.0068 | tie |
| 29 | 0.6983 | 0.1226 | -0.0003 | 0.8238 | -0.0100 | 0.0083 | tie |
| 30 | 0.7060 | 0.1200 | -0.0029 | 0.8238 | -0.0110 | 0.0048 | tie |

The full curve reports all four protocols at every length; the length rule reads their minimum:

| n_terms | r2_IS | mae_IS | smape_IS | r2_LODO | mae_LODO | smape_LODO | r2_LOMO | mae_LOMO | smape_LOMO | r2_DHO | mae_DHO | smape_DHO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.2467 | 0.2199 | 41.1201 | 0.2148 | 0.2251 | 41.7083 | 0.2276 | 0.2230 | 41.4930 | 0.1984 | 0.2279 | 42.0295 |
| 2 | 0.4403 | 0.1795 | 38.0771 | 0.4005 | 0.1856 | 38.9150 | 0.4129 | 0.1839 | 38.6545 | 0.3769 | 0.1892 | 39.3901 |
| 3 | 0.5054 | 0.1665 | 36.3610 | 0.4633 | 0.1738 | 37.4804 | 0.4779 | 0.1714 | 37.0359 | 0.4412 | 0.1777 | 38.0091 |
| 4 | 0.5699 | 0.1523 | 34.5974 | 0.5311 | 0.1585 | 35.3919 | 0.5446 | 0.1569 | 35.3221 | 0.5123 | 0.1619 | 35.8623 |
| 5 | 0.5970 | 0.1473 | 34.0607 | 0.5658 | 0.1528 | 34.9640 | 0.5700 | 0.1521 | 34.9553 | 0.5466 | 0.1565 | 35.6511 |
| 6 | 0.6069 | 0.1459 | 33.9613 | 0.5629 | 0.1532 | 35.2080 | 0.5734 | 0.1514 | 35.0261 | 0.5388 | 0.1574 | 35.9027 |
| 7 | 0.6331 | 0.1374 | 32.1927 | 0.5695 | 0.1489 | 34.0446 | 0.5990 | 0.1431 | 33.1704 | 0.5448 | 0.1527 | 34.5357 |
| 8 | 0.6524 | 0.1332 | 33.1590 | 0.5997 | 0.1448 | 34.8665 | 0.6182 | 0.1393 | 34.4938 | 0.5762 | 0.1490 | 35.7633 |
| 9 | 0.6655 | 0.1302 | 32.3120 | 0.6198 | 0.1388 | 33.6864 | 0.6309 | 0.1366 | 33.3698 | 0.5954 | 0.1433 | 34.2409 |
| 10 | 0.6729 | 0.1278 | 32.4940 | 0.6416 | 0.1347 | 33.6457 | 0.6352 | 0.1345 | 32.9890 | 0.6152 | 0.1393 | 33.8487 |
| 11 | 0.6781 | 0.1256 | 32.3765 | 0.6487 | 0.1328 | 33.6992 | 0.6380 | 0.1324 | 32.9198 | 0.6204 | 0.1378 | 34.4379 |
| 12 | 0.6882 | 0.1234 | 32.0797 | 0.6552 | 0.1317 | 34.0225 | 0.6467 | 0.1308 | 33.1036 | 0.6267 | 0.1368 | 34.9487 |
| 13 | 0.6908 | 0.1214 | 31.5563 | 0.6596 | 0.1291 | 33.1332 | 0.6456 | 0.1291 | 32.5830 | 0.6267 | 0.1348 | 34.4990 |
| 14 | 0.6959 | 0.1191 | 30.8920 | 0.6706 | 0.1261 | 32.2411 | 0.6542 | 0.1260 | 31.8926 | 0.6418 | 0.1304 | 32.7609 |
| 15 | 0.6997 | 0.1176 | 30.9385 | 0.6771 | 0.1231 | 32.2077 | 0.6574 | 0.1247 | 31.9459 | 0.6478 | 0.1280 | 33.1510 |
| 16 | 0.7029 | 0.1162 | 31.1429 | 0.6702 | 0.1240 | 32.9329 | 0.6600 | 0.1237 | 32.0950 | 0.6440 | 0.1289 | 33.8552 |
| 17 | 0.7058 | 0.1150 | 30.7971 | 0.6733 | 0.1220 | 32.4237 | 0.6610 | 0.1226 | 31.6534 | 0.6471 | 0.1264 | 32.6842 |
| 18 | 0.7088 | 0.1152 | 31.0360 | 0.6812 | 0.1216 | 32.6095 | 0.6611 | 0.1237 | 32.0063 | 0.6517 | 0.1269 | 33.1530 |
| 19 | 0.7099 | 0.1151 | 31.1924 | 0.6798 | 0.1220 | 32.5899 | 0.6633 | 0.1237 | 32.1004 | 0.6518 | 0.1270 | 33.0995 |
| 20 | 0.7114 | 0.1154 | 31.3703 | 0.6782 | 0.1231 | 32.9638 | 0.6637 | 0.1239 | 32.2015 | 0.6505 | 0.1282 | 33.6932 |
| 21 | 0.7159 | 0.1146 | 30.2871 | 0.6798 | 0.1238 | 32.6745 | 0.6585 | 0.1247 | 31.4837 | 0.6441 | 0.1296 | 33.3120 |
| 22 | 0.7175 | 0.1143 | 30.3289 | 0.6804 | 0.1239 | 32.6061 | 0.6579 | 0.1251 | 31.8014 | 0.6434 | 0.1302 | 33.4032 |
| 23 | 0.7241 | 0.1136 | 31.0781 | 0.6901 | 0.1210 | 32.1751 | 0.6763 | 0.1225 | 32.2929 | 0.6653 | 0.1252 | 32.9278 |
| 24 | 0.7247 | 0.1131 | 30.9755 | 0.6950 | 0.1207 | 32.7414 | 0.6736 | 0.1225 | 32.4334 | 0.6696 | 0.1247 | 32.5231 |
| 25 | 0.7273 | 0.1131 | 31.0857 | 0.6925 | 0.1217 | 32.4378 | 0.6738 | 0.1233 | 32.6649 | 0.6616 | 0.1273 | 32.8722 |
| 26 | 0.7286 | 0.1119 | 30.9733 | 0.6837 | 0.1225 | 33.4069 | 0.6700 | 0.1231 | 32.7671 | 0.6488 | 0.1287 | 34.1027 |
| 27 | 0.7307 | 0.1115 | 30.8939 | 0.6855 | 0.1223 | 32.8934 | 0.6713 | 0.1230 | 32.6564 | 0.6489 | 0.1291 | 33.3652 |
| 28 | 0.7352 | 0.1106 | 30.1904 | 0.6938 | 0.1199 | 31.9349 | 0.6724 | 0.1226 | 31.7596 | 0.6573 | 0.1268 | 33.1661 |
| 29 | 0.7435 | 0.1102 | 29.9980 | 0.6983 | 0.1222 | 33.5189 | 0.6773 | 0.1242 | 31.9815 | 0.6584 | 0.1310 | 34.7846 |
| 30 | 0.7470 | 0.1089 | 30.4156 | 0.7060 | 0.1196 | 32.9322 | 0.6796 | 0.1232 | 32.0662 | 0.6659 | 0.1284 | 34.4542 |

## How much of the transfer is the form's selection

Every reported held-out number fixes the equation's form -- chosen once, on all observed rows -- and refits only its weights in each fold. The form was therefore chosen with the held-out dataset in view. The second row repeats the *selection* inside every LODO fold, at E3-Valid's 15 terms and configuration, and scores those predictions instead:

| LODO | r2 | mae | spearman |
|---|---|---|---|
| form chosen once, on all rows (reported) | 0.6771 | 0.1231 | 0.8300 |
| form re-chosen inside every fold | 0.0254 | 0.2158 | 0.5052 |

**LODO R² differs by 0.652** between the globally selected form (0.677) and within-fold term reselection (0.025). This is a partial selection-sensitivity diagnostic: global configuration, requested length, and library construction remain shared. Neither row independently assesses the complete discovery procedure. The fixed-form scores are conditional on selection.

<!-- end generated -->

## Limitations of the selection procedure

### Hyperparameter selection is not nested

The configuration and the equation length were chosen by reading held-out scores, and the
equation's form was chosen on all rows before its weights were refit per fold. The reported
transfer numbers are conditional on same-corpus selection and can be optimistic. The generated section *How much of the transfer is the form's selection*
measures the largest part of the gap by repeating the term selection inside every LODO fold;
the difference is large at this sample size, because choosing a handful of terms from hundreds
of candidates with twenty dataset groups is exactly where selection optimism concentrates. What
the fixed-form numbers license is the claim that *this* equation, recalibrated, transfers as
reported; what they do not license is a claim that the search would find an equally good one on
new data.

The **IS** numbers remain resubstitution scores rather than independent validation.
