# 3. Term generation and selection

*Implemented in `ml_meta_perf.fit`, `ml_meta_perf.analysis` and `ml_meta_perf.selection`.*

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

`ml_meta_perf.fit.transform_gap` is exactly $|\rho_s| - |\rho_p|$, and `guided_screen` ranks by
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

(default library; the wider library gains ~0.01 non-monotonically, which is search noise
rather than systematic improvement.)

Spending eight times the compute changes the fourth decimal place. Together with the
negative result on a richer vocabulary ([chapter 2](02-additive-model.md)), this locates
the limit in the model *form*, not in the optimiser.

**Per-length penalty tuning** was also tested: choosing $\lambda$ separately for each $k$
lifts leave-one-dataset-out R² at $k=14$ from 0.4429 to 0.4456. A gain of 0.003 does not
justify the extra configuration surface, and one global penalty is retained.

## Stage 3 — choosing the number of terms

More terms fit better and read worse. Picking the bend of that curve by eye is the kind of
judgement this project exists to remove from its results, so it is delegated to a detector.
[kneeliverse](https://github.com/mariolpantunes/knee)'s `autoelbow` takes no threshold,
sensitivity or smoothing window, so the chosen length is a property of the curve rather than
of a parameter chosen to produce a preferred answer.

Two decisions have to be made before the detector is run, and both were previously made
badly enough to invalidate the answer.

### Which curve the detector runs on

**Not in-sample R², which is what `knee_terms` used to default to.** In-sample is monotone in
the number of terms — adding a term cannot reduce the fit — so it can only ever say "more".
A length chosen on it is chosen on the one curve that cannot express the trade the choice is
about.

**And not one cross-validated curve either.** On twenty groups they wander, and
leave-one-dataset-out on this corpus has genuine craters: at 15 terms it reads 0.393 against
neighbours around 0.62. That is not noise but not a property of the length either — the
held-out `ASNM-CDX-2009` fold sits outside the convex hull of the other nineteen datasets in
term space, the equation extrapolates it to −2.41, and `validate._clip_to_training` pins the
fold to the training floor. One fold's extrapolation should not choose the published length.

So the detector runs on a **consensus across all three protocols** — `selection.consensus_curve`,
the per-length median. The median is what makes it robust: at 15 terms the three read
0.659 / 0.393 / 0.616 and the median takes 0.616, ignoring the crater. A mean would be
dragged to 0.556 by it. `min` is available as the conservative reading and is not the default.

### Which lengths the curve is reported at

**Every length from 1 to `max_terms`.** The curve used to be reported at
`(2, 4, 8, 12, 16, 20, 24, 26, 28, 32)` — non-uniform, and skipping 13, 15, 17 and 31, which
are exactly the four lengths where the transfer curve craters. The published curve was
therefore much smoother than the real one, and the detector was partly reporting the grid:
the same detector returns 4 terms on the ragged grid and 6 on the dense one. It costs
nothing to fix, because the beam search already builds the whole path and `run_equation` was
subsampling it.

![Accuracy versus equation length](../figures/term_count_curve.png)

The craters are visible in that figure. They are a real property of leave-one-dataset-out on
twenty groups and they belong on the plot.

### The rule, and everything it beats

**The length is the argmax of the consensus curve** — `selection.best_length`. It has no
threshold, no smoothing window and no sensitivity parameter, so it is a property of the curve
rather than of a value chosen to produce a preferred answer, and it re-derives itself when the
corpus changes. Applied under the two grammars the study reports it selects **15 terms** under
arity 2 and **23** under arity 3. Neither number appears anywhere in the code.

Every alternative that was computed is reported beside it, because a selection rule is only
defensible if what it beats is on the page:

| rule | terms | in-sample R² | LOO-dataset R² |
|---|---|---|---|
| Pareto front, closest to the utopia point | 4 | 0.539 | 0.505 |
| Pareto front, furthest from the nadir | 4 | 0.539 | 0.505 |
| Pareto front, furthest from the chord | 4 | 0.539 | 0.505 |
| knee detectors, gRDP-smoothed *(removed)* | 8 | 0.601 | 0.597 |
| parsimony: shortest not significantly worse | 10 | 0.634 | 0.615 |
| **argmax of the consensus curve — the rule** | **15** | **0.658** | **0.638** |

**The geometric rules and the paired test disagree, and the disagreement is the finding.**
Every geometric reading of this curve — three knee detectors on four curves, raw and
gRDP-smoothed at seven tolerances, plus the Pareto-front knee by all three standard forms —
lands between 4 and 8 terms. Every one of those lengths is **significantly worse** than 15
when the two are paired fold by fold over the twenty held-out datasets; 11 of the 32 lengths
searched are. A knee finds where the *marginal* return per term collapses, which on a
saturating curve is early. It does not ask whether the accuracy still being added is real,
and here it is, for several terms past the bend.

**Knee detection has therefore been removed rather than reported.** It was tried properly
first — the gRDP simplification works exactly as intended, taking three detectors that split
6/8/4 on the raw curve to unanimous agreement at 8 — and `kneeliverse` left the dependency
list with it. What replaced it is not a different detector but a different question.

**The parsimony alternative is reported and not adopted.** Ten terms is the shortest length
whose paired interval against 15 spans zero, and it is the right answer for a reader whose
readability budget is tighter than this study's. It gives up 0.023 of leave-one-dataset-out
R², which is measurable even where it is not significant, so the study takes the accuracy;
`results/length_choice.csv` carries the whole table so that choice can be remade.

Both **Pareto fronts** are also reported. Over (length, LOO-dataset R²) the front is 1–8, 10,
11, 14, 16, 19, 21 and 23 — nothing longer than 23 terms earns its length on transfer. Over
(length, in-sample R²) *every* length is on the front, because fit is monotone in terms and
so nothing is ever dominated. That is precisely why the in-sample curve cannot choose a
length by itself.
