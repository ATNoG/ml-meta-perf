# 3. Term generation and selection

> The measurements quoted in this chapter were made on the MCC corpus of the paper (release `v1.0.0` of the
> repository). They are not regenerated: the numbers of a study are in the `report.md` and the CSV files that
> `ml-meta-perf fit` writes next to its equations.

*Implemented in `ml_meta_perf.search`, `ml_meta_perf.fit`, `ml_meta_perf.selection` and `ml_meta_perf.lengths`.*

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
transfers catastrophically — measured on the MCC corpus of the paper, against the published equation. The solver is not the hard part; the sample size
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
pair.

The timing and profiling measurements below are historical implementation benchmarks,
not timings of the updated corpus or the complete 70-configuration search. They describe
the numerical optimizations.

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
its neighbours. On the MCC curve of the paper the deepest crater was a single length, which the median ignores.

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
width are hyperparameters in `config/mcc.json`, stated rather than implied; `report.md` names the length rule a study used.

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


### E3-MAX, the bound

E3-MAX answers a separate capability question: how far the additive form reaches if it is
not asked to stay readable. It uses the configured capability arity
(`selection.capability_arity`), at the **raw** maximum of its floor
(`selection.floor_argmax`) — no smoothing and no complexity penalty, because a bound should not
be discounted for being long. A maximum at the horizon means the bound belongs to the horizon,
and the study reports it. E3-MAX is reported as a bound. The report gives the arity and the term
count of each variant: E3-Valid by the rule `selection.method` names, E3-MAX at the raw floor maximum
under every declared protocol. E3-MAX uses a wider grammar than
E3-Valid only when the configured capability arity exceeds the primary one.

### How the configuration itself was chosen

The rule chooses a length *given* a configuration; the configuration is chosen by the same
criterion. `ml-meta-perf-search` fits E3 once for every point of the grid in
`config/mcc.json`'s `sweep` section (stability cap × ridge penalty × arity), applies the rule
to each, and keeps the configurations whose chosen equation is readable (`sweep.readable_terms`
terms at most). Among those, configurations whose smoothed floor is within `sweep.band` of the
best remain eligible under this numerical tolerance, and the one whose
doubly-held-out predictions **rank the models best** (mean average precision over the twenty
datasets) is proposed. Ranking decides because it is what the equation is for, and because it
separates configurations R² cannot: at the same floor, ranking precision varies widely from
one configuration to the next. The sweep writes every candidate's scores and a proposal;
adopting it is a reviewed edit to the file. The unchanged grid contains seven stability
caps, five penalties, and arities two and three: 70 configurations. Repeating that full
procedure on the updated corpus retained the configuration now in `config/mcc.json`;
the selection criteria and length rule were not changed.

Lengths other than E3-Valid's are a sensitivity analysis and define no additional equations.
