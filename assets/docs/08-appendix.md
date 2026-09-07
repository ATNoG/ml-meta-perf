# 8. Appendix — approaches that were built, measured and rejected

Each of these was implemented and run against the same corpus and the same protocols as the
published equation. None is a candidate any longer. They are recorded here rather than in
the methods chapters because a paper's method section should say what the method *is*, not
retrace every route that was tried on the way to it — but they are recorded, because the
reason each one failed is evidence for the study's central claim, and because a reader who
wonders "why not just do X" deserves the measurement rather than an assurance.

**A caveat that applies to every number below.** Most were measured before 2026-09-05, when
the reported protocol changed from re-selecting terms inside each fold to fixing the form
and refitting only the weights. Numbers from the two protocols differ by up to 0.3 and must
not be read beside a current figure. What carries over is which of the two compared options
was better, not the magnitudes.

## Agglomerative construction, instead of enumeration

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

That is a fact about the size of this problem rather than about the idea. With 18 features,
exhaustive depth-2 enumeration is *complete*, so any pair-selection heuristic can only
return a subset of it. Selection heuristics start to pay when the space is too large to
enumerate, which at 18 features it is not.

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
[chapter 6](06-practices.md)).

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
`src/ml_meta_perf/construct.py` with its own tests until the repository was cut back to the code
the study actually runs; `git log -- src/ml_meta_perf/construct.py` recovers it. The measurement
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
missing structure is a rank-1 interaction ([chapter 4](04-equation.md)), and no amount of
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

## A richer transform vocabulary

Adding `f^3` and `1/sqrt(f)` to the unary transforms was tested and **made things worse**:

| | in-sample (k=20) | LOO-dataset (k=14) |
|---|---|---|
| base vocabulary | 0.6469 | +0.4429 |
| with `f^3`, `1/sqrt(f)` | 0.6303 | +0.4196 |

The additional transforms are high-variance, score well under screening, and displace
better terms. The vocabulary is not under-powered; [chapter 4](04-equation.md) locates the
real limit.

**Two of the five transforms already on offer are never used either.** Counting from the
published equation rather than from a sweep:

| transform | terms using it | share of weight mass |
|---|---|---|
| `log` | 19 of 24 | 79% |
| `id` | 12 of 24 | 41% |
| `sqrt` | 1 of 24 | 6% |
| `1/f` | **0** | **0%** |
| `f^2` | **0** | **0%** |

`log` is the workhorse by a wide margin, `sqrt` survives on a single term, and inversion
and squaring earn nothing at all despite being admissible on 14 and 18 of the 18 features
respectively. This is the strongest available evidence that the transform vocabulary is
already past the point of usefulness rather than short of it: the search had these shapes
available, screened them, and declined them. Trimming `1/f` and `f^2` would shrink the
library at no measured cost — they are retained only because a vocabulary chosen to fit one
meta-dataset's outcome is a worse default than one chosen on principle.
