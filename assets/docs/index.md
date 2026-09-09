# ml-meta-perf — study documentation

The study behind `ml-meta-perf`, written as a paper: what the data is, what form the model
takes, how its terms are generated and chosen, what the fitted equation says, how well it
does, and how it compares against the best practices the tabular-ML literature already
states. Figures are produced by `python -m ml_meta_perf`.

**Six chapters, each self-contained**, plus the related-work chapter that positions them.
Every chapter ends with its own limitations, so a reader never has to hold two places at once.

This page, and chapters 1, 4, 5 and 6, each carry a **generated section**, rewritten by
`ml_meta_perf.report` on every run and marked as such in the source. Everything above that
marker is hand-written prose describing the method, which does not change between runs;
everything below it is the result, computed from the fitted equation. A chapter therefore
cannot carry a stale table, and the results sit in the chapter that discusses them rather
than in a separate report a reader has to cross-reference. **The summary below is written to
match**: it states what each number means and leaves the number itself to the generated
table at the foot of this page.

The package API is documented separately, generated with `pdoc` from the module docstrings
and published to GitHub Pages by `.github/workflows/docs.yml`. Each chapter names the
modules it describes, and each of those modules opens with a docstring covering the same
ground from the implementation side.

## Chapters

| | | modules |
|---|---|---|
| 0 | [Related work and positioning](00-related-work.md) | — |
| 1 | [The dataset](01-dataset.md) — the corpus, the meta-features, and what each one means | `ml_meta_perf.data` |
| 2 | [The additive model](02-additive-model.md) — the equation form, and what a term is | `ml_meta_perf.terms`, `ml_meta_perf.model` |
| 3 | [Term generation and selection](03-term-selection.md) — screening, beam search, the ridge solve, and how many terms to keep | `ml_meta_perf.fit`, `ml_meta_perf.analysis`, `ml_meta_perf.selection` |
| 4 | [The equation](04-equation.md) — the three equations, the ceilings that bound them, and what interpretability buys | `ml_meta_perf.experiment`, `ml_meta_perf.validate` |
| 5 | [Evaluation](05-evaluation.md) — protocols, metrics, baselines, and the ranking and threshold decisions | `ml_meta_perf.validate`, `ml_meta_perf.stats` |
| 6 | [Best practices against the equation](06-practices.md) — published guidance checked term by term | `ml_meta_perf.practices`, `ml_meta_perf.attribution`, `ml_meta_perf.guidance` |


## The argument, in one page

**The problem.** Given a dataset and a classifier, predict the Matthews Correlation
Coefficient the classifier will reach. The corpus is one row per (dataset, model) pair, with
dataset meta-features and model meta-features on each; [chapter 1](01-dataset.md) generates
its exact shape.

**The form.** `MCC = w₀ + w₁t₁ + … + wₖtₖ`, where each `t` is a small named expression over
one to three raw features (`log(gravity) / log(Processing Units Number)`, and so on) and
each `w` is a real number. The form is fixed before validation and only the weights are
refit per fold, because the form *is* the claim being tested.

**The method.** Build a library of admissible terms under a grammar, screen it by paired
Pearson and Spearman correlation, and select a subset by beam search with an exact ridge
solve at every step. One term per combination of raw features, so no relationship is stated
twice.

**The contrast.** Three equations differ *only* in which features they may use — E1 dataset,
E2 model, E3 both — fitted by one function on the same rows and scored on the same rows
under both protocols. The gaps between them are the evidence about what each half of the
meta-data is worth. They share a scale but **not a ceiling**: E1 can only predict a
per-dataset constant, so its structural maximum is well below E3's, and the comparable
quantity is the fraction of its own ceiling each equation attains — the `reached` column of
the generated table below.

**The result.** E3 explains roughly two thirds of MCC, and loses very little of that when a
whole dataset or a whole model is held out — the three columns of the generated table below.
The same features under the looser arity-3 grammar do better still, which is how far the
additive form goes at all. E3 meets the additive oracle and passes the
all-single-feature-terms ceiling, both of which it can only do by representing dataset×model
interaction — which is what its mixed terms are for.

**The point.** Accuracy is what the equation is scored on; explainability is what it is
*for*. An opaque regressor reaches R² ≈ 0.9 in-sample on this meta-data and transfers at
under 0.1 leave-one-dataset-out ([chapter 5](05-evaluation.md) has the measurement), and
nothing can be read off it. Chapter 6 is the payoff: the
equation's terms are matched against published best practices one at a time, which is a
check no forest of the same accuracy admits.

<!-- generated: do not edit below -->

## The headline

The one table the study is summarised by, so that the summary cannot drift from the chapters. Every row is scored on the same 476 rows under the same protocols; the equations differ **only** in which features they may draw on.

| equation | features | terms | in-sample R2 | LOO-dataset R2 | LOO-model R2 | own ceiling | reached |
|---|---|---|---|---|---|---|---|
| E1 | dataset | 7 | 0.3485 | 0.3411 | 0.3062 | 0.3539 | 0.9848 |
| E2 | model | 6 | 0.2485 | 0.1849 | 0.2285 | 0.2821 | 0.8808 |
| E3 | both | 15 | 0.6578 | 0.6381 | 0.6218 |  |  |
| E3 capability | both | 23 | 0.6751 | 0.6439 | 0.6327 |  |  |

**Do not read these R² values as achievements against each other.** They share a scale but not a ceiling: E1 sees only dataset features, every row of a dataset shares one feature vector, and so E1 can predict nothing but a per-dataset constant. Its structural maximum is the `true dataset means` row, and reaching it means E1 is *done* rather than weak. The comparable quantity is the fraction of each equation's own ceiling, which the last column gives.

**E3's ceiling cells are blank because it has no structural one.** Nothing in the feature set stops an equation over both halves of the meta-data from predicting every cell, so there is no group-identity bound to divide by. The reference it is read against instead is the additive oracle, in the comparison table of chapter 5 -- and E3 is *expected* to pass that, because the oracle bounds only an equation additive in dataset effect plus model effect, which E3's mixed terms are not.

<!-- end generated -->
