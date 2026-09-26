# ml-meta-perf — study documentation

The abbreviations used on this page are machine learning (**ML**), application programming
interface (**API**), Matthews Correlation Coefficient (**MCC**), coefficient of determination
(**R²**), in-sample (**IS**), leave-one-dataset-out (**LODO**), and leave-one-model-out
(**LOMO**).
The equation labels are **E1** (dataset features only), **E2** (model features only),
**E3-Valid** (the plateau-selected equation using both feature groups), and **E3-MAX**
(the maximum-capability equation using both feature groups).

The study behind `ml-meta-perf`, written as a paper: what the data is, what form the model
takes, how its terms are generated and chosen, what the fitted equation says, how well it
does, and how it compares against the best practices the tabular-ML literature already
states. Figures are produced by `python -m ml_meta_perf`.

**Six chapters, each self-contained**, plus the related-work chapter that positions them.
Every chapter ends with its own limitations, so a reader never has to hold two places at once.

This page, and chapters 1, 3, 4, 5 and 6, each carry a **generated section**, rewritten by
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
| 3 | [Term generation and selection](03-term-selection.md) — screening, beam search, the ridge solve, and how many terms to keep | `ml_meta_perf.search`, `ml_meta_perf.fit`, `ml_meta_perf.analysis`, `ml_meta_perf.selection` |
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

**The contrast.** Three primary equations differ *only* in which features they may use — E1 dataset,
E2 model, E3-Valid both — fitted by one function, from one configuration, on the same rows, and
scored on the same rows under all four protocols. The gaps between them are the evidence about what each half of the
meta-data is worth. They share a scale but **not a ceiling**: E1 can only predict a
per-dataset constant, so its structural maximum is well below E3's, and the comparable
quantity is the fraction of its own ceiling each equation attains — the `reached` column of
the generated table below.

**The result.** E3-Valid explains roughly two thirds of MCC, and loses little of that when a
whole dataset or a whole model is held out — the three columns of the generated table below.
The same features under the looser arity-3 grammar do better still, which shows how far the
additive form goes under the study's readable grammars. E3-Valid passes both the
all-single-feature-terms ceiling and the additive mean-based reference. Cross-feature mixed terms therefore
add information beyond independent feature effects and recover part of the measurable
dataset×model interaction.

**The point.** Accuracy is what the equation is scored on; explainability is what it is
*for*. An opaque regressor reaches R² ≈ 0.9 under IS on this meta-data and scores below
0.1 under LODO ([chapter 5](05-evaluation.md) has the measurement), and
nothing can be read off it. Chapter 6 is the payoff: the
equation's terms are matched against published best practices one at a time, which is a
check no forest of the same accuracy admits.

<!-- generated: do not edit below -->

## The headline

The one table the study is summarised by, so that the summary cannot drift from the chapters. Every row is scored on the same 476 rows under the same protocols. E1, E2, and E3-Valid share the retained base configuration; E3-MAX uses the wider retained grammar as a capability bound.

| equation | features | terms | IS R2 | LODO R2 | LOMO R2 | DHO R2 | own ceiling | reached |
|---|---|---|---|---|---|---|---|---|
| E1 | dataset | 5 | 0.3394 | 0.3211 | 0.3022 | 0.2918 | 0.3539 | 0.9591 |
| E2 | model | 5 | 0.2537 | 0.1936 | 0.2312 | 0.1794 | 0.2821 | 0.8991 |
| E3-Valid | both | 17 | 0.6815 | 0.6513 | 0.6261 | 0.6151 |  |  |
| E3-MAX | both | 29 | 0.7271 | 0.6894 | 0.6538 | 0.6482 |  |  |

**Do not read these R² values as achievements against each other.** They share a scale but not a ceiling: E1 sees only dataset features, every row of a dataset shares one feature vector, and so E1 can predict nothing but a per-dataset constant. Its structural maximum is the `true dataset means` row, and reaching it means E1 is *done* rather than weak. The comparable quantity is the fraction of each equation's own ceiling, which the last column gives.

**E3-Valid and E3-MAX have blank ceiling cells because neither has a structural group-identity ceiling.** Nothing in the feature set stops an equation over both halves of the meta-data from predicting every cell, so there is no group-identity bound to divide by. The reference shown instead is the additive mean-based reference, in the comparison table of chapter 5. That reference bounds only an equation additive in dataset effect plus model effect; E3-Valid's mixed terms can represent interactions beyond it. The comparison table reports whether the current equation reaches or exceeds that reference.

<!-- end generated -->
