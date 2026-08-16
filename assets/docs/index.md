# metafit — study documentation

Reference material for the study behind `metafit`: why the model has the form it has, how
its terms are chosen and fitted, how it is evaluated, and what the results mean. Written
at the level of detail a paper needs, with the figures produced by `python -m metafit
--figures`.

The package API is documented separately from these pages, generated with `pdoc` from the
module docstrings — `make docs`. Each chapter below names the modules it describes, and
each of those modules opens with a docstring covering the same ground from the
implementation side.

## Chapters

| | | modules |
|---|---|---|
| 1 | [The problem and the data](01-problem.md) | `metafit.data` |
| 2 | [Equation form and term vocabulary](02-equation-form.md) | `metafit.terms`, `metafit.model` |
| 3 | [Search and fitting](03-search-and-fitting.md) | `metafit.fit`, `metafit.analysis`, `metafit.selection` |
| 4 | [Evaluation methodology](04-evaluation.md) | `metafit.validate`, `metafit.stats` |
| 5 | [Oracles and ceilings](05-oracles.md) | `metafit.validate` |
| 6 | [Results](06-results.md) | `metafit.experiment` |
| 7 | [From equation to practice](07-practices.md) | `metafit.practices`, `metafit.attribution` |
| 8 | [Limitations and threats to validity](08-limitations.md) | — |

## The one-paragraph version

`metafit` fits `MCC = w1·t1 + … + wk·tk`, where each `t` is a simple expression over one
to three meta-features of a dataset and of a classifier, and each `w` is a real number in
the features' own units. Terms are drawn from a filtered candidate library, screened by
paired Pearson and Spearman correlation, and selected by beam search with an exact ridge
solve at every step. Two equations are contrasted — one over dataset features alone, one
over dataset and model features — to isolate how much of MCC is attributable to model
capability rather than dataset difficulty. Validation is leave-one-dataset-out and
leave-one-model-out; the inflation caused by random k-fold splitting on grouped
meta-data is quantified rather than assumed away.
