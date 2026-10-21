# 2. The additive model

> The measurements quoted in this chapter were made on the MCC corpus of the paper (release `v1.0.0` of the
> repository). They are not regenerated: the numbers of a study are in the `report.md` and the CSV files that
> `ml-meta-perf fit` writes next to its equations.

*Implemented in `ml_meta_perf.terms` and `ml_meta_perf.model`.*

The abbreviations used in this chapter are Matthews Correlation Coefficient (**MCC**),
genetic programming (**GP**), JavaScript Object Notation (**JSON**), coefficient of
determination (**R²**), in-sample (**IS**), leave-one-dataset-out (**LODO**),
leave-one-model-out (**LOMO**), doubly held out (**DHO**), and Sparse Regression of
Turbulent Stress Anisotropy (**SpaRTA**).
The equation labels are **E1** (dataset features only), **E2** (model features only), **E3**
(dataset and model features), **E3-Valid** (the plateau-selected E3 equation), and
**E3-MAX** (the maximum-capability E3 equation).

## The form

$$\mathrm{MCC} = w_0 + \sum_{i=1}^{k} w_i \, t_i(\mathbf{f})$$

Each $t_i$ is a simple expression over one to three raw features; each $w_i$ is a real
number **in the features' own units**, so the printed equation evaluates as written.
Predictions are clipped to the bounds of the target: $[-1, 1]$ for MCC, no clip for an unbounded
target such as a cost ([chapter 7](07-generic-targets.md)).

The form is **linear in the weights** but not in the features. That distinction is the
whole design:

- Because it is linear in the weights, the weights are solved **exactly** by ridge
  regression. There is no coefficient search. Genetic programming, the usual approach to
  symbolic regression, must search for its coefficients and is known to do so poorly
  (Virgolin et al., 2022).
- Because it is nonlinear in the features, it can express saturation, ratios and
  interactions without ever leaving the closed-form solve.

Restricting the *form* up front, rather than searching free-form expression trees, is an
established route to interpretable symbolic regression — compare the
Transformation-Interaction-Rational representation (de França, 2025) and SpaRTA
(Schmelzer et al., 2019), which chose deterministic sparse regression over a candidate
library precisely because GP expressions are unreadable even when accurate.

## The term vocabulary

| operation | form | operands |
|---|---|---|
| atom | `f`, `log(f)`, `sqrt(f)`, `1/f`, `f^2` | 1 |
| ratio | `f1 / f2` | 2 |
| product | `f1 * f2` | 2 |
| sum_ratio | `(f1 + f2) / f3` | 3 |

Terms are **data, not closures** — `Term(operation, operands)`, where each operand is an
`Atom(feature, transform)`. This is what lets a fitted equation round-trip through JSON
and be re-evaluated on new columns.

### Log-compressed operands

Inside a ratio, product or sum, a strictly positive feature enters as `log(f)`. Dividing
raw `gravity` — which spans 16 orders of magnitude — by anything produces a term whose
weight is around 1e-16 and whose meaning cannot be read. One decision, applied
consistently, keeps composite terms on a comparable scale.

## How many raw features may one term combine?

A design choice with a measured justification, and one that is easy to get wrong for most of this
project's life.

| operation | raw features | terms (arity 3 library) |
|---|---|---|
| `atom` | 1 | generated from the corpus |
| `ratio`, `product` | 2 | generated from the corpus |
| `sum_ratio` — `(f1+f2)/f3` | 3 | generated from the corpus |
| `ratio_of_sums` — `(f1+f2)/(f3+f4)` | 4 | *available, not default* |

### The asymmetry that was there, and is now fixed

`sum_ratio` is generated over all retained features, including mixed dataset/model
combinations. Applying the same vocabulary to every feature group avoids giving the
dataset-only equation access to an operation denied to model terms.

### Comparing arities without changing the experiment

`max_arity` exists so the grammar width is measured rather than assumed. Every candidate
arity is now fitted with the same ridge penalty, stability cap, candidate pool, beam width
and search horizon. Keeping those settings fixed is essential: otherwise a difference
between two arities would also be a difference between two tuning runs.

Arity 2 offers atoms, ratios and products. Arity 3 adds `sum_ratio`, the ratio-of-sums
`(f1+f2)/f3`.

### The arity is a hyperparameter, and E3-MAX compares length rules

The arity belongs to the **one configuration** shared by E1, E2 and E3
(`search.max_arity` in `config/mcc.json`). The configuration sweep searches arities 2 and
3 alongside the stability cap and ridge penalty. The revised corpus retains **arity 3**.

**E3-MAX** uses `selection.capability_arity` and the raw maximum of its worst-protocol
curve. In the current configuration that arity is also 3: E3-Valid and E3-MAX compare
length-selection rules under the same grammar. Their scores and form stability are in the `report.md` of a study.

Arity 4 remains available through `--search.max_arity 4`, outside the readability-limited
sweep. No current arity-4 performance claim is made here.

## Admissibility: which terms are allowed to exist

Division is the only operation in the vocabulary that can manufacture a column no linear
solver can use. Eligibility is therefore decided **per feature, before any term is
built**, rather than by screening terms afterwards.

A feature may serve as a denominator only if some form of it — `log(f)` preferred, else
`f` — satisfies

$$\frac{\max |d|}{\min |d|} \le 20$$

Bounding **dynamic range** is the criterion that matches the failure mode. A divisor
spanning three orders of magnitude produces a ratio spanning three orders of magnitude,
so one row holds nearly all of the term's variance and its fitted weight describes that
row rather than a relationship. Requiring only $\min|d| > 0$ would admit exactly those
terms.

The rule excludes a raw denominator reaching zero, the log of anything reaching 1
(since $\log 1 = 0$), and counts whose raw dynamic range exceeds the limit. A logarithmic
representation can qualify where the corresponding raw column cannot.

Products are unconditional; only ratios are gated, so refusing a ratio never costs the
corresponding product.

### Why this matters, quantitatively

A nearly singular denominator can create a term dominated by one observation. Term
standardisation does not remove that extrapolation risk. The structural denominator rule
and stability cap constrain which columns reach the solver; they limit capacity without
guaranteeing generalisation. `ml_meta_perf.terms.is_admissible` remains a numerical
backstop for libraries assembled by hand.

### Two terms, one column

A repeated *name* is not the only way the same column enters the library twice, and the
other way is invisible on the page. In this meta-dataset `inst_to_attr` is
$n_{\text{inst}} / n_{\text{attr}}$, so

$$\log(\texttt{inst\_to\_attr}) + \log(\texttt{nr\_attr}) \;=\; \log(\texttt{nr\_inst})$$

exactly — and the grammar generates both sides. The beam search also rejects a candidate correlating above 0.95 with a selected term.
Removing equivalent columns preserves candidate-pool slots and avoids reporting two
forms of the same column as independent findings.

`Library` therefore drops a term whose centred column correlates with an already-kept
column above $1 - 10^{-9}$. Centring first is what makes the test match the fit: fitting
happens on standardised terms, so two columns differing by an additive or multiplicative
constant are one column to the solver however different their raw values look. The first
term of a pair wins, and since generation runs simple to complex the survivor is the
shorter form.

### What the vocabulary assumes about a feature

Every transform above assumes a **strictly positive, continuous** feature, and the library
degrades quietly rather than loudly when given anything else — the affected terms are never
generated, so a caller adding a feature sees a smaller library, not an error.

| feature reaches | `log` | `sqrt` | `1/f` | may be a denominator |
|---|---|---|---|---|
| strictly positive | yes | yes | yes | if the range rule allows |
| zero | no | no | no | no |

A **binary** feature is the degenerate case: only `f` and `f^2` survive, and for a 0/1
column those are the same numbers under two names — caught by the collinearity rule above.
What remains available to it is addition and multiplication, and the two differ in kind.
Inside a sum, an indicator contributes a level to every row: $(a + b)/c$ expands to
$a/c + b/c$, an additive offset. Inside a product it **zeroes the term entirely** on the
rows where it is off, so `[log(Processing Units Number)] * [is a tree]` does not say
"capacity helps" but "capacity helps, for trees" — a per-group slope written in product
notation, not a relationship. A continuous descriptor that *grades* the distinction is
preferable to an indicator that switches on it.

## Library sizes

Library sizes depend on the corpus, feature group, arity, and `max_abs_zscore`. The
stability cap is set in `config/mcc.json`. The run regenerates the admissible
libraries rather than retaining counts from an earlier corpus.

The cap rejects a term when one row lies too many standard deviations from its mean.
It is sample-size dependent: a lone outlier among $n$ rows can approach $\sqrt{n}$.
All three primary equations use the same observed rows and the same selected cap.

## Which transforms are admitted, and why not others

The unary vocabulary is fixed to identity, `log`, `sqrt`, reciprocal, and square.
Positive-only operations are generated only when their domain conditions hold; the
stability and denominator rules then restrict the resulting columns. Keeping this small
vocabulary makes the selected form inspectable and limits the search space.

### Polynomials

Squared features are available. Higher powers would enlarge the search and amplify large
values, so they are outside the retained vocabulary.

### `exp`

Exponential transforms can overflow or amplify heavy tails. They are outside the fixed
vocabulary. This design choice is not evidence that no expanded grammar could improve
performance on the revised corpus.

## Why the raw features are not scaled first

Raw named descriptors enter the vocabulary in their recorded units. Scaling them before
nonlinear term construction changes the candidate functions: for example, an additive
shift changes a logarithm or reciprocal, and centring can violate positive-domain rules.
Multiplicative rescaling alone does not change a column's max/min dynamic range.

### Yeo-Johnson and rank transforms

A fitted preprocessing transform would define a different library and require its own
training-only estimation. The retained method instead offers a small set of explicit
transforms within each term.

### Reading impact off the weights

The raw weights evaluate the equation; standardised weights compare coefficients on a
common term scale. `attribution.term_effects` also reports effects in MCC units. Their
interpretation remains conditional on the selected terms and descriptor encoding.

## Standardisation

Terms are standardised using per-term means and standard deviations learned on
**training rows only** before the ridge fit. The standardisation is folded back into
the raw weights:

$$w^{\text{raw}}_i = \frac{w_i}{\sigma_i}, \qquad w_0^{\text{raw}} = \bar{y} - \sum_i \frac{w_i \mu_i}{\sigma_i}$$

A common ridge penalty depends on the scale of its columns, which is why this step
matters. Correlation itself is invariant to linear rescaling. Both raw and standardised
weight vectors are retained on the fitted object.

## Limitations of the form

### The additive form

The equation is additive in its selected terms; mixed terms can represent some
interactions between dataset and model descriptors. They do not establish a universal upper bound on alternative symbolic forms.
