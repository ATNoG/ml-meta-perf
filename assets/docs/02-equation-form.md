# 2. Equation form and term vocabulary

*Implemented in `metafit.terms` and `metafit.model`.*

## The form

$$\mathrm{MCC} = w_0 + \sum_{i=1}^{k} w_i \, t_i(\mathbf{f})$$

Each $t_i$ is a simple expression over one to three raw features; each $w_i$ is a real
number **in the features' own units**, so the printed equation evaluates as written.
Predictions are clipped to $[-1, 1]$.

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

In this meta-dataset the rule excludes `nr_norm`, `nr_bin` and `nr_outliers` (which reach
0), the log of anything reaching 1 (since $\log 1 = 0$), and wide-range counts such as
`nr_inst` in raw form — though `log(nr_inst)` qualifies comfortably, which is the
compression the preference order exists to find.

Products are unconditional; only ratios are gated, so refusing a ratio never costs the
corresponding product.

### Why this matters, quantitatively

Allowing unrestricted division raises in-sample R² to 0.611 and drives leave-one-dataset-out
R² to **-1.7**. Standardisation hides the problem while fitting — the spike simply becomes
the scale — but the term explodes on a held-out dataset outside the training range.

`metafit.terms.is_admissible` remains as a cheap numerical backstop for libraries
assembled by hand, but nothing `build_library` produces depends on it. Its effect is
visible at loose stability caps: at `max_abs_zscore=6` the worst leave-one-dataset-out R²
across the sweep improved from **-1.66 to -0.39** once the rule became structural.

## Library sizes

| configuration | `max_abs_zscore` | terms |
|---|---|---|
| E1 (dataset only, 20 aggregated rows) | 3.0 | 136 |
| E2 default | 3.0 | 172 |
| E2 accuracy-leaning | 4.0 | 360 |

The `max_abs_zscore` cap rejects a term when a single row sits more than that many
standard deviations from its mean. The cap is **sample-size dependent**: a lone outlier
among $n$ rows can reach a z-score of at most about $\sqrt{n}$, so a cap of 8 constrains
the 476-row fit but would be inert on the 20-row aggregated fit. The two configurations
therefore do not share a value.

## Which transforms are admitted, and why not others

`log` and `f^2` are already in the vocabulary; `log` is the workhorse, since it is also the
compression applied to every composite operand. The obvious extensions — higher
polynomials and `exp` — were measured rather than argued about. **None is technically
difficult; each is a one-line addition. The constraint is statistical.**

How many of the 17 features survive each transform's admissibility rules, and the best
absolute correlation with MCC among those that do:

| transform | admissible | overflow | single-row spike | best \|r\| |
|---|---|---|---|---|
| `f` | 17/17 | 0 | 0 | **0.376** |
| `log(f)` | 11/17 | 0 | 0 | 0.356 |
| `sqrt(f)` | 11/17 | 0 | 0 | 0.356 |
| `1/f` | 11/17 | 0 | 0 | 0.341 |
| `f^2` | 16/17 | 0 | 1 | 0.360 |
| `f^3` | 16/17 | 0 | 1 | 0.333 |
| `f^4` | 16/17 | 0 | 1 | 0.317 |
| `exp(f)` | **8/17** | **5** | 4 | 0.303 |
| `exp(-f)` | 16/17 | 0 | 1 | 0.317 |
| `exp(f / max f)` | 17/17 | 0 | 0 | 0.372 |

`log` and `1/f` reach only 11 of 17 features because the remaining six contain zeros or
negatives — that is a property of the data, not a restriction of the grammar.

### Polynomials

Admissible beyond squared, but **monotonically less useful**: best correlation falls 0.360
(`f^2`) → 0.333 (`f^3`) → 0.317 (`f^4`). Adding `f^3` to the vocabulary was measured
end-to-end and dropped leave-one-dataset-out R² from 0.443 to 0.420. Squared is kept;
higher powers are not.

### `exp`

Raw `exp(f)` **overflows on five features** — `gravity` reaches 1.0e16 and `nr_inst` 7.1e6,
so $e^f$ is not representable — and four more produce single-row spikes, leaving 8 of 17.
Where it does survive it scores *worse* than the untransformed feature (0.303 against
0.376). Scaling the argument (`exp(f / max f)`) makes all 17 admissible but is then close
to the identity over its range and still scores below it (0.372 against 0.376), so it adds
nothing the grammar does not already have.

The decisive argument is directional. Every feature with a meaningful gap between its
Spearman and Pearson correlations has a **positive** one:

| feature | \|r\| | \|ρ\| | gap |
|---|---|---|---|
| `gravity` | 0.047 | 0.286 | **+0.239** |
| `nr_cor_attr` | 0.074 | 0.197 | +0.122 |
| `inst_to_attr` | 0.027 | 0.137 | +0.110 |
| `ns_ratio` | 0.296 | 0.395 | +0.099 |

A positive gap means monotone but curved with a heavy tail — the relationship is there, and
a straight line misses it because a few large values dominate. That is a request for
**compression**, which `log`, `sqrt` and `1/f` provide. `exp` expands. It is the wrong
direction for every feature in this meta-dataset, which is why it is absent rather than
merely untested.

## Negative result: a richer vocabulary does not help

Adding `f^3` and `1/sqrt(f)` to the unary transforms was tested and **made things worse**:

| | in-sample (k=20) | LOO-dataset (k=14) |
|---|---|---|
| base vocabulary | 0.6469 | +0.4429 |
| with `f^3`, `1/sqrt(f)` | 0.6303 | +0.4196 |

The additional transforms are high-variance, score well under screening, and displace
better terms. The vocabulary is not under-powered; [chapter 5](05-oracles.md) locates the
real limit.

## Standardisation

Terms are standardised (per-term mean and standard deviation, learned on **training rows
only**) before selection and fitting, and the standardisation is then folded back into
the weights so the published equation reads in raw units:

$$w^{\text{raw}}_i = \frac{w_i}{\sigma_i}, \qquad w_0^{\text{raw}} = \bar{y} - \sum_i \frac{w_i \mu_i}{\sigma_i}$$

Raw term scales span four orders of magnitude here (std 1.3e-2 to 1.5e+2), so this is not
cosmetic:

| | standardised | raw |
|---|---|---|
| same 12 terms, penalty = 0 | 0.569611 | 0.569611 |
| same 12 terms, penalty = 20 | 0.556 | 0.559 |
| same 12 terms, penalty = 200 | 0.461 | 0.503 |
| **terms chosen by the search** | **0/12 overlap between the two** | |

Ordinary least squares is exactly scale-invariant, so with no penalty the two agree to
six decimals — the correctness check. Standardisation matters for the two other things
the pipeline does: the ridge penalty is one number applied to every weight and is only
meaningful when terms share a scale, and selection compares candidates by correlation
with the residual, which on raw scales is dominated by whichever term happens to be
largest. On this data the two designs select **completely disjoint** sets of 12 terms.

Both weight vectors are retained on the fitted object: the raw weights *are* the
equation, the standardised weights ($\beta$) are how terms rank against each other.
