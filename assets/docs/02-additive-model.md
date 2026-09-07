# 2. The additive model

*Implemented in `ml_meta_perf.terms` and `ml_meta_perf.model`.*

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

## How many raw features may one term combine?

A design choice with a measured justification, and one that was wrong for most of this
project's life.

| operation | raw features | terms (arity 3 library) |
|---|---|---|
| `atom` | 1 | 30 |
| `ratio`, `product` | 2 | 75 |
| `sum_ratio` — `(f1+f2)/f3` | 3 | 211 |
| `ratio_of_sums` — `(f1+f2)/(f3+f4)` | 4 | *available, not default* |

### The asymmetry that was there, and is now fixed

`sum_ratio` was originally generated over the **dataset features only**. Every other
operation could pair a dataset feature with a model one; the highest-arity operation could
not. That made the richest part of the grammar the only part unable to express a
dataset×model interaction — the exact combination carrying E3's entire lift over E1, and
the exact structure the oracle ladder says is missing.

Generating it over all features instead takes mixed terms in the library from **19 to 126**
and is worth, on its own, most of the improvement reported in
[chapter 4](04-equation.md).

### Is three enough? — the design point, measured properly

`max_arity` exists so this is answered rather than assumed, and answering it correctly
requires **retuning the ridge penalty inside each arity**. A wider grammar needs more
shrinkage; comparing arities at a fixed penalty measures the penalty as much as the arity.

The fixed-penalty comparison (λ = 20 throughout, 32 terms) suggests arity 2 is best for
transfer:

| max arity | library | in-sample | LOO-dataset |
|---|---|---|---|
| 2 | 92 | 0.5687 | +0.458 |
| 3 | 281 | 0.5758 | +0.375 |
| 4 | 1599 | 0.6186 | +0.321 |

**That reading is an artefact.** λ = 20 was tuned for the narrow grammar, so the wider ones
are being shown under-shrunk. Sweeping λ ∈ {1, 5, 20, 50} and k ∈ {12 … 32} *within* each
arity and reporting each one's best gives a different picture:

| max arity | library | best in-sample | best LOO-dataset |
|---|---|---|---|
| 2 | 92 | 0.6222 (λ=1, k=32) | +0.4629 (λ=20, k=24) |
| **3** | **281** | **0.6361** (λ=1, k=32) | **+0.4658** (λ=5, k=24) |
| 4 | 1599 | **0.6714** (λ=1, k=32) | +0.2628 (λ=50, k=16) |

<sub>Measured over the five model descriptors the corpus ships, before `Model Capability`
joined them; the library sizes and R² values in this table are that configuration's. The
*decision* — arity 3 — was re-checked against the enlarged feature set and is unchanged,
which is why the sweep has not been re-run to refresh four numbers that support a
conclusion they still support.</sub>

Three conclusions, and the first two were invisible at fixed penalty:

1. **Arity 3 dominates arity 2 on both axes.** Better fit (0.6361 vs 0.6222) *and* better
   transfer (+0.4658 vs +0.4629). There is no reason to prefer two-feature terms; the
   apparent transfer advantage of arity 2 was the fixed penalty.
2. **Arity 4 is a fit-only option, and an expensive one.** It buys +0.035 in-sample over
   arity 3 and gives up **0.203** of transfer — roughly six units of transfer per unit of
   fit. That is the trade, stated properly.
3. **The optimal penalty falls as arity rises** for transfer (20 → 5 → 50 is not monotone,
   but arity 4's best transfer needs both the heaviest shrinkage *and* the shortest
   equation, k=16, which is the signature of a grammar the sample cannot support).

So `max_arity = 3` is the default because it is the only setting that is not dominated:
arity 2 is beaten outright, arity 4 wins one axis at a ruinous price on the other.

**The fitted equation confirms it directly.** In the published 20-term E3 the three-feature
`sum_ratio` accounts for 7 terms and **37% of the standardised weight mass** — the search
did not merely tolerate the extra arity, it built better than a third of the equation out
of it. A
two-feature grammar would have had to express that half some other way, and the 0.6222
ceiling above is what happens when it tries. Counted from the equation by
`report.operation_usage`; see [chapter 6](06-practices.md#3-which-operations-the-equation-needed).

The reason not to go past four is different and does not need a measurement. A
`(f1+f2)/(f3+f4)` term already names four features and two operations, and the grammar
exists so a reader can hold a term in their head. **Arity is capped by legibility before it
is capped by evidence** — four is already at the edge of what belongs in a printed
equation, and the evidence happens to agree.

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

`ml_meta_perf.terms.is_admissible` remains as a cheap numerical backstop for libraries
assembled by hand, but nothing `build_library` produces depends on it. Its effect is
visible at loose stability caps: at `max_abs_zscore=6` the worst leave-one-dataset-out R²
across the sweep improved from **-1.66 to -0.39** once the rule became structural.

### Two terms, one column

A repeated *name* is not the only way the same column enters the library twice, and the
other way is invisible on the page. In this meta-dataset `inst_to_attr` is
$n_{\text{inst}} / n_{\text{attr}}$, so

$$\log(\texttt{inst\_to\_attr}) + \log(\texttt{nr\_attr}) \;=\; \log(\texttt{nr\_inst})$$

exactly — and the grammar generates both sides. Three such pairs existed in the 281-term
library. The beam search would not have put both members in one equation — it refuses a
candidate correlating above 0.95 with a selected term — so nothing published was ever at
risk. What the duplicates cost is candidate-pool slots, search time, and a place in the
reported term rankings, where the two forms appear as two independent findings and are one.

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

| configuration | `max_abs_zscore` | terms |
|---|---|---|
| E1 (dataset features only) | 3.0 | 137 |
| E2 (model features only) | 3.0 | 30 |
| **E3 default** | **3.0** | **316** |
| arity-4 grammar (measured, not published) | 4.0 | 5224 |

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

How many of the 18 features survive each transform's admissibility rules, and the best
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

`log` and `1/f` reach only 14 of 18 features because `nr_bin`, `nr_cor_attr`, `nr_norm` and
`nr_outliers` contain zeros — that is a property of the data, not a restriction of the
grammar. **All four columns that block them are dataset features**: every model feature is
strictly positive, which is a requirement the model side is now held to by
`tests/test_model_features.py` rather than a coincidence.

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

## Why the raw features are not scaled first

An obvious alternative to log-compressing operands and capping term stability is to scale
every feature column up front. It was tested, at the same admissibility cap throughout.
(Measured against the pre-symmetry grammar, whose baseline was 0.5582; the comparison
between rows is what matters and none of it depends on the baseline.)

| feature scaling | library | in-sample (k=14) | LOO-dataset (k=14) |
|---|---|---|---|
| **none (current)** | 172 | 0.5582 | **+0.443** |
| divide by max (multiplicative) | 120 | 0.5668 | +0.384 |
| min-max to [1, 2] (affine) | 544 | 0.5230 | **-0.285** |
| min-max to [1, 10] (affine) | 571 | 0.5578 | -0.093 |
| standard scaler (mean 0, sd 1) | 97 | 0.5168 | +0.037 |

Three separate reasons it does not help.

**Standard scaling destroys the vocabulary.** Centring makes **all 18 features take
non-positive values**, so `log`, `sqrt` and `1/f` become undefined for every one of them.
The library collapses from 172 terms to 97 — only identity, squares and products survive.
The transforms doing most of the work are precisely the ones that require positivity.

**Multiplicative scaling cannot fix dynamic range, because dynamic range is invariant to
it.** `gravity` has a max/min ratio of 6.24e15, and dividing the column by its maximum
leaves that ratio at 6.24e15 exactly. Since the admissibility rules are about *ratios*
rather than magnitudes, a constant factor changes nothing they test. (It does shift the
library, from 172 terms to 120, because `log` turns a constant factor into an additive
shift and the denominator rule is not shift-invariant — but that is a side effect, not a
fix.)

**Affine scaling fixes dynamic range by removing the signal.** Squeezing every feature into
[1, 2] does make everything admissible — the library grows to 544 terms — but over that
range `log`, `sqrt` and the identity are nearly the same function, so the library fills
with near-duplicates and leave-one-dataset-out R² collapses to **-0.285**. The compression
`log` was providing is exactly what the scaling removed.

The magnitude problem here is a *dynamic-range* problem, and log-compression is the
response to it. Scaling addresses magnitude, which was never the difficulty.

### Yeo-Johnson and rank transforms

Two more principled options than min-max were tested, since the classical answer to
skewed predictors is a power transform (Box & Cox, 1964; Yeo & Johnson, 2000) or a
rank-based transform (van der Waerden).

| feature transform | library | in-sample (k=14) | LOO-dataset (k=14) |
|---|---|---|---|
| **none (current)** | 172 | 0.5582 | **+0.443** |
| Yeo-Johnson, λ per feature | 392 | 0.5597 | -0.124 |
| Yeo-Johnson, shifted positive | 675 | 0.5673 | +0.203 |
| rank → uniform [1, 2] | 1011 | 0.5500 | -0.001 |
| rank → uniform [1, 100] | 171 | 0.5372 | +0.236 |

Yeo-Johnson does exactly what it promises — it is chosen over Box-Cox here because it
accepts zero and negative values, which six of these features have:

| feature | fitted λ | skewness before | after |
|---|---|---|---|
| `gravity` | -0.05 | 3.97 | **0.01** |
| `nr_inst` | +0.10 | 3.46 | 0.02 |
| `nr_outliers` | -0.05 | 3.87 | 0.10 |
| `ns_ratio` | -0.35 | 3.73 | -0.02 |

**And the λ it chooses is ≈ 0 for every heavy-tailed feature, which is the log transform.**
The principled method converges on what the grammar already offers. In-sample changes
little (0.5597 against 0.5582) and transfer falls sharply, because a λ fitted to the
observed feature distribution extrapolates poorly on a held-out dataset whose features lie
outside that range.

The distinction that matters: applying a power transform as **preprocessing** commits every
term to one compression per feature. Keeping `log`, `sqrt` and `1/f` in the **grammar**
lets the search decide per term whether compression helps and how much. Same functions,
strictly more freedom, and the freedom is worth 0.24 R² of transfer.

### Reading impact off the weights

The second motivation for scaling — being able to compare terms by their weights — is
already met without it. Selection runs on standardised terms and both weight vectors are
kept, so every printed equation carries the standardised weight beside the raw one:

```
+0.00702424 * [log(gravity)] * [log(Model Capability)]   # beta=+0.1381
```

The raw weight is what you evaluate; `beta` is what you compare. `attribution.term_effects`
goes further and reports each term's effect in MCC units, which is comparable across terms
*and* against the target.

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
