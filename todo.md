# Open work: describing the models better

> **Current status (2026-08-18): the feature search is closed and it produced a negative
> result, which is the honest outcome and a usable one.** Nothing derivable from the five
> existing model columns survives the gate below; everything else needs a measurement that
> has been ruled out. The sequence of failures is not a series of near-misses — every one of
> them independently confirms the study's central claim, that this corpus lacks model
> descriptors and no re-encoding of what it has can supply them. See *What the negatives are
> worth* at the end.


The study's own measurements say where the remaining headroom is, and they agree with each
other. Model meta-features recover 63% of what model identity explains, against 99% on the
dataset side (ch. 6). The AMMI model latent is 0.32 predictable from those features against
0.597 for the dataset latent (ch. 9). A ranking objective fails for the same reason (ch. 9).
**Replacing the five model descriptors with model identity itself is worth +0.121
leave-one-dataset-out R²** — that is the budget every idea below competes for.

Nothing here is committed to the study. `python -m metafit` is unchanged.

---

## The gate every proposal below must pass

**Read this before evaluating any feature idea, including your own.** The study exists to
produce a short additive equation whose terms, singly and in groups, can be reasoned about.
Accuracy is what the equation is scored on; *explainability is what it is for*. A proposal
that improves transfer while making a term unreadable has not made a trade — it has failed.

Every session in this file's history has drifted the same way: measure R², find a gain,
then treat the interpretability loss as a price to quote. That inverts the goal. The gate:

1. **Named and physical.** A reader must be able to say what the feature *is* without a
   rotation matrix, a loadings table or a fitted lookup. `Training Operations` passes.
   `Model Structure 4` does not.
2. **Continuous, strictly positive, narrow range.** The grammar assumes it (see
   `terms.py`); a binary or zero-reaching column loses `log`, `sqrt`, `1/f` and division,
   and inside a product it zeroes the term entirely, which is a per-group slope rather
   than a relationship.
3. **A single equation.** No per-model or per-family output, no tables, no branching, no
   sub-models.
4. **Generic.** Definable for any classifier, not just the 25 here.
5. **Computable now.** No re-running models, no corpus rebuild.

**Rejected against this gate, not against their numbers** — do not resurrect them by
citing R²: the complexity ordinal, one-hot family dummies, five binary mechanistic axes
(0.494/0.535), axes graded to {0.30, 0.95}, and **principal-coordinate embeddings of the
axis matrix (0.457/0.547)**. The last was the best-transferring variant measured anywhere
in this study and it is still inadmissible: two of its five columns have no reading, so an
equation carrying them cannot be reasoned about term by term. That is disqualifying on its
own, and no gain makes it otherwise.

---

## Tier 1 — recomputable from what is already on disk

Free, generic, no new training runs. Both items below were **measured on 2026-08-17 and
both hurt leave-one-dataset-out**; they are kept here with their numbers so the next
session does not re-run them, and because the *reason* they fail is the actionable part.

### 1.1 Log-differences among the three cost columns — measured, rejected

`Processing Units Number`, `Training Operations` and `Prediction Operations` are already
log-scaled, so their differences are log-ratios. The grammar cannot reach them: it offers
`[log(a)] / [log(b)]`, the ratio of two logs, never `log(a) − log(b)`. Three mechanically
meaningful quantities are therefore unreachable today.

| column | reading | range |
|---|---|---|
| `Training Operations − Prediction Operations` | how much more a model costs to fit than to use | 0.0 .. 19.7 |
| `Training Operations − Processing Units Number` | how hard each parameter was worked while fitting | −5.8 .. 18.2 |
| `Processing Units Number − Prediction Operations` | how much of the model one prediction touches | −14.7 .. 13.3 |

All three satisfy every upstream design rule (continuous, per-trained-instance, bounded,
`log`-wrapped). Measured at the published configuration:

| | in-sample | LOO-dataset | LOO-model |
|---|---|---|---|
| E3 baseline | 0.5998 | **0.4658** | 0.4887 |
| E3 + 3 log-differences | 0.6222 | 0.3527 | 0.5009 |
| E2 baseline | 0.1772 | **0.0902** | 0.1198 |
| E2 + 3 log-differences | 0.1956 | 0.0812 | 0.1267 |

Fit up, transfer down. Same signature as §2 below.

### 1.2 Correcting `Training Operations` for the 100k cap — blocked

Every model was trained on a stratified sample capped at 100,000 rows, and ten of the
twenty datasets exceed it, but `Training Operations` was computed from the *source*
`nr_inst`. So the column carries a spurious dataset-size signal above the cap.

Recomputing it needs the tuned hyperparameters of each trained instance. **The committed
`results_stage_ml_eval.csv` upstream covers 348 of our 476 rows** — the 128 gaps are
exactly the 8 GPU-trained models × 16 datasets. Until the GPU eval outputs are recovered
this cannot be done exactly, and approximating it would put a hand-made formula in the
middle of the study's only cost column. **Blocked, not rejected.**

### 1.3 Not worth doing

- **Hyperparameter-grid size** from `model_utils.param_grids`. Generic and on disk, but it
  does not separate `BernoulliNB` from `LinearSVC`, which is the collision that matters.
- **`MCC_Fold_Std`.** Tempting and unusable: it is an outcome of the run being predicted,
  not a descriptor of the model.

---

## Tier 3 — behavioural probing (landmarking, transposed)

**This is the one with room in it, and it does not require re-running the meta-dataset.**

Landmarking (Pfahringer, Bensusan & Giraud-Carrier, 2000) characterises a *dataset* by
running fast learners on it. Invert it: characterise a *model* by running it on a fixed
battery of tiny synthetic tasks with known structure, and use its scores as descriptors.
Generic by construction — anything with `fit`/`predict` qualifies — measured rather than
asserted, knowable before touching the target dataset, and it satisfies the upstream rule
against nominal-as-ordinal labels because every column is a measured score.

Probes chosen to break the collisions actually observed in the corpus:

| probe | separates |
|---|---|
| correlated / redundant features | `BernoulliNB` (independence assumption fails) from `LinearSVC` |
| XOR / parity | linear from non-linear hypothesis classes |
| label noise at 10–20% | `PassiveAggressive` from `SGD` and `Perceptron` |
| rotated vs axis-aligned boundaries | trees from linear models |
| extreme class imbalance | generative from discriminative losses |
| irrelevant-feature padding | `DT` from `ExtraTree` (correctly finding little) |

**Cost:** ~25 models × ~10 probes of a few hundred rows. Seconds per model, minutes in
total, once, completely independent of the 20 real datasets and the 100k-row training
runs. Nothing about the corpus is regenerated.

**Where it would live:** the meta2perf-symbolic-regression repo, as a new stage script
writing one row per model. `metafit` would consume it as extra `MODEL_FEATURES`.

**The caveat §2 raises applies here too.** Probe scores are continuous and per-model, so
they enter the library the same way the ordinal did, and the 24-term budget is already
saturated. Any expansion of the model side has to be evaluated on transfer at *every*
equation length, not at the tuned one, and probably alongside a re-tuned `headline_terms`.

---

## 2. Family taxonomy — the ordinal fails, the discretization works

Two forms of the same idea, and they behave oppositely. §2.5 has the numbers, §2.6 the
grammar defect they exposed, and **§2.7 the interpretability objection that has to be
settled before the accuracy question means anything**. §3 is the plan.

Proposal as first stated: one ordinal column scoring the complexity of the underlying
learner (linear → probabilistic → tree → latent-static → latent-dynamic → ensemble),
spacing not necessarily unit.

### What the corpus does to that ladder

- **Rung "latent static (SVM, hyperspace provided)" is empty.** The only SVM present is
  `LinearSVC` — a linear SVM, no kernel, no latent space. It belongs on rung 1.
- **`LDA` is a linear classifier**, so it sits on rung 1 too, not with the probabilistic
  models, even though it is fitted generatively.
- **`KNN`, `TabPFN` and `TabICL` have no rung** in the six-level version. The ladder used
  for the measurement below adds `KNN` at 4 and the pretrained in-context models at 7.

### It is capped before it is tried

Free levels per group, fitted out of fold on E3's residual — the ceiling for **any**
encoding of that group, at any spacing:

| grouping | LOO-dataset | LOO-model |
|---|---|---|
| E3 alone | 0.4658 | 0.4887 |
| + per-model levels + slope (25 groups) | 0.5874 (**+0.122**) | 0.4887 (**+0.000**) |
| + per-family levels + slope (10 groups) | 0.5446 (**+0.079**) | 0.5641 (**+0.075**) |
| + per-rung levels + slope (7 rungs) | 0.4977 (**+0.032**) | 0.5152 (**+0.027**) |

So a rung-based column can be worth at most ~+0.03. Measured as an actual library feature
it delivers **−0.169**:

| E3 | in-sample | LOO-dataset | LOO-model |
|---|---|---|---|
| baseline | 0.5998 | **0.4658** | 0.4887 |
| + ordinal, unconfined | 0.6493 | 0.2974 | 0.4825 |
| + ordinal, confined to unary terms | 0.6176 | 0.3391 | 0.4553 |
| + ordinal, unconfined, penalty 20 | 0.6200 | 0.3642 | 0.4680 |

Checked at every equation length from 2 to 32, not just the tuned one: **the ordinal is
below baseline on leave-one-dataset-out at every length ≥ 4** (its best is 0.322 at k=32
against the baseline's 0.466 at k=24). On leave-one-model-out it is ahead at short lengths
(+0.075 at k=6, +0.033 at k=8) and roughly level at the published length.

### Why a monotone axis cannot work here

The per-family residual levels are not monotone in complexity — and they run *backwards*
relative to it:

| family | level | slope on `log(gravity)` | rows |
|---|---|---|---|
| discriminant | +0.140 | −0.014 | 40 |
| linear | +0.103 | −0.008 | 120 |
| tabular NN | +0.000 | +0.008 | 51 |
| generic NN | −0.005 | −0.007 | 34 |
| naive bayes | −0.035 | −0.001 | 40 |
| instance | −0.041 | +0.004 | 20 |
| single tree | −0.051 | +0.006 | 40 |
| tabular foundation | −0.074 | +0.008 | 34 |
| boosted trees | −0.085 | +0.008 | 77 |
| bagged trees | −0.104 | +0.004 | 20 |

These are corrections to E3, not raw performance: the equation reads capacity as good
(`Processing Units Number` and the operation counts are capacity proxies), so it
over-credits the complex families and under-credits the simple ones. The order is
linear(1) > NN(5) > naive bayes(2) > KNN(4) > tree(3) > foundation(7) > ensemble(6) —
no single monotone axis reproduces that, at any spacing.

### 2.5 Discretization — measured, and it works

Separation was never the problem. Counting rows whose full descriptor vector is shared by
another model on the same dataset:

| encoding | tied rows | tied groups | worst MCC gap inside a group |
|---|---|---|---|
| the five descriptors (today) | 151 | 65 | 0.894 |
| + complexity ordinal | 121 | 55 | 0.683 |
| + family label | 121 | 55 | 0.683 |
| + five mechanistic axes | 121 | 55 | 0.683 |

**All three break exactly the same 30 rows.** Any family-level encoding does; the
remaining 121 are collisions *within* a family (`DT`/`ExtraTree`,
`LightGBM_RF`/`LightGBM_ExtraTrees`, `SGD`/`Perceptron`/`PassiveAggressive`) and only
tier 3 can reach those. So the ordinal did not fail for want of distinction — it failed
because it forces one global ordering, and §2 above shows the residuals are not monotone.
Discretization keeps the distinction and drops the ordering.

**One-hot family dummies are the obvious form and the wrong one.** Seven of the ten
families fail the library's own admissibility filter — a binary column needs ≥ 10% of the
rows to keep its worst z-score under the study's cap of 3.0, and only `linear` (120),
`boosted trees` (77) and `tabular NN` (51) clear it. Forcing them in by loosening the cap
to 6 degrades everything, baseline included.

**Five binary mechanistic axes are the form that fits.** Each is a genuine property of the
hypothesis class rather than a category name, a model can carry several at once (boosted
trees are both), so five columns separate ten families and each one reads as a term group:

| axis | rows | members |
|---|---|---|
| `Axis Aligned Splits` | 137 | trees, bagged and boosted |
| `Combines Base Learners` | 97 | `Bagging`, `AdaBoost`, `XGBoost`, `LightGBM_*` |
| `Learns Representation` | 119 | MLP/DNN, tabular NN, `TabPFN`, `TabICL` |
| `Models Class Densities` | 80 | `GaussianNB`, `BernoulliNB`, `QDA`, `LDA` |
| `Reuses Training Rows` | 54 | `KNN`, `TabPFN`, `TabICL` |

All five are admissible at the study's own cap. `BernoulliNB` and `LinearSVC` differ on
`Models Class Densities`, which is the collision that prompted this.

**They need the ridge retuned.** At the published penalty 5 they take 11 of 24 terms and
LOO-dataset falls to 0.322. At penalty 20 they do not:

| k | baseline λ=5 (published) | baseline λ=20 | **axes λ=20** |
|---|---|---|---|
| | lodo / lomo | lodo / lomo | lodo / lomo |
| 20 | 0.4128 / 0.4816 | 0.3487 / 0.4576 | **0.4297 / 0.5182** |
| 24 | **0.4658** / 0.4887 | 0.3664 / 0.4680 | **0.4659 / 0.5250** |
| 28 | 0.3726 / 0.4912 | 0.3679 / 0.4694 | **0.4941 / 0.5347** |
| 32 | 0.4217 / 0.4924 | 0.3748 / 0.4708 | **0.4302 / 0.5320** |

The middle column is the control that matters: **heavier shrinkage on its own makes the
baseline worse** (0.4658 → 0.3664 at k=24). The gain is the axes, not the penalty. At
matched penalty the axes lead on leave-one-model-out at *every* length and on
leave-one-dataset-out at every length ≥ 20.

Against the published equation (λ=5, k=24, 0.4658 / 0.4887), the axes at λ=20, k=28 give
**0.4941 / 0.5347 — ahead on both protocols**, +0.028 and +0.046. It is the first feature
addition in this study that improves both.

### 2.6 Do binary columns survive the grammar? Mostly, with one defect

**Reachability.** A 0/1 column admits only two of the five transforms: `log(f)` and `1/f`
are undefined at 0 and rejected. What remains is `f` and `f^2` — and for a binary column
**those are numerically identical**, confirmed on the built library. So each axis
contributes one useful column and one exact duplicate of it, distinct only by name.
`Library` de-duplicates on `Term.name`, not on values, so the duplicates survive. Five
axes ship five perfectly collinear pairs.

The consequence is waste rather than danger, and the correction matters: the beam search
already refuses a candidate correlating above `fit.COLLINEARITY_LIMIT` (0.95) with a selected
term, so a duplicate pair can never both enter one equation and no singular system arises on
that path. What the duplicates cost is candidate-pool slots, search time, and a place in the
reported term rankings, where the two forms read as two independent findings.

**Fix before anything else: de-duplicate `Library` on column values, not names.** It is a
few lines, it is correct independent of this proposal, and it should land on its own with
its own test.

**Selection.** They are not crowded out — the opposite. **11 of 28 terms use an axis**, and
they enter as products and sum-ratios rather than bare indicators, which is where the
grammar earns its keep:

    [Robust to Outliers] * [Learns Representation]          rank 3 overall, beta 0.0912
    ([Axis Aligned Splits] + [Learns Representation]) / [log(nr_class)]        beta 0.0821
    [log(Training Operations)] * [Models Class Densities]                      beta -0.0668
    [log(Processing Units Number)] * [Axis Aligned Splits]                     beta 0.0764

The strongest purely model-side term in the equation is an axis term. The library grows
281 → 438 terms (+157), so the axes are competing against 280 incumbents and winning
40% of the slots.

**Interpretability improves rather than degrades**, which was the risk worth checking
given the whole point is term-group analysis.

*These are not R² values and have nothing to do with E1/E2/E3.* Both rows below are E3 —
the mixed equation — and both describe **how E3's own output variance splits across its
own terms**, classified by which features each term touches (`attribution.group_shares`,
`results/shares.csv`). Each row sums to 1. A term over `nr_attr` alone is *dataset*, a term
over `Training Operations` alone is *model*, a term multiplying one by the other is
*mixed*. E3's in-sample R² is 0.600 and is untouched by this table.

| E3's terms, by what they touch | dataset-only | model-only | mixed | in-sample R² |
|---|---|---|---|---|
| published E3 (24 terms) | 0.102 | 0.140 | **0.758** | 0.600 |
| E3 + axes (28 terms) | 0.217 | 0.185 | **0.598** | — |

Less of the equation hides in mixed terms, not more. And the axes are themselves a second,
orthogonal grouping — "what combining base learners is worth" is a term-group question the
current equation cannot even pose.

### 2.7 The objection that matters: a zero cuts the whole term

An indicator inside a **product** does not describe a relationship — it switches one off.
`[log(Processing Units Number)] * [Axis Aligned Splits]` is identically zero on the 339
non-tree rows, so it is not "capacity helps"; it is "capacity helps, **for trees**". That
is a per-family slope written in product notation, which is the per-family model the study
has already refused twice.

An indicator inside a **sum** behaves differently and acceptably.
`([nr_cor_attr] + [Learns Representation]) / [log(nr_class)]` is
`nr_cor_attr/log(nr_class) + [indicator]/log(nr_class)` — every row gets a contribution,
and the indicator adds a dataset-scaled *level*, not a slope. That is an additive offset,
which is what a family effect should be.

Read off the 28-term equation of §2.6, the eleven axis terms split roughly **4 switching**
(products, and one ratio with the indicator as numerator) against **7 shifting** (indicator
inside a sum). So the objection lands on part of the equation, not all of it — but the part
it lands on includes the single strongest model-side term,
`[Robust to Outliers] * [Learns Representation]`.

**This has to be decided before the accuracy question, not after.** If switching terms are
inadmissible on interpretability grounds then the honest variant is axes restricted to
additive use, and its accuracy has not been measured. If they are admissible, that needs
saying explicitly in chapter 2 alongside a rule for reading them, because a reader will
otherwise read a switched term as a general statement.

More generally: **the grammar assumes strictly positive continuous features.** Binary
columns are second-class in it — three of five transforms undefined, the surviving two
identical, division impossible, multiplication degenerating into a switch. That is an
argument that the right encoding of these distinctions is *continuous*, which is what
tier 3's probe scores are, and it promotes tier 3 from "more information" to "the
representation this grammar actually wants".

---

## 3. Plan

Staged so that each stage is worth doing on its own and no stage depends on a decision
that has not been taken yet. **Nothing below is started.**

### Stage 0 — correctness, independent of every proposal here — **done**

0.1 **De-duplicate `Library` on column values, not on `Term.name`.** Done. A term is now
dropped when its centred column correlates with an already-kept column above
`COLLINEARITY_TOLERANCE` (`1 - 1e-9`). Centring is what makes the test match the fit, which
runs on standardised terms: two columns differing by an additive or multiplicative constant
are one column to the solver. First offered wins, and generation runs simple to complex, so
the survivor is the shorter form. Six tests in `TestCollinearTermsAreDropped`, including one
that asserts the published library contains no collinear pair at all.

0.2 **Document the grammar's domain assumption.** Done, in the `terms.py` module docstring,
in `Library`'s docstring, and in chapter 2 under *Two terms, one column* and *What the
vocabulary assumes about a feature* — including the sum-versus-product distinction of §2.7.

0.3 **What it moved: nothing that is published.** Measured by running the full study either
side of the change:

- **E1, E2 and E3 are byte-identical.** `e1.json`, `e2.json`, `e3.json` all `cmp`-clean.
- The library goes **281 → 278 terms** at `max_arity=3`. The three dropped terms are the
  algebraic duplicates found below, not a threshold effect.
- `correlations.csv` changes as it should: the redundant
  `([log(inst_to_attr)] + [log(nr_attr)]) / [log(Processing Units Number)]` leaves the
  top-15 candidate list and a genuine term takes the slot.
- `curve_e3.csv`, `pareto.csv` and `term_choice.csv` differ **in the 16th decimal only**
  (0.26876735731338475 → 0.2687673573133842) — floating-point reassociation from a
  differently shaped batched solve, not a change in result.
- `assets/docs/10-report.md` is unchanged. `ci.sh` green: 417 tests, ruff, basedpyright,
  vulture.

**The finding 0.3 anticipated is real, and it predates the axes entirely.** The published
281-term library already contained **three perfectly collinear pairs**, because
`inst_to_attr` is `nr_inst / nr_attr` in this meta-dataset and therefore

    log(inst_to_attr) + log(nr_attr)  ==  log(nr_inst)

exactly, while the grammar generates both sides:

| kept | dropped |
|---|---|
| `[log(nr_inst)] / [log(Processing Units Number)]` | `([log(inst_to_attr)] + [log(nr_attr)]) / [log(Processing Units Number)]` |
| `[log(nr_inst)] / [log(Training Operations)]` | `([log(inst_to_attr)] + [log(nr_attr)]) / [log(Training Operations)]` |
| `([log(inst_to_attr)] + [nr_norm]) / [log(nr_attr)]` | `([log(nr_inst)] + [nr_norm]) / [log(nr_attr)]` |

None was selected into a published equation, so nothing that has been reported is wrong.
Had one been, the equation would have carried two terms that read as different statements
and are the same column. **Any feature defined as a ratio of two others will do this
again** — worth remembering before the next feature is added.

### Stage 1 — **closed by decision, 2026-08-17: no binary indicators**

The author ruled binary indicators out on the grounds that they do not suit an additive
model, and asked for continuous descriptors instead. Options A (unary only), B (sums only)
and C (unrestricted) are all withdrawn; the 0.494/0.535 result of §2.5 stands as a
measurement and **is not a candidate for merging**. Do not reopen it without new grounds —
the reasoning is §2.7 and it does not depend on any number.

**What that decision costs, stated plainly:** it gives up the only thing measured so far
that improves leave-one-model-out (+0.046), and it is not yet known whether a continuous
descriptor recovers that. Stage 2 is the attempt.

**What it rules out beyond the axes:** the complexity ordinal (§2), one-hot family dummies
(§2.5), and any future encoding of a categorical model property. It does *not* rule out a
continuous quantity that happens to correlate with family.

### Stage 2 — continuous coordinates derived from the axes — **measured, then rejected**

**Rejected on 2026-08-18 against gate rule 1, not on its numbers.** A principal-coordinate
rotation is an embedding: s4 and s5 have no plain-language reading, and an equation whose
terms cannot be reasoned about individually is not the deliverable this study exists to
produce. The measurements below stand as measurements. They are **not** a merge candidate
and must not be re-proposed on the strength of the +0.058.

The record is kept because it answers a question that was asked directly — *can one or two
continuous features be derived from the five binary ones?* — and the answer, **no, it takes
five and they are unreadable**, is itself the useful finding.


Two further decisions closed the field before this was run: **no re-running any model or
experiment from the meta-dataset** (which closes tier 3 probing, not just the corpus
rebuild), and **no binary features**. What is left is a continuous re-encoding of the
taxonomy, computed from the axis definitions alone.

**Derivation, and why it does not leak.** The five binary axes give a 25×5 matrix. Centre
it, take its principal axes by SVD, and rescale each coordinate into `[0.30, 0.95]`. MCC
never enters, so a coordinate is a structural embedding of the taxonomy and not a fitted
per-family level in disguise. The scaling is chosen against the grammar's requirements:
strictly positive (`log`, `sqrt`, `1/f` all defined), dynamic range 3.2 (≤ 20, so the column
may divide), clear of 1.0, no single-model outlier — and **no term is ever zeroed**, which
was the §2.7 objection.

**The axes distinguish only 7 groups, not 10.** `bagged trees` and `boosted trees` share a
profile, as do `naive bayes` and `discriminant`. So does every coordinate set derived from
them, at any dimension.

| encoding (λ=20 unless noted) | best lodo | lomo there | best lomo | lodo there |
|---|---|---|---|---|
| published E3, λ=5 | **0.4658** (k24) | 0.4887 | 0.4924 (k32) | 0.4217 |
| baseline, matched λ=20 | 0.3864 (k16) | 0.4565 | 0.4708 (k32) | 0.3748 |
| 1 coordinate | 0.3910 (k20) | 0.4706 | 0.4709 (k12) | 0.3778 |
| 2 coordinates | 0.4594 (k16) | 0.4828 | 0.5042 (k32) | 0.3844 |
| 3 coordinates | 0.4580 (k20) | 0.4883 | 0.5249 (k32) | 0.4511 |
| **5 coordinates** | 0.4569 (k28) | **0.5468** | **0.5468** (k28) | 0.4569 |
| 5 axes graded to {0.30, 0.95} | 0.3489 (k28) | 0.5284 | 0.5360 (k32) | 0.3258 |
| *5 binary axes — rejected by decision* | *0.4941 (k28)* | *0.5347* | *0.5347* | *0.4941* |

**One or two coordinates is not enough.** One is indistinguishable from the baseline —
the same result the complexity ordinal gave, for the same reason: a single monotone axis
cannot reproduce a non-monotone set of family effects. Two recovers about half. The
information is worth roughly five dimensions, and the first two principal axes hold only
76% of the taxonomy's variance.

**Five coordinates is the best-transferring variant measured anywhere in this study**, and
notably it beats the binary axes on leave-one-model-out (0.547 vs 0.535) while giving up
leave-one-dataset-out (0.457 vs 0.494). Against the *published* equation it is −0.009 lodo
— inside the ±0.05 band that 20 datasets make meaningless — and **+0.058 lomo**. Both
optima fall at the same length, k=28, which is a good sign rather than a coincidence to
explain away.

**Grading the axes onto {0.30, 0.95} instead of rotating them** keeps the readable names and
fixes every grammar objection, but transfers badly across datasets (0.349). With no zero to
confine it, each axis term now acts on all 476 rows, 19 of 28 terms take one, and they
displace the dataset terms that carried leave-one-dataset-out. Good lomo, poor lodo.

**The cost of the rotation is the reading.** Loadings:

| axis | s1 | s2 | s3 |
|---|---|---|---|
| `Axis Aligned Splits` | −0.66 | +0.24 | −0.02 |
| `Combines Base Learners` | −0.55 | +0.23 | −0.04 |
| `Learns Representation` | +0.46 | +0.64 | −0.44 |
| `Models Class Densities` | +0.09 | −0.64 | −0.10 |
| `Reuses Training Rows` | +0.19 | +0.26 | **+0.89** |
| variance | 0.488 | 0.268 | 0.108 |

s1 reads as *partition-based versus representation-based*, s2 as *discriminative-deep versus
generative*, s3 as *reuses the training set at inference*. **s4 and s5 have no reading** —
13% of the variance between them, and they are mixtures. `FEATURE_GLOSSARY`, `practices` and
`guidance` all build sentences out of feature names, so shipping five coordinates means
shipping two features nobody can explain. That is the trade to decide: +0.058 on
leave-one-model-out against two uninterpretable columns in an equation whose whole claim is
interpretability.

#### Correction to an earlier claim in this file

§2 said a rung-based column was "capped at +0.032 before it is tried", from the free-levels
residual ceiling. **That reasoning is wrong for a feature inside the search.** The ceiling
bounds a post-hoc *level plus one slope* correction; a feature in the library can be
combined independently with many dataset features, which is strictly more expressive. The
measured proof is in this file: the ceiling for the 7 axis profiles is +0.030 lodo, and the
binary axes delivered +0.126 over the matched baseline — four times it. The ceilings remain
valid for what they measure (a residual correction, as chapter 9 uses them) and must not be
quoted as bounds on feature sets. The conclusions about the ordinal and the dummies stand on
their measured results, not on the ceiling argument.

### Stage 2b — **closed by decision: no re-running models**

Behavioural probing would give continuous descriptors that are *measured* rather than
asserted, and it reaches the within-family collisions (`DT`/`ExtraTree`, `SGD`/`Perceptron`)
that no family-level encoding can touch — 121 of the 151 tied rows. **It requires running
the 25 models, which the author has ruled out**, so it is closed alongside the corpus
rebuild. Recorded here as the direction that remains open on the merits, not as work to do.

Everything derived in Stage 2 is therefore **asserted** taxonomy, not measurement. That is
the standing limitation on any descriptor this study can now add, and it belongs in
chapter 8 if a coordinate set is ever merged.

**First, the honest position on "computable right now".** From the CSV alone the continuous
well is dry. The five model descriptors have been recombined every way the grammar cannot
reach: log-differences were the last untested continuous idea and they cost 0.11 of
leave-one-dataset-out (§1.1). Anything further from those five columns is a
re-parameterisation of information already in the equation. **A new continuous descriptor
has to measure something new**, and the only measurement available without re-running the
corpus is behavioural probing.

2.1 **Environment.** Nothing is installed — no sklearn, torch, xgboost, lightgbm, tabpfn,
tabicl. The battery needs its own venv **in the meta2perf repo**, never in `metafit`;
`metafit` consumes a 25-row CSV and its dependency list stays polars/numpy/matplotlib/
kneeliverse.

  *Partial coverage is worthless*, which sizes the job. A descriptor column missing ten
  models cannot be fitted at all — `data.load` rejects nulls, and rightly. So it is
  all-or-nothing across the 25 models: ~17 are cheap (sklearn, xgboost, lightgbm) but the
  remaining 8 need torch, tabpfn, tabicl, pytorch-tabnet, tab-transformer-pytorch and rtdl.
  Call it 3–5 GB and a long install. **Needs authorisation before anyone starts.**

2.2 **Design the scores to survive the grammar.** This is the trap Stage 0 documented, and
it has to be designed in rather than patched later. Probe MCC is in $[-1, 1]$ and therefore
**unusable raw**: at zero or below, `log`, `sqrt` and `1/f` are all undefined and the column
can never be a denominator, which strands it in the same corner the binary axes were in.
Requirements for every probe column:

  - **strictly positive** — use balanced accuracy, or map MCC to $(0, 1]$ with a floor;
  - **narrow dynamic range** — `denominator_atom` needs $\max/\min \le 20$ to allow
    division, and a wide-range column loses three quarters of the grammar;
  - **clear of 1.0** where possible, since $\log 1 = 0$ blocks the preferred denominator
    form;
  - **no single-model outlier** beyond `max_abs_zscore`, or `is_admissible` drops the term.

  A score in roughly $[0.3, 0.95]$ satisfies all four. Fix this before generating anything.

2.3 **Probes**, per the tier 3 table: correlated/redundant features, XOR/parity, label noise
at 10–20%, rotated vs axis-aligned boundaries, extreme class imbalance, irrelevant-feature
padding. Each is a few hundred rows; the whole battery is minutes, once, and independent of
the 20 real datasets.

2.4 **Guard against the ratio trap.** Stage 0 found three collinear pairs because
`inst_to_attr` is `nr_inst / nr_attr`. If probe columns are defined as ratios of each other
the same thing happens. `Library` now catches it, but the right move is not to define them
that way.

2.5 Then measure as in §2.5's protocol — **sweep λ and length, with a matched-penalty
baseline as the control** — and judge on the whole transfer curve.

### Stage 3 — the merge decision

Only after a continuous descriptor has survived Stage 2 with a swept configuration. Merging changes the
published equation, the report, every figure, and chapters 2, 6 and 7. It is one commit
behind `ci.sh`, and it is the author's call, not a consequence of the numbers.

### Not on the path

- Publishing the per-family effect table (§"What the ceiling says"). Still a table, not an
  equation. Recorded as a ceiling, undecided, and deliberately not in this plan.
- Tier 1 log-differences and the complexity ordinal. Measured, rejected, §1.1 and §2.

### What the ceiling says survives beyond that

**The one genuinely new finding: a per-family effect table is not empty under
leave-one-model-out.** Per-*model* identity buys exactly 0.000 there, by construction — a
held-out model has no training row. A *family* effect buys **+0.075**, because the other
members of its family stay in the training fold. It is the only model-side description in
this corpus that transfers to a model nobody has run.

That makes it a stronger candidate than E4 ever was, on the specific ground E4 was
withdrawn for. It is still a **table, not an equation**, so publishing it re-opens three
of E4's four objections. The difference is that the fourth objection is gone, and that the
tabular-ML literature already states its guidance per family ("prefer tree ensembles"), so
the rows read as practices rather than as per-model advice.

**Open question for the author, not for the code:** publish the family table as a second
component alongside E3, or keep it as a ceiling measurement like ch. 9 does?

---

## The pattern behind the negatives, and the way past it

Log-differences and the complexity ordinal both showed the same signature: **in-sample up,
leave-one-dataset-out down.** New model-side columns win terms away from the dataset-side
terms that were carrying the transfer, inside a fixed budget at a penalty tuned for a
smaller library.

**§2.5 is the counter-example, and it says what to do about it.** The mechanistic axes
show that signature too at λ=5 and stop showing it at λ=20. So the rule for the next
model-side feature — probe scores included — is: **never evaluate a library expansion at
the incumbent penalty and length.** Sweep both, and include a matched-penalty baseline as
the control, or the ridge gets credited with the feature's effect or blamed for its cost.
That control is what turned this idea from a failure into a result.

The leave-one-dataset-out curve is spiky at 20 datasets (baseline: 0.41 at k=16, 0.41 at
k=20, 0.47 at k=24, 0.37 at k=28), so differences under about ±0.05 at a single length
should not be read as real. Judge a change on the whole curve.

## A capability-ordered rung — measured 2026-08-18, and it is *not* the complexity ladder

Sorting the rungs by **capability** rather than by complexity is a different proposal and it
measures differently, so the earlier result does not settle it. Order taken from the
tabular-ML literature (Grinsztajn et al. 2022; Shwartz-Ziv & Armon 2022; McElfresh et al.
2023; Hollmann et al. 2023), never from this corpus — ordering families by their observed
MCC here would be fitting on the target.

| family | rung | raw mean MCC | E3 residual |
|---|---|---|---|
| naive bayes | 1 | 0.537 | −0.035 |
| discriminant | 2 | 0.636 | +0.140 |
| linear | 3 | 0.584 | +0.103 |
| instance | 4 | 0.827 | −0.041 |
| single tree | 5 | **0.930** | −0.051 |
| generic NN | 6 | **0.454** | −0.005 |
| tabular NN | 7 | 0.797 | +0.000 |
| bagged trees | 8 | 0.942 | −0.104 |
| boosted trees | 9 | 0.905 | −0.085 |
| tabular foundation | 10 | 0.953 | −0.074 |

**It performs far better than the complexity ladder** — at λ=20 it reaches 0.4779
leave-one-dataset-out at k=20, against 0.4658 published and 0.3864 for the matched-penalty
baseline. The complexity version reached 0.297. **Ordering matters, and this ordering is
much closer to right.** But leave-one-model-out falls to 0.4276 against 0.4887 published, so
it buys a within-noise gain on one protocol by giving up 0.06 on the other.

**The reason to reject it is not the numbers — it is that the label would be false.**
Against this corpus the asserted order is barely better than chance: raw per-family mean MCC
rises at **6 of 9 steps** (9 = monotone, ~4.5 = unrelated), and the E3 residual at **5 of 9**.
`generic NN` has the *lowest* mean MCC of any family here, 0.454, while sitting at rung 6 of
10; `single tree` is near the top at 0.930 from rung 5. Whatever the equation extracts from
that column, it is not capability.

That is a gate rule 1 failure of the most damaging kind. An opaque feature is unreadable and
a reader knows it. **A transparently *mis*named feature reads fine and is wrong** — a term
carrying `Model Capability` would be quoted as a statement about capability, and chapter 7
would turn it into advice. For an explainability-first study a confidently wrong reading is
worse than no feature.

If a capability descriptor is wanted, the admissible form is **measured elsewhere, not
asserted here**: published mean normalised rank of each algorithm across external tabular
benchmarks. That is continuous, citable, non-circular (other datasets), and it is a real
quantity with a real name — the "average rank" prior of algorithm selection (Brazdil &
Soares, 2000). It is not currently in the corpus and would have to be assembled by hand,
with coverage gaps for `TabICL` and `TabPFN`. **Recorded as the one surviving version of
this idea, not started.**

## What λ actually is, and why 20 is not a large number

Worth stating because every table in this file quotes a λ and the number looks alarming.

**It is the ridge penalty in the least-squares solve, not a beam-search parameter.**
`fit.ridge_solve` solves $(G + \lambda I)w = X^\top y$ on a **centred and standardised**
design, so every Gram diagonal entry is exactly $n = 476$. The beam search has its own,
separate knobs: `beam_width` (6), `pool_size`, `candidates` (24), `refine_rounds` (2), and
`COLLINEARITY_LIMIT` (0.95).

| λ | λ/n | weight shrinkage on an orthogonal term |
|---|---|---|
| 5 (published) | 0.011 | 0.990 |
| 20 | 0.042 | 0.960 |
| 60 | 0.126 | 0.888 |

So λ=20 shrinks an isolated weight by **4%**. As shrinkage it is negligible, and that is
*not* how it was doing its work.

**The penalty is also in the selection score.** `Selector` scores a candidate subset as
$\text{RSS} = \|y\|^2 - w^\top X^\top y - \lambda\, w^\top w$, so it penalises subsets whose
weights are large — which changes *which* terms the beam picks, not merely how hard they are
pulled toward zero. Measured on the published library: **12 of the 24 terms differ between
λ=5 and λ=20.** Half the equation.

**And it exposes something about the published equation.** The largest standardised weight
falls from **3.34 at λ=5 to 1.05 at λ=20**. A standardised weight of 3.34 means that term
alone moves the prediction by ±3.34 × (target scale) across ±1 sd of itself, which MCC's
range cannot absorb — so it is being cancelled by other terms. The published E3 is
partly built on large opposing weights between near-collinear terms. λ=20 mostly refuses
those combinations, and that is what it was buying, not shrinkage.

That is worth a look on its own, independent of any feature question: an equation whose
terms cancel at that magnitude is harder to reason about term by term, which is the one
property this study cannot compromise. `is_admissible` already guards the extreme case (a
near-constant term once produced a weight of −1.5e9 against an intercept of +1.5e9); nothing
currently guards the moderate case. **Not investigated. Open.**

## What the negatives are worth

Read as a list of failures this looks like a wasted search. Read as evidence it is the
strongest support the study's central claim has, because every attempt failed *for the same
reason* and each closes a different escape route a reviewer would otherwise ask about:

| attempt | what its failure rules out |
|---|---|
| log-differences of the cost columns | the five columns do not contain unexploited structure; the grammar was not the limitation |
| complexity ordinal | model capability is not monotone in any complexity ladder — the per-family residuals run *backwards* against it |
| one-hot family dummies | family is too coarse to be encoded per-family at this sample size (7 of 10 fail the z-score floor) |
| binary mechanistic axes | family information *does* help (+0.126 over matched baseline) but only as per-group slopes, which is a piecewise model, not one equation |
| principal coordinates | the information is worth ~5 dimensions, and 5 dimensions of an asserted taxonomy cannot be made readable |
| per-model identity ceiling | +0.121 is available and **none of it is reachable** from anything the corpus records |

Together they say something sharper than "we tried and could not improve it": **the missing
signal is model capability, it is worth +0.121, and it is not recoverable by re-encoding
what the corpus has.** That is a stronger and more citable statement than the study
currently makes, and it is earned rather than asserted.

Two consequences worth writing into the paper rather than leaving in this file:

- **Chapter 8** should carry the limitation explicitly: every model descriptor available
  here is a *cost* proxy (`Processing Units Number`, the operation counts) or an asserted
  property (`Robust to Outliers`), and none of them describes what a model is *good at*.
  The five negatives above are the evidence that this is a property of the corpus, not of
  the search.
- **The per-family effect table** remains the one measured thing that transfers to an unseen
  model (+0.075 LOO-model where per-model identity gives exactly 0.000). It is still a table
  and still fails gate rule 3, so it is not a candidate either — but as a *measurement* it
  quantifies the limitation above and belongs beside the +0.121 ceiling in chapter 9.

The one direction that survives on the merits is behavioural probing, and it is closed only
by the decision not to re-run models. Worth saying so in the future-work section: it is a
known, costed, specific next step, not a vague gesture.

## When the meta-dataset can be recomputed: what to record

The diagnosis is settled and it dictates the specification. Every model descriptor the
corpus currently has says what a model **costs** (`Processing Units Number`, the two
operation counts) or asserts a static property (`Robust to Outliers`,
`Active Regularization Mechanisms`). **None of them says what a model is good at**, which is
the +0.121 that model identity holds and no re-encoding recovers. So the recompute should
add *behavioural* descriptors — things the model does, measured — and nothing else.

### Group A — dataset-independent, measured once per model

Run each model on a battery of small synthetic tasks with known structure and record its
score. Never touches a real dataset, so there is no leakage question at all, and the whole
battery is minutes. This is the group most likely to reach the +0.121, because it measures
capability directly rather than proxying it.

| descriptor | probe | separates |
|---|---|---|
| `Rotation Tolerance` | axis-aligned vs rotated boundary, same task | trees from linear models |
| `Interaction Capture` | XOR / parity | linear from non-linear classes |
| `Label Noise Tolerance` | MCC drop at 10% and 20% flipped labels | `PassiveAggressive` from `SGD`, `Perceptron` |
| `Redundancy Tolerance` | correlated / duplicated features | `BernoulliNB` from `LinearSVC` |
| `Irrelevance Tolerance` | padding with pure-noise features | `DT` from `ExtraTree` |
| `Imbalance Tolerance` | 95:5 class split | generative from discriminative losses |
| `Sample Efficiency` | MCC at 100 rows / MCC at 1000 rows, same generator | in-context and instance models from the rest |

Every one is a ratio or a difference of two balanced accuracies, so it is continuous,
bounded, and namable in one clause. **Scale them into roughly [0.3, 0.95] at generation
time** — MCC in [−1, 1] is unusable in this grammar, and a column reaching zero loses `log`,
`sqrt`, `1/f` and division (see `terms.py`).

### Group B — per (model, dataset), computed inside the training fold only

| descriptor | how | why it is worth the cost |
|---|---|---|
| `Learning Curve Slope` | MCC at 25 / 50 / 100% of the training sample, slope against `log(n)` | answers the question the 100k cap made untestable, and it is the one the withdrawn "neural architectures catch up on larger datasets" practice needed |
| `Tuning Sensitivity` | IQR of cross-validated MCC across the tuning trials | already latent in the optuna studies — just record it. Measures how much a model depends on being tuned, which is real, generic capability information |
| `Calibration Error` | Brier or ECE on the training folds | describes something MCC cannot see |
| `Train Validation Gap` | train MCC − validation MCC | overfitting propensity, per instance |

These are **landmarks** and legitimate as such (Pfahringer, Bensusan & Giraud-Carrier,
2000), but they are computed on the target dataset, so the discipline is absolute: training
folds only, never the fold being predicted. Group A has no such hazard, which is why it
should come first.

### Group C — fixes to what is already recorded

- **Record the actual sampled training size** as a column, and compute `Training Operations`
  from it. Today it uses source `nr_inst`, so above the 100k cap the column carries a
  dataset-size signal that does not correspond to any training run.
- **Record per-instance hyperparameters for every row**, GPU models included. The committed
  `results_stage_ml_eval.csv` covers 348 of 476; the 128 gaps are exactly 8 GPU-trained
  models × 16 datasets, and they block any recomputation from tuned values.
- **Keep `MCC_Fold_Std` out.** It is an outcome of the run being predicted, not a descriptor.

### Rules to bake into the generator

Continuous; strictly positive; narrow dynamic range (max/min ≤ 20 or the column can never
divide); clear of 1.0; per-trained-instance where possible; named as a physical quantity;
never a nominal category as an integer; never an outcome of the run being predicted. These
are the upstream `APPENDIX C` rules plus the two this study learned the hard way — the
positivity requirement, and **no feature defined as a ratio of two others** (`inst_to_attr`
is `nr_inst / nr_attr`, which put three perfectly collinear pairs in the library).

## Reproducing

The probes are in the session scratchpad, not in the tree. To rebuild them: each is a
short script over `metafit.experiment.run_equation` (features in, three R² out) and
`metafit.identity.correct_out_of_fold` (pass family labels instead of model labels for the
ceiling rows).
