# 6. Results

*Produced by `metafit.experiment`; reproduce with `PYTHONPATH=src venv/bin/python -m metafit`.*

## The three equations

The study is built around a deliberate contrast.

**E1 — dataset features only.** Every model evaluated on a given dataset shares one
feature vector, so many MCC values map to a single input and least squares necessarily
lands on the per-dataset mean. That is not a defect to be corrected; it is the control. E1
measures how much of MCC is explained by *the data alone*. It is fitted on the 20
aggregated per-dataset means, which makes that structure explicit and keeps the fold count
honest.

**E2 — dataset and model features.** One input, one output. E2 can express variation
*within* a dataset, which is exactly what E1 structurally cannot do.

**EM — model features only.** The mirror of E1, added to settle whether model choice
outweighs dataset difficulty. There are only five model features and one is constant per
model, so EM is short by necessity rather than by design.

## Headline comparison, on one scale

R² denominators differ between the 20 aggregated means and the 476 raw rows, so all
equations are evaluated on **every row**:

| | R² | MAE | Spearman |
|---|---|---|---|
| E1 (dataset only, 5 terms) | 0.337 | 0.215 | 0.640 |
| *E1's ceiling — the true dataset means* | *0.354* | *0.204* | *0.653* |
| EM (model only, 9 terms) | 0.164 | 0.251 | 0.356 |
| *EM's ceiling — the true model means* | *0.282* | *0.226* | *0.487* |
| **E2 (dataset + model, 14 terms)** | **0.558** | **0.167** | **0.775** |
| *additive oracle* | *0.6605* | *0.145* | *0.810* |

![Equations against their ceilings](../figures/equation_comparison.png)

## On their own scales

| | terms | in-sample R² | LOO-dataset R² | LOO-model R² |
|---|---|---|---|---|
| E1 (20 dataset means) | 5 | 0.953 | 0.506 | — |
| EM (476 rows) | 9 | 0.164 | 0.055 | — |
| **E2 (476 rows)** | **14** | **0.558** | **0.443** | **0.456** |

E1's in-sample R² of 0.953 on 20 points is a fit statistic on 20 observations with 5
parameters and should be read as such; its cross-validated 0.506 is the meaningful number,
and it is unstable (negative at 1–2 terms).

## Does model choice matter more than the dataset?

Not on this meta-dataset — but the reason is more interesting than the answer.

Dataset identity explains **0.354** of MCC variance against model identity's **0.282**, and
the equations widen the gap rather than closing it:

| | equation | own ceiling | captured |
|---|---|---|---|
| dataset features | 0.337 | 0.354 | **95%** |
| model features | 0.164 | 0.282 | **58%** |

**The meta-features describe datasets far better than they describe models.** Twelve
dataset meta-features nearly exhaust what dataset identity can explain; five model
features capture barely half of what model identity can. The remaining 42% of model
capability is real and simply not written down anywhere in this data — the strongest
argument in the study for richer model descriptors.

Two things create the opposite impression and are worth stating explicitly, because both
are artefacts:

- The **within-group correlation table is dataset-centred**, which removes dataset variance
  by construction. Only model terms can score well there. It is a diagnostic for model
  effects, not a statement of relative importance.
- **Model features carry more in combination than alone.** Adding them to E1 is worth
  +0.221 R² (0.337 → 0.558), well beyond the 0.164 they achieve by themselves. The surplus
  is dataset×model interaction, which is why 10 of the accuracy-leaning equation's 20 terms
  are mixed.

![Contribution shares](../figures/contribution_shares.png)

## The two configurations

`DEFAULT_E2` and `ACCURATE_E2` are the *same equation family* — same vocabulary, same
search, same solver — differing in two knobs:

| | `max_abs_zscore` | `penalty` | headline terms |
|---|---|---|---|
| `DEFAULT_E2` | 3.0 (172-term library) | 20 | 14 |
| `ACCURATE_E2` | 4.0 (360-term library) | 1 | 20 |

| terms | `ACCURATE_E2` in-sample R² | LOO-dataset R² | LOO-model R² |
|---|---|---|---|
| 12 | 0.584 | 0.270 | 0.472 |
| 16 | 0.619 | 0.320 | 0.493 |
| 18 | 0.640 | 0.341 | — |
| **20** | **0.647** | 0.287 | **0.533** |

The accuracy-leaning configuration clears 0.6 from 14 terms onward and reaches 0.647 at
20. Both are fitted and reported by every run.

**Does it plateau?** The budget was pushed to 64 to check. The default gains 0.003 R²
between 24 and 64 terms (0.5673 → 0.5704) — flat. The accuracy-leaning one keeps creeping
in-sample (0.653 → 0.668) while its leave-one-dataset-out R² falls to between -0.3 and
-0.7, so the creep is entirely overfitting. The sweep stops at 32 because nothing past it
is real.

## The fitted equations

E1, on the 20 dataset means:

```
MCC = +1.21681
      -0.160398   * [log(eq_num_attr)] * [log(nr_class)]
      -0.0601461  * ([nr_cor_attr] + [log(ns_ratio)]) / [log(nr_class)]
      -0.0286345  * ([log(class_ent)] + [log(gravity)]) / [log(nr_attr)]
      +0.00728631 * [log(eq_num_attr)] * [log(inst_to_attr)]
      +0.000217397* [log(inst_to_attr)] * [nr_norm]
```

E2, on all 476 rows (14 terms):

```
MCC = +0.950554
      -0.0398193   * [log(gravity)] / [log(Training Operations)]
      -0.0948144   * [log(eq_num_attr)] * [log(nr_class)]
      -0.18391     * [nr_cor_attr] * [nr_norm]
      +0.013162    * ([nr_bin] + [nr_norm]) / [log(nr_attr)]
      +0.0959242   * sqrt(Prediction Operations)
      -0.248633    * [log(eq_num_attr)] / [log(Training Operations)]
      +0.0312015   * [log(Processing Units Number)] * [Robust to Outliers]
      -0.0408016   * [log(Prediction Operations)] * [Robust to Outliers]
      +0.0123365   * Training Operations
      -0.0253954   * [log(nr_inst)] / [log(Processing Units Number)]
      -0.0254359   * ([nr_cor_attr] + [log(ns_ratio)]) / [log(nr_class)]
      +0.00439947  * [log(nr_inst)] * [Robust to Outliers]
      +0.000852048 * [log(inst_to_attr)] * [nr_norm]
      +0.134305    * [nr_cor_attr] * [log(Training Operations)]
```

![What each term is worth](../figures/term_effects.png)

## Where the equation fails

![Predicted versus actual MCC](../figures/predicted_vs_actual.png)

Predictions never fall below **0.17**, while 38 rows sit at exactly MCC = 0 — a column of
points hanging well above the diagonal on the left. **E2 cannot identify the cases where a
model will simply fail on a dataset.** It is a usable estimator in the range where models
work and a poor detector of the range where they do not, which must be stated before
anyone uses it to screen candidates.

The axes start at 0; the single negative row falls outside them and
`plots.count_below_floor()` returns the count for a caption.

## Flexible models do worse, not better

Standard regressors on the same raw features, under the same protocols:

| model | in-sample R² | LOO-dataset R² | LOO-model R² |
|---|---|---|---|
| RidgeCV (linear, 17 features) | 0.418 | **-2.002** | 0.328 |
| RandomForest (300 trees) | **0.910** | **0.067** | 0.465 |
| GradientBoosting | 0.820 | 0.049 | 0.354 |
| **metafit E2 (14 terms)** | 0.558 | **0.443** | 0.456 |

Read the RandomForest row across. With 20 dataset groups a forest memorises dataset
identity almost perfectly and then transfers worse than a 14-term equation. This is also
the likely provenance of the R² ≈ 0.9 figures reported for opaque meta-models: an
in-sample or randomly-split forest reproduces them exactly, and the same forest is
near-useless on an unseen dataset.

> Measured with scikit-learn during exploration. It is not a dependency of the package.
