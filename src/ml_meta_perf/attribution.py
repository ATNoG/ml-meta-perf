"""What each term is worth, and whether dataset or model features carry the equation.

An equation is only useful as guidance if its terms can be compared, and raw weights $w_i$
cannot be: they are in the units of whatever the term $t_i(\\mathbf{f})$ happens to compute.
For row $r$ and term $i$, let the per-row term contribution be $c_{r,i} = w_i \\, t_i(\\mathbf{f}_r)$
and the unclipped linear sum be $s_r = \\sum_{i=1}^{k} c_{r,i} = \\hat{y}_{\\text{raw}}(\\mathbf{f}_r) - w_0$.
Two comparable quantities are derived from $C = (c_{r,i}) \\in \\mathbb{R}^{n \\times k}$:

* *Effect* (`term_effects`) is the interdecile swing in predicted target produced by moving a
  term from its 10th to its 90th percentile across the corpus,

  $$\\mathrm{effect}_i = P_{90}(c_{\\cdot,i}) - P_{10}(c_{\\cdot,i}),$$

  stated in the target's own fitted units so one term's importance is directly comparable
  with another's and with the target itself.

* *Share* (`group_shares`) attributes the variance of the equation's linear output to the
  three disjoint term groups $G \\in \\{\\text{dataset}, \\text{model}, \\text{mixed}\\}$. Letting
  $s_{r,G} = \\sum_{i \\in G} c_{r,i}$ so that $s_r = \\sum_G s_{r,G}$, the variance share of
  group $G$ is the covariance decomposition

  $$\\mathrm{share}_G = \\frac{\\operatorname{cov}(s_{\\cdot,G},\\, s)}{\\operatorname{var}(s)},
  \\qquad \\sum_{G} \\mathrm{share}_G = 1,$$

  which sums to 1 exactly even when the feature groups are correlated.

Study chapter: [7. Other targets][study-chapter] -- what a study is made of.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/07-generic-targets.md
"""

from __future__ import annotations

import numpy as np
import polars as pl

from ml_meta_perf.model import Equation, direction

DATASET_ONLY = "dataset"
MODEL_ONLY = "model"
MIXED = "mixed"


def classify(
    features: tuple[str, ...],
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
) -> str:
    """Label a term by which feature sets $F_d$ (``dataset``), $F_m$ (``model``) or both (``mixed``) it draws on."""
    uses_dataset = any(feature in dataset_features for feature in features)
    uses_model = any(feature in model_features for feature in features)
    if uses_dataset and uses_model:
        return MIXED
    return MODEL_ONLY if uses_model else DATASET_ONLY


def contributions(equation: Equation, columns: dict[str, np.ndarray]) -> np.ndarray:
    """Per-row contribution matrix $C \\in \\mathbb{R}^{n \\times k}$ with $C_{r,i} = w_i \\, t_i(\\mathbf{f}_r)$."""
    if not equation.terms:
        return np.zeros((len(next(iter(columns.values()))), 0))
    with np.errstate(all="ignore"):
        return np.column_stack(
            [weight * term.evaluate(columns) for term, weight in zip(equation.terms, equation.weights, strict=True)]
        )


def term_effects(
    equation: Equation,
    columns: dict[str, np.ndarray],
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
) -> pl.DataFrame:
    """Per-term effect sizes $\\mathrm{effect}_i = P_{90}(c_{\\cdot,i}) - P_{10}(c_{\\cdot,i})$ in target units.

    ``effect`` is the 10th-to-90th-percentile swing of the term's contribution $c_{r,i} = w_i t_i(\\mathbf{f}_r)$:
    how much the predicted target moves across the bulk of the observed range of that term.
    Percentiles rather than min-to-max ensure that a single extreme row cannot define the headline number.
    """
    matrix = contributions(equation, columns)
    rows: list[dict[str, object]] = []
    for index, term in enumerate(equation.terms):
        column = matrix[:, index]
        low, high = np.percentile(column, [10.0, 90.0])
        rows.append(
            {
                "term": term.name,
                "group": classify(term.features, dataset_features, model_features),
                "weight": equation.weights[index],
                "beta": equation.standardized_weights[index],
                "effect": float(high - low),
                "direction": direction(equation.standardized_weights[index], equation.target),
            }
        )
    return pl.DataFrame(rows).sort(pl.col("effect").abs(), descending=True)


def group_shares(
    equation: Equation,
    columns: dict[str, np.ndarray],
    dataset_features: tuple[str, ...],
    model_features: tuple[str, ...],
) -> pl.DataFrame:
    """How much of the equation's output variance each feature group drives.

    For group subtotal $s_G = \\sum_{i \\in G} c_{\\cdot,i}$ and total linear prediction
    $s = \\sum_G s_G$, the share is

    $$\\mathrm{share}_G = \\frac{\\operatorname{cov}(s_G,\\, s)}{\\operatorname{var}(s)},$$

    using population moments ($\\mathrm{ddof}=0$). Because bilinearity of covariance gives
    $\\sum_G \\operatorname{cov}(s_G, s) = \\operatorname{cov}(\\sum_G s_G, s) = \\operatorname{var}(s)$,
    this decomposition sums to $1$ even when the groups are correlated, which a naive
    $\\operatorname{var}(s_G) / \\operatorname{var}(s)$ does not.
    """
    matrix = contributions(equation, columns)
    total = matrix.sum(axis=1)
    variance = float(np.var(total))
    labels = [classify(term.features, dataset_features, model_features) for term in equation.terms]

    rows: list[dict[str, object]] = []
    for group in (DATASET_ONLY, MODEL_ONLY, MIXED):
        selected = [index for index, label in enumerate(labels) if label == group]
        if not selected:
            rows.append({"group": group, "n_terms": 0, "share": 0.0, "effect_sum": 0.0})
            continue
        subtotal = matrix[:, selected].sum(axis=1)
        # Population covariance (ddof=0) to match np.var. np.cov defaults to ddof=1, and
        # mixing the two inflates every share by n/(n-1) so the decomposition stops
        # summing to 1.
        covariance = float(((subtotal - subtotal.mean()) * (total - total.mean())).mean())
        share = covariance / variance if variance > 1e-15 else 0.0
        low, high = np.percentile(subtotal, [10.0, 90.0])
        rows.append(
            {
                "group": group,
                "n_terms": len(selected),
                "share": share,
                "effect_sum": float(high - low),
            }
        )
    return pl.DataFrame(rows)
