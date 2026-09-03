"""The candidate pool for `MODEL_FEATURES`: three AIs' proposed descriptors plus the six
already in production, registered against `assets/meta_dataset_all_descriptors.csv`.

Two things happen before any search runs, and both are here rather than inside the
search itself, since neither one needs an optimizer to decide.

**Admissibility.** `ml_meta_perf.terms` assumes strictly positive, continuous features
and is explicit that a graded descriptor is preferable to an indicator (`terms.py`'s
module docstring). Reviewed against that rule, a handful of the proposed descriptors are
not what they look like: a categorical choice wearing an arbitrary sign
(`perplexity__knn_metric_signal` gives Manhattan and Euclidean no real order, unlike
`chatgpt__distance_norm_order`, which is the Minkowski *p* itself), a column whose
direction of "more" flips depending on which mechanism populated a row
(`perplexity__randomness` codes `max_features=None` -- the *least* randomised choice --
above `sqrt`), or a sentinel collision (`chatgpt__inference_work_proxy` uses `0` for an
unbounded tree, which is potentially the most expensive case, not the cheapest).
`INADMISSIBLE` records these with the one-line reason each was dropped, so the search
space is smaller and the exclusion is documented rather than silent.

**Semantic exclusion.** `ai_model_descriptors.md` groups descriptors that encode the same
underlying concept (learning rate, model depth, L2 regularisation, ...) and states the
rule as: a selected feature set may contain at most one member of each group. A member
can sit in more than one group -- Claude1's `capacity` covers depth, width *and*
aggregate capacity at once, and Perplexity's `reg_l2_log` covers both L2 regularisation
and covariance stabilisation -- so the constraint is not "one slot per group" (a single
descriptor can't be forced into one slot when it legitimately spans several) but "no two
*selected* descriptors ever share a group". `conflict_graph` turns the group table into
exactly that pairwise constraint, and `resolve_conflicts` is the one place both search
backends go through to enforce it.

**Group representatives.** 55 admissible candidates is still mostly near-synonyms
competing within one exclusion group -- deciding between ChatGPT's, Claude1's and
Perplexity's three spellings of "learning rate" is not a question worth spending a
combinatorial search's budget on. `compute_strengths` plus `Registry.reduce_to_representatives`
collapse each group (and, correctly, each cluster of groups a multi-group member bridges)
to its single strongest univariate correlate with MCC *before* any optimizer runs, so the
search that follows decides which *concepts* survive together, over a roughly
half-sized, still exclusion-clean space.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from ml_meta_perf.data import MODEL_FEATURES

#: The 26 ChatGPT, 7 Claude1 and 22 Perplexity descriptors, namespaced to match the
#: headers in `assets/meta_dataset_all_descriptors.csv`.
CHATGPT_FEATURES: tuple[str, ...] = tuple(
    f"chatgpt__{name}"
    for name in (
        "component_count", "sequential_depth", "unbounded_depth", "representation_width",
        "structural_size_proxy", "train_work_proxy", "inference_work_proxy", "learning_rate",
        "update_aggressiveness", "l1_coefficient", "l2_coefficient", "elasticnet_coefficient",
        "dropout_rate", "sparsity_coefficient", "smoothing_coefficient", "covariance_shrinkage",
        "covariance_regularization", "local_support_size", "distance_norm_order",
        "distance_weighting", "feature_subset_power", "class_balance", "attention_heads",
        "relaxation_factor", "loss_smoothness", "fixed_configuration",
    )
)

CLAUDE1_FEATURES: tuple[str, ...] = tuple(
    f"claude1__{name}"
    for name in (
        "no_tunable_hp", "capacity", "regularization", "ensemble_size", "learning_rate",
        "stochasticity", "class_balance",
    )
)

PERPLEXITY_FEATURES: tuple[str, ...] = tuple(
    f"perplexity__{name}"
    for name in (
        "reg_l2_log", "reg_l1_log", "capacity_depth", "capacity_width", "lr_log", "iters_log",
        "randomness", "class_weight_signal", "knn_k", "knn_weights_signal", "knn_metric_signal",
        "var_smoothing_log", "attn_depth", "attn_width", "attn_heads", "attn_dropout_shifted",
        "n_layers", "total_units", "max_units", "ensemble_n", "ensemble_lr_log", "no_tunable_hp",
    )
)

#: Every column the search is allowed to choose from: the six features already in
#: production (eligible for removal, not just the newcomers eligible for addition) plus
#: the three AIs' proposals.
CANDIDATE_FEATURES: tuple[str, ...] = MODEL_FEATURES + CHATGPT_FEATURES + CLAUDE1_FEATURES + PERPLEXITY_FEATURES

#: `ai_model_descriptors.md` section 2, transcribed. A descriptor absent from every group
#: (e.g. `chatgpt__update_aggressiveness`, `chatgpt__train_work_proxy`) is ungrouped: nothing
#: else claims its concept, so it competes freely.
EXCLUSION_GROUPS: dict[str, tuple[str, ...]] = {
    "E01_fixed_configuration": (
        "chatgpt__fixed_configuration", "claude1__no_tunable_hp", "perplexity__no_tunable_hp",
    ),
    "E02_learning_rate": (
        "chatgpt__learning_rate", "claude1__learning_rate", "perplexity__lr_log", "perplexity__ensemble_lr_log",
    ),
    "E03_class_balance": (
        "chatgpt__class_balance", "claude1__class_balance", "perplexity__class_weight_signal",
    ),
    "E04_knn_k": ("chatgpt__local_support_size", "perplexity__knn_k"),
    "E05_knn_metric": ("chatgpt__distance_norm_order", "perplexity__knn_metric_signal"),
    "E06_knn_weighting": ("chatgpt__distance_weighting", "perplexity__knn_weights_signal"),
    "E07_attention_heads": ("chatgpt__attention_heads", "perplexity__attn_heads"),
    "E08_ensemble_size": (
        "chatgpt__component_count", "claude1__ensemble_size", "perplexity__iters_log", "perplexity__ensemble_n",
    ),
    "E09_depth": (
        "chatgpt__sequential_depth", "claude1__capacity", "perplexity__capacity_depth",
        "perplexity__attn_depth", "perplexity__n_layers",
    ),
    "E10_width": (
        "chatgpt__representation_width", "claude1__capacity", "perplexity__capacity_width",
        "perplexity__attn_width", "perplexity__max_units",
    ),
    "E11_aggregate_capacity": ("chatgpt__structural_size_proxy", "claude1__capacity", "perplexity__total_units"),
    "E16_l2": ("chatgpt__l2_coefficient", "perplexity__reg_l2_log"),
    "E17_l1_sparsity": (
        "chatgpt__l1_coefficient", "chatgpt__elasticnet_coefficient",
        "chatgpt__sparsity_coefficient", "perplexity__reg_l1_log",
    ),
    "E18_covariance": (
        "chatgpt__covariance_shrinkage", "chatgpt__covariance_regularization", "perplexity__reg_l2_log",
    ),
    "E19_smoothing": ("chatgpt__smoothing_coefficient", "perplexity__var_smoothing_log"),
    "E20_dropout": (
        "chatgpt__dropout_rate", "claude1__stochasticity",
        "perplexity__randomness", "perplexity__attn_dropout_shifted",
    ),
    "E21_feature_subset": ("chatgpt__feature_subset_power", "perplexity__randomness"),
    "E22_tabnet_relaxation": ("chatgpt__relaxation_factor", "perplexity__randomness"),
}

#: Deterministic exclusions applied before any search -- see the module docstring. Kept
#: as a dict so the reason travels with the column into every report this module writes.
INADMISSIBLE: dict[str, str] = {
    "perplexity__knn_metric_signal": (
        "Manhattan/Euclidean encoded as an arbitrary +-1 sign; no real order between the "
        "two. chatgpt__distance_norm_order (the Minkowski p itself, 1 vs 2) is the "
        "admissible sibling in E05."
    ),
    "perplexity__knn_weights_signal": (
        "distance/uniform weighting encoded as an arbitrary +-1 sign rather than a graded "
        "quantity; chatgpt__distance_weighting (0/1, degree of distance emphasis) is the "
        "admissible sibling in E06."
    ),
    "perplexity__class_weight_signal": (
        "0 means 'class_weight does not exist for this model' and -1 means 'exists, set "
        "to None' -- applicability and direction collapsed onto one signed axis, not a "
        "single ordered quantity."
    ),
    "chatgpt__loss_smoothness": (
        "0 covers several unrelated cases at once (hinge loss, PassiveAggressive, "
        "Perceptron, and every model with no loss choice at all) -- a label, not an order."
    ),
    "perplexity__randomness": (
        "averages feature-subsampling power with dropout/gamma; max_features=None (the "
        "least randomised choice) is coded 1.0, above sqrt's 0.5 (more randomised) -- the "
        "column's own 'more randomness -> higher value' direction inverts depending on "
        "which mechanism populated the row."
    ),
    "chatgpt__inference_work_proxy": (
        "0 for an unbounded-depth tree collides with the 'not applicable' sentinel on "
        "what should be among the largest values in the column, not the smallest."
    ),
}


def _greedy_independent_set(ordered: Sequence[str], conflicts: dict[str, frozenset[str]]) -> frozenset[str]:
    """Walk ``ordered`` once, keeping a name iff it conflicts with nothing kept so far.

    Exact for this problem under *any* priority order, not just a good heuristic: every
    conflict is a simple pairwise edge (no weighted or higher-arity constraint), so
    keeping the first-seen member of a clique and dropping everything it conflicts with
    never blocks a later member that would otherwise have survived. That is what lets
    the same routine serve two different priorities below -- declaration order for
    `Registry.resolve_conflicts`, correlation strength for `Registry.reduce_to_representatives`
    -- with the same correctness argument for both.
    """
    kept: set[str] = set()
    for name in ordered:
        if kept.isdisjoint(conflicts.get(name, frozenset())):
            kept.add(name)
    return frozenset(kept)


@dataclass(frozen=True)
class Registry:
    """The admissible candidate pool for one meta-dataset, with its conflict graph."""

    admissible: tuple[str, ...]
    dropped: dict[str, str]
    conflicts: dict[str, frozenset[str]]

    def resolve_conflicts(self, selected: frozenset[str]) -> frozenset[str]:
        """The largest subset of ``selected`` with no two members sharing an exclusion
        group, choosing deterministically in `self.admissible` order.

        A candidate can belong to several groups at once (`claude1__capacity` spans
        depth, width and aggregate capacity), so the constraint is not "one per group
        slot" -- it is "no edge of the conflict graph has both endpoints selected", i.e.
        ``selected`` restricted to an independent set.
        """
        ordered = [name for name in self.admissible if name in selected]
        return _greedy_independent_set(ordered, self.conflicts)

    def reduce_to_representatives(self, strengths: dict[str, float]) -> Registry:
        """Collapse every exclusion group (and, transitively, every cluster of groups
        overlapping through a multi-group member) to its single strongest-correlated
        candidate, by ``strengths`` -- typically each candidate's univariate association
        with MCC.

        This is a pre-filter, not a replacement for the combinatorial search: it removes
        the part of the search space spent choosing *within* one concept (ChatGPT's
        `learning_rate` vs Claude1's vs Perplexity's `lr_log`, three near-synonyms in the
        same exclusion group) so the optimizer's budget goes to deciding which *concepts*
        survive together, which is the question that actually needs a search. A
        multi-group candidate is handled correctly by construction: since the ordering is
        global (not per-group), a candidate spanning several groups is compared against
        every rival in every one of them at once, and the same `_greedy_independent_set`
        argument that makes `resolve_conflicts` exact under declaration order makes this
        exact under strength order.

        Ties (or candidates `strengths` has no entry for) fall back to `self.admissible`'s
        declaration order, so the result is fully deterministic.
        """
        priority = {name: index for index, name in enumerate(self.admissible)}
        ordered = sorted(self.admissible, key=lambda name: (-strengths.get(name, 0.0), priority[name]))
        kept = _greedy_independent_set(ordered, self.conflicts)

        dropped = dict(self.dropped)
        for name in self.admissible:
            if name in kept:
                continue
            rival = next((other for other in self.conflicts.get(name, frozenset()) if other in kept), None)
            dropped[name] = (
                f"weaker univariate correlation with MCC than {rival} "
                f"({strengths.get(name, 0.0):.3f} vs {strengths.get(rival, 0.0):.3f})"
                if rival is not None
                else "dropped while reducing to group representatives"
            )
        return Registry(
            admissible=tuple(name for name in self.admissible if name in kept),
            dropped=dropped,
            conflicts=self.conflicts,
        )

    def with_correlation_conflicts(self, frame: pl.DataFrame, *, threshold: float = 0.9) -> Registry:
        """Extend the conflict graph with **empirical** redundancy, on top of the named
        exclusion groups: any two admissible candidates whose raw values correlate above
        ``threshold`` (the stronger of Pearson and Spearman) become a conflict edge.

        `EXCLUSION_GROUPS` catches redundancy the source documents already named --
        ChatGPT's `l2_coefficient` and Perplexity's `reg_l2_log` are the same concept by
        construction. It cannot catch two descriptors from *different* concepts that
        happen to be near-duplicates on one corpus (two counts that both track ensemble
        size in slightly different units, say). This does, and feeds the same
        `reduce_to_representatives` machinery, so a correlated cluster collapses to its
        one strongest-correlated-with-MCC member exactly like a named group does.
        """
        from ml_meta_perf.stats import pearson, spearman

        values = {name: frame[name].to_numpy().astype(np.float64) for name in self.admissible}
        conflicts = {name: set(members) for name, members in self.conflicts.items()}
        for index, left in enumerate(self.admissible):
            for right in self.admissible[index + 1 :]:
                strength = max(abs(pearson(values[left], values[right])), abs(spearman(values[left], values[right])))
                if strength >= threshold:
                    conflicts[left].add(right)
                    conflicts[right].add(left)
        return Registry(
            admissible=self.admissible,
            dropped=dict(self.dropped),
            conflicts={name: frozenset(members) for name, members in conflicts.items()},
        )


def _near_constant(values: np.ndarray, tolerance: float = 1e-9) -> bool:
    """Whether a column carries no usable variance on this corpus."""
    spread = float(values.max() - values.min())
    scale = max(abs(float(values.mean())), 1.0)
    return spread <= tolerance * scale


def build_registry(frame: pl.DataFrame, candidates: tuple[str, ...] = CANDIDATE_FEATURES) -> Registry:
    """The admissible pool for ``frame``: `INADMISSIBLE` plus whatever is constant here.

    Constancy is checked against the data rather than hard-coded, since a candidate that
    is degenerate on one corpus need not be degenerate on another -- `Robust to Outliers`
    is the one this repo's own corpus is expected to flag, but nothing here assumes it.
    """
    dropped = dict(INADMISSIBLE)
    for name in candidates:
        if name in dropped:
            continue
        if name not in frame.columns:
            dropped[name] = "column not present in this meta-dataset"
            continue
        if _near_constant(frame[name].to_numpy().astype(np.float64)):
            dropped[name] = "near-constant on this corpus (no usable variance)"

    admissible = tuple(name for name in candidates if name not in dropped)

    conflicts: dict[str, set[str]] = {name: set() for name in admissible}
    for members in EXCLUSION_GROUPS.values():
        present = [name for name in members if name in conflicts]
        for left in present:
            for right in present:
                if left != right:
                    conflicts[left].add(right)

    return Registry(
        admissible=admissible,
        dropped=dropped,
        conflicts={name: frozenset(members) for name, members in conflicts.items()},
    )


def compute_strengths(frame: pl.DataFrame, candidates: Sequence[str], target_column: str = "MCC") -> dict[str, float]:
    """Each candidate's univariate association with the target: the stronger of its
    Pearson and Spearman correlation, absolute.

    The stronger-of-two mirrors `fit.transform_gap`'s screening elsewhere in this
    project: a feature that is only monotonically (not linearly) related to MCC still
    carries real signal a linear correlation alone would understate.
    """
    from ml_meta_perf.stats import pearson, spearman

    truth = frame[target_column].to_numpy().astype(np.float64)
    strengths: dict[str, float] = {}
    for name in candidates:
        values = frame[name].to_numpy().astype(np.float64)
        strengths[name] = max(abs(pearson(values, truth)), abs(spearman(values, truth)))
    return strengths
