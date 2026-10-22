"""The fitted object: a sparse additive equation

$$\\hat{y}(\\mathbf{f}) = \\operatorname{clip}_B\\!\\left(w_0 + \\sum_{i=1}^{k} w_i \\, t_i(\\mathbf{f})\\right).$$

Selection runs on standardised terms, because greedy selection compares candidates by
correlation with the residual and raw scales would make that comparison meaningless.
What gets published, though, must be evaluable as written, so the weights are folded
back into raw units before they are stored:

$$w_i = \\frac{\\beta_i}{\\sigma_i}, \\qquad w_0 = \\bar{y} - \\sum_{i=1}^{k} w_i \\, \\mu_i.$$

Both sets are kept: the raw weights $w_i$ are the equation, the standardised weights
$\\beta_i$ are how the terms rank against each other.

Study chapter: [2. The additive model][study-chapter] -- the rationale, in prose.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/02-additive-model.md
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ml_meta_perf.config import Bounds
from ml_meta_perf.terms import Term

MCC_LOWER = -1.0
MCC_UPPER = 1.0


@dataclass(frozen=True)
class Equation:
    """A sparse linear equation over named terms."""

    intercept: float
    terms: tuple[Term, ...]
    weights: tuple[float, ...]
    standardized_weights: tuple[float, ...]
    name: str = "equation"
    bounds: Bounds | None = (MCC_LOWER, MCC_UPPER)
    #: What the equation predicts, as the first line of its text names it.
    target: str = "MCC"

    def __post_init__(self) -> None:
        if not (len(self.terms) == len(self.weights) == len(self.standardized_weights)):
            raise ValueError("terms and weights must have the same length")

    @property
    def n_terms(self) -> int:
        """Number of terms $k$ in the equation (excluding the intercept $w_0$)."""
        return len(self.terms)

    def evaluate(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Apply $\\hat{y}_{\\text{raw}}(\\mathbf{f}) = w_0 + \\sum_{i=1}^{k} w_i t_i(\\mathbf{f})$ unclipped."""
        n = len(next(iter(columns.values())))
        total = np.full(n, self.intercept, dtype=np.float64)
        with np.errstate(all="ignore"):
            for term, weight in zip(self.terms, self.weights, strict=True):
                total += weight * term.evaluate(columns)
        return total

    def predict(self, columns: dict[str, np.ndarray]) -> np.ndarray:
        """Clip into $B$: $\\hat{y}(\\mathbf{f}) = \\operatorname{clip}_B(\\hat{y}_{\\text{raw}}(\\mathbf{f}))$.

        A linear form has no idea that a bounded target stops at $b$ (e.g. $1.0$ for MCC),
        and the meta-dataset includes exact values at that bound. Clipping is the domain
        knowledge imposed on the output, and it is applied identically during cross-validation.
        An unbounded target sets ``bounds=None`` ($B = \\varnothing$) and is returned as written.
        """
        values = self.evaluate(columns)
        return values if self.bounds is None else np.clip(values, *self.bounds)

    def ranked_terms(self) -> list[tuple[Term, float, float]]:
        """Terms $(t_i, w_i, \\beta_i)$ ordered by descending standardised weight magnitude $|\\beta_i|$."""
        triples = list(zip(self.terms, self.weights, self.standardized_weights, strict=True))
        return sorted(triples, key=lambda item: abs(item[2]), reverse=True)

    def __str__(self) -> str:
        lines = [f"{self.target} = {self.intercept:+.6g}"]
        for term, weight, standardized in self.ranked_terms():
            # Long terms overflow the column rather than being truncated, so the padding
            # has to keep a separator of its own -- ljust alone glues the comment to a
            # name that is already past the stop.
            lines.append(f"      {weight:+.6g} * {term.name}".ljust(64) + f"  # beta={standardized:+.4f}")
        return "\n".join(lines)

    def to_latex(self) -> str:
        """A LaTeX rendering of the equation, for dropping straight into a paper."""
        parts = [f"{self.intercept:+.4g}"]
        for term, weight, _ in self.ranked_terms():
            parts.append(f"{weight:+.4g} \\cdot \\mathrm{{{_escape(term.name)}}}")
        return f"\\mathrm{{{_escape(self.target)}}} = " + " ".join(parts)

    def to_dict(self) -> dict[str, object]:
        """Serialise the equation to a JSON-compatible dictionary."""
        return {
            "name": self.name,
            "intercept": self.intercept,
            "terms": [term.to_dict() for term in self.terms],
            "weights": list(self.weights),
            "standardized_weights": list(self.standardized_weights),
            "bounds": list(self.bounds) if self.bounds is not None else None,
            "target": self.target,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> Equation:
        """Reconstruct an `Equation` from a dictionary produced by `to_dict`."""
        terms = payload["terms"]
        weights = payload["weights"]
        standardized = payload["standardized_weights"]
        bounds = payload.get("bounds", [MCC_LOWER, MCC_UPPER])
        if not isinstance(terms, list) or not isinstance(weights, list) or not isinstance(standardized, list):
            raise ValueError("malformed equation payload")
        return cls(
            intercept=float(payload["intercept"]),  # pyright: ignore[reportArgumentType]
            terms=tuple(Term.from_dict(item) for item in terms),  # pyright: ignore[reportArgumentType]
            weights=tuple(float(value) for value in weights),  # pyright: ignore[reportArgumentType]
            standardized_weights=tuple(float(value) for value in standardized),  # pyright: ignore[reportArgumentType]
            name=str(payload.get("name", "equation")),
            bounds=tuple(bounds) if bounds is not None else None,  # pyright: ignore[reportArgumentType]
            target=str(payload.get("target", "MCC")),
        )

    def save(self, path: str | Path) -> None:
        """Write the equation as formatted JSON to ``path``."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Equation:
        """Load an `Equation` from a JSON file written by `save`."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _escape(name: str) -> str:
    """Escape underscores and carets in ``name`` for LaTeX math mode."""
    return name.replace("_", r"\_").replace("^", r"\^{}")


#: How a term's direction is written in every exported table.
#:
#: One spelling, in one place, because `attribution.term_effects` and `report` both write a
#: ``direction`` column and used to disagree -- "increases MCC" in one CSV, "raises MCC" in
#: the next -- which reads as two different quantities to anyone joining them.
RAISES, LOWERS = "raises MCC", "lowers MCC"


def direction(signed: float, target: str = "MCC") -> str:
    """The word for the sign of a standardised weight or effect, on ``target``."""
    return f"{'raises' if signed > 0.0 else 'lowers'} {target}"
