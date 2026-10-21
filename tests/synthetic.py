"""Corpora with a planted equation: the true number of terms is known, so a length rule can be scored.

``planted`` draws a dataset-by-model corpus whose target is a sum of ``k`` grammar terms, each
standardised so that every one carries the same share of the signal, plus Gaussian noise. A length
rule is right when it returns ``k``. ``cost_shaped`` is the same corpus read as a positive cost that
spans orders of magnitude, the shape of the FlexFL targets (``response.transform`` is ``log1p``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ml_meta_perf.config import Schema

DATASET_FEATURES = ("size", "width", "depth", "rate")
MODEL_FEATURES = ("lr", "batch", "epochs")
MAX_TERMS = 6


def _true_terms(columns: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
    """The planted terms, simplest first; each is a term the grammar can build from these features."""
    return [
        ("log(size)", np.log(columns["size"])),
        ("lr", columns["lr"]),
        ("[log(width)] * [log(epochs)]", np.log(columns["width"]) * np.log(columns["epochs"])),
        ("sqrt(rate)", np.sqrt(columns["rate"])),
        ("[log(depth)] / [log(batch)]", np.log(columns["depth"]) / np.log(columns["batch"])),
        ("[log(size)] * [lr]", np.log(columns["size"]) * columns["lr"]),
    ][:MAX_TERMS]


@dataclass(frozen=True)
class Planted:
    """A planted corpus and what was planted in it."""

    frame: pl.DataFrame
    true_terms: tuple[str, ...]
    #: Noise standard deviation relative to the standard deviation of the noise-free signal.
    noise: float

    @property
    def k(self) -> int:
        return len(self.true_terms)

    def schema(self, *, bounds: tuple[float | None, float | None] | None = None, **kwargs: object) -> Schema:
        return Schema("d", "m", DATASET_FEATURES, MODEL_FEATURES, "y", bounds, **kwargs)  # pyright: ignore[reportArgumentType]


def planted(k: int, *, datasets: int = 14, models: int = 8, noise: float = 0.1, seed: int = 0) -> Planted:
    """``datasets * models`` rows whose target is ``sum of k unit-variance terms + noise``."""
    if not 1 <= k <= MAX_TERMS:
        raise ValueError(f"k must be in 1..{MAX_TERMS}, got {k}")
    rng = np.random.default_rng(seed)
    per_dataset = {
        "size": rng.uniform(3.0, 60.0, datasets),
        "width": rng.uniform(3.0, 40.0, datasets),
        "depth": rng.uniform(2.0, 20.0, datasets),
        "rate": rng.uniform(1.0, 30.0, datasets),
    }
    per_model = {
        "lr": rng.uniform(0.5, 9.0, models),
        "batch": rng.uniform(4.0, 40.0, models),
        "epochs": rng.uniform(3.0, 30.0, models),
    }
    cells = [(i, j) for i in range(datasets) for j in range(models)]
    columns = {name: np.array([values[i] for i, _ in cells]) for name, values in per_dataset.items()}
    columns |= {name: np.array([values[j] for _, j in cells]) for name, values in per_model.items()}
    chosen = _true_terms(columns)[:k]
    signal = sum((values - values.mean()) / values.std() for _, values in chosen)
    assert isinstance(signal, np.ndarray)
    response = 1.0 + signal + rng.normal(0.0, noise * signal.std(), len(cells))
    frame = pl.DataFrame({"d": [f"d{i}" for i, _ in cells], "m": [f"m{j}" for _, j in cells], **columns, "y": response})
    return Planted(frame, tuple(name for name, _ in chosen), noise)


def cost_shaped(k: int, **kwargs: object) -> Planted:
    """The planted corpus read as a positive cost: ``y = exp(z) - 1`` with ``z`` the planted target."""
    base = planted(k, **kwargs)  # pyright: ignore[reportArgumentType]
    z = base.frame["y"].to_numpy()
    cost = np.expm1(0.8 * (z - z.min()) + 0.1)
    return Planted(base.frame.with_columns(pl.Series("y", cost)), base.true_terms, base.noise)
