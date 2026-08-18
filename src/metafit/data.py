"""Loading and shaping of the meta-dataset.

The meta-dataset has one row per (dataset, model) pair. Two groups of columns matter
and they behave very differently, which is why they are named separately here:

* dataset features are constant across every row of a given dataset, so on their own
  they can only ever predict a per-dataset constant;
* model features vary with the model, and three of them (the operation counts) also
  vary with the dataset, since they are functions of the dataset size.

That asymmetry is the whole point of the two-equation comparison, so the split is
part of the public API rather than something each caller re-derives.

**``nr_inst`` describes the source dataset, not the training set.** Every model was
trained on a stratified sample capped at 100,000 rows, and ten of the twenty datasets are
larger than that -- up to seven million. Nothing in the CSV records the sampled size,
because it is the same cap for every dataset above it. So ``nr_inst`` and
``inst_to_attr`` are properties of the corpus a dataset was drawn from, and no statement
about "more training data" can be tested against them.

Study chapter: [1. The problem and the data](../../assets/docs/01-problem.md) -- the rationale, in
prose, with the figures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

DATASET_COLUMN = "Dataset"
MODEL_COLUMN = "Model"
TARGET_COLUMN = "MCC"

DATASET_FEATURES: tuple[str, ...] = (
    "class_ent",
    "eq_num_attr",
    "gravity",
    "inst_to_attr",
    "nr_attr",
    "nr_bin",
    "nr_class",
    "nr_cor_attr",
    "nr_inst",
    "nr_norm",
    "nr_outliers",
    "ns_ratio",
)

MODEL_FEATURES: tuple[str, ...] = (
    "Processing Units Number",
    "Training Operations",
    "Prediction Operations",
    "Active Regularization Mechanisms",
    "Robust to Outliers",
    "Model Capability",
)

#: Where each learner family sits on a capability ladder, low to high.
#:
#: This is the **provenance** of the ``Model Capability`` column, which the CSV carries like
#: any other feature. It is kept here, and checked against the CSV by ``test_data``, so the
#: column can be regenerated and so a reader can see where the numbers came from rather than
#: finding twenty-five unexplained integers in a data file.
#:
#: **The order is asserted from the tabular-ML literature, not fitted here** -- Grinsztajn
#: et al. (2022), Shwartz-Ziv & Armon (2022), McElfresh et al. (2023), and Hollmann et al.
#: (2023) for the in-context models. Ordering families by their observed MCC in this corpus
#: would fit the column on the target and make every coefficient over it circular.
#:
#: That independence is also the column's weakness, and it is measured rather than
#: suspected: against this corpus the asserted order rises at only 6 of 9 steps in raw
#: per-family mean MCC, where 9 would be monotone and about 4.5 is unrelated. `generic NN`
#: has the *lowest* family mean here, 0.454, from rung 6 of 10; `single tree` is near the
#: top at 0.930 from rung 5. Anything the equation reads off this column is therefore only
#: loosely "capability", and a term over it must not be quoted as though it were more.
MODEL_CAPABILITY: dict[str, int] = {
    "naive bayes": 1,
    "discriminant": 2,
    "linear": 3,
    "instance": 4,
    "single tree": 5,
    "generic NN": 6,
    "tabular NN": 7,
    "bagged trees": 8,
    "boosted trees": 9,
    "tabular foundation": 10,
}

ALL_FEATURES: tuple[str, ...] = DATASET_FEATURES + MODEL_FEATURES

#: Plain-language readings of each feature, used when the fitted equation is turned into
#: Learner family for each model in the meta-dataset. Model *features* describe capacity
#: and cost; they do not say what kind of learner a row refers to, and the tabular-ML
#: literature states its guidance in exactly those terms ("prefer tree ensembles"), so
#: checking that guidance against this corpus needs the taxonomy written down.
#:
#: The split between ``tabular NN`` and ``generic NN`` is the one that matters: an
#: architecture designed for tabular data and a plain multilayer perceptron are both
#: "deep learning" and behave nothing alike here.
MODEL_FAMILY: dict[str, str] = {
    "TabICL": "tabular foundation",
    "TabPFN": "tabular foundation",
    "Bagging": "bagged trees",
    "DT": "single tree",
    "ExtraTree": "single tree",
    "XGBoost": "boosted trees",
    "LightGBM_RF": "boosted trees",
    "LightGBM_ExtraTrees": "boosted trees",
    "AdaBoost": "boosted trees",
    "TabNet": "tabular NN",
    "FT-Transformer": "tabular NN",
    "TabTransformer": "tabular NN",
    "DNN": "generic NN",
    "MLP": "generic NN",
    "KNN": "instance",
    "Ridge": "linear",
    "LR": "linear",
    "LinearSVC": "linear",
    "PassiveAggressive": "linear",
    "SGD": "linear",
    "Perceptron": "linear",
    "QDA": "discriminant",
    "LDA": "discriminant",
    "BernoulliNB": "naive bayes",
    "GaussianNB": "naive bayes",
}

#: Families the tabular-ML literature groups together as "tree-based".
TREE_FAMILIES = ("bagged trees", "single tree", "boosted trees")

#: Families that are neural networks, split by whether the architecture targets tabular
#: data specifically.
NEURAL_FAMILIES = ("tabular NN", "generic NN")

#: written guidance. Without these a "best practice" degenerates into restating a column
#: name, which is not advice anyone can act on.
FEATURE_GLOSSARY: dict[str, str] = {
    "class_ent": "class entropy (how evenly the labels are spread)",
    "eq_num_attr": "equivalent number of attributes (effective feature count)",
    "gravity": "gravity (separation between the majority and minority class centres)",
    "inst_to_attr": "instances per attribute",
    "nr_attr": "number of attributes",
    "nr_bin": "number of binary attributes",
    "nr_class": "number of classes",
    "nr_cor_attr": "proportion of correlated attribute pairs",
    "nr_inst": "number of instances in the source dataset (before sampling)",
    "nr_norm": "number of normally distributed attributes",
    "nr_outliers": "number of attributes containing outliers",
    "ns_ratio": "noise-to-signal ratio",
    "Processing Units Number": "model capacity (log processing units)",
    "Training Operations": "training cost (log operations)",
    "Prediction Operations": "inference cost (log operations)",
    "Active Regularization Mechanisms": "number of active regularisation mechanisms",
    "Robust to Outliers": "built-in robustness to outliers",
    "Model Capability": "learner family's capability rank in the tabular-ML literature (1-10)",
}

#: The shipped corpus, resolved next to this module rather than relative to a source
#: checkout. `pip install metafit && metafit` has no repository around it, and the previous
#: form -- ``parents[2] / "data"`` -- pointed inside `site-packages` and failed there.
DEFAULT_PATH = Path(__file__).resolve().parent / "meta_dataset.csv"


class SchemaError(ValueError):
    """The CSV does not carry the columns the meta-model needs."""


def load(path: str | Path | None = None) -> pl.DataFrame:
    """Read the meta-dataset and check it has the expected shape.

    Every feature is cast to Float64. The integer-valued columns (``nr_attr``,
    ``nr_class``, ...) are still continuous as far as the equations are concerned,
    and carrying two dtypes through the term library buys nothing.
    """
    resolved = Path(path) if path is not None else DEFAULT_PATH
    if not resolved.is_file():
        raise SchemaError(f"meta-dataset not found: {resolved}")

    frame = pl.read_csv(resolved)
    expected = (DATASET_COLUMN, MODEL_COLUMN, *ALL_FEATURES, TARGET_COLUMN)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    frame = frame.select(expected).with_columns(
        [pl.col(column).cast(pl.Float64) for column in (*ALL_FEATURES, TARGET_COLUMN)]
    )

    nulls = sum(frame.null_count().row(0))
    if nulls:
        raise SchemaError(f"meta-dataset contains {nulls} null values")
    return frame


def aggregate_by_dataset(frame: pl.DataFrame) -> pl.DataFrame:
    """Collapse to one row per dataset: the dataset features plus the mean MCC.

    **E1 is no longer fitted against this**, and the reason is worth recording. Least
    squares on a predictor that is constant within a group lands on that group's mean
    either way, so aggregating first looked free. It was not: it put E1's R2 on a
    20-point denominator, which cannot be compared with E3's 476-row one, and the 0.506
    it produced read as *better* transfer than E3's 0.466 when on the common scale it is
    0.217. All three equations are now fitted on all 476 rows.

    Kept because the aggregated view is still the right one for describing the corpus --
    20 datasets, one row each -- and because chapter 6 quotes the comparison.
    """
    return (
        frame.group_by(DATASET_COLUMN)
        .agg(
            [pl.col(column).first() for column in DATASET_FEATURES]
            + [
                pl.col(TARGET_COLUMN).mean().alias(TARGET_COLUMN),
                pl.len().alias("n_models"),
            ]
        )
        .sort(DATASET_COLUMN)
    )


def columns_as_arrays(frame: pl.DataFrame, features: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Extract the named columns as a mapping of float arrays, the form terms consume."""
    return {name: frame[name].to_numpy().astype(np.float64) for name in features}


def target(frame: pl.DataFrame) -> np.ndarray:
    """The MCC column as a float array."""
    return frame[TARGET_COLUMN].to_numpy().astype(np.float64)


def groups(frame: pl.DataFrame, column: str) -> np.ndarray:
    """The grouping labels used by the leave-one-out splitters."""
    return frame[column].to_numpy()
