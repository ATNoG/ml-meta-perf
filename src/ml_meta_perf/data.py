"""Loading and shaping of the meta-dataset.

The meta-dataset has one row per (dataset, model) pair. Two groups of columns matter
and they behave very differently, which is why they are named separately here:

* dataset features are constant across every row of a given dataset, so on their own
  they can only ever predict a per-dataset constant;
* model features vary with the model, and one of them -- ``Processing Units Number`` --
  also varies with the dataset, since it is a function of the dataset's shape. The
  meta-dataset pipeline is the source of this schema; this module loads the generated
  corpus and keeps the package-level column contract explicit.

That asymmetry is the whole point of the two-equation comparison, so the split is
part of the public API rather than something each caller re-derives.

**Why there are eighteen features when the equation uses twelve.**

The two numbers answer different questions, asked at different stages, and reading the
second as a criticism of the first is the most natural mistake to make about this study.

*Designing the corpus* comes first, before any equation exists and before anyone knows
which terms will be worth having. The requirement there is **identification**: the
features must name every dataset and every learner the corpus contains, because two rows
sharing a feature vector are two rows no equation over those features can ever tell
apart, and a difference between them is then unexplainable rather than merely
unexplained. The right move at that stage is a wide feature set chosen for coverage. This
corpus meets the requirement exactly -- the twelve dataset features give twenty distinct
vectors for twenty datasets, and the six model features separate all twenty-five learners
on every dataset, with zero ambiguous rows of 476. ``tests/test_model_features.py`` pins
both.

Two caveats belong with that claim rather than after it. The model side is **joint**, not
standalone: five of the six are constant per model and separate only 19 of the 25 on their
own -- ``FT-Transformer``/``TabNet``/``TabTransformer``, ``LightGBM_RF``/``XGBoost``,
``DNN``/``MLP``, ``TabICL``/``TabPFN`` and ``BernoulliNB``/``GaussianNB`` collide -- and it
is ``Processing Units Number``, which varies with the dataset, that breaks those ties. So
the six identify a learner *on a given dataset*. And identification was bought with
redundancy: ``nr_attr`` and ``nr_outliers`` correlate at 0.9995, and
``log(inst_to_attr) + log(nr_attr)`` **is** ``log(nr_inst)`` to 2e-15, because
``inst_to_attr`` is defined as their ratio.

*Fitting the equation* comes second, and its criterion is not coverage but **compression**.
An equation is a statement about families of datasets and families of learners, not about
individuals, so it is expected to need fewer features as it gets better -- and the
redundancy above is part of what it is compressing away. E3 uses twelve of the eighteen.
That is the mechanism working, not a shortfall in the corpus, and an equation that used all
eighteen would be one that had failed to generalise.

So a column may earn its place at either stage. ``Solution Stochasticity`` and
``Input Distribution Modelling`` remain in the six-column corpus because the complete
descriptor tuple is used for learner identification. **Do not read absence from the equation
as evidence against a feature.** The corrected-corpus sweep selected a four-feature
model-side subset for E3, while the full schema remains available for identification and
future analyses.

**``nr_inst`` describes the source dataset, not the training set.** Every model was
trained on a stratified sample capped at 100,000 rows, and ten of the twenty datasets are
larger than that -- up to seven million. Nothing in the CSV records the sampled size,
because it is the same cap for every dataset above it. So ``nr_inst`` and
``inst_to_attr`` are properties of the corpus a dataset was drawn from, and no statement
about "more training data" can be tested against them.

Study chapter: [1. The dataset][study-chapter] -- the rationale, in
prose, with the figures.

[study-chapter]: https://github.com/mariolpantunes/ml-meta-perf/blob/main/assets/docs/01-dataset.md
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import polars as pl

from ml_meta_perf.model import MCC_LOWER, MCC_UPPER

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

#: The six generated columns an equation may use to describe a *learner*.
#:
#: This is the package-level schema expected in ``meta_dataset.csv``. The generation logic
#: lives in ``meta_dataset_pipeline``; the package keeps only the validated feature contract
#: used by loading, fitting, plotting and reporting.
MODEL_FEATURES: tuple[str, ...] = (
    "Processing Units Number",
    "Model Capability",
    "Solution Stochasticity",
    "Loss Margin Behaviour",
    "Input Distribution Modelling",
    "Fitting Regime",
)


@dataclasses.dataclass(frozen=True)
class Schema:
    """Which columns a meta-dataset's rows, groups, features and target live in."""

    dataset_column: str
    model_column: str
    dataset_features: tuple[str, ...]
    model_features: tuple[str, ...]
    target_column: str
    bounds: tuple[float, float] | None
    task_type: str | None = None
    log_target: bool = False

    def __post_init__(self) -> None:
        if self.log_target and self.bounds is not None:
            raise ValueError("log1p requires bounds=None; only the unbounded FlexFL cost targets support it")

    @property
    def features(self) -> tuple[str, ...]:
        return self.dataset_features + self.model_features

    @property
    def slug(self) -> str:
        base = self.target_column if self.task_type is None else f"{self.target_column}-{self.task_type}"
        return f"{base}-log1p" if self.log_target else base

    @property
    def label(self) -> str:
        return f"log1p({self.target_column})" if self.log_target else self.target_column

    def tag(self, name: str) -> str:
        """``name`` with a ``_log1p`` suffix when this schema fits a log-transformed target."""
        return f"{name}_log1p" if self.log_target else name


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

#: The four asserted mechanism ordinals, by learner.
#:
#: These are the **provenance** of four columns the CSV carries like any other feature. They
#: live here, and are checked against the CSV by ``test_data``, so the columns can be
#: regenerated and a reader can see where the numbers came from.
#:
#: Each is defined for all twenty-five learners, so none carries a "this learner has no such
#: thing" sentinel, and every rung is occupied and positive, so the whole grammar is defined
#: on all of them. That is what distinguishes them from the sixty-one hyperparameter
#: descriptors measured and rejected during corpus design: a hyperparameter a learner does not have
#: has no value, and encoding that absence as zero collapses applicability into magnitude.
#:
#: **They are asserted, not measured**, and carry chapter 4's caveat in full: a term over one
#: of them is evidence about the ordering claimed here, not about a quantity anyone observed.

#: How deep randomisation reaches into the fitted solution, low to high.
#:
#: 1 deterministic given the data | 2 stochastic optimisation (random init or shuffling) |
#: 3 randomised over training examples (bootstrap) | 4 randomised over features (subspace or
#: column subsampling) | 5 randomised over the split or parameter values themselves.
#:
#: Asserted from published descriptions of the learners: Breiman (1996) on bagging, Ho (1998)
#: on the random subspace method, Breiman (2001) on random forests, and Geurts, Ernst &
#: Wehenkel (2006) on extremely randomized trees. The top two rungs are what separate ``DT``
#: from ``ExtraTree`` and ``LightGBM_RF`` from ``LightGBM_ExtraTrees`` -- pairs that no
#: measured descriptor in this corpus, and none of sixty-one proposed ones, can tell apart.
SOLUTION_STOCHASTICITY: dict[str, int] = {
    "BernoulliNB": 1,
    "DT": 1,
    "GaussianNB": 1,
    "KNN": 1,
    "LDA": 1,
    "LR": 1,
    "LinearSVC": 1,
    "QDA": 1,
    "Ridge": 1,
    "DNN": 2,
    "FT-Transformer": 2,
    "MLP": 2,
    "PassiveAggressive": 2,
    "Perceptron": 2,
    "SGD": 2,
    "TabICL": 2,
    "TabNet": 2,
    "TabPFN": 2,
    "TabTransformer": 2,
    "AdaBoost": 3,
    "Bagging": 3,
    "LightGBM_RF": 4,
    "XGBoost": 4,
    "ExtraTree": 5,
    "LightGBM_ExtraTrees": 5,
}


#: How hard the objective penalises points far from the decision boundary.
#:
#: 1 squared error or an impurity criterion | 2 logistic / cross-entropy | 3 exponential |
#: 4 hinge | 5 the perceptron criterion.
#:
#: The standard robustness ordering over losses. This is the weakest claim of the four --
#: placing the perceptron criterion above hinge is a choice rather than a consensus -- and
#: chapter 4's caveat about ``Model Capability`` applies here with more force.
LOSS_MARGIN_BEHAVIOUR: dict[str, int] = {
    "Bagging": 1,
    "DT": 1,
    "ExtraTree": 1,
    "KNN": 1,
    "LDA": 1,
    "QDA": 1,
    "Ridge": 1,
    "BernoulliNB": 2,
    "DNN": 2,
    "FT-Transformer": 2,
    "GaussianNB": 2,
    "LR": 2,
    "LightGBM_ExtraTrees": 2,
    "LightGBM_RF": 2,
    "MLP": 2,
    "TabICL": 2,
    "TabNet": 2,
    "TabPFN": 2,
    "TabTransformer": 2,
    "XGBoost": 2,
    "AdaBoost": 3,
    "LinearSVC": 4,
    "PassiveAggressive": 4,
    "SGD": 4,
    "Perceptron": 5,
}


#: How much of P(x) the learner commits to modelling.
#:
#: 1 discriminative, models P(y|x) only | 2 instance-based, retains the sample and models no
#: density | 3 generative assuming conditional independence | 4 generative with a shared
#: covariance | 5 generative with a per-class covariance.
#:
#: Ng & Jordan (2001) on discriminative versus generative classifiers, and the classical
#: LDA/QDA covariance hierarchy. **Lopsided**: rung 1 holds 376 of 476 rows and the other four
#: hold 20-40 each, so three of its five rungs sit below the ten-percent-of-rows floor the
#: z-score cap imposes elsewhere. Every value is a real reading -- there is no
#: not-applicable sentinel -- but a term over this column speaks mostly about one rung.
INPUT_DISTRIBUTION_MODELLING: dict[str, int] = {
    "AdaBoost": 1,
    "Bagging": 1,
    "DNN": 1,
    "DT": 1,
    "ExtraTree": 1,
    "FT-Transformer": 1,
    "LR": 1,
    "LightGBM_ExtraTrees": 1,
    "LightGBM_RF": 1,
    "LinearSVC": 1,
    "MLP": 1,
    "PassiveAggressive": 1,
    "Perceptron": 1,
    "Ridge": 1,
    "SGD": 1,
    "TabICL": 1,
    "TabNet": 1,
    "TabPFN": 1,
    "TabTransformer": 1,
    "XGBoost": 1,
    "KNN": 2,
    "BernoulliNB": 3,
    "GaussianNB": 3,
    "LDA": 4,
    "QDA": 5,
}


#: How the parameters are reached.
#:
#: 1 closed form | 2 batch iterative | 3 mini-batch stochastic | 4 per-sample online |
#: 5 amortised, fitted in-context at prediction time.
FITTING_REGIME: dict[str, int] = {
    "BernoulliNB": 1,
    "GaussianNB": 1,
    "KNN": 1,
    "LDA": 1,
    "QDA": 1,
    "Ridge": 1,
    "AdaBoost": 2,
    "Bagging": 2,
    "DT": 2,
    "ExtraTree": 2,
    "LR": 2,
    "LightGBM_ExtraTrees": 2,
    "LightGBM_RF": 2,
    "LinearSVC": 2,
    "XGBoost": 2,
    "DNN": 3,
    "FT-Transformer": 3,
    "MLP": 3,
    "SGD": 3,
    "TabNet": 3,
    "TabTransformer": 3,
    "PassiveAggressive": 4,
    "Perceptron": 4,
    "TabICL": 5,
    "TabPFN": 5,
}

#: The four ordinals by name, for regeneration and for the test that checks them.
MODEL_ORDINALS: dict[str, dict[str, int]] = {
    "Solution Stochasticity": SOLUTION_STOCHASTICITY,
    "Loss Margin Behaviour": LOSS_MARGIN_BEHAVIOUR,
    "Input Distribution Modelling": INPUT_DISTRIBUTION_MODELLING,
    "Fitting Regime": FITTING_REGIME,
}

ALL_FEATURES: tuple[str, ...] = DATASET_FEATURES + MODEL_FEATURES
MCC_SCHEMA = Schema(
    dataset_column=DATASET_COLUMN,
    model_column=MODEL_COLUMN,
    dataset_features=DATASET_FEATURES,
    model_features=MODEL_FEATURES,
    target_column=TARGET_COLUMN,
    bounds=(MCC_LOWER, MCC_UPPER),
)

FLEXFL_DATASET_COLUMN = "dataset"
FLEXFL_MODEL_COLUMN = "fl_algo"
FLEXFL_TARGETS: tuple[str, ...] = (
    "performance", "total_time_s", "comm_bytes_total", "n_epochs", "compute_time_total_s",
    "compute_time_max_s", "comm_time_total_s", "validation_time_s",
)
FLEXFL_COST_TARGETS: tuple[str, ...] = (
    "total_time_s", "comm_bytes_total", "compute_time_total_s", "compute_time_max_s",
    "comm_time_total_s", "validation_time_s",
)
TASK_TYPES: tuple[str, ...] = ("classification", "regression")
FLEXFL_DATASET_FEATURES: tuple[str, ...] = (
    "n_samples",
    "n_features",
    "n_classes",
    "is_classification",
    "is_categorical",
    "total_parameters",
    "n_layers",
    "mean_layer_width",
    "max_layer_width",
    "weight_decay",
)
FLEXFL_MODEL_FEATURES: tuple[str, ...] = (
    "fl_algo_CentralizedSync",
    "fl_algo_CentralizedAsync",
    "fl_algo_DecentralizedSync",
    "fl_algo_DecentralizedAsync",
    "strategy_iid",
    "strategy_non_iid",
    "strategy_dirichlet",
    "learning_rate",
    "batch_size",
    "patience",
    "delta",
    "local_epochs",
    "alpha",
    "distribution_percentage",
    "feat_entropy_mean",
    "feat_entropy_min",
    "feat_entropy_max",
    "feat_entropy_std",
    "num_workers",
    "n_atnog_test1",
    "n_hobbit",
    "n_samwise",
    "worker_rate_mean",
    "worker_rate_min",
    "worker_rate_max",
    "worker_rate_std",
    "worker_rate_cv",
)
SMAPE_LOWER = 0.0
SMAPE_UPPER = 2.0
FLEXFL_PERFORMANCE_BOUNDS: dict[str, tuple[float, float]] = {
    "classification": (MCC_LOWER, MCC_UPPER),
    "regression": (SMAPE_LOWER, SMAPE_UPPER),
}


def flexfl_schema(target: str, task_type: str | None = None, log_target: bool = False) -> Schema:
    """The FlexFL per-run schema for one target, optionally restricted to one task type and fitted on ``log1p``."""
    if target not in FLEXFL_TARGETS:
        raise ValueError(f"unknown FlexFL target {target!r}; expected one of {FLEXFL_TARGETS}")
    if task_type is not None and task_type not in TASK_TYPES:
        raise ValueError(f"unknown task type {task_type!r}; expected one of {TASK_TYPES}")
    if target == "performance" and task_type is None:
        raise ValueError("the performance target needs a task type: MCC and SMAPE do not share a scale")
    if log_target and target not in FLEXFL_COST_TARGETS:
        raise ValueError(f"log1p applies only to the cost targets {FLEXFL_COST_TARGETS}")
    bounds = FLEXFL_PERFORMANCE_BOUNDS[task_type] if target == "performance" and task_type is not None else None
    return Schema(
        dataset_column=FLEXFL_DATASET_COLUMN,
        model_column=FLEXFL_MODEL_COLUMN,
        dataset_features=FLEXFL_DATASET_FEATURES,
        model_features=FLEXFL_MODEL_FEATURES,
        target_column=target,
        bounds=bounds,
        task_type=task_type,
        log_target=log_target,
    )

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

#: Plain-language readings of each feature, used wherever a fitted equation is turned into
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
    "Processing Units Number": "model capacity (number of fitted processing units)",
    "Model Capability": "learner family's capability rank in the tabular-ML literature (1-10)",
    "Solution Stochasticity": "how deep randomisation reaches into the fit (1-5)",
    "Loss Margin Behaviour": "how hard the loss penalises points far from the boundary (1-5)",
    "Input Distribution Modelling": "how much of the input distribution the learner models (1-5)",
    "Fitting Regime": "how the parameters are reached, closed form to in-context (1-5)",
}

#: The shipped corpus, resolved next to this module rather than relative to a source
#: checkout. `pip install ml-meta-perf && ml-meta-perf` has no repository around it, and the previous
#: form -- ``parents[2] / "data"`` -- pointed inside `site-packages` and failed there.
DEFAULT_PATH = Path(__file__).resolve().parent / "meta_dataset.csv"


class SchemaError(ValueError):
    """The CSV does not carry the columns the meta-model needs, or a value is out of domain for its transform."""


def load(path: str | Path | None = None, schema: Schema = MCC_SCHEMA) -> pl.DataFrame:
    """Read the meta-dataset and check it has the expected shape.

    The schema selects the required columns and optional task type.
    Every feature is cast to Float64. The integer-valued columns (``nr_attr``,
    ``nr_class``, ...) are still continuous as far as the equations are concerned,
    and carrying two dtypes through the term library buys nothing.
    """
    resolved = Path(path) if path is not None else DEFAULT_PATH
    if not resolved.is_file():
        raise SchemaError(f"meta-dataset not found: {resolved}")

    frame = pl.read_csv(resolved)
    if schema.task_type is not None:
        if "is_classification" not in frame.columns:
            raise SchemaError("missing columns: ['is_classification']")
        wanted = schema.task_type == "classification"
        frame = frame.filter(pl.col("is_classification").cast(pl.Boolean) == wanted)
        if frame.is_empty():
            raise SchemaError(f"meta-dataset has no {schema.task_type} rows")
    expected = (schema.dataset_column, schema.model_column, *schema.features, schema.target_column)
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    frame = frame.select(expected).with_columns(
        [pl.col(column).cast(pl.Float64) for column in (*schema.features, schema.target_column)]
    )

    nulls = sum(frame.null_count().row(0))
    if nulls:
        raise SchemaError(f"meta-dataset contains {nulls} null values")
    return frame


def aggregate_by_dataset(frame: pl.DataFrame, schema: Schema = MCC_SCHEMA) -> pl.DataFrame:
    """Collapse to one row per schema dataset: its features plus its mean target.

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
        frame.group_by(schema.dataset_column)
        .agg(
            [pl.col(column).first() for column in schema.dataset_features]
            + [
                pl.col(schema.target_column).mean().alias(schema.target_column),
                pl.len().alias("n_models"),
            ]
        )
        .sort(schema.dataset_column)
    )


def columns_as_arrays(frame: pl.DataFrame, features: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Extract the named columns as a mapping of float arrays, the form terms consume."""
    return {name: frame[name].to_numpy().astype(np.float64) for name in features}


def target(frame: pl.DataFrame, schema: Schema = MCC_SCHEMA) -> np.ndarray:
    """The schema's target column as a float array, ``log1p`` of it when the schema asks."""
    values = frame[schema.target_column].to_numpy().astype(np.float64)
    if not schema.log_target:
        return values
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise SchemaError(f"{schema.target_column} has negative or non-finite values; log1p needs non-negative costs")
    return np.log1p(values)


def groups(frame: pl.DataFrame, column: str) -> np.ndarray:
    """The grouping labels used by the leave-one-out splitters."""
    return frame[column].to_numpy()


def drop_constant_features(frame: pl.DataFrame, schema: Schema) -> Schema:
    """The schema without the features that take a single value in this frame.

    A constant feature cannot explain anything, and a term dividing it by a varying one
    survives the library's constant-term screen as a relabelled copy of that other feature.
    """
    constant = {name for name in schema.features if frame[name].n_unique() <= 1}
    return dataclasses.replace(
        schema,
        dataset_features=tuple(name for name in schema.dataset_features if name not in constant),
        model_features=tuple(name for name in schema.model_features if name not in constant),
    )


def corpus_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """The shape of the corpus: how many rows, groups, features, and how many cells absent.

    Every fact a chapter would otherwise state about the size of the meta-dataset, computed
    from the file rather than transcribed from it. Counts only -- the distribution of the
    target is `target_summary`, kept separate so each table can print at its own precision
    and neither has to render a count as ``476.0000``.
    """
    datasets = int(frame[DATASET_COLUMN].n_unique())
    models = int(frame[MODEL_COLUMN].n_unique())
    return pl.DataFrame(
        [
            {"quantity": "rows", "count": frame.height},
            {"quantity": "datasets", "count": datasets},
            {"quantity": "models", "count": models},
            {"quantity": "cells absent from the dataset-by-model grid", "count": datasets * models - frame.height},
            {"quantity": "dataset features", "count": len(DATASET_FEATURES)},
            {"quantity": "model features", "count": len(MODEL_FEATURES)},
        ],
        schema={"quantity": pl.String, "count": pl.Int64},
    )


def target_summary(frame: pl.DataFrame) -> pl.DataFrame:
    """How MCC is distributed across the corpus, including how much of it is pinned.

    The counts at exactly 0 and exactly 1 are here because they are the reason MAE and not
    SMAPE is the reported error, and a reader checking that argument should be able to see
    the counts it rests on. Together they are one fifth of the corpus; the single negative
    row is reported separately.

    Not included, because it cannot be: the variation across the five seeds from which each
    row's maximum was selected. That lives upstream, in the corpus builder, and chapter 1
    cites it as an external audit rather than pretending this file can recompute it.
    """
    values = target(frame)
    pinned = [("at exactly 1", values == 1.0), ("at exactly 0", values == 0.0), ("below 0", values < 0.0)]
    rows: list[dict[str, object]] = [
        {"quantity": "mean", "MCC": float(values.mean()), "rows": frame.height},
        {"quantity": "standard deviation", "MCC": float(values.std()), "rows": frame.height},
        {"quantity": "minimum", "MCC": float(values.min()), "rows": int((values == values.min()).sum())},
        {"quantity": "maximum", "MCC": float(values.max()), "rows": int((values == values.max()).sum())},
    ]
    rows += [{"quantity": label, "MCC": float("nan"), "rows": int(mask.sum())} for label, mask in pinned]
    return pl.DataFrame(rows, schema={"quantity": pl.String, "MCC": pl.Float64, "rows": pl.Int64})


def missing_cells(frame: pl.DataFrame) -> pl.DataFrame:
    """Which datasets are short of models, and how large those datasets are.

    The 24 absent cells are **not** missing at random, and the table says so on its face
    rather than in a sentence beside it: the datasets with models missing are the smallest
    ones in the corpus. Reported per dataset with its instance count, so a reader can see the
    pattern instead of being told about it.
    """
    models = int(frame[MODEL_COLUMN].n_unique())
    counts = (
        frame.group_by(DATASET_COLUMN)
        .agg(pl.len().alias("models_present"), pl.col("nr_inst").first().alias("nr_inst"))
        .filter(pl.col("models_present") < models)
        .sort("nr_inst")
    )
    return counts.with_columns((models - pl.col("models_present")).alias("models_absent")).select(
        pl.col(DATASET_COLUMN).alias("dataset"), "nr_inst", "models_absent"
    )
