"""
Script to run the experiment.
"""
import ast
import os
from pathlib import Path

import pandas as pd

PIPELINE_DIR = Path(__file__).resolve().parent
os.chdir(PIPELINE_DIR)

from utils.dataset_utils import dataset_description_header

# opening the dataset with the ML results (results_stage_ml_eval)
df_ml_results = pd.read_csv("results/results_stage_ml_eval.csv")

# opening the dataset with the meta-features (results_stage_dataset_desc)
df_meta_features = pd.read_csv("results/results_stage_dataset_desc.csv")


df_meta_dataset = pd.merge(
    df_ml_results,
    df_meta_features,
    on=["Dataset"],
    how="inner"
)


# /// Auxiliary functions \\

def parse_hyperparameters(hyperparameters):
    """
    Parse the format saved by model_utils.py.

    Tuned configurations are saved without their outer braces, while models
    without tunable parameters are saved as ``unique``.
    """
    if isinstance(hyperparameters, dict):
        return hyperparameters

    if hyperparameters is None or pd.isna(hyperparameters):
        return {}

    hyperparameters = str(hyperparameters).strip()

    if hyperparameters in [
        "",
        "unique",
        "skipped_dataset_too_small",
        "skipped_tuning_error"
    ]:
        return {}

    if hyperparameters.startswith("{") and hyperparameters.endswith("}"):
        parsed_hyperparameters = ast.literal_eval(hyperparameters)
    else:
        parsed_hyperparameters = ast.literal_eval("{" + hyperparameters + "}")

    if not isinstance(parsed_hyperparameters, dict):
        raise ValueError(
            f"Hyperparameters {hyperparameters!r} could not be parsed as a dictionary."
        )

    return parsed_hyperparameters


def get_maximum_tree_nodes(max_depth, no_instances):
    """
    Return the maximum feasible number of nodes in a binary tree.

    max_depth=None is valid for DT and ExtraTree. In that case, the maximum
    number of nodes is bounded using the number of training instances.
    """
    no_instances = max(1, int(no_instances))
    maximum_nodes_given_instances = 2 * no_instances - 1

    if max_depth is None:
        return maximum_nodes_given_instances

    maximum_nodes_given_depth = 2 ** (int(max_depth) + 1) - 1
    return min(maximum_nodes_given_depth, maximum_nodes_given_instances)


# /// Adding Model's related columns \\
#
# The output schema mirrors ml_meta_perf/meta_dataset.csv: one instance-specific
# capacity descriptor plus five canonical ordinal descriptors. "Model" itself
# is never used as a meta-model feature downstream, so these columns are the
# signal the meta-model receives about learner identity.

MODEL_FEATURES = (
    "Processing Units Number",
    "Model Capability",
    "Solution Stochasticity",
    "Loss Margin Behaviour",
    "Input Distribution Modelling",
    "Fitting Regime",
)

MODEL_FAMILY = {
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

MODEL_CAPABILITY = {
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

SOLUTION_STOCHASTICITY = {
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

LOSS_MARGIN_BEHAVIOUR = {
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

INPUT_DISTRIBUTION_MODELLING = {
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

FITTING_REGIME = {
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

MODEL_ORDINALS = {
    "Solution Stochasticity": SOLUTION_STOCHASTICITY,
    "Loss Margin Behaviour": LOSS_MARGIN_BEHAVIOUR,
    "Input Distribution Modelling": INPUT_DISTRIBUTION_MODELLING,
    "Fitting Regime": FITTING_REGIME,
}


def add_canonical_model_descriptors(frame):
    """
    Add the same model descriptors used by ml_meta_perf/meta_dataset.csv.
    """
    models = frame["Model"].astype(str)
    missing_family = sorted(set(models).difference(MODEL_FAMILY))
    if missing_family:
        raise ValueError(f"Models without canonical family mapping: {missing_family}")

    result = frame.copy()
    families = models.map(MODEL_FAMILY)
    result["Model Capability"] = families.map(MODEL_CAPABILITY).astype("int8")

    for column, mapping in MODEL_ORDINALS.items():
        missing = sorted(set(models).difference(mapping))
        if missing:
            raise ValueError(f"Models without {column} mapping: {missing}")
        result[column] = models.map(mapping).astype("int8")

    return result


### adding the Processing Units Number column ###

# Real parameter counts of the fixed pretrained checkpoints used for TabICL
# and TabPFN (utils/model_wrappers.py). Unlike every other model here,
# their capacity is not a function of any tunable hyperparameter - it is
# whatever the frozen backbone contains, so this is detected dynamically
# (by actually fitting a tiny dummy batch and counting real torch
# parameters - see utils/foundation_model_introspection.py) rather than
# hardcoded, so it stays correct automatically if the checkpoint changes.
# Environments without tabicl/tabpfn installed (for example, a machine that
# only runs this lightweight join stage) use the recorded counts below. Those
# constants must be refreshed whenever the corresponding checkpoint changes.
_FOUNDATION_MODEL_PARAM_COUNT_DEFAULTS = {
    "TabICL": 27552258,
    "TabPFN": 53153144,
}
_foundation_model_parameter_count_memo = {}


def get_foundation_model_parameter_count(model):
    """
    Returns the real parameter count of a pretrained foundation model
    checkpoint (TabICL, TabPFN), detecting it dynamically when possible and
    falling back to the recorded checkpoint count otherwise.
    """
    if model in _foundation_model_parameter_count_memo:
        return _foundation_model_parameter_count_memo[model]

    try:
        from utils.foundation_model_introspection import (
            detect_foundation_model_parameter_count,
        )
        count = detect_foundation_model_parameter_count(model)
    except ImportError:
        if model not in _FOUNDATION_MODEL_PARAM_COUNT_DEFAULTS:
            raise ValueError(
                f"{model} is not installed in this environment and no fallback "
                "parameter count is recorded for it."
            ) from None
        count = _FOUNDATION_MODEL_PARAM_COUNT_DEFAULTS[model]

    _foundation_model_parameter_count_memo[model] = count
    return count

def get_processing_units_number_given_model_and_hyperparameters(
        model,
        hyperparameters,
        no_class,
        no_features,
        no_instances):
    """
    Returns a model-size/active-unit proxy based on the model and its selected
    hyperparameters.
    """
    hyperparameters_as_dict = parse_hyperparameters(hyperparameters)

    no_class = max(1, int(no_class))
    no_features = max(1, int(no_features))
    no_instances = max(1, int(no_instances))

    if model in [
        "LR", "Ridge", "LinearSVC", "SGD", "Perceptron",
        "PassiveAggressive", "BernoulliNB"
    ]:
        return no_class * (no_features + 1)

    if model == "GaussianNB":
        return no_class * (2 * no_features + 1)

    if model == "LDA":
        covariance_parameters = no_features * (no_features + 1) // 2
        return no_class * no_features + covariance_parameters

    if model == "QDA":
        covariance_parameters = no_features * (no_features + 1) // 2
        return no_class * (no_features + covariance_parameters)

    if model in ["DT", "ExtraTree"]:
        return get_maximum_tree_nodes(
            hyperparameters_as_dict.get("max_depth"),
            no_instances
        )

    if model == "XGBoost":
        number_of_trees = int(hyperparameters_as_dict.get("n_estimators", 100))
        nodes_per_tree = get_maximum_tree_nodes(
            hyperparameters_as_dict.get("max_depth", 6),
            no_instances
        )
        return number_of_trees * nodes_per_tree

    if model in ["LightGBM_RF", "LightGBM_ExtraTrees"]:
        number_of_trees = int(hyperparameters_as_dict.get("n_estimators", 100))
        number_of_leaves = int(hyperparameters_as_dict.get("num_leaves", 31))
        max_depth = hyperparameters_as_dict.get("max_depth")

        if max_depth is not None:
            number_of_leaves = min(number_of_leaves, 2 ** int(max_depth))

        number_of_leaves = min(number_of_leaves, no_instances)
        nodes_per_tree = max(1, 2 * number_of_leaves - 1)
        return number_of_trees * nodes_per_tree

    if model == "AdaBoost":
        # The default sklearn base estimator is a depth-1 decision stump.
        return int(hyperparameters_as_dict.get("n_estimators", 50)) * 3

    if model == "Bagging":
        number_of_trees = int(hyperparameters_as_dict.get("n_estimators", 10))
        # The configured max_depth=160 is effectively unbounded, so use the
        # maximum tree size allowed by the sample.
        return number_of_trees * get_maximum_tree_nodes(160, no_instances)

    if model in ["MLP", "DNN"]:
        hidden_layer_sizes = hyperparameters_as_dict.get(
            "hidden_layer_sizes", (100,)
        )
        if isinstance(hidden_layer_sizes, int):
            hidden_layer_sizes = (hidden_layer_sizes,)
        return sum(hidden_layer_sizes)

    if model == "TabNet":
        return (
            int(hyperparameters_as_dict.get("n_d", 8))
            + int(hyperparameters_as_dict.get("n_a", 8))
        ) * int(hyperparameters_as_dict.get("n_steps", 3))

    if model == "TabTransformer":
        hidden_multipliers = hyperparameters_as_dict.get(
            "mlp_hidden_mults", (4, 2)
        )
        return no_features * sum(hidden_multipliers)

    if model == "FT-Transformer":
        return (
            int(hyperparameters_as_dict.get("n_blocks", 3))
            * int(hyperparameters_as_dict.get("d_block", 128))
        )

    if model in ["TabICL", "TabPFN"]:
        return get_foundation_model_parameter_count(model)

    if model == "KNN":
        return int(hyperparameters_as_dict.get("n_neighbors", 5)) * no_features

    raise ValueError(
        f"Model {model} not recognized for processing units calculation."
    )


df_meta_dataset["Processing Units Number"] = df_meta_dataset.apply(
    lambda row: get_processing_units_number_given_model_and_hyperparameters(
            row["Model"],
            row["Model Parameters"],
            row["nr_class"],
            row["nr_attr"],
            row["nr_inst"]
        )
    ,
    axis=1
)


df_meta_dataset = add_canonical_model_descriptors(df_meta_dataset)


### formating the meta-dataset ###
# setup_columns = ["Seed", "Dataset", "Model"]
setup_columns = ["Dataset", "Model"]
dataset_description_columns = dataset_description_header
model_description_columns = list(MODEL_FEATURES)
target_columns = ["MCC"]

columns_to_save = (
    setup_columns
    + dataset_description_columns
    + model_description_columns
    + target_columns
)

idx_best = df_meta_dataset.groupby(["Dataset", "Model"])[target_columns[0]].idxmax()
df_meta_dataset = df_meta_dataset.loc[idx_best]

df_meta_dataset = df_meta_dataset[columns_to_save].copy()

df_meta_dataset.to_csv("results/meta_dataset.csv", index=False)
