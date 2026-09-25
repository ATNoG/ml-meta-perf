"""
This file contains auxiliary methods sample, preprocess and generate the dataset description
"""

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer


def read_csv_fast(path: str) -> DataFrame:
    """
    Multi-threaded CSV reader using Polars.
    Falls back to pandas if Polars is unavailable.
    Eliminates DtypeWarning spam without requiring low_memory=False.
    """
    try:
        import polars as pl
        return pl.scan_csv(path, infer_schema_length=10_000).collect().to_pandas()
    except ImportError:
        return pd.read_csv(path, low_memory=False)


def limit_dataset_size(
        df: DataFrame,
        class_name: str,
        seed: int,
        max_size: int = 100000
) -> DataFrame:
    total_rows = df.shape[0]

    if class_name not in df.columns:
        raise ValueError(f"Class column '{class_name}' does not exist in df.")

    if total_rows <= max_size:
        return df

    print(f" - Limiting dataset size to {max_size} rows")

    try:
        import polars as pl
        fraction = max_size / total_rows
        lf = pl.from_pandas(df)
        sampled = (
            lf.group_by(class_name)
              .map_groups(lambda g: g.sample(fraction=fraction, seed=seed))
        )
        if len(sampled) > max_size:
            sampled = sampled.sample(n=max_size, seed=seed)
        return sampled.to_pandas().reset_index(drop=True)
    except ImportError:
        pass

    class_counts = df[class_name].value_counts(dropna=False)

    raw_sizes = class_counts / total_rows * max_size
    sampling_sizes = np.floor(raw_sizes).astype(int)

    remaining = max_size - sampling_sizes.sum()

    if remaining > 0:
        decimal_parts = raw_sizes - sampling_sizes
        classes_to_increment = decimal_parts.sort_values(ascending=False).index[:remaining]
        sampling_sizes.loc[classes_to_increment] += 1

    sampled_parts = []

    for class_value, group in df.groupby(class_name, dropna=False, sort=False):
        n_samples = int(sampling_sizes.loc[class_value])

        if n_samples <= 0:
            continue

        sampled_group = group.sample(
            n=min(len(group), n_samples),
            random_state=seed
        )

        sampled_parts.append(sampled_group)

    if not sampled_parts:
        return df.iloc[0:0].copy()

    df_sampled = pd.concat(sampled_parts, ignore_index=True)

    if df_sampled.shape[0] > max_size:
        df_sampled = df_sampled.sample(n=max_size, random_state=seed)

    return df_sampled.reset_index(drop=True)


def get_dataset_sample(
        df: DataFrame,
        seed: int,
        sample_percent: float,
        class_name: str,
        n_splits: int = 5
) -> DataFrame | None:

    if class_name not in df.columns:
        raise ValueError(f"Class column '{class_name}' does not exist in df.")

    if not 0 < sample_percent <= 1:
        raise ValueError("sample_percent must be in the interval (0, 1].")

    sampled_df = (
        df.groupby(class_name, group_keys=False, dropna=False)
          .sample(frac=sample_percent, random_state=seed)
          .sort_index()
    )

    original_classes = df[class_name].value_counts(dropna=False).index

    sampled_class_counts = (
        sampled_df[class_name]
        .value_counts(dropna=False)
        .reindex(original_classes, fill_value=0)
    )

    min_class_samples_after_sampling = sampled_class_counts.min()

    if min_class_samples_after_sampling >= n_splits:
        return sampled_df

    return None

MAX_ONEHOT_CATEGORIES = 5


def categorical_to_string(X):
    """
    Convert categorical values to string while preserving missing values as np.nan.
    This avoids sklearn encoder errors with mixed int/str columns.
    """
    X_df = pd.DataFrame(X).copy()

    for col in X_df.columns:
        missing_mask = X_df[col].isna()

        # Convert non-missing values to string
        X_df[col] = X_df[col].astype(str)

        # Restore real missing values
        X_df.loc[missing_mask, col] = np.nan

    return X_df.to_numpy(dtype=object)


def preprocess_dataset(x_train: DataFrame, x_test: DataFrame|None):
    x_train = x_train.copy()

    # Replace infinite values
    x_train = x_train.replace([np.inf, -np.inf], np.nan)

    if x_test is not None:
        x_test = x_test.copy()
        x_test = x_test.replace([np.inf, -np.inf], np.nan)

    # Numeric columns are only real numeric dtype columns
    numeric_columns = x_train.select_dtypes(include=[np.number]).columns.tolist()

    # Everything else is categorical: IPs, mixed int/str columns, text labels, etc.
    categorical_columns = [
        col for col in x_train.columns
        if col not in numeric_columns
    ]

    numeric_pipeline = Pipeline(steps=[
        (
            "imputer",
            SimpleImputer(strategy="mean")
        )
    ])

    categorical_pipeline = Pipeline(steps=[
        (
            "to_string",
            FunctionTransformer(
                categorical_to_string,
                validate=False,
                feature_names_out="one-to-one"
            )
        ),
        (
            "imputer",
            SimpleImputer(
                strategy="constant",
                fill_value="__MISSING__"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                max_categories=MAX_ONEHOT_CATEGORIES,
                sparse_output=False,
                dtype=np.float32
            )
        )
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                numeric_columns
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_columns
            )
        ],
        remainder="drop",
        verbose_feature_names_out=False,

        # Force dense output
        sparse_threshold=0.0
    )

    x_train_processed = preprocessor.fit_transform(x_train)

    x_train_processed = np.asarray(x_train_processed, dtype=np.float32)

    if x_test is not None:
        x_test_processed = preprocessor.transform(x_test)
        x_test_processed = np.asarray(x_test_processed, dtype=np.float32)
        return x_train_processed, x_test_processed

    return x_train_processed, None

# set with general dataset characteristics
dataset_description_header = ["class_ent", "eq_num_attr", "gravity", "inst_to_attr",
                              "nr_attr", "nr_bin", "nr_class", "nr_cor_attr", "nr_inst", "nr_norm",
                              "nr_outliers", "ns_ratio"]
# cat_to_num, num_to_cat, and nr_cat  was omitted because since the dataset was codified its value are 0
# nr_num was omitted because it is the same as nr_attr

def describe_dataset_using_pymfe(
    X,
    y
) -> dict:
    from pymfe.mfe import MFE

    X = np.asarray(X)
    y = np.asarray([
        "label " + str(value)
        for value in y
    ])

    mfe = MFE(
        groups=[
            "general",
            "info-theory",
            "statistical",
        ],
        features=dataset_description_header
    )

    mfe.fit(
        X,
        y,
        cat_cols="auto",
        transform_cat=None,
    )
    ft = mfe.extract()

    attr_value = {}

    for attr in dataset_description_header:
        if attr in ft[0]:
            # attr_values.append(ft[1][ft[0].index(attr)])
            attr_value[attr] = ft[1][ft[0].index(attr)]
        else:
            raise Exception(f"Attribute {attr} not found in the extracted features")

    return attr_value
