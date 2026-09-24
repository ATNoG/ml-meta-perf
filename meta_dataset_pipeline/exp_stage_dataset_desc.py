"""
Script for dataset description
"""

import gc
import os
import warnings
from pathlib import Path

import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

PIPELINE_DIR = Path(__file__).resolve().parent
os.chdir(PIPELINE_DIR)

from exp_setup import dataset_list
from utils.dataset_utils import describe_dataset_using_pymfe, read_csv_fast, preprocess_dataset

# Ignore warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
# Ignore warnings of type UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
# Ignore warnings of type ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
# Ignore warnings of type FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# Ignore warnings of type UserWarning
warnings.filterwarnings("ignore", category=UserWarning)


# loading the results dataset to fill the dataset descriptions
df_name = f"results/results_stage_ml_tune.csv"

results_df = pd.read_csv(df_name)

# keeping only unique datasets
unique_datasets = results_df['Dataset'].unique()

# counter
training_number = 1
number_of_datasets = len(unique_datasets)

print(f"Number of datasets to describe: {number_of_datasets}")

# descriptions accumulator
descriptions = []

# iterating over unique datasets
for dataset_name in unique_datasets:
    # searching for the useful_columns in the dataset_list
    useful_columns = None
    class_name = None
    for dataset_setup in dataset_list:
        if dataset_setup[0] == dataset_name:
            useful_columns = dataset_setup[1]
            class_name = dataset_setup[2]
            break

    # loading the dataset
    dataset_folder = f"datasets/{dataset_name}"
    df = read_csv_fast(
        f"{dataset_folder}/{[file for file in os.listdir(dataset_folder) if file.endswith('.csv')][0]}")

    if len(useful_columns) > 0:
        print(f"Removing columns {useful_columns}")
        df.drop(columns=useful_columns, inplace=True)

    print(f"\033[92m\nStarted execution with {dataset_name} ({df.shape[0]} rows).\nPreprocessing ...\033[0m")

    # Separate predictors and target exactly as in tuning/evaluation
    X = df.drop(columns=[class_name])
    y = df[class_name].to_numpy()

    # Apply exactly the same preprocessing used by the ML models
    X_processed, _ = preprocess_dataset(X, None)


    print(f"\033[92mDescribing ...\033[0m")

    attr_value: dict = describe_dataset_using_pymfe(
        X_processed,
        y
    )

    attr_value['Dataset'] = dataset_name
    descriptions.append(attr_value)

    training_number += 1

    # cleaning up
    del df
    gc.collect()


# building and exporting the results
desc_df = pd.DataFrame(descriptions)
desc_df.to_csv("results/results_stage_dataset_desc.csv", index=False)
