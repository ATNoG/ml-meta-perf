"""The MCC study's data description as the tests read it: from ``config/study.json``.

The tests used to import these names from ``ml_meta_perf.data``, where they were constants. They
are the shipped study file's columns and features now, so a test that needs "the dataset features"
asks the file, as the code does.
"""

from ml_meta_perf.config import default_schema

SCHEMA = default_schema()
DATASET_COLUMN = SCHEMA.dataset_column
MODEL_COLUMN = SCHEMA.model_column
TARGET_COLUMN = SCHEMA.target_column
DATASET_FEATURES = SCHEMA.dataset_features
MODEL_FEATURES = SCHEMA.model_features
ALL_FEATURES = SCHEMA.features
