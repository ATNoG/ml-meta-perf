# Federated OCPP 1.6 IDS APP

Tabular application-layer representation of the Federated OCPP 1.6 Intrusion Detection Dataset. The problem separates normal traffic from four families of attacks against electric-vehicle charging infrastructure.

## Source and reproducibility

- Official source: https://zenodo.org/records/14887131
- Original file: https://zenodo.org/records/14887131/files/Balanced_OCPP16_APP_Layer.zip?download=1
- Dataset DOI: https://doi.org/10.21227/v1f0-9t13
- Associated publication: https://doi.org/10.48550/arXiv.2502.01569
- Official, verified MD5 of the ZIP: `CBFAAB0941629CB2CB093AA7EF7891E8`
- SHA-256 of the prepared CSV: `A343A11A556A9A169EEFE3641EE4F05B09CE10BB1B92BE4593845D4AC1748619`
- File ready for the experiment: `federated_ocpp16_app_ids.csv`

## Prepared schema

- Rows: **4,255**.
- Columns: **42** (41 predictors and the target).
- Classification target: `label`.
- Classes: three with 863 rows, `normal` with 861 and `cyberattack_ocpp16_fdi_chargingprofile` with 805.

## Preparation applied

- Only `Combined/Train.csv` and `Combined/Test.csv` were joined. The `Client_1` to `Client_3` partitions are federated views of the same corpus and were not counted again.
- `flow_id`, source/destination IP and start/end timestamps were removed.
- Eight constant variables were removed.
- 60 exact repetitions that appeared after removing the identifiers were dropped; there were no predictor signatures with contradictory labels.
- Ports were divided by 65,535; duration was converted from microseconds to seconds; rates were converted to units per second. `log1p` was then applied to non-negative magnitudes. These are fixed transformations, with no full-dataset statistics and no train/test leakage.
- The source is the "Balanced" variant of the corpus.
- The preparation was done by hand; the steps above and the SHA-256 are its only record. A script that reproduces the prepared CSV from the original archive is still missing.

## Local validation

- 0 missing values, 0 infinities and 0 duplicate rows.
- All predictors are numeric and finite; after the transformation they lie in a compact range.
- The stratified 80/20 split leaves 3,404 training rows.
- The minority class leaves 644 rows in train and 161 in test with `random_state=5`.
- Compatible with five stratified folds; with fewer than 10,000 rows, models with a limited context do not resample it.
