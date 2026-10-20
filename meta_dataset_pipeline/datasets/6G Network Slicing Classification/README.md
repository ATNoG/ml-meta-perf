# 6G Network Slicing Classification

Tabular 6G network-slicing classification corpus, associated with the article by Botez et al. It is kept as a single dataset; its five slice types are classes of the same problem.

## Source and reproducibility

- Official source: https://github.com/robertbotez/6g-network-slicing-dataset
- Original CSV: https://raw.githubusercontent.com/robertbotez/6g-network-slicing-dataset/main/network_slicing_dataset%20-%20v3.csv
- Publication: https://doi.org/10.3390/electronics14020368
- SHA-256 of the downloaded original CSV: `D7CD452F816411BB117BD61B46FFD0C713F5422FC65581469EAEA9A8EB8C499D`
- SHA-256 of the prepared CSV: `319C1EC357D9DA8056CF70E489241D23DC2CA7E200829C3F050B01015BE6BDEB`
- File ready for the experiment: `6g_network_slicing_classification.csv`

## Prepared schema

- Rows: **10,000**.
- Columns: **11** (10 predictors and the target).
- Classification target: `Slice Type`.
- Classes: `ERLLC` 2,262; `feMBB` 2,237; `umMTC` 2,204; `MBRLLC` 2,203; `mURLLC` 1,094.

## Preparation applied

- The `;` separator and decimal commas were converted to the standard CSV the project expects.
- `Use Case Type` was removed because it directly determines `Slice Type` and would leak the target.
- `Slice Handover` was removed because it behaves as a nearly deterministic co-target.
- `Required Mobility` and `Required Connectivity` were encoded as `no=0`, `yes=1`.
- Magnitudes in nanoseconds were expressed in milliseconds and rates in Gbps were expressed in Tbps. No statistic of the full dataset was used for scaling, so no train/test leakage is introduced.
- The preparation was done by hand; the steps above and the two hashes are its only record. A script that reproduces the prepared CSV from the original one is still missing.

## Local validation

- 0 missing values, 0 infinities and 0 duplicate rows.
- All predictors are numeric and finite; the maximum absolute value is 10.
- The stratified 80/20 split leaves 8,000 training rows.
- The minority class leaves 875 rows in train and 219 in test with `random_state=5`.
- Compatible with five stratified folds and with the internal context of up to 10,000 rows.
