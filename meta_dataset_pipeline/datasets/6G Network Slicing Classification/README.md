# 6G Network Slicing Classification

Corpus tabular de clasificación de *network slicing* 6G asociado al artículo de Botez et al. Se conserva como un único dataset; sus cinco tipos de *slice* son clases del mismo problema.

## Fuente y reproducibilidad

- Fuente oficial: https://github.com/robertbotez/6g-network-slicing-dataset
- CSV original: https://raw.githubusercontent.com/robertbotez/6g-network-slicing-dataset/main/network_slicing_dataset%20-%20v3.csv
- Publicación: https://doi.org/10.3390/electronics14020368
- SHA-256 del CSV original descargado: `D7CD452F816411BB117BD61B46FFD0C713F5422FC65581469EAEA9A8EB8C499D`
- SHA-256 del CSV preparado: `319C1EC357D9DA8056CF70E489241D23DC2CA7E200829C3F050B01015BE6BDEB`
- Archivo listo para el experimento: `6g_network_slicing_classification.csv`

## Esquema preparado

- Filas: **10.000**.
- Columnas: **11** (10 predictores y el target).
- Target de clasificación: `Slice Type`.
- Clases: `ERLLC` 2.262; `feMBB` 2.237; `umMTC` 2.204; `MBRLLC` 2.203; `mURLLC` 1.094.

## Preparación aplicada

- Se convirtieron el separador `;` y las comas decimales al CSV estándar esperado por el proyecto.
- Se eliminó `Use Case Type` porque determina directamente `Slice Type` y produciría fuga del target.
- Se eliminó `Slice Handover` por comportarse como cotarget casi determinista.
- `Required Mobility` y `Required Connectivity` se codificaron como `no=0`, `yes=1`.
- Las magnitudes en nanosegundos se expresaron en milisegundos y las tasas en Gbps se expresaron en Tbps. No se usaron estadísticos del conjunto completo para escalar, por lo que no se introduce fuga entre train y test.

## Validación local

- 0 valores faltantes, 0 infinitos y 0 filas duplicadas.
- Todos los predictores son numéricos y finitos; el valor absoluto máximo es 10.
- El split estratificado 80/20 deja 8.000 filas de entrenamiento.
- La clase minoritaria deja 875 filas en train y 219 en test con `random_state=5`.
- Compatible con cinco folds estratificados y con el contexto interno de hasta 10.000 filas.
