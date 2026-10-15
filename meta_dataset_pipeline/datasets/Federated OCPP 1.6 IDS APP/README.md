# Federated OCPP 1.6 IDS APP

Representación tabular de capa de aplicación del corpus Federated OCPP 1.6 Intrusion Detection Dataset. El problema distingue tráfico normal y cuatro familias de ataques contra infraestructura de carga de vehículos eléctricos.

## Fuente y reproducibilidad

- Fuente oficial: https://zenodo.org/records/14887131
- Archivo original: https://zenodo.org/records/14887131/files/Balanced_OCPP16_APP_Layer.zip?download=1
- DOI del dataset: https://doi.org/10.21227/v1f0-9t13
- Publicación asociada: https://doi.org/10.48550/arXiv.2502.01569
- MD5 oficial/verificado del ZIP: `CBFAAB0941629CB2CB093AA7EF7891E8`
- SHA-256 del CSV preparado: `A343A11A556A9A169EEFE3641EE4F05B09CE10BB1B92BE4593845D4AC1748619`
- Archivo listo para el experimento: `federated_ocpp16_app_ids.csv`

## Esquema preparado

- Filas: **4.255**.
- Columnas: **42** (41 predictores y el target).
- Target de clasificación: `label`.
- Clases: tres con 863 filas, `normal` con 861 y `cyberattack_ocpp16_fdi_chargingprofile` con 805.

## Preparación aplicada

- Se unieron solamente `Combined/Train.csv` y `Combined/Test.csv`. Las particiones `Client_1` a `Client_3` son vistas federadas del mismo corpus y no se volvieron a contar.
- Se retiraron `flow_id`, IP de origen/destino y timestamps de inicio/fin.
- Se eliminaron ocho variables constantes.
- Se eliminaron 60 repeticiones exactas que aparecían tras retirar identificadores; no había firmas de predictores con etiquetas contradictorias.
- Los puertos se dividieron por 65.535; la duración se convirtió de microsegundos a segundos; las tasas se convirtieron a unidades por segundo. Después se aplicó `log1p` a magnitudes no negativas. Son transformaciones fijas, sin estadísticos del conjunto completo y sin fuga entre train y test.

## Validación local

- 0 valores faltantes, 0 infinitos y 0 filas duplicadas.
- Todos los predictores son numéricos y finitos; después de la transformación quedan en un rango compacto.
- El split estratificado 80/20 deja 3.404 filas de entrenamiento.
- La clase minoritaria deja 644 filas en train y 161 en test con `random_state=5`.
- Compatible con cinco folds estratificados; al tener menos de 10.000 filas, los modelos con contexto limitado no vuelven a muestrearlo.
