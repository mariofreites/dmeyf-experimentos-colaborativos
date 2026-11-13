# Resultados del baseline

Estos son los valores contra los que se comparan todos los experimentos, y los hiperparámetros que usan los ensambles. Salen de la corrida completa de `01_baseline/baseline.py` (commit `d8e9caf`), del 13/11/2026, con el dataset `competencia_02_crudo.csv.gz` completo. Usó LightGBM 4.7.0 y Optuna 5.0.0 con 4 hilos.

## El número a superar

| | |
|---|---|
| **Ganancia en 201910** | **280,4 millones** |
| Envíos (corte elegido en 201908) | 4.322 |
| AUC en 201908 (validación) | 0,9329 |
| AUC en 201910 (test) | 0,9344 |
| Modelo | corrida 48 de la bayesiana, semilla 101737 (`bo048_s101737`) |

**Variación entre semillas.** El mismo modelo con 10 semillas, sin promediarlas, da una ganancia media en 201910 de **278,9** con un **desvío de 2,9**. El mínimo es 271,8 y el máximo 282,8. Una diferencia de pocos millones contra el baseline queda dentro de lo que cambia con solo mover la semilla. Para afirmar que un ensamble mejora, la comparación tiene que superar esa dispersión. El detalle por semilla está en `modelos.csv`.

## Hiperparámetros (los que usan los ensambles)

| Hiperparámetro | Valor |
|---|---|
| `num_leaves` | 545 |
| `learning_rate` | 0,010474 |
| `min_data_in_leaf` | 2474 |
| `feature_fraction` | 0,756760 |
| `bagging_fraction` | 0,629812 |
| `bagging_freq` | 1 (fijo) |
| `max_bin` | 31 (fijo) |
| Iteraciones (early stopping en 201908) | 852 |

Los valores exactos, con todos los decimales y los parámetros fijos, están en `mejores_hiperparametros.json`. `comun.hiperparametros_baseline()` los lee de ese archivo, así que todos los experimentos usan los mismos sin volver a correr la bayesiana.

## Las 5 mejores corridas de la bayesiana

Todas con la semilla 101737 y el corte elegido en 201908 de cada una:

| Ranking | Corrida | AUC 201908 | `num_leaves` | `learning_rate` | `min_data_in_leaf` | `feature_fraction` | `bagging_fraction` | Iteraciones | Corte | Ganancia 201910 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 48 | 0,9329 | 545 | 0,0105 | 2474 | 0,757 | 0,630 | 852 | 4.322 | 280,4 |
| 2 | 33 | 0,9325 | 1388 | 0,0112 | 1436 | 0,732 | 0,760 | 487 | 3.397 | 280,6 |
| 3 | 22 | 0,9323 | 1998 | 0,0117 | 2663 | 0,717 | 0,612 | 762 | 4.594 | 281,8 |
| 4 | 39 | 0,9323 | 1627 | 0,0156 | 1497 | 0,642 | 0,484 | 532 | 4.776 | 281,2 |
| 5 | 5 | 0,9322 | 249 | 0,0131 | 2604 | 0,754 | 0,701 | 592 | 3.517 | 278,4 |

Las 5 están dentro de un desvío de semillas entre sí. La bayesiana encontró una meseta (learning rate de alrededor de 0,01, `min_data_in_leaf` de 1.400 a 2.700 y `feature_fraction` de alrededor de 0,7), no un óptimo puntual.

## Archivos

| Archivo | Qué tiene |
|---|---|
| `mejores_hiperparametros.json` | los 5 hiperparámetros, la mejor iteración, la semilla y los fijos |
| `resumen.json` | el baseline, la dispersión entre semillas y el resumen de las mejores configuraciones |
| `modelos.csv` | cada modelo (las 50 corridas y las semillas): AUC de 201907 a 201910, corte de 201908, ganancia de 201908 y 201910 |
| `corridas.csv` | las 50 corridas de la bayesiana, ordenadas por AUC de 201908 |
| `importancia_mejor_modelo.csv` | importancia (gain) de cada variable en el mejor modelo |
| `config.json`, `log.txt` | configuración y log completo de la corrida |

## Qué no está en el repo

Las **predicciones** de cada modelo para 201907 a 201910 (60 modelos, 126 MB en `.npy`) no están en el repo. Están en la carpeta compartida del proyecto, en `baseline_2019/predicciones/`, con la misma estructura que deja el script (`<mes>/_ids.parquet` y `<mes>/<modelo>.npy`). Las usan el semillero y la comparación contra semilleros del mismo tamaño.

**Semillas que faltan.** De las 4 configuraciones que siguen a la mejor no se corrieron sus 10 semillas: solo hay una semilla extra de la corrida 33. No cambian el baseline. Se completan relanzando el script, que retoma donde quedó.
