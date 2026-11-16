# Resultados del ensamble 3: undersampling de variables

Corrida completa de `04_ensamble3_variables/ens3_variables.py` (commit `d383596`), del 14 al 16/11/2026, en una VM de la cátedra con 8 vCPU. Son 8 bolsas de 50 modelos (400 en total). Google interrumpió la máquina una vez y la corrida retomó desde el último modelo guardado.

## La respuesta

Con la regla de la consigna (corte elegido en 201908, ganancia en 201910), sortear el **75 %** de las variables por modelo le gana al semillero del mismo tamaño en los 5 bloques de 10 modelos, en los dos pasos. **Pero no ordena mejor a los clientes:** con el corte fijo del baseline, en AUC y en 201909, empata con el semillero. La ventaja sale de que los ensambles de subconjuntos eligen en 201908 un corte más alto, y en 201910 conviene enviar más (632 BAJA+2 contra 543 en 201908). Con el 25 % y el 50 %, los modelos son demasiado débiles y el ensamble no lo compensa.

## Comparación principal (10 modelos, 5 bloques pareados)

Diferencia media contra el semillero pareado, en millones, y bloques ganados:

| Bolsa | Ganancia 201910, corte de 201908 | Ganancia 201910, corte fijo 4.322 | Ganancia 201909 |
|---|---|---|---|
| V75-ff (paso 2) | **+8,3 (5/5)** | −0,7 (2/5) | −0,8 (3/5) |
| V50-ff (paso 2) | +3,6 (4/5) | −4,0 (1/5) | −5,4 (1/5) |
| V25-ff (paso 2) | −6,9 (1/5) | −14,5 (1/5) | −15,7 (0/5) |
| V75-1 (paso 1) | **+6,7 (5/5)** | −1,1 (2/5) | −0,4 (2/5) |
| V50-1 (paso 1) | **+5,4 (5/5)** | −2,6 (1/5) | −4,3 (1/5) |
| V25-1 (paso 1) | −7,0 (1/5) | −15,6 (1/5) | −17,5 (0/5) |

## Valores absolutos (paso 2, 10 modelos)

| | Ganancia 201910 | Corte fijo 4.322 | AUC 201910 | Ganancia 201909 | Corte elegido en 201908 |
|---|---|---|---|---|---|
| Un modelo (S-ff, media de 50 semillas) | 278,3 | 281,9 | 0,9350 | 250,3 | 3.803 |
| S-ff, semillero de 10 | 273,8 | 280,4 | 0,9355 | 251,8 | 3.378 |
| V75-ff | 282,1 | 279,8 | 0,9356 | 251,1 | 4.206 |
| V50-ff | 277,5 | 276,5 | 0,9342 | 246,5 | 4.270 |
| V25-ff | 267,0 | 265,9 | 0,9276 | 236,2 | 3.732 |

## Diversidad y costo (paso 2)

| Bolsa | Correlación media entre modelos | AUC de un modelo solo | Minutos por modelo (8 vCPU) |
|---|---|---|---|
| S-ff | 0,996 | 0,9350 | 4,4 |
| V75-ff | 0,961 | 0,9325 | 3,5 |
| V50-ff | 0,911 | 0,9274 | 2,4 |
| V25-ff | 0,832 | 0,9134 | 1,5 |

El paso 1 (`feature_fraction` = 1) da el mismo patrón.

## Archivos

| Archivo | Qué tiene |
|---|---|
| `resumen.json` | la comparación principal (k = 10) y la correlación media de cada bolsa |
| `comparacion.csv` | cada técnica contra su semillero, para cada k y cada métrica: bloques ganados, diferencias y veredicto |
| `curva.csv` | media y rango de cada métrica por bolsa y tamaño de ensamble |
| `bloques.csv` | cada ensamble (bolsa, k, bloque) con sus semillas, corte y métricas |
| `modelos.csv` | cada uno de los 400 modelos: semilla, iteraciones, AUC de 201908, columnas y minutos |

Las predicciones (400 modelos × 4 meses) no están en el repo: quedan en el bucket, en `exp/ens3_variables/predicciones/`. La bolsa `S-ff` (50 semillas del modelo del baseline, con early stopping) sirve como semillero para el ensamble 1 y como comparación para el ensamble 2.
