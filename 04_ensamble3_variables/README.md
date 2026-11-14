# Ensamble 3: undersampling de variables

**Pregunta:** ¿entrenar cada modelo con un subconjunto distinto de variables suma algo más de lo que LightGBM ya hace con `feature_fraction`?

## Diseño

Cada tratamiento es una **bolsa** de 50 modelos LightGBM, todos con los hiperparámetros del baseline (`resultados/baseline/mejores_hiperparametros.json`) y early stopping con el AUC de 201908.

| Paso | Bolsas con subconjuntos | Semillero de comparación | `feature_fraction` |
|---|---|---|---|
| 1. ¿Mejora por sí solo? | `V25-1`, `V50-1`, `V75-1` | `S-1` (todas las variables) | 1 en todas |
| 2. ¿Suma a LightGBM? | `V25-ff`, `V50-ff`, `V75-ff` | `S-ff` (todas las variables) | 0,757 (la del baseline) en todas |

- **Subconjuntos:** cada modelo sortea el 25 %, 50 % o 75 % de las 152 variables originales, y cada variable entra con su lag1, delta1, lag2 y delta2 (`Datos.columnas_al_azar`).
- **Semillas pareadas:** el modelo i usa la misma semilla en todas las bolsas. Fija el bagging de LightGBM y el sorteo de las variables, así el modelo i de `V25-1` y el de `S-1` difieren solo en las variables.
- **Ensambles:** promedio de probabilidades. Se arman sin reentrenar, en bloques disjuntos de k = 1, 5, 10, 25 y 50 modelos consecutivos (50, 10, 5, 2 y 1 bloques por bolsa).
- **Métrica:** la de la consigna. El corte se elige en 201908 sobre cada ensamble y la ganancia se mide en 201910. Además se informan la ganancia de 201910 con el corte fijo del baseline (4.322), el AUC de 201910, la ganancia de 201909 (mes que ningún modelo vio) con el corte de 201908, la correlación media entre los modelos de cada bolsa y los minutos de entrenamiento.
- **Criterio:** con k = 10, cada bloque se compara con el bloque del semillero que tiene las mismas semillas (5 pares). Ganar los 5 es "mejora": por azar pasaría 1 de cada 32 veces. 4 de 5 es un indicio no concluyente, 2 o 3 no se distingue del semillero, y 0 o 1 no aporta.

## Correrlo

```bash
cd ~/dmeyf-experimentos-colaborativos/04_ensamble3_variables

MODO_PRUEBA=1 python ens3_variables.py                      # prueba: 5 % de los clientes, 6 modelos por bolsa, unos minutos
nohup python ens3_variables.py --paso 1 > paso1.txt 2>&1 &  # paso 1 (4 bolsas, 200 modelos)
nohup python ens3_variables.py --paso 2 > paso2.txt 2>&1 &  # paso 2 (4 bolsas, 200 modelos)
tail -f ~/buckets/b1/exp/ens3_variables/log.txt
```

Sin `--paso` corre los dos, primero el 1.

## Si se corta

La corrida es larga. Medido con el dataset completo y 4 hilos, una ronda del paso 1 (un modelo de cada bolsa) tarda unos 17 minutos: 6,0 min `S-1`, 2,2 min `V25-1`, 4,2 min `V50-1` y 5,0 min `V75-1`. Cada modelo hace early stopping con una paciencia de unas 525 rondas, así que entrena bastante más allá de su mejor iteración. Con 50 modelos por bolsa son unas 14 horas por paso con 4 hilos, y menos en una máquina con más hilos. Por eso está pensada para cortarse y retomarse:

- **Cada modelo se guarda apenas termina:** sus predicciones de 201907 a 201910 y un `.json` con sus datos (semilla, iteraciones, minutos), escritos de forma atómica. El `.json` va último, así que si existe el modelo está completo.
- **Relanzar el mismo comando retoma.** Saltea los modelos que ya están: como mucho se pierde el que estaba en curso. Sirve también en otra máquina, porque todo queda en el bucket. Las variables se vuelven a armar solas (unos 2 minutos).
- **Orden por rondas:** primero el modelo 1 de cada bolsa del paso, después el 2, y así. Si se corta, todas las bolsas tienen la misma cantidad de modelos, y cada 10 rondas se completa un bloque más para la comparación.
- **Resultados parciales:** `python ens3_variables.py --solo-evaluar` arma los resultados con lo que haya. Con 30 modelos por bolsa, por ejemplo, ya hay 3 bloques de 10.
- **Cortar prolijo:** `touch ~/buckets/b1/exp/ens3_variables/PARAR` termina el modelo en curso, evalúa y sale. Ctrl-C o un `kill` también sirven: se pierde solo el modelo en curso.
- **Agregar modelos:** `N_MODELOS=100 python ens3_variables.py` sigue desde los 50 que ya estén, con las mismas primeras 50 semillas. Si cambia cualquier otra cosa de la configuración (hiperparámetros, fracciones), se niega a mezclar modelos y pide empezar en otra carpeta.
- Como el baseline, en una máquina de la cátedra la borra al terminar o si falla (`APAGAR_AL_TERMINAR=0` lo evita). Si se interrumpe a mano, no la borra.

## Salidas

En `~/buckets/b1/exp/ens3_variables/` (o en `outputs/ens3_variables/` fuera de la cátedra):

```
log.txt, config.json
modelos/<bolsa>_mNNN_sSEMILLA.json        datos de cada modelo terminado
predicciones/<mes>/<bolsa>_mNNN_sSEMILLA.npy
resultados/modelos.csv                    un modelo por fila
resultados/bloques.csv                    cada ensamble (bolsa, k, bloque) con sus métricas
resultados/curva.csv                      ganancia media y rango por bolsa y k
resultados/comparacion.csv                cada técnica contra su semillero, por k y métrica: bloques ganados y veredicto
resultados/resumen.json                   la comparación principal (k = 10) y la diversidad de cada bolsa
```
