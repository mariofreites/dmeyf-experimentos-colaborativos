#!/usr/bin/env python3
"""Ensamble 3: undersampling de variables (DMEyF 2026, experimentos colaborativos).

Pregunta: ¿entrenar cada modelo con un subconjunto distinto de variables suma algo más de lo que
LightGBM ya hace con feature_fraction? Se responde en dos pasos (el diseño completo está en README.md):

    Paso 1, ¿mejora por sí solo?   V25-1, V50-1, V75-1  contra  S-1    (feature_fraction = 1 en todas)
    Paso 2, ¿suma a LightGBM?      V25-ff, V50-ff, V75-ff contra S-ff  (feature_fraction del baseline)

Cada bolsa tiene N_MODELOS modelos con los hiperparámetros del baseline y early stopping en 201908.
El modelo i usa la misma semilla en todas las bolsas (bagging de LightGBM y sorteo de las variables),
así cada par de modelos difiere solo en las variables. Después, sin reentrenar, se arman ensambles
(promedio de probabilidades) en bloques disjuntos de 1, 5, 10, 25 y 50 modelos, y cada bloque se compara
con el bloque del semillero que tiene las mismas semillas.

Se puede cortar en cualquier momento y retomar:
- Cada modelo se guarda apenas termina (predicciones y un .json con sus datos, escritos de forma atómica).
  Al relanzar el mismo comando, saltea los modelos que ya están: como mucho se pierde el modelo en curso.
- El orden de entrenamiento es por rondas: el modelo 1 de cada bolsa del paso, después el 2, etc. Así, si
  la corrida se corta, todas las bolsas quedan con la misma cantidad de modelos y cada 10 rondas hay un
  bloque completo más para comparar. Primero se completa el paso 1 y después el paso 2.
- --solo-evaluar arma los resultados con lo que haya, aunque la corrida esté a medias.

Uso:
    python ens3_variables.py                  # los dos pasos (primero el 1)
    python ens3_variables.py --paso 1         # solo el paso 1 (el 2 se corre otro día con --paso 2)
    MODO_PRUEBA=1 python ens3_variables.py    # 5 % de los clientes, pocos modelos, unos minutos
    python ens3_variables.py --solo-evaluar   # resultados con las predicciones ya guardadas

Seguimiento:     tail -f ~/buckets/b1/exp/ens3_variables/log.txt
Parar prolijo:   touch ~/buckets/b1/exp/ens3_variables/PARAR   (termina el modelo en curso, evalúa y sale)
"""
from pathlib import Path
import argparse
import json
import os
import signal
import sys
import time

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from comun import (  # noqa: E402
    EN_GCP, MODO_PRUEBA, NUM_THREADS, PRUEBA_FRACCION_CLIENTES, SEEDS, TEST_MONTH, VALID_MONTH,
    Datos, Experimento, armar_variables, auc_score, cargar_predicciones, config_hash, ensure_packages,
    evaluar_modelo, gain_at, release_memory, save_json,
)

EXP_ID = "ens3_variables"
N_MODELOS = int(os.environ.get("N_MODELOS", 50))  # por bolsa
TAMANOS = [1, 5, 10, 25, 50]  # tamaños de ensamble que se evalúan (bloques disjuntos)
K_PRINCIPAL = 10  # tamaño de la comparación que decide
FRACCIONES = [0.25, 0.50, 0.75]
MAX_ROUNDS = 5000  # tope; el early stopping corta antes
CONFIRMACION_MONTH = 201909  # nadie lo usa para nada: se informa como segundo mes de test
CORTE_BASELINE = 4322  # envíos del baseline: la ganancia con este corte fijo saca el ruido de elegir el corte

if MODO_PRUEBA:
    N_MODELOS = int(os.environ.get("N_MODELOS", 6))
    TAMANOS, K_PRINCIPAL = [1, 2, 3, 6], 2
    MAX_ROUNDS = 300
    CORTE_BASELINE = round(CORTE_BASELINE * PRUEBA_FRACCION_CLIENTES)

# Semillas: las 20 de comun.py y, si hacen falta más, otras sorteadas siempre igual.
_extra = np.random.default_rng(20261114).choice(np.arange(100_000, 1_000_000), 400, replace=False)
SEMILLAS = (list(SEEDS) + [int(s) for s in _extra if s not in SEEDS])[:N_MODELOS]
assert len(set(SEMILLAS)) == N_MODELOS

EXP = Experimento(EXP_ID)
log = EXP.log
META_DIR = EXP.out_dir / "modelos"
RES_DIR = EXP.out_dir / "resultados"
APAGAR_AL_TERMINAR = os.environ.get("APAGAR_AL_TERMINAR", "1" if (EN_GCP and not MODO_PRUEBA) else "0") == "1"


def hiperparametros():
    """Los del baseline, registrados en el repo (también en modo prueba, para no depender de otra corrida)."""
    return json.loads((RAIZ / "resultados" / "baseline" / "mejores_hiperparametros.json").read_text("utf-8"))


def bolsas(hp):
    """nombre -> definición. ff = feature_fraction; fraccion = None es el semillero (todas las variables)."""
    ff_baseline = float(hp["valores"]["feature_fraction"])
    out = {}
    for paso, ff, sufijo in ((1, 1.0, "1"), (2, ff_baseline, "ff")):
        out[f"S-{sufijo}"] = {"paso": paso, "fraccion": None, "ff": ff, "semillero": None}
        for fr in FRACCIONES:
            out[f"V{round(fr * 100)}-{sufijo}"] = {"paso": paso, "fraccion": fr, "ff": ff, "semillero": f"S-{sufijo}"}
    return out


def nombre_modelo(bolsa, i):
    return f"{bolsa}_m{i:03d}_s{SEMILLAS[i]}"


def meta_path(nombre):
    return META_DIR / f"{nombre}.json"


# ---------------------------------------------------------------------------------------------
# Entrenamiento (reanudable)
# ---------------------------------------------------------------------------------------------
def entrenar_bolsas(data, hp, pasos):
    defs = bolsas(hp)
    META_DIR.mkdir(parents=True, exist_ok=True)
    for paso in pasos:
        nombres = [b for b, d in defs.items() if d["paso"] == paso]
        pendientes = [(i, b) for i in range(N_MODELOS) for b in nombres if not meta_path(nombre_modelo(b, i)).exists()]
        total = N_MODELOS * len(nombres)
        log(f"Paso {paso}: bolsas {', '.join(nombres)} | hechos {total - len(pendientes)} de {total}")
        tiempos = []
        for n, (i, bolsa) in enumerate(pendientes, start=1):
            if EXP.stop_file.exists():
                log(f"Encontré {EXP.stop_file.name}: no empiezo más modelos.")
                return False
            d, seed, nombre = defs[bolsa], SEMILLAS[i], nombre_modelo(bolsa, i)
            start = time.perf_counter()
            columnas = None if d["fraccion"] is None else data.columnas_al_azar(d["fraccion"], seed)
            valores = {**hp["valores"], "feature_fraction": d["ff"]}
            booster = data.entrenar(valores, seed, MAX_ROUNDS, columnas=columnas, early_stopping=True)
            best_iter = int(booster.best_iteration) or booster.current_iteration()
            data.predecir_y_guardar(booster, EXP.pred_dir, nombre, best_iter)
            minutos = (time.perf_counter() - start) / 60
            # El .json se escribe último: si existe, el modelo está completo.
            save_json(meta_path(nombre), {
                "modelo": nombre, "bolsa": bolsa, "paso": d["paso"], "indice": i, "semilla": seed,
                "fraccion": d["fraccion"], "feature_fraction": d["ff"], "best_iteration": best_iter,
                "auc_valid": float(booster.best_score[str(VALID_MONTH)]["auc"]),
                "columnas": len(data.features) if columnas is None else int(len(columnas)),
                "minutos": round(minutos, 3)})
            del booster
            release_memory()
            tiempos.append(minutos)
            faltan = (len(pendientes) - n) * float(np.mean(tiempos[-3 * len(nombres):]))
            log(f"  {n}/{len(pendientes)} {nombre} | {best_iter} iter | {minutos:.1f} min | "
                f"faltan ~{faltan / 60:.1f} h del paso {paso}")
    return True


# ---------------------------------------------------------------------------------------------
# Evaluación (solo lee lo guardado; sirve con la corrida a medias)
# ---------------------------------------------------------------------------------------------
def p_signo(ganados, n):
    """P(ganar >= `ganados` de `n` bloques por azar), si la técnica no aportara nada (test de signo, una cola)."""
    from math import comb
    return sum(comb(n, j) for j in range(ganados, n + 1)) / 2 ** n


def veredicto(ganados, n):
    """Con 5 bloques: 5 = mejora, 4 = indicio, 2 o 3 = no se distingue, 0 o 1 = no aporta."""
    if n == 0:
        return "sin bloques"
    if p_signo(ganados, n) <= 0.05:
        return "mejora"
    if p_signo(ganados, n) <= 0.20:
        return "indicio de mejora, no concluyente"
    if p_signo(n - ganados, n) <= 0.20:
        return "empeora o no aporta"
    return "no se distingue del semillero"


def evaluar():
    metas = [json.loads(p.read_text("utf-8")) for p in sorted(META_DIR.glob("*.json"))] if META_DIR.exists() else []
    if not metas:
        log("Todavía no hay modelos terminados para evaluar.")
        return None
    modelos = pd.DataFrame(metas).sort_values(["paso", "bolsa", "indice"]).reset_index(drop=True)
    meses = [VALID_MONTH, CONFIRMACION_MONTH, TEST_MONTH]
    preds = {m: cargar_predicciones(EXP.pred_dir, m) for m in meses}
    b2 = {m: (preds[m]["clase_ternaria"] == "BAJA+2").to_numpy() for m in meses}
    pos_test = preds[TEST_MONTH]["clase_ternaria"].isin(["BAJA+1", "BAJA+2"]).to_numpy()
    defs = {r.bolsa: r for r in modelos.drop_duplicates("bolsa").itertuples()}

    # Cada modelo solo (k = 1) y cada bloque disjunto de k modelos consecutivos.
    filas = []
    for bolsa, grupo in modelos.groupby("bolsa"):
        por_indice = grupo.set_index("indice")
        for k in TAMANOS:
            for j in range(N_MODELOS // k):
                indices = list(range(j * k, (j + 1) * k))
                if not all(i in por_indice.index for i in indices):
                    continue
                cols = list(por_indice.loc[indices, "modelo"])
                score = {m: preds[m][cols].to_numpy(np.float64).mean(axis=1) for m in meses}
                r = evaluar_modelo(preds[VALID_MONTH], score[VALID_MONTH], preds[TEST_MONTH], score[TEST_MONTH])
                filas.append({
                    "bolsa": bolsa, "paso": defs[bolsa].paso, "fraccion": defs[bolsa].fraccion,
                    "feature_fraction": defs[bolsa].feature_fraction, "k": k, "bloque": j,
                    "semillas": " ".join(str(SEMILLAS[i]) for i in indices),
                    f"corte_{VALID_MONTH}": r["corte"], f"ganancia_{TEST_MONTH}": r["ganancia_test"],
                    f"ganancia_{TEST_MONTH}_corte_fijo": gain_at(b2[TEST_MONTH], score[TEST_MONTH], CORTE_BASELINE),
                    f"ganancia_{CONFIRMACION_MONTH}": gain_at(b2[CONFIRMACION_MONTH], score[CONFIRMACION_MONTH], r["corte"]),
                    f"auc_{TEST_MONTH}": auc_score(pos_test, score[TEST_MONTH]),
                    "minutos": float(por_indice.loc[indices, "minutos"].sum())})
    bloques = pd.DataFrame(filas)
    RES_DIR.mkdir(parents=True, exist_ok=True)
    modelos.to_csv(RES_DIR / "modelos.csv", index=False)
    bloques.to_csv(RES_DIR / "bloques.csv", index=False)
    g = f"ganancia_{TEST_MONTH}"

    # Curva de tamaño: media y rango de la ganancia por bolsa y k.
    curva = (bloques.groupby(["paso", "bolsa", "k"])
             .agg(bloques=(g, "size"), ganancia_media=(g, "mean"), ganancia_desvio=(g, "std"),
                  ganancia_min=(g, "min"), ganancia_max=(g, "max"),
                  ganancia_corte_fijo_media=(f"{g}_corte_fijo", "mean"),
                  ganancia_confirmacion_media=(f"ganancia_{CONFIRMACION_MONTH}", "mean"),
                  auc_media=(f"auc_{TEST_MONTH}", "mean"), minutos_medio=("minutos", "mean"))
             .reset_index())
    curva.to_csv(RES_DIR / "curva.csv", index=False)

    # Comparación pareada: cada bloque contra el bloque del semillero con las mismas semillas.
    comp = []
    for bolsa, d in bolsas(hiperparametros()).items():
        if d["semillero"] is None:
            continue
        a = bloques[bloques["bolsa"] == bolsa].set_index(["k", "bloque"])
        b = bloques[bloques["bolsa"] == d["semillero"]].set_index(["k", "bloque"])
        comunes = a.index.intersection(b.index)
        for k in TAMANOS:
            idx = [x for x in comunes if x[0] == k]
            if not idx:
                continue
            for metrica in (g, f"{g}_corte_fijo", f"ganancia_{CONFIRMACION_MONTH}"):
                diff = (a.loc[idx, metrica] - b.loc[idx, metrica]).to_numpy()
                ganados = int((diff > 0).sum())
                comp.append({"paso": d["paso"], "tecnica": bolsa, "semillero": d["semillero"], "k": k,
                             "metrica": metrica, "pares": len(idx), "ganados": ganados,
                             "diferencia_media": float(diff.mean()), "diferencia_min": float(diff.min()),
                             "diferencia_max": float(diff.max()), "p_signo": p_signo(ganados, len(idx)),
                             "veredicto": veredicto(ganados, len(idx))})
    comparacion = pd.DataFrame(comp)
    comparacion.to_csv(RES_DIR / "comparacion.csv", index=False)

    # Diversidad: correlación media entre las predicciones de 201910 de los modelos de cada bolsa.
    diversidad = {}
    for bolsa, grupo in modelos.groupby("bolsa"):
        if len(grupo) > 1:
            c = np.corrcoef(preds[TEST_MONTH][list(grupo["modelo"])].to_numpy(np.float64).T)
            diversidad[bolsa] = float(c[np.triu_indices_from(c, 1)].mean())
    resumen = {
        "modelos_por_bolsa": modelos.groupby("bolsa").size().to_dict(), "n_modelos_objetivo": N_MODELOS,
        "k_principal": K_PRINCIPAL, "corte_fijo": CORTE_BASELINE,
        "correlacion_media_entre_modelos": diversidad,
        "comparacion_principal": json.loads(
            comparacion[(comparacion["k"] == K_PRINCIPAL) & (comparacion["metrica"] == g)]
            .to_json(orient="records")) if len(comparacion) else [],
    }
    save_json(RES_DIR / "resumen.json", resumen)

    log("=" * 70)
    log(f"Modelos terminados por bolsa: {resumen['modelos_por_bolsa']}")
    if len(comparacion):
        principal = comparacion[(comparacion["metrica"] == g) & (comparacion["k"] == K_PRINCIPAL)]
        for r in principal.itertuples():
            log(f"  paso {r.paso} | {r.tecnica} vs {r.semillero} (k={r.k}): gana {r.ganados}/{r.pares} bloques | "
                f"diferencia media {r.diferencia_media:+.2f} M (de {r.diferencia_min:+.2f} a {r.diferencia_max:+.2f}) | "
                f"{r.veredicto}")
    log("Resultados en", RES_DIR)
    return resumen


# ---------------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--paso", type=int, choices=[1, 2], action="append",
                        help="qué paso correr (se puede repetir). Por defecto, los dos: primero el 1")
    parser.add_argument("--solo-evaluar", action="store_true", help="solo arma los resultados con lo guardado")
    args = parser.parse_args()
    pasos = sorted(set(args.paso or [1, 2]))
    if args.solo_evaluar:
        evaluar()
        return

    ensure_packages()
    import lightgbm

    hp = hiperparametros()
    config = {"hiperparametros": hp["valores"], "semillas": SEMILLAS, "fracciones": FRACCIONES,
              "max_rounds": MAX_ROUNDS, "bolsas": bolsas(hp), "prueba": MODO_PRUEBA, "version": 1}
    config_file = EXP.out_dir / "config.json"
    if config_file.exists():
        anterior = json.loads(config_file.read_text("utf-8"))
        # Agregar modelos (N_MODELOS más grande) es compatible: las semillas anteriores no cambian.
        n = min(len(anterior["semillas"]), len(SEMILLAS))
        if (config_hash({**anterior, "semillas": anterior["semillas"][:n]})
                != config_hash({**config, "semillas": SEMILLAS[:n]})):
            raise RuntimeError(f"{EXP.out_dir} tiene modelos de otra configuración. Movelos para empezar de cero.")
    save_json(config_file, config)

    if EXP.stop_file.exists():  # un PARAR viejo frenaría la corrida apenas empieza
        EXP.stop_file.unlink()

    def al_recibir_senal(signum, frame):  # p. ej. Google avisando que borra la máquina spot
        raise KeyboardInterrupt(f"señal {signum}")
    signal.signal(signal.SIGTERM, al_recibir_senal)

    log("=" * 70)
    log(f"{EXP_ID} | pasos {pasos} | {N_MODELOS} modelos por bolsa | prueba: {MODO_PRUEBA} | "
        f"Google Cloud: {EN_GCP} | hilos: {NUM_THREADS} | lightgbm {lightgbm.__version__}")
    log("Resultados:", EXP.out_dir)
    try:
        data = Datos(armar_variables(log), log)
        completo = entrenar_bolsas(data, hp, pasos)
        log("Entrenamiento completo." if completo else "Entrenamiento parcial: relanzar el mismo comando para seguir.")
        evaluar()
    except KeyboardInterrupt as error:
        log(f"[interrumpido] {error}. Relanzar el mismo comando retoma desde el último modelo guardado.")
        raise
    except BaseException as error:
        log(f"[error] {type(error).__name__}: {error}")
        if APAGAR_AL_TERMINAR:
            EXP.apagar_vm("falló la corrida")
        raise
    if APAGAR_AL_TERMINAR:
        EXP.apagar_vm("terminó")


if __name__ == "__main__":
    main()
