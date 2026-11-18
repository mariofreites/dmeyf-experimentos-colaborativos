"""Ensamble 3: distribución de ensambles de 10 armados al azar con los 50 modelos de cada bolsa.

No entrena nada: lee las predicciones que ya guardó ens3_variables.py. En cada repetición sortea 10
índices de los 50 y arma con ellos el ensamble de cada bolsa (mismas semillas en todas, así la
comparación con el semillero queda pareada). Sirve para dibujar la campana de los ensambles de 10.

Ojo: los ensambles comparten modelos entre sí, así que no son independientes. La campana sirve para
mostrar; la prueba estadística sigue siendo la de los bloques disjuntos (comparacion.csv).

    python remuestreo.py              # 1000 repeticiones, unos minutos
    REPETICIONES=200 python remuestreo.py
"""
from pathlib import Path
import os
import subprocess
import sys

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
from comun import TEST_MONTH, VALID_MONTH, Experimento, auc_score, cargar_predicciones, evaluar_modelo, gain_at  # noqa: E402

try:
    import matplotlib
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "matplotlib"], check=True)
    import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

K = int(os.environ.get("K", 10))
REPETICIONES = int(os.environ.get("REPETICIONES", 1000))
CONFIRMACION_MONTH = 201909
CORTE_BASELINE = 4322
PAREJAS = {"V25-1": "S-1", "V50-1": "S-1", "V75-1": "S-1", "V25-ff": "S-ff", "V50-ff": "S-ff", "V75-ff": "S-ff"}

EXP = Experimento("ens3_variables")
SALIDA = EXP.out_dir / "resultados"
log = EXP.log


def main():
    meses = [VALID_MONTH, CONFIRMACION_MONTH, TEST_MONTH]
    preds = {m: cargar_predicciones(EXP.pred_dir, m) for m in meses}
    clase = {m: preds[m]["clase_ternaria"].to_numpy() for m in meses}
    b2 = {m: clase[m] == "BAJA+2" for m in meses}
    pos_test = np.isin(clase[TEST_MONTH], ["BAJA+1", "BAJA+2"])

    # Una matriz (clientes x modelos) por bolsa y mes, con las columnas ordenadas por índice de modelo.
    nombres = [c for c in preds[TEST_MONTH].columns if "_m" in c and "_s" in c]
    bolsas = {}
    for c in nombres:
        bolsa, resto = c.split("_m", 1)
        bolsas.setdefault(bolsa, []).append((int(resto.split("_s")[0]), c))
    matrices, n = {}, None
    for bolsa, cols in sorted(bolsas.items()):
        cols = [c for _, c in sorted(cols)]
        n = len(cols) if n is None else min(n, len(cols))
        matrices[bolsa] = {m: preds[m][cols].to_numpy(np.float32) for m in meses}
    del preds
    log(f"[remuestreo] {len(matrices)} bolsas, {n} modelos en común, {REPETICIONES} ensambles de {K} al azar")

    rng = np.random.default_rng(20261117)
    filas = []
    for r in range(REPETICIONES):
        idx = np.sort(rng.choice(n, K, replace=False))
        for bolsa, mat in matrices.items():
            score = {m: mat[m][:, idx].mean(axis=1) for m in meses}
            ev = evaluar_modelo(pd.DataFrame({"clase_ternaria": clase[VALID_MONTH]}), score[VALID_MONTH],
                                pd.DataFrame({"clase_ternaria": clase[TEST_MONTH]}), score[TEST_MONTH])
            filas.append({
                "repeticion": r, "bolsa": bolsa, "modelos": " ".join(map(str, idx)),
                f"corte_{VALID_MONTH}": ev["corte"], f"ganancia_{TEST_MONTH}": ev["ganancia_test"],
                f"ganancia_{TEST_MONTH}_corte_fijo": gain_at(b2[TEST_MONTH], score[TEST_MONTH], CORTE_BASELINE),
                f"ganancia_{CONFIRMACION_MONTH}": gain_at(b2[CONFIRMACION_MONTH], score[CONFIRMACION_MONTH], ev["corte"]),
                f"auc_{TEST_MONTH}": auc_score(pos_test, score[TEST_MONTH]),
            })
        if (r + 1) % 100 == 0:
            log(f"[remuestreo] {r + 1}/{REPETICIONES}")
    tabla = pd.DataFrame(filas)
    SALIDA.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(SALIDA / f"remuestreo_k{K}.csv", index=False)

    # Resumen pareado: en qué porcentaje de las repeticiones la técnica le gana al semillero.
    metricas = [f"ganancia_{TEST_MONTH}", f"ganancia_{TEST_MONTH}_corte_fijo", f"ganancia_{CONFIRMACION_MONTH}", f"auc_{TEST_MONTH}"]
    ancha = tabla.pivot(index="repeticion", columns="bolsa")
    resumen = []
    for bolsa, semillero in PAREJAS.items():
        if bolsa not in matrices or semillero not in matrices:
            continue
        for met in metricas:
            d = ancha[met][bolsa] - ancha[met][semillero]
            resumen.append({"bolsa": bolsa, "semillero": semillero, "metrica": met,
                            "media_tecnica": ancha[met][bolsa].mean(), "media_semillero": ancha[met][semillero].mean(),
                            "diferencia_media": d.mean(), "dif_p05": d.quantile(0.05), "dif_p95": d.quantile(0.95),
                            "pct_gana": 100 * (d > 0).mean()})
    resumen = pd.DataFrame(resumen)
    resumen.to_csv(SALIDA / f"remuestreo_k{K}_resumen.csv", index=False)
    log(resumen.round(3).to_string(index=False))

    # La campana de cada bolsa, una fila por paso y una columna por métrica de ganancia.
    colores = {"S": "black", "V75": "tab:blue", "V50": "tab:orange", "V25": "tab:red"}
    etiquetas = {"S": "Semillero (todas las variables)", "V75": "75 % variables", "V50": "50 % variables", "V25": "25 % variables"}
    titulos = {f"ganancia_{TEST_MONTH}": "Octubre, corte elegido en agosto",
               f"ganancia_{TEST_MONTH}_corte_fijo": f"Octubre, mismo corte para todos ({CORTE_BASELINE:,})".replace(",", "."),
               f"ganancia_{CONFIRMACION_MONTH}": "Septiembre, corte elegido en agosto"}
    pasos = [("1", "feature_fraction 1"), ("ff", "feature_fraction 0,757")]
    fig, ejes = plt.subplots(len(pasos), len(titulos), figsize=(18, 9), squeeze=False)
    for fila, (sufijo, nombre_paso) in enumerate(pasos):
        for col, (met, titulo) in enumerate(titulos.items()):
            ax = ejes[fila][col]
            for tipo, color in colores.items():
                bolsa = f"{tipo}-{sufijo}"
                if bolsa not in matrices:
                    continue
                v = ancha[met][bolsa]
                ax.hist(v, bins=40, density=True, histtype="step", lw=2.5 if tipo == "S" else 1.8, color=color,
                        label=f"{etiquetas[tipo]} (media {v.mean():.1f})")
                ax.axvline(v.mean(), color=color, ls=":", lw=1)
            ax.set_title(f"{titulo} · {nombre_paso}", fontsize=10)
            ax.set_xlabel("Ganancia (millones)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Ganancia de {REPETICIONES} ensambles de {K} modelos armados al azar con los {n} de cada bolsa")
    fig.tight_layout()
    fig.savefig(SALIDA / f"remuestreo_k{K}.png", dpi=110)
    log(f"[remuestreo] listo: {SALIDA}/remuestreo_k{K}.png, remuestreo_k{K}.csv y remuestreo_k{K}_resumen.csv")


if __name__ == "__main__":
    main()
