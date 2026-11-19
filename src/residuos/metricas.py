"""Métricas de la evaluación sobre el 30 % de prueba, por material y por caneca.

Uso: ``uv run python -m residuos.metricas``. Lee los CSV del arnés en ``resultados/`` y escribe:

- ``metricas-entrenamiento-1.json`` y ``metricas-entrenamiento-2.json``: cada copia local sobre la prueba.
- ``metricas-elegido.json``: las cifras oficiales (el elegido cargado desde su enlace), la curva de
  cobertura y precisión de la caneca blanca, los errores más confiados y la comparación de los dos
  entrenamientos con la prueba exacta de McNemar.
- ``metricas-robustez.json``: el elegido sobre cada perturbación, con sus parámetros.

Cada archivo guarda también los parámetros del bootstrap con que se calcularon sus intervalos.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import label_binarize

from residuos import rutas
from residuos.datos import PRUEBA, leer_particion
from residuos.robustez import PERTURBACIONES, archivo_alterado
from residuos.robustez import parametros as parametros_robustez

BOOTSTRAP = {
    "remuestreos": 1000,
    "semilla": 42,
    "nivel": 0.95,
    "metodo": "percentil",
    "remuestreo": "simple, con reemplazo, de las imágenes de prueba; los mismos índices para todas las métricas de un modelo",
}
MCNEMAR = {
    "prueba": "McNemar exacta: binomial bilateral sobre los pares discordantes con p = 0,5",
    "alfa": 0.05,
    "b": "aciertos del entrenamiento 1 que el entrenamiento 2 falla",
    "c": "aciertos del entrenamiento 2 que el entrenamiento 1 falla",
}
TOLERANCIA_SUMA = 1e-4
UMBRALES_COBERTURA = tuple(round(u, 2) for u in np.arange(0.35, 1.0, 0.01))
ERRORES_MAS_CONFIADOS = 12
PARES_MAS_CONFUNDIDOS = 5
BLANCA, VERDE, NEGRA = "blanca", "verde", "negra"
PREFIJO_CANECA = "prob_caneca_"

CSV_ENTRENAMIENTO = "probabilidades-entrenamiento-{}.csv"
CSV_ELEGIDO = "probabilidades-prueba-elegido.csv"
CSV_ROBUSTEZ = "probabilidades-robustez-{}.csv"


# Lectura


@dataclass(frozen=True)
class Predicciones:
    """Una tarea de clasificación sobre la prueba: materiales o canecas."""

    clases: list[str]
    real: np.ndarray  # índice de la clase real
    predicha: np.ndarray  # índice de la decisión que tomó el navegador
    probabilidades: np.ndarray  # una columna por clase, en el orden de `clases`


def leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta, keep_default_na=False)


def columnas_material(tabla: pd.DataFrame) -> list[str]:
    return [c for c in tabla.columns if c.startswith("prob_") and not c.startswith(PREFIJO_CANECA)]


def columnas_caneca(tabla: pd.DataFrame) -> list[str]:
    return [c for c in tabla.columns if c.startswith(PREFIJO_CANECA)]


def indices(valores: Sequence[str], clases: list[str]) -> np.ndarray:
    posicion = {clase: i for i, clase in enumerate(clases)}
    return np.array([posicion[v] for v in valores])


def caneca_de(canecas: dict) -> dict[str, str]:
    return {etiqueta: datos["caneca"] for etiqueta, datos in canecas["materiales"].items()}


def predicciones_material(tabla: pd.DataFrame) -> Predicciones:
    columnas = columnas_material(tabla)
    clases = [c.removeprefix("prob_") for c in columnas]
    return Predicciones(clases, indices(tabla["clase_real"], clases), indices(tabla["material"], clases), tabla[columnas].to_numpy())


def predicciones_caneca(tabla: pd.DataFrame, canecas: dict) -> Predicciones:
    columnas = columnas_caneca(tabla)
    clases = [c.removeprefix(PREFIJO_CANECA) for c in columnas]
    reales = tabla["clase_real"].map(caneca_de(canecas))
    return Predicciones(clases, indices(reales, clases), indices(tabla["caneca"], clases), tabla[columnas].to_numpy())


# Verificación de los CSV del arnés


def recalcular_canecas(tabla: pd.DataFrame, canecas: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Suma por caneca en el mismo orden que `traducirACaneca` de inferencia.js y su argmax."""
    destino = caneca_de(canecas)
    sumas = pd.DataFrame(0.0, index=tabla.index, columns=list(canecas["canecas"]))
    for columna in columnas_material(tabla):
        sumas[destino[columna.removeprefix("prob_")]] += tabla[columna]
    return sumas, sumas.idxmax(axis=1)


def verificar(tabla: pd.DataFrame, archivos_esperados: Sequence[str], canecas: dict) -> dict:
    sumas, ganadora = recalcular_canecas(tabla, canecas)
    navegador = tabla[columnas_caneca(tabla)].set_axis(list(canecas["canecas"]), axis=1)
    desviacion_suma = float((tabla[columnas_material(tabla)].sum(axis=1) - 1).abs().max())
    resultado = {
        "filas": len(tabla),
        "una_fila_por_imagen_de_prueba": len(tabla) == len(archivos_esperados) and sorted(tabla["archivo"]) == sorted(archivos_esperados),
        "desviacion_maxima_de_la_suma": desviacion_suma,
        "suman_uno": desviacion_suma <= TOLERANCIA_SUMA,
        "caneca_igual_a_python": bool((ganadora == tabla["caneca"]).all()),
        "diferencia_maxima_prob_caneca": float((sumas - navegador).abs().to_numpy().max()),
    }
    resultado["valido"] = (
        resultado["una_fila_por_imagen_de_prueba"]
        and resultado["suman_uno"]
        and resultado["caneca_igual_a_python"]
        and resultado["diferencia_maxima_prob_caneca"] == 0
    )
    return resultado


# Métricas puntuales


def matriz_confusion(real: np.ndarray, predicha: np.ndarray, k: int) -> np.ndarray:
    return np.bincount(real * k + predicha, minlength=k * k).reshape(k, k)


def dividir(numerador: np.ndarray, denominador: np.ndarray) -> np.ndarray:
    return np.divide(numerador, denominador, out=np.zeros_like(numerador, dtype=float), where=denominador > 0)


def desde_matriz(matriz: np.ndarray) -> dict[str, np.ndarray | float]:
    """Precisión, recall y F1 por clase y sus promedios, solo con la matriz de confusión."""
    aciertos = np.diag(matriz).astype(float)
    soporte, asignadas = matriz.sum(axis=1), matriz.sum(axis=0)
    precision, recall = dividir(aciertos, asignadas), dividir(aciertos, soporte)
    f1 = dividir(2 * precision * recall, precision + recall)
    pesos = soporte / soporte.sum()
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exactitud": aciertos.sum() / matriz.sum(),
        "exactitud_balanceada": recall.mean(),
        "precision_macro": precision.mean(),
        "recall_macro": recall.mean(),
        "f1_macro": f1.mean(),
        "precision_ponderada": float(pesos @ precision),
        "recall_ponderado": float(pesos @ recall),
        "f1_ponderado": float(pesos @ f1),
    }


def desde_probabilidades(real: np.ndarray, probabilidades: np.ndarray) -> dict[str, np.ndarray | float]:
    """AUC y AP una clase contra el resto, por clase, macro y micro."""
    binaria = label_binarize(real, classes=range(probabilidades.shape[1]))
    return {
        "auc": roc_auc_score(binaria, probabilidades, average=None),
        "ap": average_precision_score(binaria, probabilidades, average=None),
        "auc_macro": roc_auc_score(binaria, probabilidades, average="macro"),
        "auc_micro": roc_auc_score(binaria, probabilidades, average="micro"),
        "ap_macro": average_precision_score(binaria, probabilidades, average="macro"),
        "ap_micro": average_precision_score(binaria, probabilidades, average="micro"),
    }


POR_CLASE = ("precision", "recall", "f1", "auc", "ap")


def puntuales(p: Predicciones, real: np.ndarray, predicha: np.ndarray, probabilidades: np.ndarray, con_curvas: bool = True) -> dict:
    valores = desde_matriz(matriz_confusion(real, predicha, len(p.clases)))
    if con_curvas:
        valores |= desde_probabilidades(real, probabilidades)
    return valores


def aplanar(valores: dict, clases: list[str]) -> dict[str, float]:
    """Una clave por cifra: `f1.Cardboard`, `f1_macro`, etc."""
    plano = {}
    for clave, valor in valores.items():
        if clave in POR_CLASE:
            plano |= {f"{clave}.{clase}": float(v) for clase, v in zip(clases, valor, strict=True)}
        else:
            plano[clave] = float(valor)
    return plano


# Bootstrap


def remuestreos(n: int) -> np.ndarray:
    generador = np.random.default_rng(BOOTSTRAP["semilla"])
    return generador.integers(0, n, size=(BOOTSTRAP["remuestreos"], n))


def percentiles(muestras: np.ndarray) -> tuple[float, float]:
    cola = (1 - BOOTSTRAP["nivel"]) / 2 * 100
    inferior, superior = np.nanpercentile(muestras, [cola, 100 - cola])
    return float(inferior), float(superior)


def intervalos(calcular: Callable[[np.ndarray], dict[str, float]], n: int) -> dict[str, list[float]]:
    muestras = [calcular(indice) for indice in remuestreos(n)]
    return {clave: list(percentiles(np.array([m[clave] for m in muestras]))) for clave in muestras[0]}


def evaluar_tarea(p: Predicciones, con_curvas: bool = True) -> dict:
    """Cifras puntuales e intervalos de una tarea, con la matriz y el soporte."""
    def calcular(indice: np.ndarray) -> dict[str, float]:
        return aplanar(puntuales(p, p.real[indice], p.predicha[indice], p.probabilidades[indice], con_curvas), p.clases)

    matriz = matriz_confusion(p.real, p.predicha, len(p.clases))
    return {
        "clases": p.clases,
        "matriz": matriz.tolist(),
        "soporte": dict(zip(p.clases, matriz.sum(axis=1).tolist(), strict=True)),
        "valores": calcular(np.arange(len(p.real))),
        "intervalos": intervalos(calcular, len(p.real)),
    }


# Lecturas propias del problema


def caneca_vs_material(tabla: pd.DataFrame, canecas: dict) -> dict:
    """Caneca por suma de probabilidades frente a la caneca del material más probable."""
    destino = caneca_de(canecas)
    real = tabla["clase_real"].map(destino)
    por_material = tabla["material"].map(destino)
    acierta_suma, acierta_material = tabla["caneca"] == real, por_material == real
    return {
        "difieren": int((por_material != tabla["caneca"]).sum()),
        "exactitud_suma": float(acierta_suma.mean()),
        "exactitud_material": float(acierta_material.mean()),
        "aciertos_solo_suma": int((acierta_suma & ~acierta_material).sum()),
        "aciertos_solo_material": int((~acierta_suma & acierta_material).sum()),
    }


def errores(tabla: pd.DataFrame, canecas: dict) -> dict:
    destino = caneca_de(canecas)
    real_caneca = tabla["clase_real"].map(destino)
    falla_material = tabla["material"] != tabla["clase_real"]
    pares = tabla[falla_material].groupby(["clase_real", "material"]).size().sort_values(ascending=False, kind="stable")
    return {
        "material": int(falla_material.sum()),
        "material_con_caneca_correcta": int((falla_material & (tabla["caneca"] == real_caneca)).sum()),
        "caneca": int((tabla["caneca"] != real_caneca).sum()),
        "contaminan_la_blanca": int(((tabla["caneca"] == BLANCA) & (real_caneca != BLANCA)).sum()),
        "organicos_en_la_blanca": int(((tabla["caneca"] == BLANCA) & (real_caneca == VERDE)).sum()),
        "no_aprovechables_en_la_blanca": int(((tabla["caneca"] == BLANCA) & (real_caneca == NEGRA)).sum()),
        "aprovechables_perdidos_en_la_negra": int(((tabla["caneca"] == NEGRA) & (real_caneca == BLANCA)).sum()),
        "pares_mas_confundidos": [
            {"real": real, "predicho": predicho, "imagenes": int(n)}
            for (real, predicho), n in pares.head(PARES_MAS_CONFUNDIDOS).items()
        ],
    }


def errores_mas_confiados(tabla: pd.DataFrame, canecas: dict) -> list[dict]:
    destino = caneca_de(canecas)
    fallos = tabla[tabla["material"] != tabla["clase_real"]].copy()
    fallos["confianza"] = [fila[f"prob_{fila['material']}"] for _, fila in fallos.iterrows()]
    fallos = fallos.sort_values(["confianza", "archivo"], ascending=[False, True], kind="stable").head(ERRORES_MAS_CONFIADOS)
    return [
        {
            "archivo": fila["archivo"],
            "real": fila["clase_real"],
            "predicho": fila["material"],
            "confianza": float(fila["confianza"]),
            "caneca_real": destino[fila["clase_real"]],
            "caneca_predicha": fila["caneca"],
        }
        for _, fila in fallos.iterrows()
    ]


@dataclass(frozen=True)
class SenalesBlanca:
    """Por imagen: si es blanca de verdad, si el modelo la manda a la blanca y con qué probabilidad."""

    real: np.ndarray
    asignada: np.ndarray
    confianza: np.ndarray

    @classmethod
    def desde(cls, tabla: pd.DataFrame, canecas: dict) -> SenalesBlanca:
        return cls(
            real=(tabla["clase_real"].map(caneca_de(canecas)) == BLANCA).to_numpy(),
            asignada=(tabla["caneca"] == BLANCA).to_numpy(),
            confianza=tabla[f"{PREFIJO_CANECA}{BLANCA}"].to_numpy(),
        )

    def submuestra(self, indice: np.ndarray) -> SenalesBlanca:
        return SenalesBlanca(self.real[indice], self.asignada[indice], self.confianza[indice])


def punto_cobertura(senales: SenalesBlanca, umbral: float | None) -> dict:
    real_blanca, asignada = senales.real, senales.asignada
    cubierta = asignada if umbral is None else asignada & (senales.confianza >= umbral)
    return {
        "umbral": umbral,
        "imagenes": int(cubierta.sum()),
        "cobertura": float(cubierta.sum() / asignada.sum()),
        "precision": float((cubierta & real_blanca).sum() / cubierta.sum()) if cubierta.any() else None,
        "recall": float((cubierta & real_blanca).sum() / real_blanca.sum()),
        "contaminantes": int((cubierta & ~real_blanca).sum()),
    }


def cobertura_blanca(tabla: pd.DataFrame, canecas: dict) -> dict:
    """Precisión de la blanca si el sistema se abstiene cuando su probabilidad no llega al umbral."""
    senales = SenalesBlanca.desde(tabla, canecas)
    return {
        "definicion": (
            "Entre las imágenes que el modelo manda a la blanca, se conservan las que tienen probabilidad de "
            "blanca igual o mayor que el umbral; cobertura es la fracción conservada, precisión la fracción "
            "de ellas que de verdad es blanca y recall la fracción de las blancas reales que se conserva."
        ),
        "sin_abstencion": punto_cobertura(senales, None),
        "puntos": [punto_cobertura(senales, u) for u in UMBRALES_COBERTURA],
    }


def mcnemar(acierta_1: np.ndarray, acierta_2: np.ndarray) -> dict:
    b = int((acierta_1 & ~acierta_2).sum())
    c = int((~acierta_1 & acierta_2).sum())
    p_valor = float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0
    return {"b": b, "c": c, "discordantes": b + c, "p_valor": p_valor, "significativo": p_valor < MCNEMAR["alfa"]}


def diferencia_pareada(acierta_1: np.ndarray, acierta_2: np.ndarray) -> dict:
    """Exactitud del entrenamiento 2 menos la del 1, con intervalo por bootstrap pareado."""
    diferencias = [float(acierta_2[i].mean() - acierta_1[i].mean()) for i in remuestreos(len(acierta_1))]
    return {"valor": float(acierta_2.mean() - acierta_1.mean()), "intervalo": list(percentiles(np.array(diferencias)))}


def comparar_entrenamientos(tabla_1: pd.DataFrame, tabla_2: pd.DataFrame, canecas: dict) -> dict:
    if list(tabla_1["archivo"]) != list(tabla_2["archivo"]):
        raise ValueError("Los dos CSV deben tener las mismas imágenes en el mismo orden")
    destino = caneca_de(canecas)
    real_caneca = tabla_1["clase_real"].map(destino)
    material = [(t["material"] == t["clase_real"]).to_numpy() for t in (tabla_1, tabla_2)]
    caneca = [(t["caneca"] == real_caneca).to_numpy() for t in (tabla_1, tabla_2)]
    return {
        "parametros_mcnemar": MCNEMAR,
        "material": {"mcnemar": mcnemar(*material), "diferencia_exactitud": diferencia_pareada(*material)},
        "caneca": {"mcnemar": mcnemar(*caneca), "diferencia_exactitud": diferencia_pareada(*caneca)},
    }


# Ensamble de cada archivo


def relativa(ruta: Path) -> str:
    return ruta.relative_to(rutas.RAIZ_CLASIFICADOR).as_posix()


def metricas_de_modelo(ruta_csv: Path, modelo: str, archivos: Sequence[str], canecas: dict) -> dict:
    tabla = leer_csv(ruta_csv)
    return {
        "modelo": modelo,
        "csv": relativa(ruta_csv),
        "imagenes": len(tabla),
        "bootstrap": BOOTSTRAP,
        "verificacion": verificar(tabla, archivos, canecas),
        "material": evaluar_tarea(predicciones_material(tabla)),
        "caneca": evaluar_tarea(predicciones_caneca(tabla, canecas)),
        "caneca_vs_material_mas_probable": caneca_vs_material(tabla, canecas),
        "errores": errores(tabla, canecas),
    }


def metricas_elegido(modelos: dict, archivos: Sequence[str], canecas: dict) -> dict:
    elegido = modelos["elegido"]
    tabla = leer_csv(rutas.RESULTADOS / CSV_ELEGIDO)
    tablas = {n: leer_csv(rutas.RESULTADOS / CSV_ENTRENAMIENTO.format(n)) for n in (1, 2)}
    otro = 1 if elegido == "entrenamiento_2" else 2
    return {
        **metricas_de_modelo(rutas.RESULTADOS / CSV_ELEGIDO, f"{elegido}, cargado desde {modelos[elegido]['url']}", archivos, canecas),
        "elegido": elegido,
        "cobertura_blanca": cobertura_blanca(tabla, canecas),
        "errores_mas_confiados": errores_mas_confiados(tabla, canecas),
        "comparacion_entrenamientos": {
            "csv_entrenamiento_1": relativa(rutas.RESULTADOS / CSV_ENTRENAMIENTO.format(1)),
            "csv_entrenamiento_2": relativa(rutas.RESULTADOS / (CSV_ELEGIDO if otro == 1 else CSV_ENTRENAMIENTO.format(2))),
            **comparar_entrenamientos(tablas[1], tabla if otro == 1 else tablas[2], canecas),
        },
    }


def tarea_resumida(p: Predicciones) -> dict:
    """Las cifras de la robustez: sin curvas, que no cambian la lectura y alargan el bootstrap."""
    return evaluar_tarea(p, con_curvas=False)


def metricas_robustez(modelos: dict, prueba: pd.DataFrame, canecas: dict) -> dict:
    original = leer_csv(rutas.RESULTADOS / CSV_ELEGIDO)
    base = {
        "material": tarea_resumida(predicciones_material(original)),
        "caneca": tarea_resumida(predicciones_caneca(original, canecas)),
    }
    resultado = {
        "modelo": f"{modelos['elegido']}, cargado desde {modelos[modelos['elegido']]['url']}",
        "bootstrap": BOOTSTRAP,
        "parametros": parametros_robustez(),
        "original": {"csv": CSV_ELEGIDO, **base},
        "perturbaciones": {},
    }
    material_original = original.set_index("archivo")["material"]
    for perturbacion in PERTURBACIONES:
        ruta = rutas.RESULTADOS / CSV_ROBUSTEZ.format(perturbacion.nombre)
        tabla = leer_csv(ruta)
        esperados = [archivo_alterado(a, perturbacion) for a in prueba["archivo"]]
        material = tarea_resumida(predicciones_material(tabla))
        caneca = tarea_resumida(predicciones_caneca(tabla, canecas))
        originales = [str(Path(a).with_suffix(".jpg").as_posix()) for a in tabla["archivo"]]
        resultado["perturbaciones"][perturbacion.nombre] = {
            "csv": relativa(ruta),
            "verificacion": verificar(tabla, esperados, canecas),
            "material": material,
            "caneca": caneca,
            "caida_exactitud_material": base["material"]["valores"]["exactitud"] - material["valores"]["exactitud"],
            "caida_exactitud_caneca": base["caneca"]["valores"]["exactitud"] - caneca["valores"]["exactitud"],
            "cambian_de_material": int((tabla["material"].to_numpy() != material_original.loc[originales].to_numpy()).sum()),
        }
    return resultado


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def escribir_json(ruta: Path, datos: dict) -> None:
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    canecas = cargar_json(rutas.CANECAS)
    modelos = cargar_json(rutas.MODELOS_JSON)
    particion = leer_particion()
    prueba = particion[particion["particion"] == PRUEBA]
    archivos = prueba["archivo"].tolist()
    salidas = {
        f"metricas-entrenamiento-{n}.json": lambda n=n: metricas_de_modelo(
            rutas.RESULTADOS / CSV_ENTRENAMIENTO.format(n), f"entrenamiento_{n}, copia local", archivos, canecas
        )
        for n in (1, 2)
    }
    salidas["metricas-elegido.json"] = lambda: metricas_elegido(modelos, archivos, canecas)
    salidas["metricas-robustez.json"] = lambda: metricas_robustez(modelos, prueba, canecas)
    for nombre, calcular in salidas.items():
        datos = calcular()
        escribir_json(rutas.RESULTADOS / nombre, datos)
        print(f"{nombre}: escrito")


if __name__ == "__main__":
    main()
