"""Datos de prueba de la app web.

Escribe dos archivos:

- `web/pruebas/fixture-prueba.json`, para `node --test`: las probabilidades del modelo elegido sobre el
  30 % de prueba (`resultados/probabilidades-prueba-elegido.csv`), las métricas por material y por caneca
  de `resultados/metricas-elegido.json` y dos casos pequeños (empates, una clase que nunca se predice y una
  clase ausente). Antes de escribirlo coteja `metricas-elegido.json` con scikit-learn; los casos pequeños
  se calculan con scikit-learn.
- `../datos/etiquetas-prueba.csv` (`archivo,clase`), la lista de etiquetas del 30 % que la prueba de
  paridad sube a la pestaña Evaluar. Vive con los datos pesados, que no se versionan.

Uso, desde `clasificador/`:

    uv run python -m residuos.fixture_app
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from residuos.metricas import (
    CSV_ELEGIDO,
    Predicciones,
    leer_csv,
    predicciones_caneca,
    predicciones_material,
)
from residuos.rutas import CANECAS, DATOS_PESADOS, PARTICION, RAIZ_CLASIFICADOR, RESULTADOS

FIXTURE = RAIZ_CLASIFICADOR / "web" / "pruebas" / "fixture-prueba.json"
ETIQUETAS_PRUEBA = DATOS_PESADOS / "etiquetas-prueba.csv"
METRICAS_ELEGIDO = RESULTADOS / "metricas-elegido.json"

# Las claves de metricas-elegido.json que calcula la app (no calcula AP ni intervalos).
CLAVES_POR_CLASE = ("precision", "recall", "f1", "auc")
CLAVES_GLOBALES = (
    "exactitud",
    "exactitud_balanceada",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "precision_ponderada",
    "recall_ponderado",
    "f1_ponderado",
    "auc_macro",
    "auc_micro",
)
TOLERANCIA_COTEJO = 1e-12


def por_clase(clases: Sequence[str], valores: Sequence) -> dict:
    return dict(zip(clases, valores, strict=True))


def metricas_sklearn(clases: Sequence[str], real: np.ndarray, predicha: np.ndarray, probabilidades: np.ndarray) -> dict:
    """Las métricas de la app calculadas con scikit-learn.

    Por clase, con todas las clases; los promedios, con las clases presentes en lo real o en lo predicho
    (lo que hace scikit-learn sin `labels`). AUC por clase solo donde hay positivos y negativos; su
    media, sobre esas clases.
    """
    k = len(clases)
    todas = list(range(k))
    precision, recall, f1, _ = precision_recall_fscore_support(real, predicha, labels=todas, zero_division=0)
    macro = precision_recall_fscore_support(real, predicha, average="macro", zero_division=0)
    ponderado = precision_recall_fscore_support(real, predicha, average="weighted", zero_division=0)
    binaria = label_binarize(real, classes=todas)
    auc = [
        float(roc_auc_score(binaria[:, c], probabilidades[:, c])) if 0 < binaria[:, c].sum() < len(real) else None
        for c in todas
    ]
    definidas = [a for a in auc if a is not None]
    return {
        "precision": por_clase(clases, [float(v) for v in precision]),
        "recall": por_clase(clases, [float(v) for v in recall]),
        "f1": por_clase(clases, [float(v) for v in f1]),
        "auc": por_clase(clases, auc),
        "exactitud": float(accuracy_score(real, predicha)),
        "exactitud_balanceada": float(balanced_accuracy_score(real, predicha)),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "precision_ponderada": float(ponderado[0]),
        "recall_ponderado": float(ponderado[1]),
        "f1_ponderado": float(ponderado[2]),
        "auc_macro": float(np.mean(definidas)) if definidas else None,
        "auc_micro": float(roc_auc_score(binaria.ravel(), probabilidades.ravel())),
    }


def valores_de_la_app(valores: dict, clases: Sequence[str]) -> dict:
    """Las métricas que calcula la app, anidadas como ella las da (`precision.Cardboard` pasa a
    `precision` → `Cardboard`)."""
    anidados = {clave: {clase: valores[f"{clave}.{clase}"] for clase in clases} for clave in CLAVES_POR_CLASE}
    return anidados | {clave: valores[clave] for clave in CLAVES_GLOBALES}


def diferencia_maxima(a: dict, b: dict) -> float:
    """Mayor diferencia absoluta entre dos diccionarios de métricas con las mismas claves."""
    diferencias = []
    for clave, valor in a.items():
        if isinstance(valor, dict):
            diferencias.append(diferencia_maxima(valor, b[clave]))
        elif valor is None or b[clave] is None:
            diferencias.append(0.0 if valor is b[clave] else float("inf"))
        else:
            diferencias.append(abs(valor - b[clave]))
    return max(diferencias)


def tarea_esperada(nombre: str, predicciones: Predicciones, metricas: dict) -> dict:
    """La tarea de metricas-elegido.json, después de comprobar que coincide con scikit-learn."""
    guardada = metricas[nombre]
    valores = valores_de_la_app(guardada["valores"], guardada["clases"])
    calculada = metricas_sklearn(predicciones.clases, predicciones.real, predicciones.predicha, predicciones.probabilidades)
    diferencia = diferencia_maxima(valores, calculada)
    if diferencia > TOLERANCIA_COTEJO:
        raise ValueError(f"metricas-elegido.json ({nombre}) difiere de scikit-learn en {diferencia}")
    return {"clases": guardada["clases"], "matriz": guardada["matriz"], "soporte": guardada["soporte"], "valores": valores}


def caso_pequeno(nombre: str, clases: list[str], real: list[int], probabilidades: list[list[float]]) -> dict:
    """Un caso a mano: la clase predicha es la más probable y lo esperado sale de scikit-learn."""
    matriz_prob = np.array(probabilidades)
    predicha = matriz_prob.argmax(axis=1)
    return {
        "nombre": nombre,
        "clases": clases,
        "reales": real,
        "predichas": predicha.tolist(),
        "probabilidades": probabilidades,
        "esperado": metricas_sklearn(clases, np.array(real), predicha, matriz_prob),
    }


def casos_pequenos() -> list[dict]:
    return [
        caso_pequeno(
            "empates y una clase que nunca se predice",
            ["a", "b", "c"],
            [0, 0, 0, 1, 1, 1, 2, 2],
            [
                [0.6, 0.3, 0.1],
                [0.5, 0.25, 0.25],
                [0.4, 0.5, 0.1],
                [0.2, 0.7, 0.1],
                [0.3, 0.6, 0.1],
                [0.5, 0.3, 0.2],
                [0.5, 0.25, 0.25],
                [0.6, 0.2, 0.2],
            ],
        ),
        caso_pequeno(
            "una clase ausente en lo real y en lo predicho",
            ["a", "b", "c"],
            [0, 0, 1, 1, 0, 1],
            [
                [0.7, 0.2, 0.1],
                [0.4, 0.5, 0.1],
                [0.2, 0.7, 0.1],
                [0.3, 0.6, 0.1],
                [0.8, 0.1, 0.1],
                [0.5, 0.4, 0.1],
            ],
        ),
    ]


def construir_fixture() -> dict:
    tabla = leer_csv(RESULTADOS / CSV_ELEGIDO)
    canecas = json.loads(CANECAS.read_text(encoding="utf-8"))
    metricas = json.loads(METRICAS_ELEGIDO.read_text(encoding="utf-8"))
    material = predicciones_material(tabla)
    caneca = predicciones_caneca(tabla, canecas)
    columnas_material = [f"prob_{e}" for e in material.clases]
    columnas_caneca = [f"prob_caneca_{c}" for c in caneca.clases]
    filas = [
        [
            fila["archivo"],
            fila["clase_real"],
            [float(fila[c]) for c in columnas_material],
            fila["material"],
            fila["caneca"],
            [float(fila[c]) for c in columnas_caneca],
        ]
        for fila in tabla.to_dict("records")
    ]
    return {
        "descripcion": (
            "Probabilidades del modelo elegido sobre el 30 % de prueba y sus métricas, cotejadas con scikit-learn. "
            "Cada fila: archivo, clase real, probabilidades por etiqueta, material, caneca y probabilidades por caneca."
        ),
        "etiquetas": material.clases,
        "clases_caneca": caneca.clases,
        "filas": filas,
        "esperado": {
            "material": tarea_esperada("material", material, metricas),
            "caneca": tarea_esperada("caneca", caneca, metricas),
        },
        "casos_pequenos": casos_pequenos(),
    }


def texto_json(datos: dict) -> str:
    return json.dumps(datos, ensure_ascii=False, separators=(",", ":")) + "\n"


def escribir_fixture(destino: Path = FIXTURE) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto_json(construir_fixture()), encoding="utf-8")
    return destino


def escribir_etiquetas_prueba(destino: Path = ETIQUETAS_PRUEBA) -> Path:
    particion = pd.read_csv(PARTICION, keep_default_na=False)
    prueba = particion.loc[particion["particion"] == "prueba", ["archivo", "clase"]]
    destino.parent.mkdir(parents=True, exist_ok=True)
    prueba.to_csv(destino, index=False)
    return destino


def main() -> None:
    print(f"Fixture: {escribir_fixture().relative_to(RAIZ_CLASIFICADOR)}")
    print(f"Etiquetas de prueba: {escribir_etiquetas_prueba()}")


if __name__ == "__main__":
    main()
