"""Métricas de la prueba en campo: el modelo elegido sobre las fotos propias, frente a la prueba del 30 %.

Uso, desde ``clasificador/``, después de correr el arnés sobre las fotos:

    uv run python -m residuos.metricas_campo

Lee ``resultados/probabilidades-campo-elegido.csv`` y su ``.entorno.json`` (los escribe el arnés),
``prueba-campo/registro.csv``, ``canecas.json``, ``modelos.json``, ``probabilidades-prueba-elegido.csv``
y ``metricas-elegido.json``, y escribe ``resultados/metricas-campo.json``. Se niega a escribir si el
CSV no trae una fila por foto del registro o si el modelo no es el elegido cargado desde su enlace.

La prueba de campo está balanceada por diseño (las mismas fotos por clase) y el 30 % conserva el
desbalance de RealWaste, así que la comparación de material es contra la exactitud balanceada del
30 %, no contra su exactitud cruda. Los intervalos usan el bootstrap del proyecto
(``metricas.BOOTSTRAP``); en la diferencia, cada muestra se remuestrea por separado.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from residuos import rutas
from residuos.arnes import ruta_entorno
from residuos.campo import ORIENTACIONES_QUE_GIRAN, describir_foto, leer_registro
from residuos.metricas import (
    BOOTSTRAP,
    CSV_ELEGIDO,
    Predicciones,
    SenalesBlanca,
    caneca_de,
    caneca_vs_material,
    cargar_json,
    cobertura_blanca,
    desde_matriz,
    errores,
    escribir_json,
    evaluar_tarea,
    leer_csv,
    matriz_confusion,
    percentiles,
    predicciones_caneca,
    predicciones_material,
    punto_cobertura,
    remuestreos,
    verificar,
)

UMBRAL_BLANCA = 0.9  # el que el informe propone para el aviso de caneca dudosa
NOTA_COMPARACION = (
    "La prueba de campo está balanceada por diseño (las mismas fotos por clase), mientras que el 30 % "
    "conserva el desbalance de RealWaste. Por eso la exactitud de material de campo se compara con la "
    "exactitud balanceada del 30 % (media del recall por clase) y no con su exactitud cruda; en una muestra "
    "balanceada, exactitud y exactitud balanceada coinciden. La caneca se compara por exactitud, como se "
    "pidió, aunque la mezcla de canecas difiere entre las dos muestras (ver el soporte de cada una). El "
    "intervalo de la diferencia remuestrea cada muestra por separado, con los parámetros de bootstrap."
)
CONDICIONES = {  # nivel del registro: (nombre corto para las macros, descripción)
    "fondo": {
        "A": ("mesón", "mesón de la cocina"),
        "B": ("piso", "piso"),
        "C": ("mesa", "mesa del comedor"),
        "D": ("tabla", "tabla de picar o bandeja"),
        "E": ("bolsa", "bolsa de basura abierta o boca de una caneca"),
    },
    "luz": {
        "L1": ("natural", "natural de día, junto a una ventana"),
        "L2": ("techo", "blanca del techo"),
        "L3": ("cálida", "cálida de noche"),
    },
    "encuadre": {
        "cenital": ("cenital", "cenital, desde arriba"),
        "45°": ("inclinado", "inclinado a 45°"),
    },
}


def tarea(p: Predicciones) -> dict:
    """Cifras puntuales, intervalos y matriz, sin ROC ni AP (con pocas fotos por clase no aportan)."""
    return evaluar_tarea(p, con_curvas=False)


def muestras(p: Predicciones, clave: str) -> np.ndarray:
    """La cifra en cada remuestreo, con los mismos índices que usa `evaluar_tarea` para esa muestra."""
    k = len(p.clases)
    return np.array([desde_matriz(matriz_confusion(p.real[i], p.predicha[i], k))[clave] for i in remuestreos(len(p.real))])


def puntual(p: Predicciones, clave: str) -> float:
    return float(desde_matriz(matriz_confusion(p.real, p.predicha, len(p.clases)))[clave])


def comparar(campo: Predicciones, prueba: Predicciones, clave: str) -> dict:
    """Campo menos prueba en una cifra, con su intervalo por bootstrap independiente."""
    diferencias = muestras(campo, clave) - muestras(prueba, clave)
    inferior, superior = percentiles(diferencias)
    valor_campo, valor_prueba = puntual(campo, clave), puntual(prueba, clave)
    return {
        "metrica": clave,
        "campo": valor_campo,
        "prueba": valor_prueba,
        "soporte_campo": len(campo.real),
        "soporte_prueba": len(prueba.real),
        "diferencia": valor_campo - valor_prueba,
        "intervalo": [inferior, superior],
        "excluye_cero": superior < 0 or inferior > 0,
    }


def umbral_blanca(tabla: pd.DataFrame, canecas: dict, prueba: dict) -> dict:
    """La precisión y la cobertura de la blanca con el umbral del informe, en campo y en el 30 %."""
    senales = SenalesBlanca.desde(tabla, canecas)
    remuestreados = [punto_cobertura(senales.submuestra(i), UMBRAL_BLANCA) for i in remuestreos(len(tabla))]

    def intervalo(clave: str) -> list[float]:
        return list(percentiles(np.array([np.nan if p[clave] is None else p[clave] for p in remuestreados], dtype=float)))

    campo = punto_cobertura(senales, UMBRAL_BLANCA)
    base = punto_cobertura(senales, None)
    en_prueba = next(p for p in prueba["cobertura_blanca"]["puntos"] if p["umbral"] == UMBRAL_BLANCA)
    return {
        "umbral": UMBRAL_BLANCA,
        "campo": {"valores": campo, "intervalos": {"precision": intervalo("precision"), "cobertura": intervalo("cobertura")}},
        "campo_sin_abstencion": base,
        "prueba": en_prueba,
        "prueba_sin_abstencion": prueba["cobertura_blanca"]["sin_abstencion"],
        "sube_la_precision_en_campo": campo["precision"] is not None and campo["precision"] > base["precision"],
    }


def por_condicion(tabla: pd.DataFrame, registro: pd.DataFrame, canecas: dict) -> dict:
    """Exactitud de material y de caneca por fondo, luz y encuadre (cifras puntuales)."""
    datos = tabla.merge(registro[["archivo", *CONDICIONES]], on="archivo", validate="one_to_one")
    datos["acierta_material"] = datos["material"] == datos["clase_real"]
    datos["acierta_caneca"] = datos["caneca"] == datos["clase_real"].map(caneca_de(canecas))
    resultado = {}
    for factor, niveles in CONDICIONES.items():
        resultado[factor] = {
            nivel: {
                "corto": niveles[nivel][0],
                "nombre": niveles[nivel][1],
                "imagenes": len(grupo),
                "exactitud_material": float(grupo["acierta_material"].mean()),
                "exactitud_caneca": float(grupo["acierta_caneca"].mean()),
            }
            for nivel, grupo in datos.groupby(factor, sort=True)
        }
    return resultado


def orientacion(carpeta: Path, archivos: Sequence[str]) -> dict:
    etiquetas = [describir_foto(carpeta, a).orientacion_exif for a in archivos]
    return {
        "aplicada_por": "el navegador al decodificar, igual que ImageOps.exif_transpose (tests/test_campo.py)",
        "fotos_con_etiqueta_distinta_de_1": sum(e not in (None, 1) for e in etiquetas),
        "fotos_que_giran": sum(e in ORIENTACIONES_QUE_GIRAN for e in etiquetas),
    }


def con_sustituto(registro: pd.DataFrame) -> dict:
    usados = registro[registro["objeto_usado"].str.strip() != ""]
    return {"definicion": "filas del registro con objeto_usado lleno", "fotos": len(usados), "archivos": usados["archivo"].tolist()}


def calcular(
    tabla: pd.DataFrame, entorno: dict, registro: pd.DataFrame, canecas: dict, modelos: dict, prueba: pd.DataFrame, elegido: dict, carpeta: Path
) -> dict:
    """`carpeta` es la de las fotos evaluadas; `prueba` y `elegido`, el CSV y las métricas del 30 %."""
    url_elegido = modelos[modelos["elegido"]]["url"]
    comprobacion = verificar(tabla, registro["archivo"].tolist(), canecas)
    comprobacion["modelo_es_el_elegido"] = entorno["modelo"] == url_elegido
    if not (comprobacion["valido"] and comprobacion["modelo_es_el_elegido"]):
        raise ValueError(f"El CSV de campo no pasa la verificación: {comprobacion}")
    material, material_prueba = predicciones_material(tabla), predicciones_material(prueba)
    caneca, caneca_prueba = predicciones_caneca(tabla, canecas), predicciones_caneca(prueba, canecas)
    return {
        "modelo": f"{modelos['elegido']}, cargado desde {url_elegido}",
        "base_imagenes": entorno["base_imagenes"],
        "imagenes": len(tabla),
        "bootstrap": BOOTSTRAP,
        "verificacion": comprobacion,
        "con_sustituto": con_sustituto(registro),
        "orientacion_exif": orientacion(carpeta, registro["archivo"].tolist()),
        "material": tarea(material),
        "caneca": tarea(caneca),
        "caneca_vs_material_mas_probable": caneca_vs_material(tabla, canecas),
        "errores": errores(tabla, canecas),
        "comparacion_con_prueba": {
            "nota": NOTA_COMPARACION,
            "csv_prueba": f"resultados/{CSV_ELEGIDO}",
            "material_exactitud_balanceada": comparar(material, material_prueba, "exactitud_balanceada"),
            "caneca_exactitud": comparar(caneca, caneca_prueba, "exactitud"),
        },
        "umbral_blanca": umbral_blanca(tabla, canecas, elegido),
        "cobertura_blanca": cobertura_blanca(tabla, canecas),
        "por_condicion": por_condicion(tabla, registro, canecas),
    }


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(prog="residuos.metricas_campo", description=__doc__.splitlines()[0])
    analizador.add_argument("--probabilidades", type=Path, default=rutas.PROBABILIDADES_CAMPO, help="CSV del arnés sobre las fotos")
    analizador.add_argument("--registro", type=Path, default=rutas.REGISTRO_CAMPO)
    analizador.add_argument("--salida", type=Path, default=rutas.METRICAS_CAMPO)
    return analizador.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    argumentos = _argumentos(argv)
    entorno = cargar_json(ruta_entorno(argumentos.probabilidades))
    registro = leer_registro(argumentos.registro)
    datos = calcular(
        tabla=leer_csv(argumentos.probabilidades),
        entorno=entorno,
        registro=registro,
        canecas=cargar_json(rutas.CANECAS),
        modelos=cargar_json(rutas.MODELOS_JSON),
        prueba=leer_csv(rutas.RESULTADOS / CSV_ELEGIDO),
        elegido=cargar_json(rutas.RESULTADOS / "metricas-elegido.json"),
        carpeta=rutas.RAIZ_PROYECTO.parent / entorno["base_imagenes"],
    )
    escribir_json(argumentos.salida, datos)
    material, caneca = datos["material"]["valores"], datos["caneca"]["valores"]
    print(
        f"{datos['imagenes']} fotos; exactitud balanceada de material {material['exactitud_balanceada']:.3f}; "
        f"exactitud de caneca {caneca['exactitud']:.3f}"
    )
    print(f"Escrito {argumentos.salida}")


if __name__ == "__main__":
    main()
