"""Conservación de los modelos alojados y selección del modelo que se evalúa.

Uso, desde ``clasificador/``:

    uv run python -m residuos.seleccion elegir
    uv run python -m residuos.seleccion comparar --local resultados/probabilidades-entrenamiento-2.csv \\
        --alojado resultados/probabilidades-prueba-elegido.csv

``elegir`` compara los pesos y las etiquetas de cada modelo descargado de su enlace (``modelo/alojado-N/``,
bajado antes con ``curl -L --fail <url><archivo>`` para ``model.json``, ``metadata.json`` y
``model.weights.bin``) con la copia exportada (``modelo/entrenamiento-N/``), aplica la regla de selección
fijada antes de entrenar y escribe ``resultados/modelos.json``. ``comparar`` registra en ese archivo si
las probabilidades del modelo elegido cargado desde su enlace son idénticas a las de la copia local.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from residuos import rutas

ENTRENAMIENTOS = (1, 2)
BALANCEADO = 2
DECIMALES_MOSTRADOS = 2
REGLA = {
    "texto": (
        "Se elige el entrenamiento con la mayor exactitud media por clase en la validación interna de "
        "Teachable Machine; si empatan en la cifra que muestra la herramienta, el balanceado. El conjunto "
        "de prueba no interviene en la elección."
    ),
    "criterio": "media de la exactitud por clase de la tabla «Precisión por clase», con los 2 decimales que muestra la herramienta",
    "empate": "medias iguales al redondearlas a 2 decimales; gana el balanceado (entrenamiento 2)",
    "fijada": "antes de entrenar (plan de la Actividad 5, §2.3)",
}


def sha256_de_archivo(ruta: Path) -> str:
    return hashlib.sha256(ruta.read_bytes()).hexdigest()


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def escribir_json(ruta: Path, datos: dict) -> None:
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def relativa(ruta: Path) -> str:
    return ruta.relative_to(rutas.RAIZ_PROYECTO).as_posix()


def comparar_model_json(alojado: dict, local: dict) -> dict:
    """El `model.json` alojado cambia la ruta de los pesos y suma claves informativas."""
    manifiesto_alojado, manifiesto_local = alojado["weightsManifest"], local["weightsManifest"]
    return {
        "topologia_identica": alojado["modelTopology"] == local["modelTopology"],
        "manifiesto_de_pesos_identico_salvo_ruta": [m["weights"] for m in manifiesto_alojado]
        == [m["weights"] for m in manifiesto_local],
        "ruta_de_pesos_alojado": [p for m in manifiesto_alojado for p in m["paths"]],
        "ruta_de_pesos_local": [p for m in manifiesto_local for p in m["paths"]],
        "claves_solo_en_alojado": sorted(set(alojado) - set(local)),
    }


def media_mostrada(entrenamiento: dict) -> float:
    exactitudes = [clase["exactitud"] for clase in entrenamiento["validacion_por_clase"].values()]
    return sum(exactitudes) / len(exactitudes)


def describir_modelo(numero: int, fecha_descarga: str) -> dict:
    entrenamiento = cargar_json(rutas.entrenamiento_json(numero))
    local = rutas.MODELOS / f"entrenamiento-{numero}"
    alojado = rutas.MODELOS / f"alojado-{numero}"
    sha_local = sha256_de_archivo(local / "weights.bin")
    sha_alojado = sha256_de_archivo(alojado / "model.weights.bin")
    etiquetas_local = cargar_json(local / "metadata.json")["labels"]
    etiquetas_alojado = cargar_json(alojado / "metadata.json")["labels"]
    resumen = entrenamiento["resumen_validacion"]
    return {
        "descripcion": entrenamiento["descripcion"],
        "url": entrenamiento["url"],
        "carpeta_local": relativa(local),
        "carpeta_alojada": relativa(alojado),
        "fecha_descarga_alojado": fecha_descarga,
        "sha256_local": sha_local,
        "sha256_alojado": sha_alojado,
        "pesos_identicos": sha_local == sha_alojado,
        "bytes_pesos": (alojado / "model.weights.bin").stat().st_size,
        "etiquetas": etiquetas_alojado,
        "etiquetas_identicas": etiquetas_local == etiquetas_alojado,
        "model_json": comparar_model_json(cargar_json(alojado / "model.json"), cargar_json(local / "model.json")),
        "exactitud_media_validacion": media_mostrada(entrenamiento),
        "exactitud_media_validacion_redondeada": round(media_mostrada(entrenamiento), DECIMALES_MOSTRADOS),
        "exactitud_por_clase_validacion": {c: v["exactitud"] for c, v in entrenamiento["validacion_por_clase"].items()},
        "exactitud_global_validacion": resumen["exactitud_global_desde_matriz"],
        "imagenes_validacion": resumen["imagenes_validacion"],
        "imagenes_entrenamiento": sum(entrenamiento["conteos"].values()),
    }


def elegir(modelos: dict[int, dict]) -> tuple[int, bool]:
    """Devuelve el entrenamiento elegido y si hubo empate en la cifra mostrada."""
    redondeadas = {n: m["exactitud_media_validacion_redondeada"] for n, m in modelos.items()}
    mejor = max(redondeadas.values())
    empatados = [n for n, valor in redondeadas.items() if valor == mejor]
    empate = len(empatados) > 1
    return (BALANCEADO if empate else empatados[0]), empate


def registrar_seleccion(fecha_descarga: str) -> dict:
    modelos = {n: describir_modelo(n, fecha_descarga) for n in ENTRENAMIENTOS}
    elegido, empate = elegir(modelos)
    anterior = cargar_json(rutas.MODELOS_JSON) if rutas.MODELOS_JSON.exists() else {}
    registro = {
        **{f"entrenamiento_{n}": m for n, m in modelos.items()},
        "elegido": f"entrenamiento_{elegido}",
        "regla": {
            **REGLA,
            "comparacion": {f"entrenamiento_{n}": m["exactitud_media_validacion_redondeada"] for n, m in modelos.items()},
            "hubo_empate": empate,
            "nota_validacion": (
                "Las validaciones internas tienen tamaños distintos (15 % de cada clase, redondeado hacia arriba): "
                + "; ".join(f"entrenamiento {n}, {m['imagenes_validacion']} imágenes" for n, m in modelos.items())
                + ". Teachable Machine las sortea sin semilla en cada entrenamiento."
            ),
        },
        "predicciones_alojado_vs_local": anterior.get(
            "predicciones_alojado_vs_local", {"identicas": None, "diferencia_maxima": None}
        ),
    }
    escribir_json(rutas.MODELOS_JSON, registro)
    return registro


def leer_probabilidades(ruta: Path) -> dict[str, dict[str, str]]:
    with ruta.open(newline="", encoding="utf-8") as archivo:
        return {fila["archivo"]: fila for fila in csv.DictReader(archivo)}


def comparar_predicciones(local: Path, alojado: Path) -> dict:
    """Igualdad exacta de cada probabilidad, material y caneca entre la copia local y el enlace."""
    filas_local, filas_alojado = leer_probabilidades(local), leer_probabilidades(alojado)
    if filas_local.keys() != filas_alojado.keys():
        raise ValueError("Los dos CSV no tienen las mismas imágenes")
    columnas = [c for c in next(iter(filas_local.values())) if c.startswith("prob_")]
    diferencia = max(
        abs(float(filas_local[a][c]) - float(filas_alojado[a][c])) for a in filas_local for c in columnas
    )
    mismas_decisiones = all(
        filas_local[a]["material"] == filas_alojado[a]["material"] and filas_local[a]["caneca"] == filas_alojado[a]["caneca"]
        for a in filas_local
    )
    return {
        "identicas": diferencia == 0 and mismas_decisiones,
        "diferencia_maxima": diferencia,
        "mismo_material_y_caneca": mismas_decisiones,
        "imagenes": len(filas_local),
        "columnas_comparadas": len(columnas),
        "csv_local": local.relative_to(rutas.RAIZ_CLASIFICADOR).as_posix(),
        "csv_alojado": alojado.relative_to(rutas.RAIZ_CLASIFICADOR).as_posix(),
    }


def registrar_comparacion(local: Path, alojado: Path) -> dict:
    registro = cargar_json(rutas.MODELOS_JSON)
    registro["predicciones_alojado_vs_local"] = comparar_predicciones(local.resolve(), alojado.resolve())
    escribir_json(rutas.MODELOS_JSON, registro)
    return registro["predicciones_alojado_vs_local"]


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(prog="residuos.seleccion", description=__doc__.splitlines()[0])
    sub = analizador.add_subparsers(dest="orden", required=True)
    eleccion = sub.add_parser("elegir", help="compara pesos y etiquetas y aplica la regla de selección")
    eleccion.add_argument("--fecha-descarga", default=datetime.now().astimezone().date().isoformat())
    comparacion = sub.add_parser("comparar", help="probabilidades de la copia local frente al enlace")
    comparacion.add_argument("--local", type=Path, required=True)
    comparacion.add_argument("--alojado", type=Path, required=True)
    return analizador.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    argumentos = _argumentos(argv)
    if argumentos.orden == "elegir":
        registro = registrar_seleccion(argumentos.fecha_descarga)
        for n in ENTRENAMIENTOS:
            modelo = registro[f"entrenamiento_{n}"]
            print(
                f"Entrenamiento {n}: pesos idénticos {modelo['pesos_identicos']}, etiquetas idénticas "
                f"{modelo['etiquetas_identicas']}, media de validación {modelo['exactitud_media_validacion_redondeada']:.2f}"
            )
        print(f"Elegido: {registro['elegido']} (empate: {registro['regla']['hubo_empate']})")
    else:
        resultado = registrar_comparacion(argumentos.local, argumentos.alojado)
        print(f"Idénticas: {resultado['identicas']}; diferencia máxima {resultado['diferencia_maxima']}")


if __name__ == "__main__":
    main()
