"""Pruebas del README: ninguna cifra ni dato del modelo se escribe a mano.

Las métricas y sus intervalos salen de `resultados/metricas-elegido.json`, los conteos y el bootstrap de ese
mismo archivo y de `datos/inventario.json`, el enlace del modelo de `resultados/modelos.json` y la tabla de
canecas de `config/canecas.json`. El formato es el del informe (`residuos.formato`).
"""

import json
from pathlib import Path

import pytest

from residuos import formato, rutas

README = rutas.RAIZ_CLASIFICADOR / "README.md"
METRICAS_ELEGIDO = rutas.RESULTADOS / "metricas-elegido.json"

# Primera celda de la fila del README: (tarea, clave) en metricas-elegido.json.
CIFRAS = {
    "F1 macro por material": ("material", "f1_macro"),
    "Exactitud balanceada por material": ("material", "exactitud_balanceada"),
    "Precisión de la caneca blanca": ("caneca", "precision.blanca"),
}


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def filas_de_tablas(texto: str) -> dict[str, list[str]]:
    """Cada fila de las tablas de Markdown, por su primera celda."""
    filas = {}
    for linea in texto.splitlines():
        if linea.startswith("|"):
            primera, *resto = (celda.strip() for celda in linea.strip().strip("|").split("|"))
            filas[primera] = resto
    return filas


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def metricas() -> dict:
    return cargar_json(METRICAS_ELEGIDO)


@pytest.mark.parametrize("etiqueta", CIFRAS)
def test_cada_metrica_y_su_intervalo_salen_de_metricas_elegido(readme, metricas, etiqueta):
    tarea, clave = CIFRAS[etiqueta]
    esperado = [formato.metrica(metricas[tarea]["valores"][clave]), formato.intervalo(metricas[tarea]["intervalos"][clave])]
    assert filas_de_tablas(readme)[etiqueta] == esperado


def test_conteos_y_bootstrap_salen_de_los_json(readme, metricas):
    inventario = cargar_json(rutas.INVENTARIO)
    bootstrap = metricas["bootstrap"]
    esperadas = [
        f"{formato.entero(inventario['zip']['imagenes'])} imágenes de RealWaste",
        f"{formato.entero(metricas['imagenes'])} imágenes de prueba",
        f"{formato.entero(bootstrap['remuestreos'])} remuestreos",
        f"confianza al {round(bootstrap['nivel'] * 100)} %",
        f"IC {round(bootstrap['nivel'] * 100)} %",
    ]
    assert [cadena for cadena in esperadas if cadena not in readme] == []


def test_el_enlace_del_modelo_es_el_del_elegido(readme):
    modelos = cargar_json(rutas.MODELOS_JSON)
    assert f"MODELO={modelos[modelos['elegido']]['url']}\n" in readme


def test_la_tabla_de_canecas_es_la_de_canecas_json(readme):
    canecas = cargar_json(rutas.CANECAS)
    filas = filas_de_tablas(readme)
    for etiqueta, material in canecas["materiales"].items():
        assert filas[material["nombre"]] == [etiqueta, material["caneca"].capitalize()]
