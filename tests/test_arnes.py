"""Pruebas del arnés: paridad con la vista previa de Teachable Machine y traducción a caneca.

Las marcadas `navegador` usan la instancia de Brave del puerto 9333 y se saltan si no responde. La
de paridad escribe `resultados/paridad-a2.csv`; las demás leen ese archivo sin abrir el navegador. Las de
paridad se saltan si faltan sus archivos locales, que no se publican.
"""

import json
import random

import pytest
from requisitos import requiere_paridad, requiere_paridad_ejecutable

from residuos.arnes import (
    CANECAS,
    PARIDAD_A2,
    RAIZ_CLASIFICADOR,
    Imagen,
    cargar_canecas,
    cdp_responde,
    comparar_paridad,
    correr_paridad,
    escribir_csv,
    leer_csv,
    pagina_del_arnes,
    ruta_entorno,
    url_local,
)

TOLERANCIA_PARIDAD = 0.0005  # la vista previa quedó registrada con 4 decimales
TOLERANCIA_SUMA = 1e-4
IMAGENES_DE_PARIDAD = 20

NAVEGADOR_DISPONIBLE = cdp_responde()


def requiere_navegador(prueba):
    prueba = pytest.mark.navegador(prueba)
    return pytest.mark.skipif(not NAVEGADOR_DISPONIBLE, reason="la instancia de Brave no responde en el puerto 9333")(prueba)


def suma_por_caneca(probabilidades: dict[str, float], canecas: dict) -> dict[str, float]:
    """Oráculo independiente de la traducción a caneca del navegador."""
    suma = dict.fromkeys(canecas["canecas"], 0.0)
    for etiqueta, probabilidad in probabilidades.items():
        suma[canecas["materiales"][etiqueta]["caneca"]] += probabilidad
    return suma


def casos_de_probabilidades(etiquetas: list[str]) -> list[dict[str, float]]:
    """Un caso seguro por material, el uniforme, uno donde material y caneca difieren y 30 al azar."""
    seguros = [{e: float(e == elegida) for e in etiquetas} for elegida in etiquetas]
    uniforme = {e: 1 / len(etiquetas) for e in etiquetas}
    difieren = dict.fromkeys(etiquetas, 0.0) | {"Miscellaneous Trash": 0.4, "Cardboard": 0.3, "Paper": 0.3}
    azar = random.Random(42)
    aleatorios = []
    for _ in range(30):
        pesos = [azar.random() for _ in etiquetas]
        aleatorios.append({e: p / sum(pesos) for e, p in zip(etiquetas, pesos, strict=True)})
    return [*seguros, uniforme, difieren, *aleatorios]


def canecas_temporal() -> dict:
    """Copia mínima con el esquema de config/canecas.json, solo para cuando ese archivo aún no existe."""
    por_caneca = {
        "blanca": ["Cardboard", "Glass", "Metal", "Paper", "Plastic", "Textile Trash"],
        "verde": ["Food Organics", "Vegetation"],
        "negra": ["Miscellaneous Trash"],
    }
    return {
        "fuente": "copia temporal de prueba",
        "canecas": {c: {"nombre": c, "descripcion": c} for c in por_caneca},
        "materiales": {
            e: {"nombre": e, "caneca": c, "fuente": "copia temporal de prueba"}
            for c, etiquetas in por_caneca.items()
            for e in etiquetas
        },
    }


@pytest.fixture(scope="module")
def canecas(tmp_path_factory) -> dict:
    if CANECAS.exists():
        return cargar_canecas(CANECAS)
    ruta = tmp_path_factory.mktemp("config") / "canecas.json"
    ruta.write_text(json.dumps(canecas_temporal(), ensure_ascii=False), encoding="utf-8")
    return cargar_canecas(ruta)


@requiere_navegador
@requiere_paridad_ejecutable
def test_paridad_corre_en_el_navegador():
    entorno = correr_paridad()
    assert entorno["imagenes"] == IMAGENES_DE_PARIDAD
    assert entorno["tensores_al_final"] == entorno["tensores_antes"]


@requiere_paridad
def test_paridad_misma_clase_en_todas():
    resumen = comparar_paridad()
    assert (resumen.misma_clase, resumen.total) == (IMAGENES_DE_PARIDAD, IMAGENES_DE_PARIDAD)


@requiere_paridad
def test_paridad_diferencia_maxima():
    assert comparar_paridad().diferencia_maxima <= TOLERANCIA_PARIDAD


@requiere_paridad
def test_paridad_probabilidades_suman_uno():
    for fila in leer_csv(PARIDAD_A2.salida):
        suma = sum(float(v) for c, v in fila.items() if c.startswith("prob_"))
        assert suma == pytest.approx(1, abs=TOLERANCIA_SUMA)


@requiere_paridad
def test_paridad_en_webgl_con_la_rtx_5080():
    entorno = json.loads(ruta_entorno(PARIDAD_A2.salida).read_text(encoding="utf-8"))
    assert entorno["backend"] == "webgl"
    assert "RTX 5080" in entorno["renderizador_gpu"]


@requiere_navegador
def test_caneca_del_navegador_igual_a_la_de_python(canecas):
    casos = casos_de_probabilidades(list(canecas["materiales"]))
    modulo = url_local(RAIZ_CLASIFICADOR / "web" / "inferencia.js")
    with pagina_del_arnes() as (pagina, _):
        obtenidos = pagina.evaluate(
            """async ({modulo, canecas, casos}) => {
                const { traducirACaneca } = await import(modulo);
                return casos.map((probabilidades) => traducirACaneca(probabilidades, canecas));
            }""",
            {"modulo": modulo, "canecas": canecas, "casos": casos},
        )
    for probabilidades, obtenido in zip(casos, obtenidos, strict=True):
        esperado = suma_por_caneca(probabilidades, canecas)
        assert obtenido["probabilidadesCaneca"] == pytest.approx(esperado, abs=1e-12)
        assert obtenido["caneca"] == max(esperado, key=esperado.get)


def test_csv_con_canecas_conserva_columnas_y_valores(tmp_path, canecas):
    etiquetas = list(canecas["materiales"])
    probabilidades = casos_de_probabilidades(etiquetas)[-1]
    por_caneca = suma_por_caneca(probabilidades, canecas)
    fila = {
        "probabilidades": probabilidades,
        "material": max(probabilidades, key=probabilidades.get),
        "caneca": max(por_caneca, key=por_caneca.get),
        "probabilidadesCaneca": por_caneca,
    }
    salida = tmp_path / "probabilidades.csv"
    escribir_csv(salida, [Imagen("Cardboard/Cardboard_1.jpg", "Cardboard")], [fila], etiquetas, canecas)
    [leida] = leer_csv(salida)
    assert list(leida) == [
        "archivo",
        "clase_real",
        *(f"prob_{e}" for e in etiquetas),
        "material",
        "caneca",
        *(f"prob_caneca_{c}" for c in canecas["canecas"]),
    ]
    assert {e: float(leida[f"prob_{e}"]) for e in etiquetas} == probabilidades
    assert {c: float(leida[f"prob_caneca_{c}"]) for c in canecas["canecas"]} == por_caneca
