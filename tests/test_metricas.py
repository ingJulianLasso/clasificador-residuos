"""Pruebas de la evaluación: CSV del arnés, conservación del modelo, métricas recalculadas por otra vía,
agregación por caneca, robustez, cifras del informe y figuras.

Ninguna abre el navegador: leen lo que dejaron el arnés y `residuos.metricas` en `resultados/`.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from requisitos import requiere_informe, requiere_paridad
from sklearn.metrics import f1_score

from residuos import cifras, metricas, rutas, seleccion
from residuos.datos import PRUEBA, leer_particion
from residuos.robustez import (
    PERTURBACIONES,
    POR_NOMBRE,
    acercar,
    alterar,
    archivo_alterado,
    girar,
)

TOLERANCIA = 1e-9
FUENTES_PERMITIDAS = re.compile(
    r"^(particion\.csv|inventario\.json|canecas\.json|modelos\.json|paridad-a2\.csv"
    r"|tm-entrenamiento-[12]\.json|metricas-(entrenamiento-[12]|elegido|robustez|campo)\.json)$"
)


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def canecas() -> dict:
    return cargar_json(rutas.CANECAS)


@pytest.fixture(scope="module")
def archivos_prueba() -> list[str]:
    particion = leer_particion()
    return particion.loc[particion["particion"] == PRUEBA, "archivo"].tolist()


@pytest.fixture(scope="module")
def elegido() -> dict:
    return cargar_json(rutas.RESULTADOS / "metricas-elegido.json")


@pytest.fixture(scope="module")
def tabla_elegido() -> pd.DataFrame:
    return metricas.leer_csv(rutas.RESULTADOS / metricas.CSV_ELEGIDO)


def csv_de_prueba() -> list[tuple[str, Path, str | None]]:
    """Cada CSV del arnés sobre la prueba, con la perturbación que lleva (si alguna)."""
    casos = [(f"entrenamiento-{n}", rutas.RESULTADOS / metricas.CSV_ENTRENAMIENTO.format(n), None) for n in (1, 2)]
    casos.append(("elegido", rutas.RESULTADOS / metricas.CSV_ELEGIDO, None))
    casos += [(p.nombre, rutas.RESULTADOS / metricas.CSV_ROBUSTEZ.format(p.nombre), p.nombre) for p in PERTURBACIONES]
    return casos


# CSV del arnés


@pytest.mark.parametrize(("nombre", "ruta", "perturbacion"), csv_de_prueba(), ids=[c[0] for c in csv_de_prueba()])
def test_csv_del_arnes_es_valido(nombre, ruta, perturbacion, canecas, archivos_prueba):
    tabla = metricas.leer_csv(ruta)
    esperados = archivos_prueba if perturbacion is None else [archivo_alterado(a, POR_NOMBRE[perturbacion]) for a in archivos_prueba]
    resultado = metricas.verificar(tabla, esperados, canecas)
    assert resultado["una_fila_por_imagen_de_prueba"], nombre
    assert resultado["suman_uno"], nombre
    assert resultado["caneca_igual_a_python"], nombre
    assert resultado["diferencia_maxima_prob_caneca"] == 0, nombre


# Conservación del modelo y regla de selección


def test_el_modelo_alojado_es_el_evaluado():
    modelos = cargar_json(rutas.MODELOS_JSON)
    for clave in ("entrenamiento_1", "entrenamiento_2"):
        assert modelos[clave]["sha256_alojado"] == modelos[clave]["sha256_local"]
        assert modelos[clave]["etiquetas_identicas"]
    assert modelos["predicciones_alojado_vs_local"]["identicas"] is True
    assert modelos["predicciones_alojado_vs_local"]["diferencia_maxima"] == 0


def test_la_regla_elige_por_la_validacion_y_desempata_con_el_balanceado():
    modelos = cargar_json(rutas.MODELOS_JSON)
    medias = {}
    for n in (1, 2):
        exactitudes = [c["exactitud"] for c in cargar_json(rutas.entrenamiento_json(n))["validacion_por_clase"].values()]
        medias[n] = round(sum(exactitudes) / len(exactitudes), 2)
    esperado = 2 if medias[2] >= medias[1] else 1
    assert modelos["elegido"] == f"entrenamiento_{esperado}"
    empate = {n: {"exactitud_media_validacion_redondeada": 0.8} for n in (1, 2)}
    assert seleccion.elegir(empate) == (2, True)


# Métricas recalculadas por otra vía


def test_f1_desde_la_matriz_coincide(elegido):
    """F1 = 2 VP / (2 VP + FP + FN), sin pasar por la precisión ni el recall."""
    for tarea in ("material", "caneca"):
        datos = elegido[tarea]
        matriz = np.array(datos["matriz"])
        vp = np.diag(matriz)
        f1 = 2 * vp / (2 * vp + (matriz.sum(axis=0) - vp) + (matriz.sum(axis=1) - vp))
        for clase, valor in zip(datos["clases"], f1, strict=True):
            assert datos["valores"][f"f1.{clase}"] == pytest.approx(valor, abs=TOLERANCIA)
        assert datos["valores"]["f1_macro"] == pytest.approx(f1.mean(), abs=TOLERANCIA)


def test_f1_coincide_con_scikit_learn_sobre_el_csv(elegido, tabla_elegido):
    clases = elegido["material"]["clases"]
    por_clase = f1_score(tabla_elegido["clase_real"], tabla_elegido["material"], labels=clases, average=None)
    for clase, valor in zip(clases, por_clase, strict=True):
        assert elegido["material"]["valores"][f"f1.{clase}"] == pytest.approx(valor, abs=TOLERANCIA)


def auc_por_trapecios(positiva: np.ndarray, puntaje: np.ndarray) -> float:
    """ROC propia: un punto por umbral distinto, y área con la regla del trapecio."""
    orden = np.argsort(-puntaje, kind="stable")
    positiva, puntaje = positiva[orden], puntaje[orden]
    cortes = np.r_[np.flatnonzero(np.diff(puntaje)), len(puntaje) - 1]
    vp = np.r_[0, np.cumsum(positiva)[cortes]]
    fp = np.r_[0, np.cumsum(~positiva)[cortes]]
    return float(np.trapezoid(vp / positiva.sum(), fp / (~positiva).sum()))


def test_auc_por_trapecios_coincide(elegido, tabla_elegido, canecas):
    for clase in elegido["material"]["clases"]:
        auc = auc_por_trapecios((tabla_elegido["clase_real"] == clase).to_numpy(), tabla_elegido[f"prob_{clase}"].to_numpy())
        assert elegido["material"]["valores"][f"auc.{clase}"] == pytest.approx(auc, abs=TOLERANCIA)
    real_caneca = tabla_elegido["clase_real"].map(metricas.caneca_de(canecas))
    for caneca in elegido["caneca"]["clases"]:
        auc = auc_por_trapecios((real_caneca == caneca).to_numpy(), tabla_elegido[f"prob_caneca_{caneca}"].to_numpy())
        assert elegido["caneca"]["valores"][f"auc.{caneca}"] == pytest.approx(auc, abs=TOLERANCIA)


def test_los_intervalos_contienen_la_cifra_puntual(elegido):
    for tarea in ("material", "caneca"):
        for clave, (inferior, superior) in elegido[tarea]["intervalos"].items():
            assert inferior <= elegido[tarea]["valores"][clave] + TOLERANCIA, clave
            assert elegido[tarea]["valores"][clave] <= superior + TOLERANCIA, clave


def test_el_bootstrap_es_reproducible():
    assert np.array_equal(metricas.remuestreos(50), metricas.remuestreos(50))


# Casos a mano


def test_agregacion_por_caneca_con_un_caso_a_mano(canecas):
    etiquetas = list(canecas["materiales"])
    fila = dict.fromkeys(etiquetas, 0.0) | {"Miscellaneous Trash": 0.4, "Cardboard": 0.3, "Paper": 0.3}
    tabla = pd.DataFrame([{f"prob_{e}": p for e, p in fila.items()}])
    sumas, ganadora = metricas.recalcular_canecas(tabla, canecas)
    assert sumas.loc[0].to_dict() == pytest.approx({"blanca": 0.6, "verde": 0.0, "negra": 0.4})
    assert ganadora.loc[0] == "blanca"


def test_mcnemar_con_un_caso_a_mano():
    acierta_1 = np.array([True, False, False, False, False, False, True, True])
    acierta_2 = np.array([False, True, True, True, True, True, True, False])
    resultado = metricas.mcnemar(acierta_1, acierta_2)
    assert (resultado["b"], resultado["c"]) == (2, 5)
    assert resultado["p_valor"] == pytest.approx(2 * (1 + 7 + 21) / 2**7)


def test_cobertura_con_un_caso_a_mano(canecas):
    tabla = pd.DataFrame(
        {
            "clase_real": ["Plastic", "Plastic", "Food Organics", "Glass"],
            "caneca": ["blanca", "blanca", "blanca", "verde"],
            "prob_caneca_blanca": [0.95, 0.6, 0.5, 0.2],
        }
    )
    curva = metricas.cobertura_blanca(tabla, canecas)
    assert curva["sin_abstencion"]["precision"] == pytest.approx(2 / 3)
    noventa = next(p for p in curva["puntos"] if p["umbral"] == 0.9)
    assert (noventa["imagenes"], noventa["precision"], noventa["cobertura"]) == (1, 1.0, pytest.approx(1 / 3))
    assert noventa["recall"] == pytest.approx(1 / 3)


# Robustez


def test_las_perturbaciones_conservan_el_tamano_y_hacen_lo_que_dicen():
    generador = np.random.default_rng(0)
    imagen = Image.fromarray(generador.integers(0, 256, size=(40, 40, 3), dtype=np.uint8))
    for perturbacion in PERTURBACIONES:
        assert alterar(imagen, perturbacion).size == imagen.size, perturbacion.nombre
    media = np.asarray(imagen).mean()
    assert np.asarray(alterar(imagen, POR_NOMBRE["oscura"])).mean() < media < np.asarray(alterar(imagen, POR_NOMBRE["clara"])).mean()
    assert np.array_equal(np.asarray(girar(imagen, 90)), np.rot90(np.asarray(imagen)))
    recorte = imagen.crop((6, 6, 34, 34)).resize(imagen.size, Image.Resampling.BICUBIC)
    assert np.array_equal(np.asarray(acercar(imagen, 70)), np.asarray(recorte))


def test_robustez_guarda_sus_parametros():
    datos = cargar_json(rutas.RESULTADOS / "metricas-robustez.json")
    assert {n: p["valor"] for n, p in datos["parametros"].items()} == {p.nombre: p.valor for p in PERTURBACIONES}
    assert datos["bootstrap"] == metricas.BOOTSTRAP
    for nombre, resultado in datos["perturbaciones"].items():
        assert resultado["verificacion"]["valido"], nombre


# Cifras del informe


def resolver(datos: dict, ruta: str):
    valor = datos
    for clave in ruta.split(cifras.SEPARADOR):
        valor = valor[int(clave)] if isinstance(valor, list) else valor[clave]
    return valor


@requiere_paridad
def test_cada_macro_sale_de_una_fuente_permitida_y_su_clave_existe():
    macros = cifras.todas_las_macros()
    for macro in macros:
        archivos_origen, _, ruta = macro.origen.partition(": ")
        archivos = re.findall(r"[\w.-]+\.(?:json|csv)", archivos_origen)
        assert archivos, macro
        assert all(FUENTES_PERMITIDAS.match(a) for a in archivos), macro
        if len(archivos) == 1 and archivos[0].endswith(".json") and cifras.SEPARADOR in ruta and "(" not in ruta:
            carpeta = rutas.RESULTADOS if archivos[0] != "inventario.json" else rutas.INVENTARIO.parent
            resolver(cargar_json(carpeta / archivos[0]), ruta)


@requiere_paridad
@requiere_informe
def test_cifras_y_tablas_estan_al_dia(tmp_path):
    assert cifras.comprobar(cifras.DESTINO.parent, tmp_path) == []


# Figuras


MANIFIESTO = rutas.FIGURAS / "manifiesto.json"


@pytest.mark.skipif(not MANIFIESTO.exists(), reason="las figuras no están generadas (proyecto-final/figuras no se versiona)")
def test_cada_figura_del_manifiesto_existe():
    manifiesto = cargar_json(MANIFIESTO)
    assert manifiesto["figuras"]
    for figura in manifiesto["figuras"]:
        assert (rutas.FIGURAS / figura["archivo"]).stat().st_size > 0, figura["archivo"]
