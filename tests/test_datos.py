"""Invariantes de la partición, del balanceado, del inventario y de canecas.json."""

import hashlib
import json
import re
import zipfile
import zlib

import pandas as pd
import pytest

from residuos import datos, rutas
from residuos.datos import ENTRENAMIENTO, EXCLUIDA, PRUEBA, Imagen

CORRESPONDENCIA = {
    "Cardboard": ("Cartón", "blanca"),
    "Food Organics": ("Orgánicos de comida", "verde"),
    "Glass": ("Vidrio", "blanca"),
    "Metal": ("Metal", "blanca"),
    "Miscellaneous Trash": ("Basura miscelánea", "negra"),
    "Paper": ("Papel", "blanca"),
    "Plastic": ("Plástico", "blanca"),
    "Textile Trash": ("Textiles", "blanca"),
    "Vegetation": ("Vegetación", "verde"),
}
CIFRAS = rutas.TEX_GENERADO / "cifras.tex"
MACRO = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{([^}]*)\}")


@pytest.fixture(scope="module")
def particion() -> pd.DataFrame:
    return datos.leer_particion()


@pytest.fixture(scope="module")
def incluidas(particion: pd.DataFrame) -> pd.DataFrame:
    return particion[particion["particion"] != EXCLUIDA]


@pytest.fixture(scope="module")
def inventario() -> dict:
    return json.loads(rutas.INVENTARIO.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def canecas() -> dict:
    return json.loads(rutas.CANECAS.read_text(encoding="utf-8"))


# Partición


def test_los_conteos_suman_el_total(particion: pd.DataFrame, inventario: dict) -> None:
    por_clase = particion["clase"].value_counts().to_dict()
    assert por_clase == datos.CONTEOS_UCI
    assert len(particion) == inventario["zip"]["imagenes"] == sum(datos.CONTEOS_UCI.values())
    assert particion["particion"].value_counts().sum() == len(particion)
    totales = inventario["particion"]["totales"]
    assert totales["entrenamiento"] + totales["prueba"] + totales["excluidas"] == totales["total"] == len(particion)
    assert totales["prueba"] == (particion["particion"] == PRUEBA).sum()


def test_ninguna_imagen_en_dos_particiones(particion: pd.DataFrame, incluidas: pd.DataFrame) -> None:
    assert particion["archivo"].is_unique
    assert set(particion["particion"]) <= {ENTRENAMIENTO, PRUEBA, EXCLUIDA}
    assert incluidas["sha256"].is_unique


def test_ningun_grupo_queda_partido(incluidas: pd.DataFrame) -> None:
    assert incluidas.groupby("grupo")["particion"].nunique().max() == 1


def test_los_pares_entre_clases_comparten_grupo(particion: pd.DataFrame, inventario: dict) -> None:
    grupo = particion.set_index("archivo")["grupo"]
    casi = inventario["casi_duplicados"]
    pares = casi["pares_entre_clases_detalle"]
    assert len(pares) == casi["pares_entre_clases"]
    assert all(grupo[par["archivo_a"]] == grupo[par["archivo_b"]] for par in pares)


def test_prueba_entre_29_y_31_por_ciento_por_clase(incluidas: pd.DataFrame) -> None:
    fracciones = incluidas.groupby("clase")["particion"].apply(lambda serie: (serie == PRUEBA).mean())
    assert len(fracciones) == 9
    assert fracciones.between(0.29, 0.31).all(), fracciones.to_dict()


def test_balanceado_igual_en_las_nueve_clases(particion: pd.DataFrame) -> None:
    balanceado = particion[particion["balanceado"] == 1]
    assert set(particion["balanceado"]) <= {0, 1}
    assert (balanceado["particion"] == ENTRENAMIENTO).all()
    por_clase = balanceado["clase"].value_counts()
    menor = (particion["particion"] == ENTRENAMIENTO).groupby(particion["clase"]).sum().min()
    assert len(por_clase) == 9
    assert por_clase.nunique() == 1
    assert por_clase.iloc[0] == menor


@pytest.mark.skipif(not rutas.IMAGENES.is_dir(), reason="las imágenes de RealWaste no están extraídas")
def test_dos_corridas_dan_el_mismo_sha256(particion: pd.DataFrame, inventario: dict) -> None:
    def corrida() -> str:
        imagenes = datos.describir_todas(rutas.IMAGENES, list(particion["archivo"]))
        tabla = datos.construir_particion(imagenes, datos.pares_casi_duplicados(imagenes))
        return hashlib.sha256(datos.a_csv(tabla)).hexdigest()

    guardado = hashlib.sha256(rutas.PARTICION.read_bytes()).hexdigest()
    assert corrida() == corrida() == guardado == inventario["particion"]["particion_csv"]["sha256"]


# Reglas de duplicados exactos, con casos a mano


def imagen(archivo: str, sha256: str) -> Imagen:
    return Imagen(archivo=archivo, clase=archivo.split("/")[0], sha256=sha256, abre=True)


def test_duplicado_exacto_en_la_misma_clase_conserva_uno() -> None:
    imagenes = [imagen("Glass/b.jpg", "x"), imagen("Glass/a.jpg", "x"), imagen("Glass/c.jpg", "y")]
    motivos = datos.motivos_de_exclusion(imagenes)
    assert motivos == {"Glass/b.jpg": f"{datos.MOTIVO_DUPLICADO} Glass/a.jpg"}


def test_duplicado_exacto_entre_clases_excluye_todos() -> None:
    motivos = datos.motivos_de_exclusion([imagen("Glass/a.jpg", "x"), imagen("Metal/a.jpg", "x")])
    assert set(motivos) == {"Glass/a.jpg", "Metal/a.jpg"}
    assert all(m.startswith(datos.MOTIVO_ENTRE_CLASES) for m in motivos.values())


def test_agrupar_une_las_cadenas_de_pares() -> None:
    grupos = datos.agrupar(6, [(1, 2, 3), (0, 1, 4), (3, 4, 0)])
    assert grupos == ["g0001", "g0001", "g0001", "g0002", "g0002", "g0003"]


def test_un_grupo_entre_clases_va_entero_a_una_particion() -> None:
    filas = [("Glass/a.jpg", "Glass", "g1"), ("Metal/a.jpg", "Metal", "g1")]
    filas += [(f"Glass/{k}.jpg", "Glass", f"v{k}") for k in range(9)]
    filas += [(f"Metal/{k}.jpg", "Metal", f"m{k}") for k in range(9)]
    tabla = pd.DataFrame(filas, columns=["archivo", "clase", "grupo"])
    asignacion = datos.asignar_particion(tabla)
    assert asignacion[tabla["grupo"] == "g1"].nunique() == 1
    assert (asignacion == PRUEBA).groupby(tabla["clase"]).sum().to_dict() == {"Glass": 3, "Metal": 3}


# Inventario


def test_inventario_trae_el_zip_y_explica_la_diferencia(inventario: dict) -> None:
    zip_ = inventario["zip"]
    assert re.fullmatch(r"[0-9a-f]{64}", zip_["sha256"])
    assert zip_["bytes"] > 0
    diferencia = inventario["diferencia_con_uci"]
    assert diferencia["archivos_zip"] - diferencia["instancias_uci"] == diferencia["diferencia"] == len(zip_["no_imagen"])
    assert diferencia["imagenes_igual_a_uci"]
    assert all(not datos.es_imagen(n["archivo"]) for n in zip_["no_imagen"])
    assert diferencia["explicacion"]
    if rutas.ZIP_REALWASTE.is_file():
        assert rutas.ZIP_REALWASTE.stat().st_size == zip_["bytes"]
        assert datos.sha256_de_archivo(rutas.ZIP_REALWASTE) == zip_["sha256"]


@pytest.mark.skipif(
    not (rutas.ZIP_REALWASTE.is_file() and rutas.IMAGENES.is_dir()), reason="falta el ZIP o su extracción"
)
def test_la_extraccion_coincide_con_el_zip() -> None:
    with zipfile.ZipFile(rutas.ZIP_REALWASTE) as archivo_zip:
        esperado = {
            datos.ruta_extraida(e.filename).as_posix(): (e.file_size, e.CRC)
            for e in archivo_zip.infolist()
            if not e.is_dir()
        }
    extraido = {r.relative_to(rutas.IMAGENES).as_posix(): r for r in rutas.IMAGENES.rglob("*") if r.is_file()}
    assert extraido.keys() == esperado.keys()
    for relativa, ruta in extraido.items():
        contenido = ruta.read_bytes()
        assert (len(contenido), zlib.crc32(contenido)) == esperado[relativa], relativa


# canecas.json


def test_canecas_cubre_las_nueve_clases_con_caneca_valida(canecas: dict, particion: pd.DataFrame) -> None:
    assert set(canecas) == {"fuente", "canecas", "materiales"}
    assert isinstance(canecas["fuente"], str) and canecas["fuente"]
    assert set(canecas["canecas"]) == {"blanca", "verde", "negra"}
    for caneca in canecas["canecas"].values():
        assert set(caneca) == {"nombre", "descripcion"} and all(caneca.values())
    assert list(canecas["materiales"]) == sorted(set(particion["clase"]))
    for clase, material in canecas["materiales"].items():
        assert set(material) == {"nombre", "caneca", "fuente"} and material["fuente"]
        assert material["caneca"] in canecas["canecas"]
        assert (material["nombre"], material["caneca"]) == CORRESPONDENCIA[clase]


# cifras.tex


def numero(texto: str) -> int:
    return int(texto.replace(".", ""))


@pytest.mark.skipif(not CIFRAS.is_file(), reason="cifras.tex no está generado")
def test_las_cifras_coinciden_con_la_particion(particion: pd.DataFrame, canecas: dict) -> None:
    macros = dict(MACRO.findall(CIFRAS.read_text(encoding="utf-8")))
    assert numero(macros["ImagenesTotal"]) == len(particion)
    assert numero(macros["ImagenesPrueba"]) == (particion["particion"] == PRUEBA).sum()
    assert numero(macros["ImagenesEntrenamiento"]) == (particion["particion"] == ENTRENAMIENTO).sum()
    assert numero(macros["ImagenesExcluidas"]) == (particion["particion"] == EXCLUIDA).sum()
    assert numero(macros["ImagenesBalanceado"]) == particion["balanceado"].sum()
    tabla = pd.crosstab(particion["clase"], particion["particion"])
    for clase, (nombre, _) in CORRESPONDENCIA.items():
        prefijo = "Clase" + re.sub(r"[^A-Za-z]", "", nombre.title().replace("á", "a").replace("é", "e").replace("ó", "o"))
        assert numero(macros[f"{prefijo}Total"]) == tabla.loc[clase].sum()
        assert numero(macros[f"{prefijo}Prueba"]) == tabla.loc[clase].get(PRUEBA, 0)
        assert numero(macros[f"{prefijo}Entrenamiento"]) == tabla.loc[clase].get(ENTRENAMIENTO, 0)
        assert numero(macros[f"{prefijo}Balanceado"]) == particion.loc[particion["clase"] == clase, "balanceado"].sum()
    caneca_de = particion["clase"].map({c: m["caneca"] for c, m in canecas["materiales"].items()})
    for caneca in canecas["canecas"]:
        assert numero(macros[f"Caneca{caneca.title()}Total"]) == (caneca_de == caneca).sum()
