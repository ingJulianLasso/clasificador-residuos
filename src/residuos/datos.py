"""Preparación de RealWaste: extracción, inventario, duplicados, partición y balanceo.

Uso: ``uv run python -m residuos.datos``. Lee el ZIP de UCI ya descargado (``curl -L --fail -o
../datos/descarga/realwaste.zip`` con ``URL_ZIP``), lo extrae en ``proyecto-final/datos/realwaste/``
y escribe ``datos/particion.csv`` y ``datos/inventario.json``.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import shutil
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
from functools import partial
from pathlib import Path, PurePosixPath

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from residuos import rutas

URL_ZIP = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
INSTANCIAS_UCI = 4752
CONTEOS_UCI = {
    "Cardboard": 461,
    "Food Organics": 411,
    "Glass": 420,
    "Metal": 790,
    "Miscellaneous Trash": 495,
    "Paper": 500,
    "Plastic": 921,
    "Textile Trash": 318,
    "Vegetation": 436,
}
PREFIJOS_ZIP = ("realwaste-main/RealWaste/", "realwaste-main/")
EXTENSION_IMAGEN = ".jpg"
ETIQUETA_ORIENTACION_EXIF = 0x0112

SEMILLA = 42
FRACCION_PRUEBA = Fraction(3, 10)
DISTANCIA_CASI_DUPLICADO = 4

ENTRENAMIENTO, PRUEBA, EXCLUIDA = "entrenamiento", "prueba", "excluida"
COLUMNAS = ["archivo", "clase", "sha256", "grupo", "particion", "balanceado", "motivo"]
MOTIVO_NO_ABRE = "no abre"
MOTIVO_DUPLICADO = "duplicado exacto de"
MOTIVO_ENTRE_CLASES = "duplicado exacto entre clases"
MOTIVO_GRUPO = "casi duplicado, grupo de"


@dataclass(frozen=True)
class Imagen:
    """Lo que se sabe de una imagen del dataset tras abrirla."""

    archivo: str
    clase: str
    sha256: str
    abre: bool
    formato: str | None = None
    ancho: int | None = None
    alto: int | None = None
    modo: str | None = None
    orientacion_exif: int | None = None
    phash: int | None = None


# Extracción


def ruta_extraida(nombre_en_zip: str) -> PurePosixPath:
    """Traduce una entrada del ZIP a su ruta dentro de la carpeta de imágenes (clase/archivo)."""
    prefijo = next((p for p in PREFIJOS_ZIP if nombre_en_zip.startswith(p)), None)
    if prefijo is None:
        raise ValueError(f"Entrada inesperada en el ZIP: {nombre_en_zip}")
    relativa = PurePosixPath(nombre_en_zip.removeprefix(prefijo))
    if relativa.is_absolute() or ".." in relativa.parts:
        raise ValueError(f"Entrada insegura en el ZIP: {nombre_en_zip}")
    return relativa


def es_imagen(nombre: str) -> bool:
    return nombre.lower().endswith(EXTENSION_IMAGEN)


def extraer(zip_path: Path, destino: Path) -> list[str]:
    """Extrae el ZIP en ``destino`` y devuelve las imágenes como rutas relativas ordenadas."""
    imagenes = []
    with zipfile.ZipFile(zip_path) as archivo_zip:
        for entrada in archivo_zip.infolist():
            if entrada.is_dir():
                continue
            relativa = ruta_extraida(entrada.filename)
            salida = destino / relativa
            salida.parent.mkdir(parents=True, exist_ok=True)
            with archivo_zip.open(entrada) as origen, salida.open("wb") as copia:
                shutil.copyfileobj(origen, copia)
            if es_imagen(entrada.filename):
                imagenes.append(relativa.as_posix())
    return sorted(imagenes)


# Descripción de cada imagen


def describir(relativa: str, raiz: Path) -> Imagen:
    contenido = (raiz / relativa).read_bytes()
    base = {
        "archivo": relativa,
        "clase": PurePosixPath(relativa).parent.name,
        "sha256": hashlib.sha256(contenido).hexdigest(),
    }
    try:
        with Image.open(io.BytesIO(contenido)) as imagen:
            imagen.load()
            return Imagen(
                **base,
                abre=True,
                formato=imagen.format,
                ancho=imagen.width,
                alto=imagen.height,
                modo=imagen.mode,
                orientacion_exif=imagen.getexif().get(ETIQUETA_ORIENTACION_EXIF),
                phash=int(str(imagehash.phash(imagen)), 16),
            )
    except (UnidentifiedImageError, OSError):
        return Imagen(**base, abre=False)


def describir_todas(raiz: Path, relativas: list[str]) -> list[Imagen]:
    with ThreadPoolExecutor() as ejecutor:
        return list(ejecutor.map(partial(describir, raiz=raiz), sorted(relativas)))


# Duplicados exactos y casi duplicados


def grupos_duplicados_exactos(imagenes: list[Imagen]) -> list[list[Imagen]]:
    por_hash = defaultdict(list)
    for imagen in imagenes:
        por_hash[imagen.sha256].append(imagen)
    return [sorted(g, key=lambda i: i.archivo) for _, g in sorted(por_hash.items()) if len(g) > 1]


def motivos_de_exclusion(imagenes: list[Imagen]) -> dict[str, str]:
    """Se conserva una copia de cada duplicado exacto; si sus clases difieren, ninguna."""
    motivos = {}
    for grupo in grupos_duplicados_exactos(imagenes):
        clases = sorted({i.clase for i in grupo})
        if len(clases) > 1:
            motivos.update({i.archivo: f"{MOTIVO_ENTRE_CLASES} ({', '.join(clases)})" for i in grupo})
        else:
            conservada, *repetidas = grupo
            motivos.update({i.archivo: f"{MOTIVO_DUPLICADO} {conservada.archivo}" for i in repetidas})
    motivos.update({i.archivo: MOTIVO_NO_ABRE for i in imagenes if not i.abre})
    return motivos


def pares_casi_duplicados(imagenes: list[Imagen]) -> list[tuple[int, int, int]]:
    """Pares (índice, índice, distancia) con phash a distancia de Hamming de 4 o menos."""
    indices = [k for k, imagen in enumerate(imagenes) if imagen.phash is not None]
    hashes = np.array([imagenes[k].phash for k in indices], dtype=np.uint64)
    pares = []
    for a in range(len(hashes) - 1):
        distancias = np.bitwise_count(hashes[a + 1 :] ^ hashes[a])
        for b in np.flatnonzero(distancias <= DISTANCIA_CASI_DUPLICADO):
            pares.append((indices[a], indices[a + 1 + int(b)], int(distancias[b])))
    return pares


def agrupar(cantidad: int, pares: list[tuple[int, int, int]]) -> list[str]:
    """Componentes conexas de los pares; los grupos se numeran por su primer archivo."""
    padre = list(range(cantidad))

    def raiz(k: int) -> int:
        while padre[k] != k:
            padre[k] = padre[padre[k]]
            k = padre[k]
        return k

    for a, b, _ in pares:
        raiz_a, raiz_b = raiz(a), raiz(b)
        if raiz_a != raiz_b:
            padre[max(raiz_a, raiz_b)] = min(raiz_a, raiz_b)
    numeros: dict[int, int] = {}
    return [f"g{numeros.setdefault(raiz(k), len(numeros) + 1):04d}" for k in range(cantidad)]


# Partición y balanceo


def tamano_de_prueba(incluidas: int) -> int:
    """30 % redondeado al entero más cercano (las mitades hacia arriba), en aritmética exacta."""
    return math.floor(FRACCION_PRUEBA * incluidas + Fraction(1, 2))


def asignar_particion(incluidas: pd.DataFrame) -> pd.Series:
    """Recorre los grupos en orden aleatorio y manda a prueba los que caben en el cupo de cada clase.

    Un grupo entra o sale entero, así que los casi duplicados nunca quedan a ambos lados.
    """
    cupo = {clase: tamano_de_prueba(n) for clase, n in incluidas["clase"].value_counts().items()}
    composicion = {grupo: Counter(filas["clase"]) for grupo, filas in incluidas.groupby("grupo", sort=True)}
    grupos = list(composicion)
    ocupado: Counter[str] = Counter()
    en_prueba = set()
    for posicion in np.random.default_rng(SEMILLA).permutation(len(grupos)):
        conteo = composicion[grupos[posicion]]
        if all(ocupado[clase] + n <= cupo[clase] for clase, n in conteo.items()):
            en_prueba.add(grupos[posicion])
            ocupado.update(conteo)
    return incluidas["grupo"].map(lambda grupo: PRUEBA if grupo in en_prueba else ENTRENAMIENTO)


def marcar_balanceado(tabla: pd.DataFrame) -> pd.Series:
    """Submuestrea el entrenamiento, sin reemplazo, al tamaño de la clase menor."""
    entrenamiento = tabla[tabla["particion"] == ENTRENAMIENTO]
    tamano = int(entrenamiento["clase"].value_counts().min())
    generador = np.random.default_rng(SEMILLA)
    elegidos = set()
    for _, filas in entrenamiento.groupby("clase", sort=True):
        archivos = sorted(filas["archivo"])
        elegidos.update(archivos[k] for k in generador.choice(len(archivos), size=tamano, replace=False))
    return tabla["archivo"].isin(elegidos).astype(int)


def construir_particion(imagenes: list[Imagen], pares: list[tuple[int, int, int]]) -> pd.DataFrame:
    tabla = pd.DataFrame(
        {
            "archivo": [i.archivo for i in imagenes],
            "clase": [i.clase for i in imagenes],
            "sha256": [i.sha256 for i in imagenes],
            "grupo": agrupar(len(imagenes), pares),
        }
    )
    motivos = motivos_de_exclusion(imagenes)
    tabla["motivo"] = [motivos.get(archivo, "") for archivo in tabla["archivo"]]
    incluida = ~tabla["archivo"].isin(motivos)
    tabla["particion"] = EXCLUIDA
    tabla.loc[incluida, "particion"] = asignar_particion(tabla[incluida])
    tabla["balanceado"] = marcar_balanceado(tabla)
    tamano_grupo = tabla[incluida].groupby("grupo")["archivo"].transform("size")
    en_grupo = tamano_grupo[tamano_grupo > 1]
    tabla.loc[en_grupo.index, "motivo"] = [f"{MOTIVO_GRUPO} {n}" for n in en_grupo]
    return tabla[COLUMNAS].sort_values("archivo").reset_index(drop=True)


def a_csv(tabla: pd.DataFrame) -> bytes:
    return tabla.to_csv(index=False, lineterminator="\n").encode("utf-8")


def leer_particion(ruta: Path = rutas.PARTICION) -> pd.DataFrame:
    return pd.read_csv(ruta, dtype={"balanceado": int}, keep_default_na=False)


# Inventario


def sha256_de_archivo(ruta: Path) -> str:
    resumen = hashlib.sha256()
    with ruta.open("rb") as archivo:
        for bloque in iter(lambda: archivo.read(1 << 20), b""):
            resumen.update(bloque)
    return resumen.hexdigest()


def ordenado(conteo: Counter) -> dict:
    return {str(clave): int(valor) for clave, valor in sorted(conteo.items(), key=lambda par: str(par[0]))}


def relativa_al_proyecto(ruta: Path) -> str:
    try:
        return ruta.resolve().relative_to(rutas.RAIZ_PROYECTO).as_posix()
    except ValueError:
        return ruta.name


def resumen_zip(zip_path: Path) -> dict:
    with zipfile.ZipFile(zip_path) as archivo_zip:
        entradas = archivo_zip.infolist()
    archivos = [e for e in entradas if not e.is_dir()]
    imagenes = [e for e in archivos if es_imagen(e.filename)]
    no_imagen = [{"archivo": e.filename, "bytes": e.file_size} for e in archivos if not es_imagen(e.filename)]
    tamano = zip_path.stat().st_size
    return {
        "url": URL_ZIP,
        "archivo": relativa_al_proyecto(zip_path),
        "bytes": tamano,
        "mib": round(tamano / 2**20, 1),
        "sha256": sha256_de_archivo(zip_path),
        "fecha_descarga": datetime.fromtimestamp(zip_path.stat().st_mtime).date().isoformat(),
        "entradas": len(entradas),
        "directorios": len(entradas) - len(archivos),
        "archivos": len(archivos),
        "imagenes": len(imagenes),
        "no_imagen": no_imagen,
        "extensiones": ordenado(Counter(PurePosixPath(e.filename).suffix.lower() for e in archivos)),
    }


def explicar_diferencia(zip_resumen: dict) -> dict:
    no_imagen = zip_resumen["no_imagen"]
    otros = "; ".join(f"{n['archivo']}, de {n['bytes']} bytes" for n in no_imagen)
    cuantos = "1 archivo que no es imagen" if len(no_imagen) == 1 else f"{len(no_imagen)} archivos que no son imagen"
    return {
        "instancias_uci": INSTANCIAS_UCI,
        "archivos_zip": zip_resumen["archivos"],
        "imagenes_zip": zip_resumen["imagenes"],
        "diferencia": zip_resumen["archivos"] - INSTANCIAS_UCI,
        "imagenes_igual_a_uci": zip_resumen["imagenes"] == INSTANCIAS_UCI,
        "explicacion": (
            f"El ZIP trae {zip_resumen['archivos']} archivos: {zip_resumen['imagenes']} imágenes JPG y "
            f"{cuantos} ({otros}). Las {INSTANCIAS_UCI} instancias que reporta UCI son las imágenes, "
            "así que la diferencia corresponde a lo que no es imagen."
        ),
    }


def resumen_imagenes(imagenes: list[Imagen]) -> dict:
    por_clase = Counter(i.clase for i in imagenes)
    abiertas = [i for i in imagenes if i.abre]
    return {
        "por_clase": {
            clase: {"archivos": n, "uci": CONTEOS_UCI.get(clase), "coincide_con_uci": CONTEOS_UCI.get(clase) == n}
            for clase, n in sorted(por_clase.items())
        },
        "formatos": ordenado(Counter(i.formato for i in abiertas)),
        "modos_de_color": ordenado(Counter(i.modo for i in abiertas)),
        "dimensiones": ordenado(Counter(f"{i.ancho}x{i.alto}" for i in abiertas)),
        "orientacion_exif": ordenado(
            Counter("sin etiqueta" if i.orientacion_exif is None else i.orientacion_exif for i in abiertas)
        ),
        "no_abren": [i.archivo for i in imagenes if not i.abre],
    }


def resumen_duplicados_exactos(imagenes: list[Imagen], tabla: pd.DataFrame) -> dict:
    grupos = grupos_duplicados_exactos(imagenes)
    return {
        "criterio": "SHA-256 idéntico. Se conserva la primera ruta en orden alfabético; si las clases difieren, se excluyen todas",
        "grupos": len(grupos),
        "imagenes_en_grupos": sum(len(g) for g in grupos),
        "grupos_entre_clases": sum(len({i.clase for i in g}) > 1 for g in grupos),
        "excluidas_por_copia": int(tabla["motivo"].str.startswith(MOTIVO_DUPLICADO).sum()),
        "excluidas_entre_clases": int(tabla["motivo"].str.startswith(MOTIVO_ENTRE_CLASES).sum()),
        "detalle": [
            {"sha256": g[0].sha256, "archivos": [i.archivo for i in g], "clases": sorted({i.clase for i in g})}
            for g in grupos
        ],
    }


def resumen_casi_duplicados(imagenes: list[Imagen], pares: list[tuple[int, int, int]], tabla: pd.DataFrame) -> dict:
    distintos = [(a, b, d) for a, b, d in pares if imagenes[a].sha256 != imagenes[b].sha256]
    entre_clases = [(a, b, d) for a, b, d in distintos if imagenes[a].clase != imagenes[b].clase]
    tamanos = tabla.loc[tabla["particion"] != EXCLUIDA, "grupo"].value_counts()
    return {
        "criterio": (
            f"phash de 64 bits (imagehash) a distancia de Hamming de {DISTANCIA_CASI_DUPLICADO} o menos; "
            "los grupos son las componentes conexas de esos pares y nunca se parten entre entrenamiento y prueba"
        ),
        "distancia_maxima": DISTANCIA_CASI_DUPLICADO,
        "pares_contando_duplicados_exactos": len(pares),
        "pares": len(distintos),
        "pares_misma_clase": len(distintos) - len(entre_clases),
        "pares_entre_clases": len(entre_clases),
        "pares_por_distancia": ordenado(Counter(d for _, _, d in distintos)),
        "grupos_de_varias_imagenes": int((tamanos > 1).sum()),
        "imagenes_en_esos_grupos": int(tamanos[tamanos > 1].sum()),
        "tamano_maximo_de_grupo": int(tamanos.max()),
        "pares_entre_clases_por_clases": ordenado(
            Counter(" y ".join(sorted((imagenes[a].clase, imagenes[b].clase))) for a, b, _ in entre_clases)
        ),
        "pares_entre_clases_detalle": [
            {
                "archivo_a": imagenes[a].archivo,
                "archivo_b": imagenes[b].archivo,
                "distancia": d,
            }
            for a, b, d in entre_clases
        ],
        "tratamiento": (
            "Los pares entre clases conservan su etiqueta y su grupo va entero a una sola partición; "
            "se reportan para revisar si son ruido de etiqueta o coincidencias del fondo"
        ),
    }


def resumen_particion(tabla: pd.DataFrame, contenido_csv: bytes) -> dict:
    por_clase = {}
    for clase, filas in tabla.groupby("clase", sort=True):
        incluidas = int((filas["particion"] != EXCLUIDA).sum())
        prueba = int((filas["particion"] == PRUEBA).sum())
        por_clase[clase] = {
            "total": len(filas),
            "excluidas": int((filas["particion"] == EXCLUIDA).sum()),
            "incluidas": incluidas,
            "entrenamiento": int((filas["particion"] == ENTRENAMIENTO).sum()),
            "prueba": prueba,
            "fraccion_prueba": round(prueba / incluidas, 4),
            "balanceado": int(filas["balanceado"].sum()),
        }
    return {
        "semilla": SEMILLA,
        "fraccion_prueba": float(FRACCION_PRUEBA),
        "criterio": (
            "Por clase, el 30 % de las imágenes incluidas va a prueba (redondeo al entero más cercano). "
            "Los grupos de casi duplicados se recorren en orden aleatorio con la semilla y entran enteros "
            "a prueba mientras quepan en el cupo de todas sus clases"
        ),
        "balanceado_criterio": "Submuestreo sin reemplazo del entrenamiento al tamaño de la clase menor, con la misma semilla",
        "por_clase": por_clase,
        "totales": {
            clave: sum(c[clave] for c in por_clase.values())
            for clave in ("total", "excluidas", "incluidas", "entrenamiento", "prueba", "balanceado")
        },
        "particion_csv": {
            "archivo": rutas.PARTICION.relative_to(rutas.RAIZ_CLASIFICADOR).as_posix(),
            "filas": len(tabla),
            "sha256": hashlib.sha256(contenido_csv).hexdigest(),
        },
    }


def construir_inventario(
    zip_path: Path, imagenes: list[Imagen], pares: list[tuple[int, int, int]], tabla: pd.DataFrame, contenido_csv: bytes
) -> dict:
    zip_resumen = resumen_zip(zip_path)
    return {
        "zip": zip_resumen,
        "diferencia_con_uci": explicar_diferencia(zip_resumen),
        "imagenes": resumen_imagenes(imagenes),
        "duplicados_exactos": resumen_duplicados_exactos(imagenes, tabla),
        "casi_duplicados": resumen_casi_duplicados(imagenes, pares, tabla),
        "particion": resumen_particion(tabla, contenido_csv),
    }


def escribir_json(ruta: Path, contenido: dict) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(contenido, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preparar(zip_path: Path = rutas.ZIP_REALWASTE, imagenes_dir: Path = rutas.IMAGENES) -> dict:
    relativas = extraer(zip_path, imagenes_dir)
    imagenes = describir_todas(imagenes_dir, relativas)
    pares = pares_casi_duplicados(imagenes)
    tabla = construir_particion(imagenes, pares)
    contenido_csv = a_csv(tabla)
    rutas.PARTICION.parent.mkdir(parents=True, exist_ok=True)
    rutas.PARTICION.write_bytes(contenido_csv)
    inventario = construir_inventario(zip_path, imagenes, pares, tabla, contenido_csv)
    escribir_json(rutas.INVENTARIO, inventario)
    return inventario


def main() -> None:
    inventario = preparar()
    particion = inventario["particion"]
    print(json.dumps({"totales": particion["totales"], "sha256_particion": particion["particion_csv"]["sha256"]}))


if __name__ == "__main__":
    main()
