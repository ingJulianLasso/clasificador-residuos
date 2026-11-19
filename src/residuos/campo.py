"""Prueba en campo: revisión de la carpeta de fotos propias frente a su registro.

Uso, desde ``clasificador/``:

    uv run python -m residuos.campo verificar

Compara ``datos/campo/`` con ``prueba-campo/registro.csv`` (la columna ``archivo`` es relativa a
``datos/campo/``) e imprime el avance por clase y por condición, las fotos que faltan, los archivos
que sobran, los nombres mal escritos, las fotos que no son JPEG o no abren, las dimensiones y la
orientación EXIF. Solo lee: no mueve, no renombra ni convierte nada. Sale con código 1 si falta una
foto o si algo no está en orden.

La orientación EXIF no se corrige aquí: el navegador la aplica al decodificar, igual que
``ImageOps.exif_transpose`` (lo comprueba ``tests/test_campo.py`` con la instancia de Brave).
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import pandas as pd
from PIL import Image, UnidentifiedImageError

from residuos import rutas
from residuos.datos import ETIQUETA_ORIENTACION_EXIF

IGNORADOS = {".gitkeep"}  # un proceso de la máquina los recrea en las carpetas vacías
FORMATO_ESPERADO = "JPEG"
FIRMAS_HEIF = {b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1", b"msf1", b"avif"}
ORIENTACIONES_QUE_GIRAN = {5, 6, 7, 8}
VALORES_TOMADA = {"sí", "si"}


def leer_registro(ruta: Path = rutas.REGISTRO_CAMPO) -> pd.DataFrame:
    """El registro como texto, leído por nombre de columna."""
    return pd.read_csv(ruta, dtype=str, keep_default_na=False)


@dataclass(frozen=True)
class Foto:
    archivo: str
    abre: bool
    formato: str | None = None
    ancho: int | None = None
    alto: int | None = None
    orientacion_exif: int | None = None

    @property
    def dimension(self) -> str:
        return f"{self.ancho}x{self.alto}"

    @property
    def vertical_al_mostrarla(self) -> bool:
        """Con la orientación EXIF aplicada, como la ve el navegador."""
        ancho, alto = (self.alto, self.ancho) if self.orientacion_exif in ORIENTACIONES_QUE_GIRAN else (self.ancho, self.alto)
        return alto > ancho


def formato_por_firma(cabecera: bytes) -> str | None:
    """Reconoce HEIC/HEIF, que Pillow no abre sin un complemento, por su caja ``ftyp``."""
    return "HEIF" if cabecera[4:8] == b"ftyp" and cabecera[8:12] in FIRMAS_HEIF else None


def describir_foto(carpeta: Path, archivo: str) -> Foto:
    ruta = carpeta / archivo
    try:
        with Image.open(ruta) as imagen:
            imagen.load()
            return Foto(archivo, True, imagen.format, imagen.width, imagen.height, imagen.getexif().get(ETIQUETA_ORIENTACION_EXIF))
    except (UnidentifiedImageError, OSError):
        with ruta.open("rb") as contenido:
            return Foto(archivo, False, formato_por_firma(contenido.read(12)))


def archivos_en(carpeta: Path) -> set[str]:
    if not carpeta.exists():
        return set()
    return {p.relative_to(carpeta).as_posix() for p in carpeta.rglob("*") if p.is_file() and p.name not in IGNORADOS}


def clave_aproximada(archivo: str) -> tuple[str, int] | None:
    """Carpeta sin mayúsculas y número del nombre: ``glass/7.JPG`` y ``Glass/07.jpg`` comparten clave."""
    ruta = PurePosixPath(archivo)
    numeros = re.findall(r"\d+", ruta.stem)
    if len(ruta.parts) != 2 or len(numeros) != 1:
        return None
    return ruta.parent.name.casefold(), int(numeros[0])


@dataclass(frozen=True)
class Revision:
    registro: pd.DataFrame
    fotos: dict[str, Foto]  # las que existen con el nombre esperado
    sobrantes: dict[str, Foto]  # archivos que el registro no nombra

    @property
    def esperados(self) -> list[str]:
        return self.registro["archivo"].tolist()

    @property
    def validas(self) -> list[str]:
        return [a for a in self.esperados if a in self.fotos and self.fotos[a].abre and self.fotos[a].formato == FORMATO_ESPERADO]

    @property
    def faltan(self) -> list[str]:
        return [a for a in self.esperados if a not in self.fotos]

    @property
    def no_abren(self) -> list[Foto]:
        return [f for f in self.fotos.values() if not f.abre]

    @property
    def no_jpeg(self) -> list[Foto]:
        return [f for f in self.fotos.values() if f.abre and f.formato != FORMATO_ESPERADO]

    @property
    def mal_escritos(self) -> dict[str, str]:
        """Archivo sobrante y el nombre esperado que parece querer tener (solo entre los que faltan)."""
        por_clave = {clave_aproximada(a): a for a in self.faltan}
        por_clave.pop(None, None)
        return {s: por_clave[clave_aproximada(s)] for s in sorted(self.sobrantes) if clave_aproximada(s) in por_clave}

    @property
    def en_orden(self) -> bool:
        return len(self.validas) == len(self.esperados) and not self.sobrantes


def revisar(registro: pd.DataFrame, carpeta: Path) -> Revision:
    presentes = archivos_en(carpeta)
    esperados = set(registro["archivo"])
    return Revision(
        registro=registro,
        fotos={a: describir_foto(carpeta, a) for a in registro["archivo"] if a in presentes},
        sobrantes={a: describir_foto(carpeta, a) for a in sorted(presentes - esperados)},
    )


# Informe


def conteo(revision: Revision, columna: str) -> list[str]:
    validas = set(revision.validas)
    partes = []
    for valor, grupo in revision.registro.groupby(columna, sort=True):
        partes.append(f"{valor} {sum(a in validas for a in grupo['archivo'])} de {len(grupo)}")
    return partes


def agrupar_por_clase(archivos: Sequence[str]) -> list[str]:
    grupos: dict[str, list[str]] = {}
    for archivo in archivos:
        ruta = PurePosixPath(archivo)
        grupos.setdefault(ruta.parent.as_posix(), []).append(ruta.stem)
    return [f"  {clase} ({len(nombres)}): {', '.join(nombres)}" for clase, nombres in grupos.items()]


def describir_formato(foto: Foto) -> str:
    return foto.formato or "desconocido"


def contador(valores: Sequence[object]) -> str:
    return "; ".join(f"{valor} ({n})" for valor, n in Counter(valores).most_common()) or "ninguna"


def informe(revision: Revision) -> list[str]:
    total = len(revision.esperados)
    lineas = [
        f"Fotos listas: {len(revision.validas)} de {total}",
        "Por clase: " + "; ".join(conteo(revision, "clase")),
        "Por condición (NN): " + "; ".join(conteo(revision, "numero")),
    ]
    if revision.faltan:
        lineas += [f"Faltan ({len(revision.faltan)}):", *agrupar_por_clase(revision.faltan)]
    mal_escritos = revision.mal_escritos
    for sobrante, esperado in mal_escritos.items():
        lineas.append(f"Nombre mal escrito: {sobrante} ({describir_formato(revision.sobrantes[sobrante])}) debería ser {esperado} en JPEG")
    otros = [s for s in revision.sobrantes if s not in mal_escritos]
    if otros:
        lineas += [f"Archivos que el registro no nombra ({len(otros)}):", *(f"  {s}" for s in otros)]
    lineas += [f"No es JPEG: {f.archivo} ({describir_formato(f)})" for f in revision.no_jpeg]
    lineas += [f"No abre: {f.archivo} (formato {describir_formato(f)})" for f in revision.no_abren]
    abiertas = [f for f in revision.fotos.values() if f.abre]
    if abiertas:
        lineas += [
            "Dimensiones: " + contador([f.dimension for f in abiertas]),
            "Orientación EXIF: " + contador(["sin etiqueta" if f.orientacion_exif is None else f.orientacion_exif for f in abiertas]),
            f"Verticales al mostrarlas (con la orientación aplicada): {sum(f.vertical_al_mostrarla for f in abiertas)} de {len(abiertas)}",
        ]
    lineas += avisos_del_registro(revision)
    lineas.append("Todo en orden." if revision.en_orden else "Hay pendientes: revisar lo anterior.")
    return lineas


def avisos_del_registro(revision: Revision) -> list[str]:
    """La columna ``tomada`` y los sustitutos; no cambian el código de salida."""
    registro = revision.registro
    tomada = registro["tomada"].str.strip().str.casefold().isin(VALORES_TOMADA)
    validas = registro["archivo"].isin(revision.validas)
    sustitutos = int((registro["objeto_usado"].str.strip() != "").sum())
    lineas = [f"registro.csv: {int(tomada.sum())} de {len(registro)} marcadas como tomadas; {sustitutos} con objeto_usado (sustituto u otro objeto)"]
    sin_marcar = registro.loc[validas & ~tomada, "archivo"].tolist()
    sin_foto = registro.loc[~validas & tomada, "archivo"].tolist()
    if sin_marcar:
        lineas.append(f"Aviso: {len(sin_marcar)} fotos están en la carpeta, pero no marcadas como tomadas.")
    if sin_foto:
        lineas.append(f"Aviso: {len(sin_foto)} filas marcadas como tomadas no tienen foto válida.")
    return lineas


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(prog="residuos.campo", description=__doc__.splitlines()[0])
    sub = analizador.add_subparsers(dest="orden", required=True)
    verificacion = sub.add_parser("verificar", help="compara la carpeta de fotos con el registro (solo lectura)")
    verificacion.add_argument("--registro", type=Path, default=rutas.REGISTRO_CAMPO)
    verificacion.add_argument("--carpeta", type=Path, default=rutas.CAMPO)
    return analizador.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    argumentos = _argumentos(argv)
    revision = revisar(leer_registro(argumentos.registro), argumentos.carpeta)
    print("\n".join(informe(revision)))
    sys.exit(0 if revision.en_orden else 1)


if __name__ == "__main__":
    main()
