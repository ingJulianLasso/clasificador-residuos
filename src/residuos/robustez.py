"""Versiones alteradas del conjunto de prueba para medir la robustez del modelo elegido.

Uso: ``uv run python -m residuos.robustez``. Escribe en ``proyecto-final/datos/robustez/<perturbación>/``
una copia alterada de cada imagen de prueba, con la misma ruta relativa que en RealWaste, y una
``lista.csv`` con las columnas ``archivo``, ``clase`` y ``particion`` que lee el arnés con ``--base``.
Las versiones se guardan en PNG, salvo la comprimida, que es un JPEG por definición.

Cada perturbación tiene una sola severidad, fijada antes de entrenar (plan, §2.3).
"""

from __future__ import annotations

import csv
import io
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath

from PIL import Image, ImageEnhance, ImageFilter

from residuos import rutas
from residuos.datos import PRUEBA, leer_particion

LISTA = "lista.csv"


@dataclass(frozen=True)
class Perturbacion:
    nombre: str
    descripcion: str
    parametro: str
    valor: float
    unidad: str
    formato: str  # "PNG" o "JPEG"
    simula: str

    @property
    def extension(self) -> str:
        return ".png" if self.formato == "PNG" else ".jpg"

    def como_dict(self) -> dict:
        return {
            "descripcion": self.descripcion,
            "parametro": self.parametro,
            "valor": self.valor,
            "unidad": self.unidad,
            "formato": self.formato,
            "simula": self.simula,
        }


PERTURBACIONES = (
    Perturbacion("oscura", "brillo multiplicado por un factor menor que 1", "factor de brillo", 0.6, "", "PNG", "poca luz"),
    Perturbacion("clara", "brillo multiplicado por un factor mayor que 1", "factor de brillo", 1.4, "", "PNG", "luz intensa"),
    Perturbacion("desenfocada", "desenfoque gaussiano", "radio", 2, "píxeles", "PNG", "foto movida o fuera de foco"),
    Perturbacion("girada", "rotación en sentido antihorario", "ángulo", 90, "grados", "PNG", "celular de lado"),
    Perturbacion(
        "acercada", "recorte central y vuelta al tamaño original", "lado del recorte", 70, "% del lado", "PNG", "residuo que llena el encuadre"
    ),
    Perturbacion("comprimida", "compresión JPEG", "calidad", 15, "", "JPEG", "foto reenviada por mensajería"),
)
POR_NOMBRE = {p.nombre: p for p in PERTURBACIONES}


def oscurecer_o_aclarar(imagen: Image.Image, factor: float) -> Image.Image:
    return ImageEnhance.Brightness(imagen).enhance(factor)


def desenfocar(imagen: Image.Image, radio: float) -> Image.Image:
    return imagen.filter(ImageFilter.GaussianBlur(radius=radio))


def girar(imagen: Image.Image, grados: float) -> Image.Image:
    return imagen.rotate(grados, expand=True)


def acercar(imagen: Image.Image, porcentaje: float) -> Image.Image:
    ancho, alto = imagen.size
    lado_x, lado_y = round(ancho * porcentaje / 100), round(alto * porcentaje / 100)
    izquierda, arriba = (ancho - lado_x) // 2, (alto - lado_y) // 2
    recorte = imagen.crop((izquierda, arriba, izquierda + lado_x, arriba + lado_y))
    return recorte.resize((ancho, alto), Image.Resampling.BICUBIC)


def comprimir(imagen: Image.Image, calidad: float) -> Image.Image:
    memoria = io.BytesIO()
    imagen.save(memoria, format="JPEG", quality=int(calidad))
    memoria.seek(0)
    return Image.open(memoria).convert("RGB")


TRANSFORMACIONES: dict[str, Callable[[Image.Image, float], Image.Image]] = {
    "oscura": oscurecer_o_aclarar,
    "clara": oscurecer_o_aclarar,
    "desenfocada": desenfocar,
    "girada": girar,
    "acercada": acercar,
}


def alterar(imagen: Image.Image, perturbacion: Perturbacion) -> Image.Image:
    """La versión alterada en memoria; la comprimida se reabre desde su JPEG."""
    if perturbacion.nombre == "comprimida":
        return comprimir(imagen, perturbacion.valor)
    return TRANSFORMACIONES[perturbacion.nombre](imagen, perturbacion.valor)


def archivo_alterado(archivo: str, perturbacion: Perturbacion) -> str:
    return PurePosixPath(archivo).with_suffix(perturbacion.extension).as_posix()


def carpeta(perturbacion: Perturbacion) -> Path:
    return rutas.ROBUSTEZ / perturbacion.nombre


def guardar_alterada(archivo: str, perturbacion: Perturbacion, origen: Path, destino: Path) -> None:
    with Image.open(origen / archivo) as original:
        imagen = original.convert("RGB")
    salida = destino / archivo_alterado(archivo, perturbacion)
    salida.parent.mkdir(parents=True, exist_ok=True)
    if perturbacion.formato == "JPEG":
        imagen.save(salida, format="JPEG", quality=int(perturbacion.valor))
    else:
        alterar(imagen, perturbacion).save(salida, format="PNG")


def escribir_lista(perturbacion: Perturbacion, archivos: list[str], clases: list[str]) -> Path:
    ruta = carpeta(perturbacion) / LISTA
    with ruta.open("w", newline="", encoding="utf-8") as salida:
        escritor = csv.writer(salida)
        escritor.writerow(["archivo", "clase", "particion"])
        for archivo, clase in zip(archivos, clases, strict=True):
            escritor.writerow([archivo_alterado(archivo, perturbacion), clase, PRUEBA])
    return ruta


def generar(perturbacion: Perturbacion, archivos: list[str], clases: list[str]) -> Path:
    destino = carpeta(perturbacion)
    tarea = partial(guardar_alterada, perturbacion=perturbacion, origen=rutas.IMAGENES, destino=destino)
    with ProcessPoolExecutor() as procesos:
        list(procesos.map(tarea, archivos, chunksize=32))
    return escribir_lista(perturbacion, archivos, clases)


def parametros() -> dict:
    return {p.nombre: p.como_dict() for p in PERTURBACIONES}


def main() -> None:
    particion = leer_particion()
    prueba = particion[particion["particion"] == PRUEBA]
    archivos, clases = prueba["archivo"].tolist(), prueba["clase"].tolist()
    for perturbacion in PERTURBACIONES:
        lista = generar(perturbacion, archivos, clases)
        print(f"{perturbacion.nombre}: {len(archivos)} imágenes; lista en {lista}")


if __name__ == "__main__":
    main()
