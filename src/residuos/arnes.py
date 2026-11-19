"""Arnés de inferencia en el navegador.

Sirve la carpeta de Visión por Computador en 127.0.0.1:8765, se conecta por CDP a la instancia de
Brave de la RTX 5080 (puerto 9333) y corre `web/evaluacion.html` con un modelo de Teachable Machine
sobre una lista de imágenes. Escribe un CSV con una fila por imagen y, a su lado, un JSON con el
entorno (librerías, backend, renderizador y tiempos).

Columnas del CSV: `archivo`, `clase_real`, `prob_<etiqueta>` (una por clase, en el orden del modelo),
`material` y, si hay canecas, `caneca` y `prob_caneca_<caneca>` (una por caneca, en el orden de
`canecas.json`).

Uso, desde `clasificador/`:

    uv run python -m residuos.arnes paridad
    uv run python -m residuos.arnes evaluar --modelo ../modelo/entrenamiento-1 \\
        --lista datos/particion.csv --particion prueba \\
        --salida resultados/probabilidades-entrenamiento-1.csv

`--modelo` acepta una carpeta local dentro de la raíz servida o la URL de un modelo alojado.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import urllib.request
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import Browser, Page, sync_playwright

from residuos.rutas import CANECAS, IMAGENES, RAIZ_CLASIFICADOR, RAIZ_PROYECTO

RAIZ_SERVIDA = RAIZ_PROYECTO.parent
HOST = "127.0.0.1"
PUERTO_SERVIDOR = 8765
URL_SERVIDOR = f"http://{HOST}:{PUERTO_SERVIDOR}"
URL_CDP = f"http://{HOST}:9333"
PAGINA_EVALUACION = RAIZ_CLASIFICADOR / "web" / "evaluacion.html"
RESULTADOS = RAIZ_CLASIFICADOR / "resultados"

SEGUNDOS_BASE = 120
SEGUNDOS_POR_IMAGEN = 2
SEGUNDOS_ENTRE_CONSULTAS = 1.0


@dataclass(frozen=True)
class Imagen:
    archivo: str
    clase_real: str


@dataclass(frozen=True)
class ReferenciaParidad:
    """Modelo de la Actividad 2, sus 20 radiografías y las probabilidades de la vista previa."""

    modelo: Path
    imagenes: Path
    probabilidades: Path
    salida: Path


PARIDAD_A2 = ReferenciaParidad(
    modelo=RAIZ_SERVIDA / "entregas" / "actividad-2" / "modelo" / "tm-my-image-model",
    imagenes=RAIZ_SERVIDA / "entregas" / "actividad-2" / "dataset" / "spr-xray" / "prueba",
    probabilidades=RAIZ_SERVIDA / "entregas" / "actividad-2" / "capturas" / "prueba-tm-probabilidades.csv",
    salida=RESULTADOS / "paridad-a2.csv",
)


@dataclass(frozen=True)
class ResumenParidad:
    total: int
    misma_clase: int
    diferencia_maxima: float


def cdp_responde(url: str = URL_CDP) -> bool:
    """La única prueba fiable de que la instancia sirve: un puerto en escucha puede no responder."""
    try:
        with urllib.request.urlopen(f"{url}/json/version", timeout=3) as respuesta:
            return respuesta.status == 200
    except OSError:
        return False


def ruta_servida(ruta: Path) -> str:
    """Ruta relativa a la raíz servida; falla si la ruta queda fuera de ella."""
    return ruta.resolve().relative_to(RAIZ_SERVIDA).as_posix()


def url_local(ruta: Path) -> str:
    return f"{URL_SERVIDOR}/{quote(ruta_servida(ruta))}"


def url_modelo(modelo: str) -> str:
    return modelo if modelo.startswith(("http://", "https://")) else url_local(Path(modelo))


def nombre_modelo(modelo: str) -> str:
    """Cómo queda el modelo en el JSON del entorno: la URL o la ruta relativa, nunca la absoluta."""
    return modelo if modelo.startswith(("http://", "https://")) else ruta_servida(Path(modelo))


class _ManejadorSilencioso(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:  # noqa: A002 (firma heredada)
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


@contextmanager
def servidor_estatico() -> Iterator[None]:
    manejador = partial(_ManejadorSilencioso, directory=str(RAIZ_SERVIDA))
    servidor = ThreadingHTTPServer((HOST, PUERTO_SERVIDOR), manejador)
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    try:
        yield
    finally:
        servidor.shutdown()
        servidor.server_close()


def _gpu_segun_cdp(navegador: Browser) -> dict:
    """Renderizador del proceso de GPU. Brave enmascara el de la página (devuelve "Brave")."""
    sesion = navegador.new_browser_cdp_session()
    try:
        atributos = sesion.send("SystemInfo.getInfo")["gpu"]["auxAttributes"]
    finally:
        sesion.detach()
    return {"renderizador_gpu": atributos.get("glRenderer"), "implementacion_gl": atributos.get("glImplementationParts")}


@contextmanager
def pagina_del_arnes() -> Iterator[tuple[Page, dict]]:
    """Abre `evaluacion.html` en una pestaña nueva de la instancia y la cierra al salir.

    Solo se cierra la pestaña: la instancia de Brave sigue abierta para los pasos siguientes.
    """
    if not cdp_responde():
        raise RuntimeError(f"La instancia de Brave no responde en {URL_CDP}; lanzarla con herramientas/lanzar-brave-tm.sh")
    with servidor_estatico(), sync_playwright() as playwright:
        navegador = playwright.chromium.connect_over_cdp(URL_CDP)
        contexto = {"navegador": navegador.version, **_gpu_segun_cdp(navegador)}
        pagina = navegador.contexts[0].new_page()
        try:
            pagina.goto(url_local(PAGINA_EVALUACION))
            pagina.wait_for_function("() => typeof window.evaluar === 'function'")
            yield pagina, contexto
        finally:
            pagina.close()


def _esperar_resultados(pagina: Page, total: int) -> dict:
    limite = time.monotonic() + SEGUNDOS_BASE + SEGUNDOS_POR_IMAGEN * total
    ultimo_aviso = 0.0
    while time.monotonic() < limite:
        avance = pagina.evaluate("() => ({estado: window.resultados.estado, hechas: window.resultados.hechas})")
        if avance["estado"] in ("terminado", "error"):
            return pagina.evaluate("() => window.resultados")
        if time.monotonic() - ultimo_aviso > 30:
            print(f"  {avance['hechas']} de {total} imágenes", file=sys.stderr)
            ultimo_aviso = time.monotonic()
        time.sleep(SEGUNDOS_ENTRE_CONSULTAS)
    raise TimeoutError(f"El arnés no terminó {total} imágenes a tiempo")


def correr_en_navegador(modelo: str, urls: Sequence[str], canecas: dict | None, libreria: str | None) -> dict:
    parametros = {"libreria": libreria, "modelo": url_modelo(modelo), "canecas": canecas, "imagenes": list(urls)}
    with pagina_del_arnes() as (pagina, contexto):
        pagina.evaluate("parametros => { window.evaluar(parametros); }", parametros)
        resultados = _esperar_resultados(pagina, len(urls))
    if resultados["estado"] != "terminado":
        raise RuntimeError(f"El arnés falló en el navegador:\n{resultados.get('error')}")
    if len(resultados["filas"]) != len(urls):
        raise RuntimeError(f"El arnés devolvió {len(resultados['filas'])} filas para {len(urls)} imágenes")
    return {**resultados, "cdp": contexto}


def encabezado_csv(etiquetas: Sequence[str], canecas: dict | None) -> list[str]:
    columnas = ["archivo", "clase_real", *(f"prob_{e}" for e in etiquetas), "material"]
    if canecas:
        columnas += ["caneca", *(f"prob_caneca_{c}" for c in canecas["canecas"])]
    return columnas


def fila_csv(imagen: Imagen, fila: dict, etiquetas: Sequence[str], canecas: dict | None) -> list:
    valores = [imagen.archivo, imagen.clase_real, *(fila["probabilidades"][e] for e in etiquetas), fila["material"]]
    if canecas:
        valores += [fila["caneca"], *(fila["probabilidadesCaneca"][c] for c in canecas["canecas"])]
    return valores


def escribir_csv(salida: Path, imagenes: Sequence[Imagen], filas: Sequence[dict], etiquetas: Sequence[str], canecas: dict | None) -> None:
    salida.parent.mkdir(parents=True, exist_ok=True)
    with salida.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.writer(archivo)
        escritor.writerow(encabezado_csv(etiquetas, canecas))
        escritor.writerows(fila_csv(i, f, etiquetas, canecas) for i, f in zip(imagenes, filas, strict=True))


def ruta_entorno(salida: Path) -> Path:
    return salida.with_suffix(".entorno.json")


def evaluar(modelo: str, imagenes: Sequence[Imagen], base: Path, salida: Path, canecas: dict | None = None, libreria: str | None = None) -> dict:
    """Corre el modelo sobre las imágenes (rutas relativas a `base`) y escribe el CSV y su entorno."""
    urls = [url_local(base / imagen.archivo) for imagen in imagenes]
    resultados = correr_en_navegador(modelo, urls, canecas, libreria)
    escribir_csv(salida, imagenes, resultados["filas"], resultados["etiquetas"], canecas)
    entorno = {
        "modelo": nombre_modelo(modelo),
        "imagenes": len(imagenes),
        "base_imagenes": ruta_servida(base),
        "etiquetas": resultados["etiquetas"],
        **resultados["entorno"],
        **resultados["cdp"],
        "segundos": resultados["segundos"],
        "tensores_antes": resultados["tensores_antes"],
        "tensores_al_final": resultados["tensores_al_final"],
        "fecha": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    ruta_entorno(salida).write_text(json.dumps(entorno, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return entorno


def leer_csv(ruta: Path) -> list[dict[str, str]]:
    with ruta.open(newline="", encoding="utf-8") as archivo:
        return list(csv.DictReader(archivo))


def leer_lista(ruta: Path, particion: str | None = None, columna_clase: str = "clase") -> list[Imagen]:
    filas = leer_csv(ruta)
    if particion is not None:
        filas = [f for f in filas if f["particion"] == particion]
    return [Imagen(f["archivo"], f[columna_clase]) for f in filas]


def cargar_canecas(ruta: Path = CANECAS) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def correr_paridad(referencia: ReferenciaParidad = PARIDAD_A2, libreria: str | None = None) -> dict:
    imagenes = leer_lista(referencia.probabilidades, columna_clase="sexo_real")
    return evaluar(str(referencia.modelo), imagenes, referencia.imagenes, referencia.salida, libreria=libreria)


def comparar_paridad(referencia: ReferenciaParidad = PARIDAD_A2) -> ResumenParidad:
    """Compara lo que dio el arnés con lo que mostró la vista previa de Teachable Machine."""
    esperadas = {f["archivo"]: f for f in leer_csv(referencia.probabilidades)}
    obtenidas = leer_csv(referencia.salida)
    columnas = [c for c in obtenidas[0] if c.startswith("prob_")]
    misma_clase = sum(f["material"] == esperadas[f["archivo"]]["clase_predicha"] for f in obtenidas)
    diferencia = max(abs(float(f[c]) - float(esperadas[f["archivo"]][c])) for f in obtenidas for c in columnas)
    return ResumenParidad(total=len(obtenidas), misma_clase=misma_clase, diferencia_maxima=diferencia)


def _argumentos(argv: Sequence[str] | None) -> argparse.Namespace:
    analizador = argparse.ArgumentParser(prog="residuos.arnes", description=__doc__.splitlines()[0])
    sub = analizador.add_subparsers(dest="orden", required=True)
    paridad = sub.add_parser("paridad", help="modelo de la A2 sobre sus 20 radiografías frente a la vista previa")
    paridad.add_argument("--libreria", help="combinación de inferencia.js (por defecto, la de LIBRERIA_POR_DEFECTO)")
    evaluacion = sub.add_parser("evaluar", help="un modelo sobre una lista de imágenes")
    evaluacion.add_argument("--modelo", required=True, help="carpeta local con model.json o URL del modelo alojado")
    evaluacion.add_argument("--lista", type=Path, required=True, help="CSV con las columnas archivo y clase")
    evaluacion.add_argument("--particion", help="filtra la lista por su columna particion")
    evaluacion.add_argument("--base", type=Path, default=IMAGENES, help="carpeta a la que son relativos los archivos")
    evaluacion.add_argument("--canecas", type=Path, default=CANECAS)
    evaluacion.add_argument("--salida", type=Path, required=True)
    evaluacion.add_argument("--libreria")
    return analizador.parse_args(argv)


def _resumen_entorno(entorno: dict) -> str:
    return f"backend {entorno['backend']}; renderizador {entorno['renderizador_gpu']}; librerías {entorno['paquetes']}; {entorno['segundos']:.1f} s"


def main(argv: Sequence[str] | None = None) -> None:
    argumentos = _argumentos(argv)
    if argumentos.orden == "paridad":
        entorno = correr_paridad(libreria=argumentos.libreria)
        resumen = comparar_paridad()
        print(f"Paridad: {resumen.misma_clase} de {resumen.total} en la misma clase; diferencia máxima {resumen.diferencia_maxima:.6f}")
    else:
        imagenes = leer_lista(argumentos.lista, argumentos.particion)
        entorno = evaluar(argumentos.modelo, imagenes, argumentos.base, argumentos.salida, cargar_canecas(argumentos.canecas), argumentos.libreria)
        print(f"{len(imagenes)} imágenes evaluadas en {argumentos.salida}")
    print(_resumen_entorno(entorno))


if __name__ == "__main__":
    main()
