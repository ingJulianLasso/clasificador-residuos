"""Pruebas de la app web (`web/`).

Sin navegador: lo heredado de la evaluación sigue intacto (`inferencia.js` y `evaluacion.html` con sus
SHA-256, y las librerías solo desde `inferencia.js`, con su combinación por defecto); el modelo y las
canecas tienen una sola fuente; las rutas sirven bajo GitHub Pages; git no rastrea pesos ni la paridad de
librerías; los umbrales son 0,80 y 0,50, y el fixture de `node --test` está al día.

Con el marcador `navegador`: la app servida en `APP_URL` (por defecto, un servidor estático de
`clasificador/` en 127.0.0.1:8766) dentro de la instancia de Brave del puerto 9333. Comprueba el origen del
modelo, la paridad con la evaluación del proyecto sobre el 30 % de prueba, Clasificar con fotos y archivos
ilegibles, el celular sin desplazamiento horizontal y axe-core en las tres pestañas. Escribe
`resultados/app-paridad.json`.

    uv run pytest -m "not navegador" tests/test_app.py
    APP_URL=http://127.0.0.1:8766/web/ uv run pytest -s -m navegador tests/test_app.py
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from functools import partial
from html.parser import HTMLParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest
from playwright.sync_api import Page, sync_playwright

from residuos.arnes import URL_CDP, cdp_responde, leer_csv
from residuos.fixture_app import (
    FIXTURE,
    METRICAS_ELEGIDO,
    construir_fixture,
    diferencia_maxima,
    escribir_etiquetas_prueba,
    texto_json,
    valores_de_la_app,
)
from residuos.metricas import CSV_ELEGIDO
from residuos.rutas import IMAGENES, MODELOS_JSON, RAIZ_CLASIFICADOR, RESULTADOS

WEB = RAIZ_CLASIFICADOR / "web"
HERENCIA = RESULTADOS / "herencia-a5.json"
APP_PARIDAD = RESULTADOS / "app-paridad.json"

APP_URL = os.environ.get("APP_URL", "http://127.0.0.1:8766/web/")
SERVIDOR_LOCAL = ("127.0.0.1", 8766)
NAVEGADOR_DISPONIBLE = cdp_responde()

ORIGENES_DEL_MODELO = {"teachablemachine.withgoogle.com", "storage.googleapis.com"}
ARCHIVO_DEL_MODELO = re.compile(r"/(model\.json|metadata\.json|[^/]+\.bin)$")
AXE_URL = "https://cdn.jsdelivr.net/npm/axe-core@4.10.2/axe.min.js"
AXE_VERSION = "4.10.2"
IMPACTOS_GRAVES = {"serious", "critical"}
PESTANAS = ("clasificar", "evaluar", "acerca")
VISTAS_AXE = {"escritorio": (1280, 800), "celular": (390, 844)}
VISTAS_ESTRECHAS = {"celular": (390, 844), "estrecho": (320, 640)}

TOLERANCIA_PROBABILIDAD = 1e-4
TOLERANCIA_AUC = 1e-3
TOLERANCIA_METRICAS = 1e-9
CLAVES_AUC = ("auc", "auc_macro", "auc_micro")
FOTOS_A_CLASIFICAR = 12
SEGUNDOS_MODELO = 180
SEGUNDOS_EVALUACION = 900
SEGUNDOS_CLASIFICACION = 180

# Pesos y modelos exportados, y la paridad de librerías, que depende de un modelo que no se publica.
NO_PUBLICABLES = re.compile(r"(^|/)(model\.json|[^/]+\.bin|paridad-a2[^/]*)$")


# ---------------------------------------------------------------------------------------------
# Utilidades


class _Scripts(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.scripts: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "script":
            self.scripts.append(dict(attrs))


def scripts_de(ruta: Path) -> list[dict[str, str | None]]:
    lector = _Scripts()
    lector.feed(ruta.read_text(encoding="utf-8"))
    return lector.scripts


def git(*argumentos: str) -> list[str]:
    salida = subprocess.run(["git", "-C", str(RAIZ_CLASIFICADOR), *argumentos], capture_output=True, text=True, check=True)
    return salida.stdout.splitlines()


def commit_actual() -> str | None:
    try:
        return git("rev-parse", "HEAD")[0]
    except subprocess.CalledProcessError:
        return None  # repositorio todavía sin commits


def umbrales_de_la_app() -> dict[str, float]:
    codigo = (
        "import * as u from './web/umbrales.js';"
        "console.log(JSON.stringify({caneca: u.UMBRAL_CANECA_DUDOSA, material: u.UMBRAL_LEJOS_DE_LO_APRENDIDO}));"
    )
    salida = subprocess.run(
        ["node", "--input-type=module", "-e", codigo], cwd=RAIZ_CLASIFICADOR, capture_output=True, text=True, check=True
    )
    return json.loads(salida.stdout)


def leer_csv_texto(texto: str) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(texto)))


def si_no(condicion: bool) -> str:
    """Cómo escribe la app los avisos en su CSV."""
    return "sí" if condicion else "no"


# ---------------------------------------------------------------------------------------------
# Sin navegador


def test_index_solo_carga_app_js_como_modulo():
    assert scripts_de(WEB / "index.html") == [{"type": "module", "src": "app.js"}]


def test_las_librerias_salen_solo_de_inferencia_js_con_la_combinacion_por_defecto():
    """La app predice como la evaluación: importa el mismo inferencia.js y no repite URLs de librerías."""
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert re.search(r"import \{[^}]*\bcargarLibrerias\b[^}]*\} from '\./inferencia\.js';", app)
    assert re.findall(r"cargarLibrerias\(([^)]*)\)", app) == [""]
    assert "from './inferencia.js'" in (WEB / "evaluacion.html").read_text(encoding="utf-8")
    for nombre in ("index.html", "app.js", "metricas.js", "umbrales.js", "estilos.css"):
        texto = (WEB / nombre).read_text(encoding="utf-8")
        for ajeno in ("cdn.jsdelivr.net", "tf.min.js", "teachablemachine-image", "integrity", "LIBRERIAS"):
            assert ajeno not in texto, f"{nombre} nombra {ajeno}"


def test_inferencia_y_evaluacion_tienen_el_sha256_heredado():
    herencia = json.loads(HERENCIA.read_text(encoding="utf-8"))
    assert set(herencia["sha256"]) == {"web/inferencia.js", "web/evaluacion.html"}
    for ruta, esperado in herencia["sha256"].items():
        assert hashlib.sha256((RAIZ_CLASIFICADOR / ruta).read_bytes()).hexdigest() == esperado, ruta


def test_el_modelo_y_las_canecas_tienen_una_sola_fuente():
    app = (WEB / "app.js").read_text(encoding="utf-8")
    assert "'../resultados/modelos.json'" in app
    assert "'../config/canecas.json'" in app
    for ruta in (WEB / "index.html", WEB / "app.js", RAIZ_CLASIFICADOR / "index.html"):
        assert "teachablemachine.withgoogle.com/models/" not in ruta.read_text(encoding="utf-8"), ruta.name


def test_las_rutas_son_relativas_para_servir_bajo_pages():
    """Pages publica el sitio en /clasificador-residuos/: ninguna ruta propia puede empezar por /."""
    for ruta in (WEB / "index.html", RAIZ_CLASIFICADOR / "index.html"):
        destinos = re.findall(r'(?:src|href)="([^"]*)"', ruta.read_text(encoding="utf-8"))
        assert [d for d in destinos if d.startswith("/")] == [], ruta.name
    for ruta in WEB.glob("*.js"):
        assert not re.search(r"""(?:from|fetch\()\s*['"]/""", ruta.read_text(encoding="utf-8")), ruta.name


def test_la_raiz_redirige_a_web_con_un_enlace_visible():
    raiz = (RAIZ_CLASIFICADOR / "index.html").read_text(encoding="utf-8")
    assert '<meta http-equiv="refresh" content="0; url=web/">' in raiz
    assert '<a href="web/">' in raiz
    assert 'lang="es-CO"' in raiz
    assert (RAIZ_CLASIFICADOR / ".nojekyll").is_file()


def test_git_no_rastrea_pesos_ni_la_paridad_de_librerias():
    rastreados = git("ls-files")
    por_agregar = git("ls-files", "--others", "--exclude-standard")
    assert [r for r in (*rastreados, *por_agregar) if NO_PUBLICABLES.search(r)] == []
    ignorados = ["resultados/paridad-a2.csv", "resultados/paridad-a2.entorno.json"]
    assert git("check-ignore", *ignorados) == ignorados


def test_los_umbrales_son_080_y_050():
    assert umbrales_de_la_app() == {"caneca": 0.8, "material": 0.5}


def test_el_fixture_de_node_esta_al_dia():
    assert FIXTURE.read_text(encoding="utf-8") == texto_json(construir_fixture())


# ---------------------------------------------------------------------------------------------
# Con navegador


def requiere_navegador(prueba):
    prueba = pytest.mark.navegador(prueba)
    return pytest.mark.skipif(not NAVEGADOR_DISPONIBLE, reason="la instancia de Brave no responde en el puerto 9333")(prueba)


class _ManejadorSinCache(SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        pass

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


@contextmanager
def servidor_si_es_local(url: str) -> Iterator[None]:
    """Sirve `clasificador/` en 127.0.0.1:8766 cuando `APP_URL` apunta ahí; con otra URL no hace nada."""
    partes = urlsplit(url)
    if (partes.hostname, partes.port) != SERVIDOR_LOCAL:
        yield
        return
    servidor = ThreadingHTTPServer(SERVIDOR_LOCAL, partial(_ManejadorSinCache, directory=str(RAIZ_CLASIFICADOR)))
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    try:
        yield
    finally:
        servidor.shutdown()
        servidor.server_close()


@pytest.fixture(scope="module")
def registro() -> Iterator[dict]:
    """Lo que dejan las pruebas en el navegador; se escribe en resultados/app-paridad.json al final."""
    datos = {
        "url": APP_URL,
        "commit": commit_actual(),
        "cambios_sin_commit": bool(git("status", "--porcelain")),
        "hora": datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    yield datos
    APP_PARIDAD.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@pytest.fixture(scope="module")
def app(registro) -> Iterator[SimpleNamespace]:
    """La app abierta en una pestaña nueva, sin caché, con el modelo ya cargado."""
    with servidor_si_es_local(APP_URL), sync_playwright() as playwright:
        navegador = playwright.chromium.connect_over_cdp(URL_CDP)
        pagina = navegador.contexts[0].new_page()
        respuestas: list[dict] = []
        pagina.on("response", lambda r: respuestas.append({"url": r.url, "estado": r.status}))
        cdp = pagina.context.new_cdp_session(pagina)
        cdp.send("Network.enable")
        cdp.send("Network.setCacheDisabled", {"cacheDisabled": True})
        try:
            pagina.set_viewport_size({"width": 1280, "height": 800})
            pagina.goto(APP_URL)
            pagina.wait_for_function("document.body.dataset.modelo !== 'cargando'", timeout=SEGUNDOS_MODELO * 1000)
            assert pagina.evaluate("document.body.dataset.modelo") == "listo", pagina.inner_text("#estado-modelo")
            registro["navegador"] = navegador.version
            yield SimpleNamespace(pagina=pagina, respuestas=respuestas)
        finally:
            pagina.close()


def ir_a(pagina: Page, pestana: str) -> None:
    pagina.click(f"#pestana-{pestana}")
    pagina.wait_for_selector(f"#panel-{pestana}:not([hidden])")


def esperar_listo(pagina: Page, panel: str, segundos: int) -> None:
    pagina.wait_for_function(
        f"['listo', 'error'].includes(document.querySelector('#panel-{panel}').dataset.estado)", timeout=segundos * 1000
    )
    assert pagina.evaluate(f"document.querySelector('#panel-{panel}').dataset.estado") == "listo"


def poner_archivos(pagina: Page, selector: str, rutas: Sequence[str]) -> None:
    """Elige archivos en un `<input type=file>` con `DOM.setFileInputFiles` de CDP, que dispara `change`.

    Playwright no pasa más de 50 MB a un navegador conectado por CDP, y el 30 % de prueba los supera; como la
    instancia corre en esta misma máquina, basta con darle las rutas.
    """
    cdp = pagina.context.new_cdp_session(pagina)
    try:
        raiz = cdp.send("DOM.getDocument", {"depth": 0})["root"]["nodeId"]
        nodo = cdp.send("DOM.querySelector", {"nodeId": raiz, "selector": selector})["nodeId"]
        cdp.send("DOM.setFileInputFiles", {"nodeId": nodo, "files": [str(r) for r in rutas]})
    finally:
        cdp.detach()


def contenido_de_descarga(pagina: Page, selector: str) -> str:
    """Lo que bajaría el enlace de descarga, leído desde su URL de blob sin abrir el diálogo de guardar."""
    return pagina.evaluate("async (s) => (await fetch(document.querySelector(s).href)).text()", selector)


def id_del_modelo() -> str:
    modelos = json.loads(MODELOS_JSON.read_text(encoding="utf-8"))
    return urlsplit(modelos[modelos["elegido"]]["url"]).path.rstrip("/").split("/")[-1]


def comparar_probabilidades(app: list[dict[str, str]], referencia: list[dict[str, str]]) -> dict:
    por_archivo = {fila["archivo"]: fila for fila in app}
    columnas = [c for c in referencia[0] if c.startswith("prob_")]
    mismos = 0
    diferencia = 0.0
    for esperada in referencia:
        obtenida = por_archivo.get(esperada["archivo"])
        if obtenida is None:
            continue
        mismos += (obtenida["material"], obtenida["caneca"]) == (esperada["material"], esperada["caneca"])
        diferencia = max(diferencia, *(abs(float(obtenida[c]) - float(esperada[c])) for c in columnas))
    return {
        "n": len(app),
        "n_referencia": len(referencia),
        "mismos_archivos": set(por_archivo) == {f["archivo"] for f in referencia},
        "mismo_material_y_caneca": mismos,
        "diferencia_maxima": diferencia,
    }


def comparar_metricas(informe: dict, referencia: dict) -> dict:
    """Métricas del JSON de la app frente a metricas-elegido.json: AUC con 1e-3 y lo demás con 1e-9."""
    resultado = {}
    for tarea in ("material", "caneca"):
        obtenida, esperada = informe[tarea], referencia[tarea]
        valores = valores_de_la_app(esperada["valores"], esperada["clases"])
        auc = [k for k in valores if k in CLAVES_AUC]
        resto = [k for k in valores if k not in CLAVES_AUC]
        diferencia_auc = diferencia_maxima({k: valores[k] for k in auc}, {k: obtenida["valores"][k] for k in auc})
        diferencia_resto = diferencia_maxima({k: valores[k] for k in resto}, {k: obtenida["valores"][k] for k in resto})
        resultado[tarea] = {
            "matriz_igual": obtenida["matriz"] == esperada["matriz"],
            "soporte_igual": obtenida["soporte"] == esperada["soporte"],
            "diferencia_auc": diferencia_auc,
            "diferencia_resto": diferencia_resto,
            "iguales": (
                obtenida["matriz"] == esperada["matriz"]
                and obtenida["soporte"] == esperada["soporte"]
                and diferencia_auc <= TOLERANCIA_AUC
                and diferencia_resto <= TOLERANCIA_METRICAS
            ),
        }
    return resultado


def crear_heic(destino: Path) -> Path:
    """Una foto HEIC de verdad si ImageMagick la sabe escribir; si no, la cabecera de un HEIC."""
    if shutil.which("magick"):
        subprocess.run(["magick", "-size", "64x64", "gradient:green-white", str(destino)], check=True, capture_output=True)
    if not destino.exists() or destino.stat().st_size == 0:
        destino.write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic")
    return destino


def fotos_repartidas(cuantas: int) -> list[dict[str, str]]:
    """Fotos del 30 % de prueba repartidas entre las clases (una cada tantas filas)."""
    prueba = leer_csv(escribir_etiquetas_prueba())
    paso = len(prueba) // cuantas
    return prueba[::paso][:cuantas]


@requiere_navegador
def test_el_modelo_llega_de_la_nube_de_teachable_machine(app, registro):
    archivos = [r for r in app.respuestas if ARCHIVO_DEL_MODELO.search(urlsplit(r["url"]).path)]
    registro["origen_modelo"] = archivos
    servidos = {Path(urlsplit(r["url"]).path).name for r in archivos if r["estado"] == 200}
    assert {"model.json", "metadata.json"} <= servidos
    assert any(nombre.endswith(".bin") for nombre in servidos)
    assert all(urlsplit(r["url"]).hostname in ORIGENES_DEL_MODELO for r in archivos)
    assert all(id_del_modelo() in r["url"] for r in archivos)


@requiere_navegador
def test_paridad_con_la_evaluacion_del_proyecto(app, registro):
    pagina = app.pagina
    etiquetas = escribir_etiquetas_prueba()
    fotos = [str(IMAGENES / fila["archivo"]) for fila in leer_csv(etiquetas)]
    ir_a(pagina, "evaluar")
    poner_archivos(pagina, "#evaluar-csv", [etiquetas])
    poner_archivos(pagina, "#evaluar-imagenes", fotos)
    inicio = time.monotonic()
    pagina.click("#form-csv button[type=submit]")
    esperar_listo(pagina, "evaluar", SEGUNDOS_EVALUACION)
    segundos = time.monotonic() - inicio

    csv_app = leer_csv_texto(contenido_de_descarga(pagina, "#evaluar-descargar-csv"))
    informe = json.loads(contenido_de_descarga(pagina, "#evaluar-descargar-json"))
    probabilidades = comparar_probabilidades(csv_app, leer_csv(RESULTADOS / CSV_ELEGIDO))
    metricas = comparar_metricas(informe, json.loads(METRICAS_ELEGIDO.read_text(encoding="utf-8")))
    n = probabilidades["n"]
    iguales = (
        probabilidades["mismos_archivos"]
        and probabilidades["mismo_material_y_caneca"] == n == probabilidades["n_referencia"]
        and probabilidades["diferencia_maxima"] <= TOLERANCIA_PROBABILIDAD
    )
    registro.update(
        n=n,
        iguales=iguales,
        diferencia_maxima=probabilidades["diferencia_maxima"],
        metricas_iguales=all(t["iguales"] for t in metricas.values()),
        paridad=probabilidades,
        metricas=metricas,
        segundos_evaluacion=round(segundos, 1),
        entorno=informe["entorno"],
        sin_evaluar=informe["sin_evaluar"],
    )
    print(
        f"\nParidad: {probabilidades['mismo_material_y_caneca']} de {n} imágenes con el mismo material y la misma caneca "
        f"(referencia: {probabilidades['n_referencia']}); diferencia máxima {probabilidades['diferencia_maxima']}; "
        f"métricas iguales: {registro['metricas_iguales']}; {segundos:.1f} s"
    )
    assert n == len(fotos) == probabilidades["n_referencia"]
    assert iguales, probabilidades
    assert registro["metricas_iguales"], metricas
    assert informe["entorno"]["libreria"] == "fijada"
    assert not any(informe["sin_evaluar"].values()), informe["sin_evaluar"]


@requiere_navegador
def test_clasificar_doce_fotos_y_dos_archivos_ilegibles(app, registro, tmp_path):
    pagina = app.pagina
    fotos = fotos_repartidas(FOTOS_A_CLASIFICAR)
    texto = tmp_path / "notas.txt"
    texto.write_text("Esto no es una foto.\n", encoding="utf-8")
    archivos = [str(IMAGENES / f["archivo"]) for f in fotos] + [str(texto), str(crear_heic(tmp_path / "foto.heic"))]
    ir_a(pagina, "clasificar")
    poner_archivos(pagina, "#clasificar-fotos", archivos)
    esperar_listo(pagina, "clasificar", SEGUNDOS_CLASIFICACION)

    tarjetas = pagina.locator("#clasificar-tarjetas > li.tarjeta:not(.tarjeta-error)").count()
    errores = pagina.locator("#clasificar-tarjetas > li.tarjeta-error").all_inner_texts()
    filas = leer_csv_texto(contenido_de_descarga(pagina, "#clasificar-descargar-csv"))
    referencia = {Path(f["archivo"]).name: f for f in leer_csv(RESULTADOS / CSV_ELEGIDO)}
    umbrales = umbrales_de_la_app()
    coinciden = sum(
        (f["etiqueta"], f["caneca"]) == (referencia[f["archivo"]]["material"], referencia[f["archivo"]]["caneca"])
        and f["caneca_dudosa"] == si_no(float(f["probabilidad_caneca"]) < umbrales["caneca"])
        and f["lejos_de_lo_aprendido"] == si_no(float(f["probabilidad_material"]) < umbrales["material"])
        for f in filas
    )
    registro["clasificar"] = {
        "archivos": len(archivos),
        "tarjetas": tarjetas,
        "errores": len(errores),
        "filas_csv": len(filas),
        "iguales_a_la_evaluacion": coinciden,
    }
    assert (tarjetas, len(errores), len(filas)) == (FOTOS_A_CLASIFICAR, 2, FOTOS_A_CLASIFICAR)
    assert coinciden == FOTOS_A_CLASIFICAR
    assert any("HEIC" in e for e in errores) and any("notas.txt" in e for e in errores)


@requiere_navegador
def test_sin_desplazamiento_horizontal_en_el_celular(app, registro):
    pagina = app.pagina
    exceso = {}
    try:
        for vista, (ancho, alto) in VISTAS_ESTRECHAS.items():
            pagina.set_viewport_size({"width": ancho, "height": alto})
            for pestana in PESTANAS:
                ir_a(pagina, pestana)
                exceso[f"{vista} {ancho} px, {pestana}"] = pagina.evaluate(
                    "() => document.documentElement.scrollWidth - document.documentElement.clientWidth"
                )
    finally:
        pagina.set_viewport_size({"width": 1280, "height": 800})
    registro["desplazamiento_horizontal_px"] = exceso
    assert all(px <= 0 for px in exceso.values()), exceso


@requiere_navegador
def test_axe_sin_violaciones_serias_ni_criticas(app, registro):
    pagina = app.pagina
    pagina.add_script_tag(url=AXE_URL)
    pagina.wait_for_function("typeof window.axe === 'object'")
    assert pagina.evaluate("axe.version") == AXE_VERSION
    resultados: dict[str, dict] = {pestana: {} for pestana in PESTANAS}
    try:
        for vista, (ancho, alto) in VISTAS_AXE.items():
            pagina.set_viewport_size({"width": ancho, "height": alto})
            for pestana in PESTANAS:
                ir_a(pagina, pestana)
                violaciones = pagina.evaluate(
                    """async () => (await axe.run(document, { resultTypes: ['violations'] })).violations
                        .map((v) => ({ regla: v.id, impacto: v.impact, nodos: v.nodes.length }))"""
                )
                graves = [v for v in violaciones if v["impacto"] in IMPACTOS_GRAVES]
                resultados[pestana][vista] = {"serias_o_criticas": len(graves), "violaciones": violaciones}
    finally:
        pagina.set_viewport_size({"width": 1280, "height": 800})
    registro["axe"] = {"version": AXE_VERSION, **resultados}
    assert all(r["serias_o_criticas"] == 0 for vistas in resultados.values() for r in vistas.values()), resultados
