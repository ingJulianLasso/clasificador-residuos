"""Pruebas de la prueba en campo: revisión de la carpeta frente al registro, métricas con un caso armado
a mano, comparación con la exactitud balanceada del 30 % y cifras del informe con y sin datos de campo.

Ninguna necesita fotos reales. La marcada `navegador` comprueba en la instancia de Brave del puerto 9333
que el navegador aplica la orientación EXIF al decodificar; se salta si la instancia no responde. Las que
leen el registro real o regeneran las cifras del informe se saltan si faltan esos archivos, que no se
publican.
"""

import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image, ImageOps
from requisitos import requiere_informe, requiere_paridad, requiere_registro_campo

from residuos import campo, cifras, metricas, metricas_campo, rutas
from residuos.arnes import cdp_responde
from residuos.datos import ETIQUETA_ORIENTACION_EXIF

URL = "https://teachablemachine.withgoogle.com/models/prueba/"
# Foto armada por clase: el material que predice el modelo ficticio (el resto acierta).
ERRORES_ARMADOS = {
    "Plastic/02.jpg": "Cardboard",  # material mal, caneca bien (blanca)
    "Food Organics/02.jpg": "Vegetation",  # material mal, caneca bien (verde)
    "Miscellaneous Trash/02.jpg": "Plastic",  # negra a la blanca: contamina
    "Vegetation/02.jpg": "Miscellaneous Trash",  # verde a la negra
}
CON_ORIENTACION = {"Glass/01.jpg": 6, "Metal/02.jpg": 8, "Paper/01.jpg": 1}


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def guardar_jpeg(ruta: Path, tamano: tuple[int, int] = (40, 30), orientacion: int | None = None) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    if orientacion is not None:
        exif[ETIQUETA_ORIENTACION_EXIF] = orientacion
    Image.new("RGB", tamano, (120, 80, 40)).save(ruta, format="JPEG", exif=exif.tobytes())


@pytest.fixture(scope="module")
def canecas() -> dict:
    return cargar_json(rutas.CANECAS)


@pytest.fixture(scope="module")
def registro() -> pd.DataFrame:
    """Dos fotos por clase (condiciones 01 y 02) con el formato del registro real."""
    filas = campo.leer_registro()
    filas = filas[filas["numero"].isin(["01", "02"])].reset_index(drop=True)
    filas.loc[filas["archivo"] == "Glass/01.jpg", "objeto_usado"] = "otro frasco"
    return filas


# Revisión de la carpeta


@requiere_registro_campo
def test_el_registro_real_tiene_20_fotos_por_clase_y_9_por_condicion(canecas):
    registro = campo.leer_registro()
    assert len(registro) == 180
    assert set(registro["clase"].value_counts()) == {20}
    assert set(registro["numero"].value_counts()) == {9}
    assert sorted(registro["clase"].unique()) == sorted(canecas["materiales"])
    assert (registro["archivo"] == registro["clase"] + "/" + registro["numero"] + ".jpg").all()
    assert (registro["caneca_esperada"] == registro["clase"].map(metricas.caneca_de(canecas))).all()
    assert set(metricas_campo.CONDICIONES) <= set(registro.columns)
    for factor, niveles in metricas_campo.CONDICIONES.items():
        assert set(registro[factor]) == set(niveles), factor


@requiere_registro_campo
def test_verificar_reporta_cada_problema(tmp_path, registro, capsys):
    filas = registro[registro["clase"].isin(["Glass", "Metal", "Plastic"])].reset_index(drop=True)
    guardar_jpeg(tmp_path / "Glass/01.jpg", orientacion=6)
    guardar_jpeg(tmp_path / "Glass/IMG_0001.jpg")  # sobra
    Image.new("RGB", (8, 8)).save(tmp_path / "Glass/02.jpg", format="PNG")  # no es JPEG
    guardar_jpeg(tmp_path / "Metal/1.JPG")  # mal escrito; Metal/01.jpg y Metal/02.jpg faltan
    (tmp_path / "Metal/.gitkeep").touch()  # se ignora
    (tmp_path / "Plastic").mkdir()
    (tmp_path / "Plastic/01.jpg").write_bytes(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00mif1heic")  # HEIC renombrado
    guardar_jpeg(tmp_path / "Plastic/02.jpg")
    revision = campo.revisar(filas, tmp_path)
    assert revision.validas == ["Glass/01.jpg", "Plastic/02.jpg"]
    assert revision.faltan == ["Metal/01.jpg", "Metal/02.jpg"]
    assert revision.mal_escritos == {"Metal/1.JPG": "Metal/01.jpg"}
    assert set(revision.sobrantes) == {"Glass/IMG_0001.jpg", "Metal/1.JPG"}
    assert [(f.archivo, f.formato) for f in revision.no_jpeg] == [("Glass/02.jpg", "PNG")]
    assert [(f.archivo, f.formato) for f in revision.no_abren] == [("Plastic/01.jpg", "HEIF")]
    assert revision.fotos["Glass/01.jpg"].vertical_al_mostrarla  # 40 x 30 guardada, girada por la etiqueta 6
    assert not revision.en_orden
    filas.to_csv(tmp_path / "registro.csv", index=False)
    with pytest.raises(SystemExit) as salida:
        campo.main(["verificar", "--registro", str(tmp_path / "registro.csv"), "--carpeta", str(tmp_path)])
    assert salida.value.code == 1
    texto = capsys.readouterr().out
    assert "Fotos listas: 2 de 6" in texto
    assert "Por condición (NN): 01 1 de 3; 02 1 de 3" in texto
    assert "Nombre mal escrito: Metal/1.JPG (JPEG) debería ser Metal/01.jpg en JPEG" in texto


@requiere_registro_campo
def test_verificar_en_orden_sale_con_cero_y_lee_las_columnas_por_nombre(tmp_path, registro, capsys):
    filas = registro[registro["clase"] == "Paper"]
    filas = filas[list(reversed(filas.columns))]  # otro orden de columnas
    for archivo in filas["archivo"]:
        guardar_jpeg(tmp_path / "fotos" / archivo, tamano=(30, 40))
    filas.to_csv(tmp_path / "registro.csv", index=False)
    with pytest.raises(SystemExit) as salida:
        campo.main(["verificar", "--registro", str(tmp_path / "registro.csv"), "--carpeta", str(tmp_path / "fotos")])
    assert salida.value.code == 0
    texto = capsys.readouterr().out
    assert "Fotos listas: 2 de 2" in texto
    assert "Todo en orden." in texto


# Métricas con un caso armado a mano


def tabla_armada(registro: pd.DataFrame, canecas: dict) -> pd.DataFrame:
    """Lo que escribiría el arnés: 0,9 para el material predicho y el resto repartido."""
    etiquetas = list(canecas["materiales"])
    filas = []
    for archivo, clase in zip(registro["archivo"], registro["clase"], strict=True):
        predicho = ERRORES_ARMADOS.get(archivo, clase)
        probabilidades = {f"prob_{e}": 0.9 if e == predicho else 0.1 / (len(etiquetas) - 1) for e in etiquetas}
        filas.append({"archivo": archivo, "clase_real": clase, **probabilidades, "material": predicho})
    tabla = pd.DataFrame(filas)
    sumas, ganadora = metricas.recalcular_canecas(tabla, canecas)
    tabla["caneca"] = ganadora
    for caneca in canecas["canecas"]:
        tabla[f"prob_caneca_{caneca}"] = sumas[caneca]
    return tabla


@pytest.fixture(scope="module")
def carpeta_armada(tmp_path_factory, registro) -> Path:
    carpeta = tmp_path_factory.mktemp("campo")
    for archivo in registro["archivo"]:
        guardar_jpeg(carpeta / archivo, orientacion=CON_ORIENTACION.get(archivo))
    return carpeta


def calcular(registro, canecas, carpeta, tabla=None, modelo=URL) -> dict:
    return metricas_campo.calcular(
        tabla=tabla_armada(registro, canecas) if tabla is None else tabla,
        entorno={"modelo": modelo, "base_imagenes": "proyecto-final/datos/campo"},
        registro=registro,
        canecas=canecas,
        modelos={"elegido": "entrenamiento_2", "entrenamiento_2": {"url": URL}},
        prueba=metricas.leer_csv(rutas.RESULTADOS / metricas.CSV_ELEGIDO),
        elegido=cargar_json(rutas.RESULTADOS / "metricas-elegido.json"),
        carpeta=carpeta,
    )


@pytest.fixture(scope="module")
def armado(registro, canecas, carpeta_armada) -> dict:
    return calcular(registro, canecas, carpeta_armada)


@requiere_registro_campo
def test_agregacion_por_caneca_con_el_caso_armado(armado):
    caneca = armado["caneca"]
    assert caneca["clases"] == ["blanca", "verde", "negra"]
    assert caneca["matriz"] == [[12, 0, 0], [0, 3, 1], [1, 0, 1]]
    assert caneca["valores"]["exactitud"] == pytest.approx(16 / 18)
    assert caneca["valores"]["precision.blanca"] == pytest.approx(12 / 13)
    assert caneca["valores"]["recall.verde"] == pytest.approx(3 / 4)
    assert armado["errores"]["contaminan_la_blanca"] == 1
    assert armado["errores"]["material_con_caneca_correcta"] == 2
    assert np.trace(np.array(armado["material"]["matriz"])) == 14
    assert armado["material"]["valores"]["exactitud_balanceada"] == pytest.approx(7 / 9)
    assert "auc_macro" not in armado["material"]["valores"]


@requiere_registro_campo
def test_la_comparacion_de_material_es_contra_la_exactitud_balanceada_del_30(armado):
    elegido = cargar_json(rutas.RESULTADOS / "metricas-elegido.json")
    material = armado["comparacion_con_prueba"]["material_exactitud_balanceada"]
    balanceada, cruda = elegido["material"]["valores"]["exactitud_balanceada"], elegido["material"]["valores"]["exactitud"]
    assert balanceada != cruda  # el 30 % está desbalanceado: la diferencia importa
    assert material["prueba"] == balanceada
    assert material["campo"] == pytest.approx(7 / 9)
    assert material["diferencia"] == pytest.approx(7 / 9 - balanceada)
    assert material["intervalo"][0] <= material["diferencia"] <= material["intervalo"][1]
    caneca = armado["comparacion_con_prueba"]["caneca_exactitud"]
    assert caneca["prueba"] == elegido["caneca"]["valores"]["exactitud"]
    assert caneca["diferencia"] == pytest.approx(16 / 18 - caneca["prueba"])
    assert "balanceada" in armado["comparacion_con_prueba"]["nota"]


@requiere_registro_campo
def test_umbral_sustitutos_y_orientacion(armado):
    elegido = cargar_json(rutas.RESULTADOS / "metricas-elegido.json")
    umbral = armado["umbral_blanca"]
    assert umbral["umbral"] == 0.9
    assert umbral["campo"]["valores"]["precision"] == pytest.approx(12 / 13)
    assert umbral["prueba"] == next(p for p in elegido["cobertura_blanca"]["puntos"] if p["umbral"] == 0.9)
    assert armado["con_sustituto"]["fotos"] == 1
    assert armado["orientacion_exif"]["fotos_con_etiqueta_distinta_de_1"] == 2
    assert armado["orientacion_exif"]["fotos_que_giran"] == 2
    assert armado["bootstrap"] == metricas.BOOTSTRAP


@requiere_registro_campo
def test_se_niega_con_otro_modelo_o_si_falta_una_foto(registro, canecas, carpeta_armada):
    with pytest.raises(ValueError, match="verificación"):
        calcular(registro, canecas, carpeta_armada, modelo="https://teachablemachine.withgoogle.com/models/otro/")
    incompleta = tabla_armada(registro, canecas).iloc[1:]
    with pytest.raises(ValueError, match="verificación"):
        calcular(registro, canecas, carpeta_armada, tabla=incompleta)


# Cifras del informe


@requiere_paridad
@requiere_informe
def test_cifras_no_cambia_nada_sin_datos_de_campo(tmp_path):
    ausente = tmp_path / "metricas-campo.json"
    assert cifras.comprobar(cifras.DESTINO.parent, tmp_path / "generado", ausente) == []
    assert len(cifras.archivos_generados(tmp_path / "generado")) == 9
    assert not any(m.nombre.startswith(cifras.PREFIJO_CAMPO) for m in cifras.todas_las_macros(ausente))


@requiere_registro_campo
@requiere_paridad
@requiere_informe
def test_cifras_con_datos_de_campo_suma_macros_y_la_fila_de_robustez(tmp_path, armado):
    datos = tmp_path / "metricas-campo.json"
    datos.write_text(json.dumps(armado, ensure_ascii=False), encoding="utf-8")
    salida = tmp_path / "generado"
    cifras.escribir_todo(salida, datos)
    actual = cifras.DESTINO.parent
    for tabla in sorted((actual / cifras.CARPETA_TABLAS).glob("*.tex")):
        nueva = (salida / cifras.CARPETA_TABLAS / tabla.name).read_text(encoding="utf-8")
        if tabla.name == "robustez.tex":
            assert "Campo$^{a}$ & Fotos propias & 0,778 & " in nueva
        else:
            assert nueva == tabla.read_text(encoding="utf-8"), tabla.name
    lineas = (salida / cifras.DESTINO.name).read_text(encoding="utf-8").splitlines()
    de_campo = [linea for linea in lineas if linea.startswith("\\newcommand{\\Campo")]
    assert [linea for linea in lineas if linea not in de_campo] == (actual / cifras.DESTINO.name).read_text(encoding="utf-8").splitlines()
    assert "\\newcommand{\\CampoExactitudBalanceada}{0,778}" in "\n".join(de_campo)
    for macro in cifras.todas_las_macros(datos):
        if not macro.nombre.startswith(cifras.PREFIJO_CAMPO):
            continue
        archivo, _, ruta = macro.origen.partition(": ")
        assert archivo == "metricas-campo.json", macro
        if cifras.SEPARADOR in ruta and "(" not in ruta:  # la clave existe, como exige test_metricas
            valor = armado
            for clave in ruta.split(cifras.SEPARADOR):
                valor = valor[int(clave)] if isinstance(valor, list) else valor[clave]


# Orientación EXIF en el navegador

SONDA_JS = """
async (url) => {
  const { cargarImagen } = await import(new URL('inferencia.js', location.href).href);
  const img = await cargarImagen(url);
  const lienzo = document.createElement('canvas');
  lienzo.width = img.width; lienzo.height = img.height;
  const ctx = lienzo.getContext('2d');
  ctx.drawImage(img, 0, 0, img.width, img.height);
  const px = (x, y) => Array.from(ctx.getImageData(x, y, 1, 1).data.slice(0, 3));
  const [w, h] = [img.width, img.height];
  return [w, h, px(2, 2), px(w - 3, 2), px(2, h - 3), px(w - 3, h - 3)];
}
"""
COLORES = {"rojo": (255, 0, 0), "verde": (0, 255, 0), "azul": (0, 0, 255), "amarillo": (255, 255, 0)}


def cuadrantes_con_orientacion(orientacion: int) -> bytes:
    """JPEG de 64 x 32 con un color por cuadrante y la etiqueta de orientación dada."""
    imagen = Image.new("RGB", (64, 32), COLORES["rojo"])
    imagen.paste(COLORES["verde"], (32, 0, 64, 16))
    imagen.paste(COLORES["azul"], (0, 16, 32, 32))
    imagen.paste(COLORES["amarillo"], (32, 16, 64, 32))
    exif = Image.Exif()
    exif[ETIQUETA_ORIENTACION_EXIF] = orientacion
    memoria = io.BytesIO()
    imagen.save(memoria, format="JPEG", quality=95, exif=exif.tobytes())
    return memoria.getvalue()


def color(rgb) -> str:
    return min(COLORES, key=lambda c: sum((a - b) ** 2 for a, b in zip(COLORES[c], rgb, strict=True)))


def como_la_ve_pillow(contenido: bytes) -> list:
    with Image.open(io.BytesIO(contenido)) as imagen:
        derecha = ImageOps.exif_transpose(imagen)
    w, h = derecha.size
    return [w, h, *(color(derecha.getpixel(p)) for p in ((2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)))]


@pytest.mark.navegador
@pytest.mark.skipif(not cdp_responde(), reason="la instancia de Brave no responde en el puerto 9333")
def test_el_navegador_aplica_la_orientacion_exif_al_decodificar():
    """Mismo camino que el arnés: `cargarImagen` de inferencia.js y `drawImage`, como `cropTo` de la librería."""
    from residuos.arnes import pagina_del_arnes  # abre el servidor y una pestaña de la instancia

    with pagina_del_arnes() as (pagina, _):
        for orientacion in range(1, 9):
            contenido = cuadrantes_con_orientacion(orientacion)
            w, h, *esquinas = pagina.evaluate(SONDA_JS, "data:image/jpeg;base64," + base64.b64encode(contenido).decode())
            assert [w, h, *(color(e) for e in esquinas)] == como_la_ve_pillow(contenido), orientacion
