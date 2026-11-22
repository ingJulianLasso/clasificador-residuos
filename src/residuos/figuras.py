"""Figuras del informe, generadas desde la partición, los CSV del arnés y los JSON de métricas.

Uso: ``uv run python -m residuos.figuras``. Escribe en ``proyecto-final/figuras/`` las gráficas en PDF
vectorial y las rejillas de fotos en JPEG de calidad 90, más ``manifiesto.json`` con cada figura, su
archivo, la sección del informe que la usa y el ancho para el que se diseñó: ``media`` (dos por fila,
3,15 pulgadas) o ``completa`` (6,5 pulgadas). Estilo sobrio: grises con un solo acento azul oscuro.
"""

from __future__ import annotations

import json
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
from PIL import Image
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from residuos import rutas
from residuos.datos import ENTRENAMIENTO, PRUEBA, leer_particion
from residuos.metricas import CSV_ELEGIDO, leer_csv, predicciones_material
from residuos.robustez import PERTURBACIONES

ACENTO = "#1F3A5F"
GRIS_OSCURO, GRIS_MEDIO, GRIS_CLARO, GRIS_TENUE = "#2B2B2B", "#6E6E6E", "#B4B4B4", "#E3E3E3"
MAPA = LinearSegmentedColormap.from_list("blanco_acento", ["#FFFFFF", ACENTO])
MEDIA, COMPLETA = 3.15, 6.5
SEMILLA_MUESTRAS = 42
MUESTRAS_POR_CLASE = 3
COLUMNAS_ERRORES = 6
ANCHO_ROTULO_ERRORES = 20
CALIDAD_JPEG = 90
PPP_FOTOS = 250
UMBRALES_MARCADOS = (0.5, 0.7, 0.9, 0.99)

ESTILO = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "axes.edgecolor": GRIS_OSCURO,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "grid.color": GRIS_TENUE,
    "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


@dataclass(frozen=True)
class Figura:
    id: str
    archivo: str
    seccion: str
    ancho: str
    descripcion: str
    fuentes: tuple[str, ...]
    dibujar: Callable[[], plt.Figure]


# Utilidades


def coma(decimales: int) -> FuncFormatter:
    return FuncFormatter(lambda valor, _: f"{valor:.{decimales}f}".replace(".", ","))


def numero(valor: float, decimales: int = 3) -> str:
    return f"{valor:.{decimales}f}".replace(".", ",")


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def canecas() -> dict:
    return cargar_json(rutas.CANECAS)


def nombre(etiqueta: str) -> str:
    return canecas()["materiales"][etiqueta]["nombre"]


def nombre_caneca(caneca: str) -> str:
    return caneca.capitalize()


def en_dos_lineas(texto: str, ancho: int = 11) -> str:
    return "\n".join(textwrap.wrap(texto, ancho))


def metricas(archivo: str) -> dict:
    return cargar_json(rutas.RESULTADOS / archivo)


def elegido() -> pd.DataFrame:
    return leer_csv(rutas.RESULTADOS / CSV_ELEGIDO)


def guardar(figura: plt.Figure, destino: Path) -> None:
    if destino.suffix == ".jpg":
        figura.savefig(destino, dpi=PPP_FOTOS, pil_kwargs={"quality": CALIDAD_JPEG})
    else:
        figura.savefig(destino)
    plt.close(figura)


# Dataset


def muestras_por_clase() -> plt.Figure:
    particion = leer_particion()
    generador = np.random.default_rng(SEMILLA_MUESTRAS)
    clases = sorted(particion["clase"].unique())
    figura, ejes = plt.subplots(MUESTRAS_POR_CLASE, len(clases), figsize=(COMPLETA, 2.4))
    for columna, clase in enumerate(clases):
        candidatas = particion.loc[(particion["clase"] == clase) & (particion["particion"] == ENTRENAMIENTO), "archivo"].to_numpy()
        elegidas = generador.choice(candidatas, size=MUESTRAS_POR_CLASE, replace=False)
        for fila, archivo in enumerate(elegidas):
            eje = ejes[fila, columna]
            with Image.open(rutas.IMAGENES / archivo) as imagen:
                eje.imshow(imagen.convert("RGB"))
            eje.set_axis_off()
        caneca = canecas()["materiales"][clase]["caneca"]
        ejes[0, columna].set_title(f"{en_dos_lineas(nombre(clase))}\n({caneca})", fontsize=6.5, linespacing=1.1)
    figura.subplots_adjust(wspace=0.04, hspace=0.03)
    return figura


def distribucion_particion() -> plt.Figure:
    particion = leer_particion()
    conteos = particion.groupby(["clase", "particion"]).size().unstack(fill_value=0)
    conteos = conteos.assign(total=conteos.sum(axis=1)).sort_values("total")
    balanceado = int(particion.query("balanceado == 1").groupby("clase").size().min())
    prueba = round(cargar_json(rutas.INVENTARIO)["particion"]["fraccion_prueba"] * 100)
    figura, eje = plt.subplots(figsize=(MEDIA, 2.45))
    posiciones = np.arange(len(conteos))
    eje.barh(posiciones, conteos[ENTRENAMIENTO], color=GRIS_MEDIO, height=0.62, label=f"Entrenamiento ({100 - prueba} %)")
    eje.barh(posiciones, conteos[PRUEBA], left=conteos[ENTRENAMIENTO], color=ACENTO, height=0.62, label=f"Prueba ({prueba} %)")
    eje.axvline(balanceado, color=GRIS_OSCURO, linestyle=(0, (3, 2)), linewidth=0.8)
    eje.text(balanceado + 8, len(conteos) - 0.35, f"Balanceado: {balanceado} por clase", fontsize=6.5, va="center")
    eje.set_yticks(posiciones, [nombre(c) for c in conteos.index])
    eje.set_xlabel("Imágenes")
    eje.xaxis.grid(True)
    eje.set_axisbelow(True)
    eje.legend(loc="lower right")
    return figura


# Matrices de confusión


def dibujar_matriz(matriz: np.ndarray, etiquetas: list[str], normalizada: bool, alto: float) -> plt.Figure:
    proporcion = matriz / matriz.sum(axis=1, keepdims=True)
    figura, eje = plt.subplots(figsize=(MEDIA, alto))
    eje.imshow(proporcion, cmap=MAPA, vmin=0, vmax=1)
    tamano = 6 if len(etiquetas) > 3 else 7.5
    for (fila, columna), valor in np.ndenumerate(matriz):
        texto = numero(proporcion[fila, columna], 2) if normalizada else str(valor)
        color = "white" if proporcion[fila, columna] > 0.55 else GRIS_OSCURO
        eje.text(columna, fila, texto, ha="center", va="center", fontsize=tamano, color=color)
    marcas = range(len(etiquetas))
    eje.set_xticks(marcas, etiquetas, rotation=45 if len(etiquetas) > 3 else 0, ha="right" if len(etiquetas) > 3 else "center")
    eje.set_yticks(marcas, etiquetas)
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Clase real")
    eje.tick_params(length=0)
    for borde in eje.spines.values():
        borde.set_visible(False)
    return figura


def matriz_materiales(normalizada: bool) -> Callable[[], plt.Figure]:
    def dibujar() -> plt.Figure:
        tarea = metricas("metricas-elegido.json")["material"]
        etiquetas = [en_dos_lineas(nombre(c), 10) for c in tarea["clases"]]
        return dibujar_matriz(np.array(tarea["matriz"]), etiquetas, normalizada, alto=3.15)

    return dibujar


def matriz_canecas(normalizada: bool) -> Callable[[], plt.Figure]:
    def dibujar() -> plt.Figure:
        tarea = metricas("metricas-elegido.json")["caneca"]
        etiquetas = [nombre_caneca(c) for c in tarea["clases"]]
        return dibujar_matriz(np.array(tarea["matriz"]), etiquetas, normalizada, alto=2.0)

    return dibujar


# Curvas ROC y de precisión y recall


def curvas_por_clase(curva: Callable, real: np.ndarray, probabilidades: np.ndarray) -> tuple[list, tuple]:
    binaria = label_binarize(real, classes=range(probabilidades.shape[1]))
    por_clase = [curva(binaria[:, k], probabilidades[:, k]) for k in range(probabilidades.shape[1])]
    micro = curva(binaria.ravel(), probabilidades.ravel())
    return por_clase, micro


def ejes_de_curva(eje: plt.Axes, etiqueta_x: str, etiqueta_y: str) -> None:
    eje.set_xlim(0, 1)
    eje.set_ylim(0, 1.01)
    eje.set_xlabel(etiqueta_x)
    eje.set_ylabel(etiqueta_y)
    eje.xaxis.set_major_formatter(coma(1))
    eje.yaxis.set_major_formatter(coma(1))
    eje.grid(True)
    eje.set_axisbelow(True)
    eje.set_aspect("equal")


def dibujar_curvas(clave: str, por_clase: list, micro: tuple, ejes_xy: Callable, area: dict, titulo: str) -> tuple[plt.Figure, plt.Axes]:
    clases = area["clases"]
    valores = {c: area["valores"][f"{clave}.{c}"] for c in clases}
    peor = min(valores, key=valores.get)
    figura, eje = plt.subplots(figsize=(MEDIA, MEDIA))
    for k, clase in enumerate(clases):
        if clase != peor:
            eje.plot(*ejes_xy(por_clase[k]), color=GRIS_CLARO, linewidth=0.8)
    eje.plot([], [], color=GRIS_CLARO, linewidth=0.8, label="Demás materiales")
    k_peor = clases.index(peor)
    eje.plot(*ejes_xy(por_clase[k_peor]), color=GRIS_OSCURO, linewidth=1, linestyle=(0, (4, 2)),
             label=f"Menor: {nombre(peor)} ({titulo} {numero(valores[peor])})")  # fmt: skip
    eje.plot(*ejes_xy(micro), color=ACENTO, linewidth=1.4, label=f"Micropromedio ({titulo} {numero(area['valores'][f'{clave}_micro'])})")
    return figura, eje


def roc() -> plt.Figure:
    tabla = elegido()
    p = predicciones_material(tabla)
    por_clase, micro = curvas_por_clase(roc_curve, p.real, p.probabilidades)
    area = metricas("metricas-elegido.json")["material"]
    figura, eje = dibujar_curvas("auc", por_clase, micro, lambda c: (c[0], c[1]), area, "AUC")
    eje.plot([0, 1], [0, 1], color=GRIS_MEDIO, linewidth=0.6, linestyle=":")
    ejes_de_curva(eje, "Tasa de falsos positivos", "Tasa de verdaderos positivos")
    eje.legend(loc="lower right")
    return figura


def precision_recall() -> plt.Figure:
    tabla = elegido()
    p = predicciones_material(tabla)
    por_clase, micro = curvas_por_clase(precision_recall_curve, p.real, p.probabilidades)
    area = metricas("metricas-elegido.json")["material"]
    figura, eje = dibujar_curvas("ap", por_clase, micro, lambda c: (c[1], c[0]), area, "AP")
    ejes_de_curva(eje, "Recall", "Precisión")
    eje.legend(loc="lower left")
    return figura


# Comparación de los dos entrenamientos


def antes_despues() -> plt.Figure:
    uno, dos = metricas("metricas-entrenamiento-1.json"), metricas("metricas-entrenamiento-2.json")
    modelos = cargar_json(rutas.MODELOS_JSON)
    clases = uno["material"]["clases"]
    filas = [(nombre(c), uno["material"]["valores"][f"recall.{c}"], dos["material"]["valores"][f"recall.{c}"]) for c in clases]
    filas.sort(key=lambda fila: fila[2])
    resumen = [
        ("Exactitud balanceada\n(prueba)", uno["material"]["valores"]["exactitud_balanceada"], dos["material"]["valores"]["exactitud_balanceada"]),
        ("Media por clase\n(validación)", *(modelos[f"entrenamiento_{n}"]["exactitud_media_validacion"] for n in (1, 2))),
    ]
    # Las etiquetas del resumen van en dos líneas para que el panel quepa a media página sin reducirse;
    # por eso sus filas se separan un poco más que las de los materiales.
    separador = len(filas)
    posiciones = [*range(separador), *(separador + 0.9 + 1.5 * i for i in range(len(resumen)))]
    figura, eje = plt.subplots(figsize=(MEDIA, 2.58), layout="constrained")
    eje.axhline(separador, color=GRIS_CLARO, linewidth=0.6)
    for posicion, (_, valor_1, valor_2) in zip(posiciones, filas + resumen, strict=True):
        eje.plot([valor_1, valor_2], [posicion, posicion], color=GRIS_CLARO, linewidth=1.2, zorder=1)
        eje.scatter(valor_1, posicion, s=16, color=GRIS_MEDIO, zorder=2, label="Entrenamiento 1 (completo)" if posicion == 0 else None)
        eje.scatter(valor_2, posicion, s=18, marker="D", color=ACENTO, zorder=3,
                    label="Entrenamiento 2 (balanceado)" if posicion == 0 else None)  # fmt: skip
    eje.set_yticks(posiciones, [f[0] for f in filas + resumen], fontsize=7, linespacing=1.0)
    eje.set_xlabel("Recall por material (exactitud por clase)")
    eje.xaxis.set_major_formatter(coma(2))
    eje.xaxis.grid(True)
    eje.set_axisbelow(True)
    eje.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=1, handletextpad=0.3)
    return figura


# Robustez


def robustez() -> plt.Figure:
    datos = metricas("metricas-robustez.json")
    filas = [("Original", datos["original"])] + [(p.nombre.capitalize(), datos["perturbaciones"][p.nombre]) for p in PERTURBACIONES]
    figura, eje = plt.subplots(figsize=(MEDIA, 2.45))
    posiciones = np.arange(len(filas))[::-1]
    for tarea, color, marcador, desplazamiento, etiqueta in (
        ("material", GRIS_MEDIO, "o", 0.13, "Material"),
        ("caneca", ACENTO, "s", -0.13, "Caneca"),
    ):
        valores = np.array([f[1][tarea]["valores"]["exactitud"] for f in filas])
        inferior = valores - np.array([f[1][tarea]["intervalos"]["exactitud"][0] for f in filas])
        superior = np.array([f[1][tarea]["intervalos"]["exactitud"][1] for f in filas]) - valores
        eje.errorbar(valores, posiciones + desplazamiento, xerr=[inferior, superior], fmt=marcador, color=color,
                     markersize=3.5, elinewidth=0.8, capsize=1.5, label=etiqueta)  # fmt: skip
        eje.axvline(valores[0], color=color, linewidth=0.6, linestyle=":")
    eje.set_yticks(posiciones, [f[0] for f in filas])
    eje.set_xlabel("Exactitud en la prueba (intervalo de 95 %)")
    eje.xaxis.set_major_formatter(coma(2))
    eje.xaxis.grid(True)
    eje.set_axisbelow(True)
    eje.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
    return figura


# Errores y abstención


def errores_confiados() -> plt.Figure:
    errores = metricas("metricas-elegido.json")["errores_mas_confiados"]
    filas = int(np.ceil(len(errores) / COLUMNAS_ERRORES))
    figura, ejes = plt.subplots(filas, COLUMNAS_ERRORES, figsize=(COMPLETA, 1.5 * filas))
    for eje, error in zip(ejes.ravel(), errores, strict=False):
        with Image.open(rutas.IMAGENES / error["archivo"]) as imagen:
            eje.imshow(imagen.convert("RGB"))
        eje.set_xticks([])
        eje.set_yticks([])
        cambia_caneca = error["caneca_real"] != error["caneca_predicha"]
        caneca = f"{error['caneca_real']} a {error['caneca_predicha']}" if cambia_caneca else f"{error['caneca_real']}, igual"
        real = en_dos_lineas(f"Real: {nombre(error['real'])}", ANCHO_ROTULO_ERRORES)
        predicho = en_dos_lineas(f"Predicho: {nombre(error['predicho'])}", ANCHO_ROTULO_ERRORES)
        eje.set_xlabel(
            f"{real}\n{predicho}\nProbabilidad: {numero(error['confianza'], 4)}\nCaneca: {caneca}",
            fontsize=5.8, linespacing=1.1, labelpad=2, color=ACENTO if cambia_caneca else GRIS_OSCURO,
        )  # fmt: skip
        for borde in eje.spines.values():
            borde.set_color(ACENTO if cambia_caneca else GRIS_CLARO)
            borde.set_linewidth(1.4 if cambia_caneca else 0.5)
            borde.set_visible(True)
    for eje in ejes.ravel()[len(errores):]:
        eje.set_axis_off()
    figura.subplots_adjust(wspace=0.25, hspace=0.3)
    return figura


def cobertura_blanca() -> plt.Figure:
    curva = metricas("metricas-elegido.json")["cobertura_blanca"]
    puntos = pd.DataFrame(curva["puntos"])
    figura, eje = plt.subplots(figsize=(MEDIA, 2.45))
    eje.plot(puntos["umbral"], puntos["precision"], color=ACENTO, linewidth=1.4, label="Precisión de la blanca")
    eje.plot(puntos["umbral"], puntos["cobertura"], color=GRIS_MEDIO, linewidth=1.2, linestyle=(0, (4, 2)), label="Cobertura")
    marcados = puntos[puntos["umbral"].isin(UMBRALES_MARCADOS)]
    eje.scatter(marcados["umbral"], marcados["precision"], s=12, color=ACENTO, zorder=3)
    eje.scatter(marcados["umbral"], marcados["cobertura"], s=12, color=GRIS_MEDIO, zorder=3)
    eje.set_xlabel("Umbral de probabilidad de la caneca blanca")
    eje.set_ylabel("Proporción")
    eje.xaxis.set_major_formatter(coma(1))
    eje.yaxis.set_major_formatter(coma(2))
    eje.grid(True)
    eje.set_axisbelow(True)
    eje.legend(loc="lower left")
    return figura


# Catálogo y escritura

DATOS = "clasificador/datos/particion.csv"
ELEGIDO = "clasificador/resultados/metricas-elegido.json"

FIGURAS = (
    Figura("muestras-por-clase", "muestras-por-clase.jpg", "03-dataset", "completa",
           "Tres fotos de entrenamiento por material, elegidas al azar con semilla fija, con su caneca", (DATOS,), muestras_por_clase),
    Figura("distribucion-particion", "distribucion-particion.pdf", "03-dataset", "media",
           "Imágenes por material en entrenamiento y prueba, con el tamaño del conjunto balanceado", (DATOS,), distribucion_particion),
    Figura("matriz-materiales-conteos", "matriz-materiales-conteos.pdf", "05-resultados", "media",
           "Matriz de confusión de los 9 materiales del modelo elegido sobre la prueba, en conteos (el tono es la proporción por fila)",
           (ELEGIDO,), matriz_materiales(normalizada=False)),
    Figura("matriz-materiales-normalizada", "matriz-materiales-normalizada.pdf", "05-resultados", "media",
           "Matriz de confusión de los 9 materiales normalizada por fila (recall en la diagonal)", (ELEGIDO,), matriz_materiales(normalizada=True)),
    Figura("matriz-canecas-conteos", "matriz-canecas-conteos.pdf", "05-resultados", "media",
           "Matriz de confusión de las 3 canecas en conteos", (ELEGIDO,), matriz_canecas(normalizada=False)),
    Figura("matriz-canecas-normalizada", "matriz-canecas-normalizada.pdf", "05-resultados", "media",
           "Matriz de confusión de las 3 canecas normalizada por fila", (ELEGIDO,), matriz_canecas(normalizada=True)),
    Figura("roc", "roc.pdf", "05-resultados", "media",
           "Curvas ROC una contra el resto por material, con el micropromedio y el material de menor AUC",
           ("clasificador/resultados/probabilidades-prueba-elegido.csv", ELEGIDO), roc),
    Figura("precision-recall", "precision-recall.pdf", "05-resultados", "media",
           "Curvas de precisión y recall por material, con el micropromedio y el material de menor AP",
           ("clasificador/resultados/probabilidades-prueba-elegido.csv", ELEGIDO), precision_recall),
    Figura("antes-despues", "antes-despues.pdf", "05-resultados", "media",
           "Recall por material de los dos entrenamientos sobre la prueba, con la exactitud balanceada de la prueba y la media por clase de la validación interna",
           ("clasificador/resultados/metricas-entrenamiento-1.json", "clasificador/resultados/metricas-entrenamiento-2.json",
            "clasificador/resultados/modelos.json"), antes_despues),
    Figura("robustez", "robustez.pdf", "05-resultados", "media",
           "Exactitud de material y de caneca del modelo elegido en la prueba original y en cada perturbación, con su intervalo de 95 %",
           ("clasificador/resultados/metricas-robustez.json",), robustez),
    Figura("cobertura-blanca", "cobertura-blanca.pdf", "05-resultados", "media",
           "Precisión y cobertura de la caneca blanca según el umbral de probabilidad por debajo del cual el sistema se abstiene",
           (ELEGIDO,), cobertura_blanca),
    Figura("errores-confiados", "errores-confiados.jpg", "07-discusion", "completa",
           "Los 12 errores de material con mayor confianza del modelo elegido, con su efecto en la caneca",
           (ELEGIDO,), errores_confiados),
)  # fmt: skip


def escribir_manifiesto(destino: Path) -> Path:
    manifiesto = {
        "descripcion": "Figuras del informe de la Actividad 5; las genera residuos.figuras",
        "anchos": {"media": f"{MEDIA} pulgadas, dos por fila", "completa": f"{COMPLETA} pulgadas"},
        "figuras": [
            {"id": f.id, "archivo": f.archivo, "seccion": f.seccion, "ancho": f.ancho, "descripcion": f.descripcion, "fuentes": list(f.fuentes)}
            for f in FIGURAS
        ],
    }
    ruta = destino / "manifiesto.json"
    ruta.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ruta


def main() -> None:
    plt.rcParams.update(ESTILO)
    rutas.FIGURAS.mkdir(parents=True, exist_ok=True)
    for figura in FIGURAS:
        guardar(figura.dibujar(), rutas.FIGURAS / figura.archivo)
        print(f"{figura.archivo}")
    print(f"Manifiesto: {escribir_manifiesto(rutas.FIGURAS)}")


if __name__ == "__main__":
    main()
