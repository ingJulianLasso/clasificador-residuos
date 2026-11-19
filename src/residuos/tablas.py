"""Tablas del informe en LaTeX (solo el ``tabular``, con booktabs), generadas desde los datos.

El informe las incluye con ``\\input{generado/tablas/<nombre>}`` dentro de su propio entorno ``table``,
que pone el título, la etiqueta y la nota. Las escribe ``residuos.cifras``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from residuos.datos import ENTRENAMIENTO, PRUEBA
from residuos.formato import (
    con_intervalo,
    decimal,
    entero,
    metrica,
    p_valor,
    texto_latex,
)
from residuos.robustez import PERTURBACIONES, Perturbacion


@dataclass(frozen=True)
class Insumos:
    particion: pd.DataFrame
    canecas: dict
    entrenamientos: dict[int, dict]  # tm-entrenamiento-N.json
    modelos: dict  # modelos.json
    metricas: dict[str, dict]  # "1", "2", "elegido", "robustez"
    campo: dict | None = None  # metricas-campo.json, si ya existe


def nombre(insumos: Insumos, etiqueta: str) -> str:
    return insumos.canecas["materiales"][etiqueta]["nombre"]


def caneca(insumos: Insumos, etiqueta: str) -> str:
    return insumos.canecas["materiales"][etiqueta]["caneca"].capitalize()


def tabular(columnas: str, encabezado: list[str], filas: list[list[str] | str], fuente: str) -> str:
    """`filas` admite cadenas sueltas para líneas como \\midrule."""
    lineas = [
        "% Generado por residuos.cifras: no se edita a mano.",
        f"% Fuente: {fuente}.",
        f"\\begin{{tabular}}{{{columnas}}}",
        "\\toprule",
        " & ".join(encabezado) + " \\\\",
        "\\midrule",
    ]
    lineas += [fila if isinstance(fila, str) else " & ".join(fila) + " \\\\" for fila in filas]
    lineas += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(lineas) + "\n"


# Dataset y configuración


def dataset_particion(insumos: Insumos) -> str:
    p = insumos.particion
    filas = []
    for etiqueta in insumos.canecas["materiales"]:
        clase = p[p["clase"] == etiqueta]
        filas.append([
            nombre(insumos, etiqueta), caneca(insumos, etiqueta), entero(len(clase)),
            entero((clase["particion"] == ENTRENAMIENTO).sum()), entero((clase["particion"] == PRUEBA).sum()),
            entero(clase["balanceado"].sum()),
        ])  # fmt: skip
    total = ["Total", "", entero(len(p)), entero((p["particion"] == ENTRENAMIENTO).sum()),
             entero((p["particion"] == PRUEBA).sum()), entero(p["balanceado"].sum())]  # fmt: skip
    return tabular(
        "@{}llrrrr@{}",
        ["Material", "Caneca", "Total", "Entrenamiento", "Prueba", "Balanceado"],
        [*filas, "\\midrule", total],
        "particion.csv y canecas.json",
    )


def canecas(insumos: Insumos) -> str:
    filas = [
        [texto_latex(etiqueta), nombre(insumos, etiqueta), caneca(insumos, etiqueta), texto_latex(datos["fuente"])]
        for etiqueta, datos in insumos.canecas["materiales"].items()
    ]
    return tabular("@{}lllL{7cm}@{}", ["Clase de RealWaste", "Material", "Caneca", "Fuente"], filas, "canecas.json")


def entrenamientos(insumos: Insumos) -> str:
    tm, modelos = insumos.entrenamientos, insumos.modelos

    def por_modelo(valor: Callable[[int], str]) -> list[str]:
        return [valor(n) for n in (1, 2)]

    def rango(n: int) -> str:
        conteos = tm[n]["conteos"].values()
        return entero(min(conteos)) if min(conteos) == max(conteos) else f"{entero(min(conteos))} a {entero(max(conteos))}"

    filas = [
        ["Conjunto de entrenamiento", *por_modelo(lambda n: texto_latex(tm[n]["descripcion"].capitalize()))],
        ["Imágenes cargadas", *por_modelo(lambda n: entero(sum(tm[n]["conteos"].values())))],
        ["Imágenes por clase", *por_modelo(rango)],
        ["Validación interna (imágenes)", *por_modelo(lambda n: entero(tm[n]["resumen_validacion"]["imagenes_validacion"]))],
        ["Épocas", *por_modelo(lambda n: entero(tm[n]["parametros"]["epocas"]))],
        ["Tamaño del lote", *por_modelo(lambda n: entero(tm[n]["parametros"]["lote"]))],
        ["Tasa de aprendizaje", *por_modelo(lambda n: decimal(tm[n]["parametros"]["tasa_aprendizaje"], 3))],
        ["Exactitud final de entrenamiento", *por_modelo(lambda n: metrica(tm[n]["curvas"]["entrenamiento"]["exactitud"][-1]))],
        ["Exactitud final de validación", *por_modelo(lambda n: metrica(tm[n]["curvas"]["validacion"]["exactitud"][-1]))],
        ["Exactitud media por clase (validación)", *por_modelo(lambda n: decimal(modelos[f"entrenamiento_{n}"]["exactitud_media_validacion"], 2))],
        ["Elegido por la regla", *por_modelo(lambda n: "Sí" if modelos["elegido"] == f"entrenamiento_{n}" else "No")],
    ]
    return tabular(
        "@{}L{6.2cm}L{3.6cm}L{4.4cm}@{}",
        ["Concepto", "Entrenamiento 1", "Entrenamiento 2"],
        filas,
        "tm-entrenamiento-1.json, tm-entrenamiento-2.json y modelos.json",
    )


# Resultados


def antes_despues(insumos: Insumos) -> str:
    uno, dos, elegido = insumos.metricas["1"], insumos.metricas["2"], insumos.metricas["elegido"]
    comparacion = elegido["comparacion_entrenamientos"]

    def fila(etiqueta: str, tarea: str, clave: str, diferencia: str | None = None) -> list[str]:
        valores = [m[tarea]["valores"][clave] for m in (uno, dos)]
        intervalos = [m[tarea]["intervalos"][clave] for m in (uno, dos)]
        celda_diferencia = diferencia if diferencia is not None else metrica(valores[1] - valores[0])
        return [etiqueta, *(con_intervalo(v, i) for v, i in zip(valores, intervalos, strict=True)), celda_diferencia]

    def diferencia_pareada(tarea: str) -> str:
        datos = comparacion[tarea]["diferencia_exactitud"]
        return con_intervalo(datos["valor"], datos["intervalo"])

    def mcnemar(tarea: str, etiqueta: str) -> str:
        prueba = comparacion[tarea]["mcnemar"]
        texto = f"$b = {prueba['b']}$, $c = {prueba['c']}$, {p_valor(prueba['p_valor'])}"
        return f"\\multicolumn{{4}}{{@{{}}l@{{}}}}{{McNemar exacta, {etiqueta}: {texto}}} \\\\"

    filas = [
        "\\multicolumn{4}{@{}l@{}}{\\textit{Material}} \\\\",
        fila("Exactitud", "material", "exactitud", diferencia_pareada("material")),
        fila("Exactitud balanceada", "material", "exactitud_balanceada"),
        fila("F1 macro", "material", "f1_macro"),
        fila("F1 ponderado", "material", "f1_ponderado"),
        fila("AUC macro", "material", "auc_macro"),
        fila("AP macro", "material", "ap_macro"),
        mcnemar("material", "material"),
        "\\midrule",
        "\\multicolumn{4}{@{}l@{}}{\\textit{Caneca}} \\\\",
        fila("Exactitud", "caneca", "exactitud", diferencia_pareada("caneca")),
        fila("Exactitud balanceada", "caneca", "exactitud_balanceada"),
        fila("F1 macro", "caneca", "f1_macro"),
        fila("Precisión de la blanca", "caneca", "precision.blanca"),
        fila("Recall de la verde", "caneca", "recall.verde"),
        mcnemar("caneca", "caneca"),
    ]
    return tabular(
        "@{}llll@{}",
        ["Métrica", "Entrenamiento 1", "Entrenamiento 2", "Diferencia (2 menos 1)"],
        filas,
        "metricas-entrenamiento-1.json, metricas-entrenamiento-2.json y metricas-elegido.json",
    )


def por_clase(insumos: Insumos) -> str:
    material = insumos.metricas["elegido"]["material"]
    valores, intervalos = material["valores"], material["intervalos"]

    def celdas(sufijo: str) -> list[str]:
        return [con_intervalo(valores[f"{m}{sufijo}"], intervalos[f"{m}{sufijo}"]) for m in ("precision", "recall", "f1")]

    filas = [[nombre(insumos, c), entero(material["soporte"][c]), *celdas(f".{c}")] for c in material["clases"]]
    total = entero(sum(material["soporte"].values()))
    resumen = [
        ["Macro", total, *celdas("_macro")],
        ["Ponderado", total, *(con_intervalo(valores[k], intervalos[k]) for k in ("precision_ponderada", "recall_ponderado", "f1_ponderado"))],
    ]
    return tabular(
        "@{}lrlll@{}",
        ["Material", "$n$", "Precisión", "Recall", "F1"],
        [*filas, "\\midrule", *resumen],
        "metricas-elegido.json (material)",
    )


def auc_ap(insumos: Insumos) -> str:
    material = insumos.metricas["elegido"]["material"]
    valores, intervalos = material["valores"], material["intervalos"]

    def par(sufijo: str) -> list[str]:
        return [con_intervalo(valores[f"{m}{sufijo}"], intervalos[f"{m}{sufijo}"]) for m in ("auc", "ap")]

    filas = [[nombre(insumos, c), *par(f".{c}")] for c in material["clases"]]
    return tabular(
        "@{}lll@{}",
        ["Material", "AUC", "AP"],
        [*filas, "\\midrule", ["Macro", *par("_macro")], ["Micro", *par("_micro")]],
        "metricas-elegido.json (material)",
    )


def por_caneca(insumos: Insumos) -> str:
    tarea = insumos.metricas["elegido"]["caneca"]
    valores, intervalos = tarea["valores"], tarea["intervalos"]

    def celdas(sufijo: str, claves: tuple[str, ...] = ("precision", "recall", "f1")) -> list[str]:
        return [con_intervalo(valores[f"{m}{sufijo}"], intervalos[f"{m}{sufijo}"]) for m in claves]

    filas = [
        [c.capitalize(), entero(tarea["soporte"][c]), *celdas(f".{c}"), metrica(valores[f"auc.{c}"]), metrica(valores[f"ap.{c}"])]
        for c in tarea["clases"]
    ]
    total = entero(sum(tarea["soporte"].values()))
    resumen = [
        ["Macro", total, *celdas("_macro"), metrica(valores["auc_macro"]), metrica(valores["ap_macro"])],
        ["Ponderado", total, *(con_intervalo(valores[k], intervalos[k]) for k in ("precision_ponderada", "recall_ponderado", "f1_ponderado")), "", ""],
    ]
    return tabular(
        "@{}lrlllll@{}",
        ["Caneca", "$n$", "Precisión", "Recall", "F1", "AUC", "AP"],
        [*filas, "\\midrule", *resumen],
        "metricas-elegido.json (caneca)",
    )


def alteracion(perturbacion: Perturbacion) -> str:
    valor = decimal(perturbacion.valor, 1) if perturbacion.valor % 1 else entero(perturbacion.valor)
    textos = {
        "oscura": f"Brillo $\\times$ {valor}",
        "clara": f"Brillo $\\times$ {valor}",
        "desenfocada": f"Gaussiano de {valor} px",
        "girada": f"{valor}$^{{\\circ}}$ antihorario",
        "acercada": f"Recorte del {valor} \\%",
        "comprimida": f"Calidad JPEG {valor}",
    }
    return textos[perturbacion.nombre]


def celdas_robustez(resultado: dict, caida_material: str, caida_caneca: str, exactitud_material: str = "exactitud") -> list[str]:
    material, caneca_ = resultado["material"]["valores"], resultado["caneca"]["valores"]
    return [
        metrica(material[exactitud_material]), caida_material,
        metrica(caneca_["exactitud"]), caida_caneca, metrica(caneca_["precision.blanca"]), metrica(caneca_["recall.verde"]),
    ]  # fmt: skip


def fila_campo(campo: dict) -> list[str]:
    """Fila «Campo, Fotos propias» (rótulos cortos: con más largos la tabla no cabe en el ancho del texto).

    La marca «a» pide una nota: la exactitud de material de campo es la balanceada (las mismas fotos por
    clase) y su caída se mide frente a la exactitud balanceada de la prueba, no frente a la cruda de la
    fila Original; la caída de caneca sí es frente a la exactitud de caneca de la prueba.
    """
    comparacion = campo["comparacion_con_prueba"]
    caida_material = metrica(-comparacion["material_exactitud_balanceada"]["diferencia"])
    caida_caneca = metrica(-comparacion["caneca_exactitud"]["diferencia"])
    return ["Campo$^{a}$", "Fotos propias", *celdas_robustez(campo, caida_material, caida_caneca, "exactitud_balanceada")]


def robustez(insumos: Insumos) -> str:
    datos = insumos.metricas["robustez"]
    original = datos["original"]
    filas = [["Original", "Ninguna", *celdas_robustez(original, "", "")], "\\midrule"]
    for p in PERTURBACIONES:
        resultado = datos["perturbaciones"][p.nombre]
        filas.append([
            p.nombre.capitalize(), alteracion(p),
            *celdas_robustez(resultado, metrica(resultado["caida_exactitud_material"]), metrica(resultado["caida_exactitud_caneca"])),
        ])  # fmt: skip
    if insumos.campo:
        filas += ["\\midrule", fila_campo(insumos.campo)]
    encabezado = [
        "Versión", "Alteración", "\\multicolumn{2}{c}{Material}", "\\multicolumn{4}{c}{Caneca}",
    ]  # fmt: skip
    segunda = ["", "", "Exactitud", "Caída", "Exactitud", "Caída", "Prec. blanca", "Rec. verde"]
    texto = tabular(
        "@{}llrrrrrr@{}",
        encabezado,
        filas,
        "metricas-robustez.json" + (" y metricas-campo.json" if insumos.campo else ""),
    )
    cabecera = " & ".join(encabezado) + " \\\\"
    return texto.replace(cabecera, cabecera + "\n\\cmidrule(lr){3-4}\\cmidrule(l){5-8}\n" + " & ".join(segunda) + " \\\\")


TABLAS: dict[str, Callable[[Insumos], str]] = {
    "dataset-particion": dataset_particion,
    "canecas": canecas,
    "entrenamientos": entrenamientos,
    "antes-despues": antes_despues,
    "por-clase": por_clase,
    "auc-ap": auc_ap,
    "por-caneca": por_caneca,
    "robustez": robustez,
}


def todas(insumos: Insumos) -> dict[str, str]:
    return {f"{nombre_}.tex": generar(insumos) for nombre_, generar in TABLAS.items()}
