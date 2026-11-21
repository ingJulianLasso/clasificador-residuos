"""Macros de LaTeX con las cifras del proyecto y tablas del informe, generadas desde los archivos de datos.

Uso, desde ``clasificador/``:

    uv run python -m residuos.cifras                # escribe generado/cifras.tex y generado/tablas/*.tex
    uv run python -m residuos.cifras --comprobar    # regenera todo en una carpeta temporal y lo compara

El destino por defecto es el informe de la Actividad 6 (``rutas.TEX_GENERADO``); ``--destino`` escribe
en otra carpeta generado/.

Ninguna cifra del texto se escribe a mano. Cada macro lleva en un comentario el archivo y el campo de
donde sale; en los JSON, la ruta de claves va separada por `` > `` y una nota entre paréntesis indica
cuando la cifra es un cálculo sobre esos campos (un mínimo, una diferencia). Las fuentes son
``particion.csv``, ``inventario.json``, ``canecas.json`` (nombres), ``tm-entrenamiento-*.json``,
``modelos.json``, ``metricas-*.json`` y ``paridad-a2.csv``; las de la aplicación (prefijos ``App`` y
``Aviso``), ``app-paridad.json``, ``web/umbrales.js`` y ``probabilidades-prueba-elegido.csv``.

Las macros con prefijo ``Campo`` y la fila «Fotos propias (campo)» de la tabla de robustez solo se
generan si existe ``resultados/metricas-campo.json`` (``--campo`` indica otra ruta); sin ese archivo,
lo generado no cambia.
"""

from __future__ import annotations

import argparse
import filecmp
import json
import re
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from residuos import rutas, tablas
from residuos.datos import (
    ENTRENAMIENTO,
    EXCLUIDA,
    MOTIVO_DUPLICADO,
    MOTIVO_ENTRE_CLASES,
    MOTIVO_NO_ABRE,
    PRUEBA,
    leer_particion,
)
from residuos.formato import (
    decimal,
    entero,
    fecha,
    identificador,
    intervalo,
    metrica,
    p_valor,
)
from residuos.robustez import PERTURBACIONES

DESTINO = rutas.TEX_GENERADO / "cifras.tex"
CARPETA_TABLAS = "tablas"
ENCABEZADO = (
    "% Archivo generado por residuos.cifras: no se edita a mano.\n"
    "% Cada macro indica en un comentario el archivo y el campo de donde sale.\n"
)
NOMBRE_VALIDO = re.compile(r"[A-Za-z]+")
SEPARADOR = " > "


@dataclass(frozen=True)
class Macro:
    nombre: str
    valor: str
    origen: str

    def como_latex(self) -> str:
        return f"\\newcommand{{\\{self.nombre}}}{{{self.valor}}} % {self.origen}"


# Conteos sobre particion.csv


def conteos(filas: pd.DataFrame) -> dict[str, int]:
    return {
        "Total": len(filas),
        "Excluidas": int((filas["particion"] == EXCLUIDA).sum()),
        "Entrenamiento": int((filas["particion"] == ENTRENAMIENTO).sum()),
        "Prueba": int((filas["particion"] == PRUEBA).sum()),
        "Balanceado": int(filas["balanceado"].sum()),
    }


def macros_de_conteos(prefijo: str, filas: pd.DataFrame, descripcion: str) -> list[Macro]:
    return [
        Macro(f"{prefijo}{campo}", entero(valor), f"particion.csv: {descripcion}, {campo.lower()}")
        for campo, valor in conteos(filas).items()
    ]


# Grupos de macros


def macros_zip(inventario: dict) -> list[Macro]:
    zip_ = inventario["zip"]
    diferencia = inventario["diferencia_con_uci"]
    return [
        Macro("ZipBytes", entero(zip_["bytes"]), "inventario.json: zip.bytes"),
        Macro("ZipMiB", decimal(zip_["mib"]), "inventario.json: zip.mib"),
        Macro("ZipSha", zip_["sha256"], "inventario.json: zip.sha256"),
        Macro("ZipFechaDescarga", fecha(zip_["fecha_descarga"]), "inventario.json: zip.fecha_descarga"),
        Macro("ZipArchivos", entero(zip_["archivos"]), "inventario.json: zip.archivos"),
        Macro("ZipImagenes", entero(zip_["imagenes"]), "inventario.json: zip.imagenes"),
        Macro("ZipNoImagen", entero(len(zip_["no_imagen"])), "inventario.json: zip.no_imagen"),
        Macro("InstanciasUCI", entero(diferencia["instancias_uci"]), "inventario.json: diferencia_con_uci.instancias_uci"),
    ]


def macros_imagenes(inventario: dict) -> list[Macro]:
    imagenes = inventario["imagenes"]
    dimension, habituales = max(imagenes["dimensiones"].items(), key=lambda par: par[1])
    ancho, alto = dimension.split("x")
    con_orientacion = sum(n for clave, n in imagenes["orientacion_exif"].items() if clave != "sin etiqueta")
    return [
        Macro("AnchoImagen", ancho, "inventario.json: imagenes.dimensiones (la más frecuente)"),
        Macro("AltoImagen", alto, "inventario.json: imagenes.dimensiones (la más frecuente)"),
        Macro(
            "ImagenesOtraDimension",
            entero(sum(imagenes["dimensiones"].values()) - habituales),
            "inventario.json: imagenes.dimensiones",
        ),
        Macro("ImagenesRGB", entero(imagenes["modos_de_color"].get("RGB", 0)), "inventario.json: imagenes.modos_de_color"),
        Macro("ImagenesConOrientacionExif", entero(con_orientacion), "inventario.json: imagenes.orientacion_exif"),
        Macro("ImagenesNoAbren", entero(len(imagenes["no_abren"])), "inventario.json: imagenes.no_abren"),
    ]


def macros_duplicados(inventario: dict, particion: pd.DataFrame) -> list[Macro]:
    exactos = inventario["duplicados_exactos"]
    casi = inventario["casi_duplicados"]
    tamanos = particion.loc[particion["particion"] != EXCLUIDA, "grupo"].value_counts()
    en_grupos = tamanos[tamanos > 1]
    motivo = particion["motivo"]
    return [
        Macro("DuplicadosExactosGrupos", entero(exactos["grupos"]), "inventario.json: duplicados_exactos.grupos"),
        Macro(
            "DuplicadosExactosEntreClases",
            entero(exactos["grupos_entre_clases"]),
            "inventario.json: duplicados_exactos.grupos_entre_clases",
        ),
        Macro(
            "ExcluidasPorCopia", entero(motivo.str.startswith(MOTIVO_DUPLICADO).sum()), "particion.csv: motivo"
        ),
        Macro(
            "ExcluidasEntreClases", entero(motivo.str.startswith(MOTIVO_ENTRE_CLASES).sum()), "particion.csv: motivo"
        ),
        Macro("ExcluidasNoAbren", entero((motivo == MOTIVO_NO_ABRE).sum()), "particion.csv: motivo"),
        Macro("DistanciaHammingMaxima", entero(casi["distancia_maxima"]), "inventario.json: casi_duplicados.distancia_maxima"),
        Macro("CasiDuplicadosPares", entero(casi["pares"]), "inventario.json: casi_duplicados.pares"),
        Macro(
            "CasiDuplicadosParesMismaClase",
            entero(casi["pares_misma_clase"]),
            "inventario.json: casi_duplicados.pares_misma_clase",
        ),
        Macro(
            "CasiDuplicadosParesEntreClases",
            entero(casi["pares_entre_clases"]),
            "inventario.json: casi_duplicados.pares_entre_clases",
        ),
        Macro("CasiDuplicadosGrupos", entero(len(en_grupos)), "particion.csv: grupos con más de una imagen"),
        Macro("CasiDuplicadosImagenes", entero(en_grupos.sum()), "particion.csv: imágenes en esos grupos"),
        Macro("CasiDuplicadosGrupoMaximo", entero(en_grupos.max() if len(en_grupos) else 0), "particion.csv: grupo mayor"),
    ]


def macros_particion(inventario: dict, particion: pd.DataFrame) -> list[Macro]:
    criterio = inventario["particion"]
    incluidas = particion[particion["particion"] != EXCLUIDA]
    fracciones = incluidas.groupby("clase")["particion"].apply(lambda serie: (serie == PRUEBA).mean() * 100)
    balanceado = particion[particion["balanceado"] == 1]["clase"].value_counts()
    return [
        Macro("NumClases", entero(particion["clase"].nunique()), "particion.csv: clases distintas"),
        Macro("SemillaParticion", entero(criterio["semilla"]), "inventario.json: particion.semilla"),
        Macro("PorcentajePrueba", entero(round(criterio["fraccion_prueba"] * 100)), "inventario.json: particion.fraccion_prueba"),
        Macro("ImagenesIncluidas", entero(len(incluidas)), "particion.csv: filas no excluidas"),
        *macros_de_conteos("Imagenes", particion, "todas las filas"),
        Macro("BalanceadoPorClase", entero(balanceado.min()), "particion.csv: balanceado por clase"),
        Macro("PruebaPorcentajeMinimo", decimal(fracciones.min()), "particion.csv: porcentaje de prueba por clase, mínimo"),
        Macro("PruebaPorcentajeMaximo", decimal(fracciones.max()), "particion.csv: porcentaje de prueba por clase, máximo"),
    ]


def macros_por_clase(particion: pd.DataFrame, canecas: dict) -> list[Macro]:
    materiales = canecas["materiales"]
    totales = particion["clase"].value_counts()
    mayor, menor = totales.idxmax(), totales.idxmin()
    macros = [
        Macro("ClaseMayorNombre", materiales[mayor]["nombre"], f"particion.csv y canecas.json: {mayor}"),
        Macro("ClaseMayorTotal", entero(totales[mayor]), f"particion.csv: filas de {mayor}"),
        Macro("ClaseMenorNombre", materiales[menor]["nombre"], f"particion.csv y canecas.json: {menor}"),
        Macro("ClaseMenorTotal", entero(totales[menor]), f"particion.csv: filas de {menor}"),
        Macro("RazonDesbalance", decimal(totales[mayor] / totales[menor]), "particion.csv: clase mayor entre clase menor"),
    ]
    for clase, filas in particion.groupby("clase", sort=True):
        macros += macros_de_conteos(f"Clase{identificador(materiales[clase]['nombre'])}", filas, clase)
    return macros


def macros_por_caneca(particion: pd.DataFrame, canecas: dict) -> list[Macro]:
    caneca_de = {clase: datos["caneca"] for clase, datos in canecas["materiales"].items()}
    por_caneca = particion["clase"].map(caneca_de)
    macros = []
    for caneca in canecas["canecas"]:
        prefijo = f"Caneca{identificador(caneca)}"
        filas = particion[por_caneca == caneca]
        macros.append(
            Macro(f"{prefijo}Materiales", entero(filas["clase"].nunique()), f"canecas.json y particion.csv: {caneca}")
        )
        macros += macros_de_conteos(prefijo, filas, f"caneca {caneca}")
    return macros


def macros_dataset(inventario: dict, particion: pd.DataFrame, canecas: dict) -> list[Macro]:
    return [
        *macros_zip(inventario),
        *macros_imagenes(inventario),
        *macros_duplicados(inventario, particion),
        *macros_particion(inventario, particion),
        *macros_por_clase(particion, canecas),
        *macros_por_caneca(particion, canecas),
    ]

# Resultados: entrenamientos, modelos, paridad y métricas


@dataclass(frozen=True)
class Fuente:
    """Un JSON de resultados y su nombre, para escribir el origen de cada macro."""

    archivo: str
    datos: dict

    def valor(self, *claves: str | int):
        valor = self.datos
        for clave in claves:
            valor = valor[clave]
        return valor

    def origen(self, *claves: str | int, nota: str = "") -> str:
        ruta = SEPARADOR.join(str(c) for c in claves)
        return f"{self.archivo}: {ruta}" + (f" ({nota})" if nota else "")

    def macro(self, nombre: str, formato: Callable, *claves: str | int) -> Macro:
        return Macro(nombre, formato(self.valor(*claves)), self.origen(*claves))


def cargar_fuente(ruta: Path) -> Fuente:
    return Fuente(ruta.name, cargar_json(ruta))


def nombre_material(canecas: dict, etiqueta: str) -> str:
    return canecas["materiales"][etiqueta]["nombre"]


def ordinal_modelo(numero: int) -> str:
    return {1: "Uno", 2: "Dos"}[numero]


def numero_de(clave_modelo: str) -> int:
    return int(clave_modelo.removeprefix("entrenamiento_"))


def segundos(inicio: str, fin: str) -> int:
    return round((datetime.fromisoformat(fin) - datetime.fromisoformat(inicio)).total_seconds())


def macros_entrenamientos(tm: dict[int, Fuente], modelos: Fuente, canecas: dict) -> list[Macro]:
    primero = tm[1]
    if any(tm[n].valor("parametros") != primero.valor("parametros") for n in tm):
        raise ValueError("Los dos entrenamientos deberían tener los mismos hiperparámetros")
    macros = [
        primero.macro("Epocas", entero, "parametros", "epocas"),
        primero.macro("Lote", entero, "parametros", "lote"),
        primero.macro("TasaAprendizaje", lambda v: decimal(v, 3), "parametros", "tasa_aprendizaje"),
        primero.macro("UnidadesDensas", entero, "parametros_internos_del_entrenamiento", "denseUnits"),
        primero.macro("TmVersion", str, "modelo_local", "metadata", "tmVersion"),
    ]
    for n, fuente in tm.items():
        prefijo = ordinal_modelo(n)
        exactitudes = fuente.valor("validacion_por_clase")
        peor = min(exactitudes, key=lambda c: exactitudes[c]["exactitud"])
        mejor = max(exactitudes, key=lambda c: exactitudes[c]["exactitud"])
        macros += [
            modelos.macro(f"{prefijo}ValMediaClase", lambda v: decimal(v, 2), f"entrenamiento_{n}", "exactitud_media_validacion"),
            fuente.macro(f"{prefijo}ValGlobal", metrica, "resumen_validacion", "exactitud_global_desde_matriz"),
            fuente.macro(f"{prefijo}ValImagenes", entero, "resumen_validacion", "imagenes_validacion"),
            Macro(f"{prefijo}ValPeorNombre", nombre_material(canecas, peor), fuente.origen("validacion_por_clase", nota="mínimo de exactitud")),
            fuente.macro(f"{prefijo}ValPeor", lambda v: decimal(v, 2), "validacion_por_clase", peor, "exactitud"),
            Macro(f"{prefijo}ValMejorNombre", nombre_material(canecas, mejor), fuente.origen("validacion_por_clase", nota="máximo de exactitud")),
            fuente.macro(f"{prefijo}ValMejor", lambda v: decimal(v, 2), "validacion_por_clase", mejor, "exactitud"),
            Macro(
                f"{prefijo}EntExactitudFinal",
                metrica(fuente.valor("curvas", "entrenamiento", "exactitud")[-1]),
                fuente.origen("curvas", "entrenamiento", "exactitud", nota="última época"),
            ),
            Macro(
                f"{prefijo}ValExactitudFinal",
                metrica(fuente.valor("curvas", "validacion", "exactitud")[-1]),
                fuente.origen("curvas", "validacion", "exactitud", nota="última época"),
            ),
            Macro(
                f"{prefijo}SegundosEntrenamiento",
                entero(segundos(fuente.valor("hora_inicio"), fuente.valor("hora_fin"))),
                fuente.origen("hora_inicio", nota="diferencia con hora_fin"),
            ),
        ]
    return macros


def macros_modelos(modelos: Fuente) -> list[Macro]:
    elegido = modelos.valor("elegido")
    otro = next(c for c in ("entrenamiento_1", "entrenamiento_2") if c != elegido)
    comparacion = modelos.valor("predicciones_alojado_vs_local")
    claves = ("entrenamiento_1", "entrenamiento_2")
    url = lambda v: f"\\url{{{v}}}"
    return [
        Macro("ElegidoNumero", entero(numero_de(elegido)), modelos.origen("elegido")),
        Macro("NoElegidoNumero", entero(numero_de(otro)), modelos.origen("elegido", nota="el otro entrenamiento")),
        modelos.macro("UnoUrl", url, "entrenamiento_1", "url"),
        modelos.macro("DosUrl", url, "entrenamiento_2", "url"),
        modelos.macro("ElegidoUrl", url, elegido, "url"),
        modelos.macro("ElegidoShaPesos", str, elegido, "sha256_alojado"),
        modelos.macro("ElegidoShaPesosCorto", lambda v: v[:16], elegido, "sha256_alojado"),
        modelos.macro("ElegidoBytesPesos", entero, elegido, "bytes_pesos"),
        Macro("ModelosPesosIdenticos", entero(sum(modelos.valor(c, "pesos_identicos") for c in claves)),
              modelos.origen("entrenamiento_N", "pesos_identicos", nota="cuenta")),
        Macro("ModelosEtiquetasIdenticas", entero(sum(modelos.valor(c, "etiquetas_identicas") for c in claves)),
              modelos.origen("entrenamiento_N", "etiquetas_identicas", nota="cuenta")),
        Macro("ModelosTotal", entero(len(claves)), modelos.origen("entrenamiento_N", nota="cuenta")),
        modelos.macro("AlojadoPrediccionesIdenticas", lambda v: "idénticas" if v else "distintas", "predicciones_alojado_vs_local", "identicas"),
        modelos.macro("AlojadoDiferenciaMaxima", lambda v: "0" if v == 0 else decimal(v, 7), "predicciones_alojado_vs_local", "diferencia_maxima"),
        modelos.macro("AlojadoImagenesComparadas", entero, "predicciones_alojado_vs_local", "imagenes"),
        Macro("AlojadoValoresComparados", entero(comparacion["imagenes"] * comparacion["columnas_comparadas"]),
              modelos.origen("predicciones_alojado_vs_local", nota="imagenes por columnas_comparadas")),
    ]


def macros_paridad() -> list[Macro]:
    from residuos.arnes import (
        comparar_paridad,  # importa Playwright: solo cuando hace falta
    )

    resumen = comparar_paridad()
    origen = "paridad-a2.csv: frente a las probabilidades de la vista previa de la Actividad 2 (arnes.comparar_paridad)"
    return [
        Macro("ParidadImagenes", entero(resumen.total), origen),
        Macro("ParidadMismaClase", entero(resumen.misma_clase), origen),
        Macro("ParidadDiferenciaMaxima", decimal(resumen.diferencia_maxima, 5), origen),
    ]


GLOBALES = {
    "exactitud": "Exactitud",
    "exactitud_balanceada": "ExactitudBalanceada",
    "precision_macro": "PrecisionMacro",
    "recall_macro": "RecallMacro",
    "f1_macro": "FMacro",
    "precision_ponderada": "PrecisionPonderada",
    "recall_ponderado": "RecallPonderado",
    "f1_ponderado": "FPonderado",
    "auc_macro": "AucMacro",
    "auc_micro": "AucMicro",
    "ap_macro": "ApMacro",
    "ap_micro": "ApMicro",
}
POR_CLASE = {"precision": "Precision", "recall": "Recall", "f1": "F", "auc": "Auc", "ap": "Ap"}


def con_limites(fuente: Fuente, nombre: str, tarea: str | tuple[str, ...], clave: str) -> list[Macro]:
    """La cifra puntual y los límites de su intervalo de bootstrap; `tarea` puede ser una ruta de claves."""
    ruta = (tarea,) if isinstance(tarea, str) else tarea
    inferior, superior = fuente.valor(*ruta, "intervalos", clave)
    origen = fuente.origen(*ruta, "intervalos", clave)
    return [
        fuente.macro(nombre, metrica, *ruta, "valores", clave),
        Macro(f"{nombre}Inf", metrica(inferior), origen + " (límite inferior)"),
        Macro(f"{nombre}Sup", metrica(superior), origen + " (límite superior)"),
        Macro(f"{nombre}IC", intervalo((inferior, superior)), origen),
    ]


def extremo(fuente: Fuente, prefijo: str, tarea: str, clave: str, nombres: Callable[[str], str], mayor: bool) -> list[Macro]:
    clases = fuente.valor(tarea, "clases")
    valores = {c: fuente.valor(tarea, "valores", f"{clave}.{c}") for c in clases}
    elegida = (max if mayor else min)(valores, key=valores.get)
    nota = "máximo" if mayor else "mínimo"
    return [
        Macro(f"{prefijo}Nombre", nombres(elegida), fuente.origen(tarea, "valores", f"{clave}.*", nota=nota)),
        fuente.macro(prefijo, metrica, tarea, "valores", f"{clave}.{elegida}"),
    ]


def calculadas(fuente: Fuente, tarea: str, claves: dict[str, str]) -> dict[str, str]:
    """Las claves que trae la tarea: sin ROC ni AP (prueba en campo) faltan `auc` y `ap`."""
    valores = fuente.valor(tarea, "valores")
    primera = fuente.valor(tarea, "clases")[0]
    return {c: n for c, n in claves.items() if c in valores or f"{c}.{primera}" in valores}


def macros_tarea_material(fuente: Fuente, prefijo: str, canecas: dict) -> list[Macro]:
    nombres = lambda e: nombre_material(canecas, e)
    globales, por_clase = calculadas(fuente, "material", GLOBALES), calculadas(fuente, "material", POR_CLASE)
    macros = [m for clave, nombre in globales.items() for m in con_limites(fuente, f"{prefijo}{nombre}", "material", clave)]
    for etiqueta in fuente.valor("material", "clases"):
        clase = identificador(nombres(etiqueta))
        macros += [fuente.macro(f"{prefijo}{n}{clase}", metrica, "material", "valores", f"{c}.{etiqueta}") for c, n in por_clase.items()]
    for clave, nombre in calculadas(fuente, "material", {"f1": "F", "recall": "Recall", "auc": "Auc", "ap": "Ap"}).items():
        macros += extremo(fuente, f"{prefijo}Mejor{nombre}", "material", clave, nombres, mayor=True)
        macros += extremo(fuente, f"{prefijo}Peor{nombre}", "material", clave, nombres, mayor=False)
    return macros


def macros_tarea_caneca(fuente: Fuente, prefijo: str) -> list[Macro]:
    base = f"{prefijo}Caneca"
    globales, por_clase = calculadas(fuente, "caneca", GLOBALES), calculadas(fuente, "caneca", POR_CLASE)
    macros = [m for clave, nombre in globales.items() for m in con_limites(fuente, f"{base}{nombre}", "caneca", clave)]
    for caneca in fuente.valor("caneca", "clases"):
        for clave, nombre in por_clase.items():
            macros += con_limites(fuente, f"{base}{nombre}{identificador(caneca)}", "caneca", f"{clave}.{caneca}")
        macros.append(fuente.macro(f"{base}Soporte{identificador(caneca)}", entero, "caneca", "soporte", caneca))
    return macros


def macros_errores(fuente: Fuente, prefijo: str, canecas: dict) -> list[Macro]:
    campos = {
        "material": "ErroresMaterial",
        "material_con_caneca_correcta": "ErroresMismaCaneca",
        "caneca": "ErroresCaneca",
        "contaminan_la_blanca": "ContaminanBlanca",
        "organicos_en_la_blanca": "OrganicosEnBlanca",
        "no_aprovechables_en_la_blanca": "NoAprovechablesEnBlanca",
        "aprovechables_perdidos_en_la_negra": "AprovechablesEnNegra",
    }
    macros = [fuente.macro(f"{prefijo}{nombre}", entero, "errores", clave) for clave, nombre in campos.items()]
    pares = fuente.valor("errores", "pares_mas_confundidos")
    for posicion, (ordinal, par) in enumerate(zip(("Uno", "Dos", "Tres"), pares)):
        base = f"{prefijo}ParConfusion{ordinal}"
        macros += [
            Macro(f"{base}Real", nombre_material(canecas, par["real"]), fuente.origen("errores", "pares_mas_confundidos", posicion, "real")),
            Macro(f"{base}Predicho", nombre_material(canecas, par["predicho"]),
                  fuente.origen("errores", "pares_mas_confundidos", posicion, "predicho")),
            fuente.macro(f"{base}Imagenes", entero, "errores", "pares_mas_confundidos", posicion, "imagenes"),
        ]
    campos_caneca = {
        "difieren": ("CanecaDifiereMaterial", entero),
        "exactitud_material": ("CanecaExactitudMaterialMasProbable", metrica),
        "aciertos_solo_suma": ("CanecaAciertosSoloSuma", entero),
        "aciertos_solo_material": ("CanecaAciertosSoloMaterial", entero),
    }
    macros += [
        fuente.macro(f"{prefijo}{nombre}", formato, "caneca_vs_material_mas_probable", clave)
        for clave, (nombre, formato) in campos_caneca.items()
    ]
    return macros


def macros_modelo(fuente: Fuente, prefijo: str, canecas: dict) -> list[Macro]:
    return [
        fuente.macro(f"{prefijo}ImagenesEvaluadas", entero, "imagenes"),
        *macros_tarea_material(fuente, prefijo, canecas),
        *macros_tarea_caneca(fuente, prefijo),
        *macros_errores(fuente, prefijo, canecas),
    ]


def macros_brecha(metricas: dict[int, Fuente], modelos: Fuente) -> list[Macro]:
    """Media por clase de la validación interna menos exactitud balanceada de la prueba."""
    macros = []
    for n, fuente in metricas.items():
        validacion = modelos.valor(f"entrenamiento_{n}", "exactitud_media_validacion")
        prueba = fuente.valor("material", "valores", "exactitud_balanceada")
        origen = (
            f"modelos.json y {fuente.archivo}: entrenamiento_{n} > exactitud_media_validacion "
            f"y material > valores > exactitud_balanceada (diferencia)"
        )
        macros.append(Macro(f"{ordinal_modelo(n)}BrechaValPrueba", metrica(validacion - prueba), origen))
    return macros


def macros_bootstrap_y_mcnemar(elegido: Fuente) -> list[Macro]:
    macros = [
        elegido.macro("BootstrapRemuestreos", entero, "bootstrap", "remuestreos"),
        elegido.macro("BootstrapSemilla", entero, "bootstrap", "semilla"),
        elegido.macro("BootstrapNivel", lambda v: entero(round(v * 100)), "bootstrap", "nivel"),
        elegido.macro("McNemarAlfa", lambda v: decimal(v, 2), "comparacion_entrenamientos", "parametros_mcnemar", "alfa"),
    ]
    for tarea, nombre in (("material", "Material"), ("caneca", "Caneca")):
        base = ("comparacion_entrenamientos", tarea)
        macros += [
            elegido.macro(f"McNemar{nombre}B", entero, *base, "mcnemar", "b"),
            elegido.macro(f"McNemar{nombre}C", entero, *base, "mcnemar", "c"),
            elegido.macro(f"McNemar{nombre}Discordantes", entero, *base, "mcnemar", "discordantes"),
            elegido.macro(f"McNemar{nombre}P", p_valor, *base, "mcnemar", "p_valor"),
            elegido.macro(f"DifExactitud{nombre}", metrica, *base, "diferencia_exactitud", "valor"),
            elegido.macro(f"DifExactitud{nombre}IC", intervalo, *base, "diferencia_exactitud", "intervalo"),
            elegido.macro(f"DifExactitud{nombre}Inf", lambda v: metrica(v[0]), *base, "diferencia_exactitud", "intervalo"),
            elegido.macro(f"DifExactitud{nombre}Sup", lambda v: metrica(v[1]), *base, "diferencia_exactitud", "intervalo"),
        ]
    return macros


UMBRALES_NOMBRADOS = {0.5: "Cincuenta", 0.7: "Setenta", 0.8: "Ochenta", 0.9: "Noventa", 0.95: "NoventaYCinco", 0.99: "NoventaYNueve"}


def macros_cobertura(elegido: Fuente) -> list[Macro]:
    base = elegido.valor("cobertura_blanca", "sin_abstencion")
    macros = [
        elegido.macro("BlancaAsignadas", entero, "cobertura_blanca", "sin_abstencion", "imagenes"),
        elegido.macro("BlancaPrecisionBase", metrica, "cobertura_blanca", "sin_abstencion", "precision"),
        elegido.macro("BlancaRecallBase", metrica, "cobertura_blanca", "sin_abstencion", "recall"),
        elegido.macro("BlancaContaminantesBase", entero, "cobertura_blanca", "sin_abstencion", "contaminantes"),
    ]
    puntos = elegido.valor("cobertura_blanca", "puntos")
    for posicion, punto in enumerate(puntos):
        if punto["umbral"] not in UMBRALES_NOMBRADOS:
            continue
        nombre = f"BlancaUmbral{UMBRALES_NOMBRADOS[punto['umbral']]}"
        ruta = ("cobertura_blanca", "puntos", posicion)
        macros += [
            elegido.macro(nombre, lambda v: decimal(v, 2), *ruta, "umbral"),
            elegido.macro(f"{nombre}Cobertura", metrica, *ruta, "cobertura"),
            elegido.macro(f"{nombre}Precision", metrica, *ruta, "precision"),
            elegido.macro(f"{nombre}Recall", metrica, *ruta, "recall"),
            elegido.macro(f"{nombre}Imagenes", entero, *ruta, "imagenes"),
            elegido.macro(f"{nombre}Contaminantes", entero, *ruta, "contaminantes"),
        ]
    if not base["imagenes"]:
        raise ValueError("Ninguna imagen asignada a la blanca")
    return macros


def macros_errores_confiados(elegido: Fuente) -> list[Macro]:
    errores = elegido.valor("errores_mas_confiados")
    cambian = sum(e["caneca_real"] != e["caneca_predicha"] for e in errores)
    return [
        Macro("ErroresConfiadosTotal", entero(len(errores)), elegido.origen("errores_mas_confiados", nota="cuenta")),
        Macro("ErroresConfiadosMinima", decimal(min(e["confianza"] for e in errores), 4),
              elegido.origen("errores_mas_confiados", "confianza", nota="mínimo")),
        Macro("ErroresConfiadosCambianCaneca", entero(cambian), elegido.origen("errores_mas_confiados", nota="caneca_real distinta de caneca_predicha")),
    ]


def macros_robustez(robustez: Fuente) -> list[Macro]:
    macros = [
        robustez.macro("RobBootstrapRemuestreos", entero, "bootstrap", "remuestreos"),
    ]
    caidas = {}
    for p in PERTURBACIONES:
        nombre = f"Rob{identificador(p.nombre)}"
        ruta = ("perturbaciones", p.nombre)
        macros += [
            robustez.macro(f"{nombre}Valor", lambda v: decimal(v, 1) if v % 1 else entero(v), "parametros", p.nombre, "valor"),
            *con_limites(robustez, f"{nombre}Exactitud", (*ruta, "material"), "exactitud"),
            robustez.macro(f"{nombre}FMacro", metrica, *ruta, "material", "valores", "f1_macro"),
            robustez.macro(f"{nombre}ExactitudBalanceada", metrica, *ruta, "material", "valores", "exactitud_balanceada"),
            robustez.macro(f"{nombre}Caida", metrica, *ruta, "caida_exactitud_material"),
            *con_limites(robustez, f"{nombre}CanecaExactitud", (*ruta, "caneca"), "exactitud"),
            robustez.macro(f"{nombre}CanecaCaida", metrica, *ruta, "caida_exactitud_caneca"),
            robustez.macro(f"{nombre}PrecisionBlanca", metrica, *ruta, "caneca", "valores", "precision.blanca"),
            robustez.macro(f"{nombre}RecallVerde", metrica, *ruta, "caneca", "valores", "recall.verde"),
            robustez.macro(f"{nombre}CambianMaterial", entero, *ruta, "cambian_de_material"),
        ]
        caidas[p.nombre] = (robustez.valor(*ruta, "caida_exactitud_material"), robustez.valor(*ruta, "caida_exactitud_caneca"))
    for indice, sufijo in ((0, ""), (1, "Caneca")):
        peor = max(caidas, key=lambda n: caidas[n][indice])
        menor = min(caidas, key=lambda n: caidas[n][indice])
        clave = "caida_exactitud_material" if indice == 0 else "caida_exactitud_caneca"
        macros += [
            Macro(f"RobPeor{sufijo}Nombre", peor, robustez.origen("perturbaciones", "*", clave, nota="máximo")),
            robustez.macro(f"RobPeor{sufijo}Caida", metrica, "perturbaciones", peor, clave),
            Macro(f"RobMenor{sufijo}Nombre", menor, robustez.origen("perturbaciones", "*", clave, nota="mínimo")),
            robustez.macro(f"RobMenor{sufijo}Caida", metrica, "perturbaciones", menor, clave),
        ]
    return macros


def macros_resultados(canecas: dict) -> list[Macro]:
    tm = {n: cargar_fuente(rutas.entrenamiento_json(n)) for n in (1, 2)}
    modelos = cargar_fuente(rutas.MODELOS_JSON)
    metricas = {n: cargar_fuente(rutas.RESULTADOS / f"metricas-entrenamiento-{n}.json") for n in (1, 2)}
    elegido = cargar_fuente(rutas.RESULTADOS / "metricas-elegido.json")
    robustez = cargar_fuente(rutas.RESULTADOS / "metricas-robustez.json")
    return [
        *macros_entrenamientos(tm, modelos, canecas),
        *macros_modelos(modelos),
        *macros_paridad(),
        *macros_bootstrap_y_mcnemar(elegido),
        *macros_modelo(metricas[1], "Uno", canecas),
        *macros_modelo(metricas[2], "Dos", canecas),
        *macros_modelo(elegido, "Elegido", canecas),
        *macros_brecha(metricas, modelos),
        *macros_cobertura(elegido),
        *macros_errores_confiados(elegido),
        *macros_robustez(robustez),
    ]


# Aplicación de la Actividad 6: sus pruebas en el navegador (app-paridad.json) y los dos avisos de Clasificar

APP_PARIDAD = rutas.RESULTADOS / "app-paridad.json"
UMBRALES_APP = rutas.RAIZ_CLASIFICADOR / "web" / "umbrales.js"
PROBABILIDADES_PRUEBA = rutas.RESULTADOS / "probabilidades-prueba-elegido.csv"
VISTA_EN_LA_CLAVE = re.compile(r"^(?P<vista>[a-z]+) (?P<ancho>\d+) px, ")
PAQUETES_APP = {"@tensorflow/tfjs": "AppVersionTfjs", "@teachablemachine/image": "AppVersionTmImage"}


@dataclass(frozen=True)
class Aviso:
    """Un aviso de Clasificar: se muestra cuando la probabilidad de lo elegido es menor que el umbral."""

    prefijo: str
    constante: str
    columna_elegida: str
    prefijo_probabilidad: str
    columna_real: str


AVISOS = (
    Aviso("AvisoCaneca", "UMBRAL_CANECA_DUDOSA", "caneca", "prob_caneca_", "caneca_real"),
    Aviso("AvisoMaterial", "UMBRAL_LEJOS_DE_LO_APRENDIDO", "material", "prob_", "clase_real"),
)


def validar_app(app: Fuente) -> None:
    """Lo que el informe da por hecho de las pruebas de la aplicación; si algo falla, no hay macros."""
    axe = [vista["serias_o_criticas"] for pestana in app.valor("axe").values() if isinstance(pestana, dict) for vista in pestana.values()]
    clasificar = app.valor("clasificar")
    fallas = {
        "probabilidades distintas de las de la evaluación": not app.valor("iguales"),
        "métricas distintas de las de la evaluación": not app.valor("metricas_iguales"),
        "violaciones serias o críticas de axe": any(axe),
        "desplazamiento horizontal en el celular": any(app.valor("desplazamiento_horizontal_px").values()),
        "tarjetas de Clasificar distintas de la evaluación": clasificar["iguales_a_la_evaluacion"] != clasificar["tarjetas"],
    }
    problemas = [descripcion for descripcion, falla in fallas.items() if falla]
    if problemas:
        raise ValueError(f"{app.archivo}: {'; '.join(problemas)}")


def macros_anchos(app: Fuente) -> list[Macro]:
    """El ancho de cada vista angosta, que va en la clave de desplazamiento_horizontal_px."""
    anchos = {}
    for clave in app.valor("desplazamiento_horizontal_px"):
        vista = VISTA_EN_LA_CLAVE.match(clave)
        anchos[vista["vista"]] = vista["ancho"]
    origen = app.origen("desplazamiento_horizontal_px", nota="ancho de la vista en la clave")
    return [Macro(f"AppAncho{identificador(vista)}", ancho, origen) for vista, ancho in anchos.items()]


def macros_pruebas_app(app: Fuente) -> list[Macro]:
    return [
        app.macro("AppParidadImagenes", entero, "paridad", "n"),
        app.macro("AppParidadReferencia", entero, "paridad", "n_referencia"),
        app.macro("AppParidadMismoMaterialYCaneca", entero, "paridad", "mismo_material_y_caneca"),
        app.macro("AppParidadDiferenciaMaxima", lambda v: "0" if v == 0 else decimal(v, 7), "paridad", "diferencia_maxima"),
        app.macro("AppSegundosEvaluacion", decimal, "segundos_evaluacion"),
        *(app.macro(nombre, str, "entorno", "paquetes", paquete) for paquete, nombre in PAQUETES_APP.items()),
        app.macro("AppClasificarArchivos", entero, "clasificar", "archivos"),
        app.macro("AppClasificarTarjetas", entero, "clasificar", "tarjetas"),
        app.macro("AppClasificarErrores", entero, "clasificar", "errores"),
        app.macro("AppClasificarFilasCsv", entero, "clasificar", "filas_csv"),
        app.macro("AppAxeVersion", str, "axe", "version"),
        *macros_anchos(app),
    ]


def umbral_de_la_app(constante: str) -> float:
    coincidencia = re.search(rf"export const {constante} = (?P<valor>\d+(?:\.\d+)?);", UMBRALES_APP.read_text(encoding="utf-8"))
    if coincidencia is None:
        raise ValueError(f"{UMBRALES_APP.name} no define {constante}")
    return float(coincidencia["valor"])


def probabilidad_de_lo_elegido(prueba: pd.DataFrame, aviso: Aviso) -> pd.Series:
    return prueba.apply(lambda fila: fila[f"{aviso.prefijo_probabilidad}{fila[aviso.columna_elegida]}"], axis=1)


def macros_aviso(prueba: pd.DataFrame, aviso: Aviso) -> list[Macro]:
    """Cobertura (fracción sin aviso) y aciertos con y sin el aviso, sobre el conjunto de prueba."""
    umbral = umbral_de_la_app(aviso.constante)
    con_aviso = probabilidad_de_lo_elegido(prueba, aviso) < umbral
    acierto = prueba[aviso.columna_elegida] == prueba[aviso.columna_real]
    if con_aviso.all() or not con_aviso.any():
        raise ValueError(f"El aviso {aviso.constante} no separa el conjunto de prueba")
    origen = f"{PROBABILIDADES_PRUEBA.name} y {UMBRALES_APP.name}: {aviso.columna_elegida} con probabilidad menor que {aviso.constante}"
    p = aviso.prefijo
    return [
        Macro(f"{p}Umbral", decimal(umbral, 2), f"{UMBRALES_APP.name}: {aviso.constante}"),
        Macro(f"{p}Imagenes", entero(con_aviso.sum()), f"{origen} (cuenta)"),
        Macro(f"{p}Cobertura", metrica(1 - con_aviso.mean()), f"{origen} (fracción sin aviso)"),
        Macro(f"{p}ExactitudSinAviso", metrica(acierto[~con_aviso].mean()), f"{origen} (aciertos sin aviso)"),
        Macro(f"{p}ExactitudConAviso", metrica(acierto[con_aviso].mean()), f"{origen} (aciertos con aviso)"),
        Macro(f"{p}ErroresAvisados", entero((con_aviso & ~acierto).sum()), f"{origen} (errores con aviso)"),
    ]


def macros_app(canecas: dict) -> list[Macro]:
    app = cargar_fuente(APP_PARIDAD)
    validar_app(app)
    prueba = pd.read_csv(PROBABILIDADES_PRUEBA)
    prueba["caneca_real"] = prueba["clase_real"].map({etiqueta: m["caneca"] for etiqueta, m in canecas["materiales"].items()})
    return [*macros_pruebas_app(app), *(m for aviso in AVISOS for m in macros_aviso(prueba, aviso))]


# Prueba en campo (solo si existe metricas-campo.json)

PREFIJO_CAMPO = "Campo"
COMPARACIONES_CAMPO = {"material_exactitud_balanceada": "ExactitudBalanceada", "caneca_exactitud": "CanecaExactitud"}


def macros_comparacion_campo(campo: Fuente) -> list[Macro]:
    """Campo frente a la prueba del 30 %; la caída es la diferencia con el signo cambiado."""
    macros = []
    for clave, nombre in COMPARACIONES_CAMPO.items():
        ruta = ("comparacion_con_prueba", clave)
        macros += [
            campo.macro(f"{PREFIJO_CAMPO}Prueba{nombre}", metrica, *ruta, "prueba"),
            campo.macro(f"{PREFIJO_CAMPO}Dif{nombre}", metrica, *ruta, "diferencia"),
            campo.macro(f"{PREFIJO_CAMPO}Dif{nombre}IC", intervalo, *ruta, "intervalo"),
            campo.macro(f"{PREFIJO_CAMPO}Dif{nombre}Inf", lambda v: metrica(v[0]), *ruta, "intervalo"),
            campo.macro(f"{PREFIJO_CAMPO}Dif{nombre}Sup", lambda v: metrica(v[1]), *ruta, "intervalo"),
            Macro(f"{PREFIJO_CAMPO}Caida{nombre}", metrica(-campo.valor(*ruta, "diferencia")), campo.origen(*ruta, "diferencia", nota="signo cambiado")),
        ]
    return macros


def metrica_o_indefinida(valor: float | None) -> str:
    return "no definida" if valor is None else metrica(valor)


def macros_umbral_campo(campo: Fuente) -> list[Macro]:
    ruta = ("umbral_blanca", "campo")
    base = f"{PREFIJO_CAMPO}BlancaUmbral"
    macros = [campo.macro(base, lambda v: decimal(v, 2), "umbral_blanca", "umbral")]
    for clave, nombre in (("precision", "Precision"), ("cobertura", "Cobertura")):
        inferior, superior = campo.valor(*ruta, "intervalos", clave)
        origen = campo.origen(*ruta, "intervalos", clave)
        macros += [
            campo.macro(f"{base}{nombre}", metrica_o_indefinida, *ruta, "valores", clave),
            Macro(f"{base}{nombre}IC", intervalo((inferior, superior)), origen),
        ]
    macros += [
        campo.macro(f"{base}Imagenes", entero, *ruta, "valores", "imagenes"),
        campo.macro(f"{base}Contaminantes", entero, *ruta, "valores", "contaminantes"),
        campo.macro(f"{PREFIJO_CAMPO}BlancaAsignadas", entero, "umbral_blanca", "campo_sin_abstencion", "imagenes"),
        campo.macro(f"{PREFIJO_CAMPO}BlancaPrecisionBase", metrica_o_indefinida, "umbral_blanca", "campo_sin_abstencion", "precision"),
        campo.macro(f"{PREFIJO_CAMPO}BlancaContaminantesBase", entero, "umbral_blanca", "campo_sin_abstencion", "contaminantes"),
    ]
    return macros


def macros_condiciones_campo(campo: Fuente) -> list[Macro]:
    macros = []
    for factor, niveles in campo.valor("por_condicion").items():
        for nivel, datos in niveles.items():
            nombre = f"{PREFIJO_CAMPO}{identificador(factor)}{identificador(datos['corto'])}"
            ruta = ("por_condicion", factor, nivel)
            macros += [
                campo.macro(f"{nombre}Imagenes", entero, *ruta, "imagenes"),
                campo.macro(f"{nombre}Exactitud", metrica, *ruta, "exactitud_material"),
                campo.macro(f"{nombre}CanecaExactitud", metrica, *ruta, "exactitud_caneca"),
            ]
    return macros


def macros_campo(campo: Fuente, canecas: dict) -> list[Macro]:
    return [
        *macros_modelo(campo, PREFIJO_CAMPO, canecas),
        campo.macro(f"{PREFIJO_CAMPO}ConSustituto", entero, "con_sustituto", "fotos"),
        campo.macro(f"{PREFIJO_CAMPO}ConOrientacionExif", entero, "orientacion_exif", "fotos_con_etiqueta_distinta_de_1"),
        *macros_comparacion_campo(campo),
        *macros_umbral_campo(campo),
        *macros_condiciones_campo(campo),
    ]


def cargar_campo(ruta: Path) -> dict | None:
    return cargar_json(ruta) if ruta.exists() else None


# Escritura


def cargar_json(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8"))


def todas_las_macros(campo: Path = rutas.METRICAS_CAMPO) -> list[Macro]:
    canecas = cargar_json(rutas.CANECAS)
    macros = [
        *macros_dataset(cargar_json(rutas.INVENTARIO), leer_particion(), canecas),
        *macros_resultados(canecas),
        *macros_app(canecas),
    ]
    if campo.exists():
        macros += macros_campo(cargar_fuente(campo), canecas)
    return macros


def insumos_de_tablas(campo: Path = rutas.METRICAS_CAMPO) -> tablas.Insumos:
    return tablas.Insumos(
        particion=leer_particion(),
        canecas=cargar_json(rutas.CANECAS),
        entrenamientos={n: cargar_json(rutas.entrenamiento_json(n)) for n in (1, 2)},
        modelos=cargar_json(rutas.MODELOS_JSON),
        metricas={
            "1": cargar_json(rutas.RESULTADOS / "metricas-entrenamiento-1.json"),
            "2": cargar_json(rutas.RESULTADOS / "metricas-entrenamiento-2.json"),
            "elegido": cargar_json(rutas.RESULTADOS / "metricas-elegido.json"),
            "robustez": cargar_json(rutas.RESULTADOS / "metricas-robustez.json"),
        },
        campo=cargar_campo(campo),
    )


def validar(macros: list[Macro]) -> None:
    nombres = [m.nombre for m in macros]
    invalidos = [n for n in nombres if not NOMBRE_VALIDO.fullmatch(n)]
    repetidos = sorted({n for n in nombres if nombres.count(n) > 1})
    if invalidos or repetidos:
        raise ValueError(f"Macros inválidas: {invalidos}; repetidas: {repetidos}")


def como_texto(macros: list[Macro]) -> str:
    validar(macros)
    return ENCABEZADO + "".join(m.como_latex() + "\n" for m in macros)


def escribir_todo(carpeta: Path, campo: Path = rutas.METRICAS_CAMPO) -> tuple[list[Path], int]:
    """Escribe cifras.tex y las tablas en `carpeta`; devuelve las rutas escritas y el número de macros."""
    carpeta_tablas = carpeta / CARPETA_TABLAS
    carpeta_tablas.mkdir(parents=True, exist_ok=True)
    macros = todas_las_macros(campo)
    escritas = [carpeta / DESTINO.name]
    escritas[0].write_text(como_texto(macros), encoding="utf-8")
    for nombre, contenido in tablas.todas(insumos_de_tablas(campo)).items():
        ruta = carpeta_tablas / nombre
        ruta.write_text(contenido, encoding="utf-8")
        escritas.append(ruta)
    return escritas, len(macros)


def archivos_generados(carpeta: Path) -> set[str]:
    return {p.relative_to(carpeta).as_posix() for p in [carpeta / DESTINO.name, *(carpeta / CARPETA_TABLAS).glob("*.tex")] if p.exists()}


def comprobar(destino: Path, temporal: Path, campo: Path = rutas.METRICAS_CAMPO) -> list[str]:
    """Diferencias entre lo que hay en `destino` y lo que se regenera en `temporal`."""
    escribir_todo(temporal, campo)
    nuevos, actuales = archivos_generados(temporal), archivos_generados(destino)
    diferencias = [f"falta en {destino}: {r}" for r in sorted(nuevos - actuales)]
    diferencias += [f"sobra en {destino}: {r}" for r in sorted(actuales - nuevos)]
    diferencias += [
        f"distinto: {r}" for r in sorted(nuevos & actuales) if not filecmp.cmp(temporal / r, destino / r, shallow=False)
    ]
    return diferencias


def main(argumentos: Sequence[str] | None = None) -> None:
    lector = argparse.ArgumentParser(description="Escribe las macros y las tablas de LaTeX del proyecto.")
    lector.add_argument("--destino", type=Path, default=DESTINO.parent, help="carpeta generado/ del informe")
    lector.add_argument("--comprobar", action="store_true", help="regenera en una carpeta temporal y compara con --destino")
    lector.add_argument("--temporal", type=Path, help="carpeta para --comprobar (por defecto, una temporal nueva)")
    lector.add_argument("--campo", type=Path, default=rutas.METRICAS_CAMPO, help="métricas de la prueba en campo (opcional)")
    opciones = lector.parse_args(argumentos)
    if opciones.comprobar:
        temporal = opciones.temporal or Path(tempfile.mkdtemp(prefix="cifras-"))
        diferencias = comprobar(opciones.destino, temporal, opciones.campo)
        for diferencia in diferencias:
            print(diferencia)
        total = len(archivos_generados(temporal))
        print(f"Sin diferencias: {total} archivos" if not diferencias else f"{len(diferencias)} diferencias en {total} archivos")
        sys.exit(1 if diferencias else 0)
    escritas, macros = escribir_todo(opciones.destino, opciones.campo)
    print(f"{macros} macros y {len(escritas) - 1} tablas en {opciones.destino}")


if __name__ == "__main__":
    main()
