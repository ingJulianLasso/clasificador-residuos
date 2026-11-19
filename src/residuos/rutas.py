"""Rutas del proyecto, resueltas desde la ubicación del paquete."""

from pathlib import Path

RAIZ_CLASIFICADOR = Path(__file__).resolve().parents[2]
RAIZ_PROYECTO = RAIZ_CLASIFICADOR.parent

# Lo pesado vive fuera del clasificador y no se versiona.
DATOS_PESADOS = RAIZ_PROYECTO / "datos"
ZIP_REALWASTE = DATOS_PESADOS / "descarga" / "realwaste.zip"
IMAGENES = DATOS_PESADOS / "realwaste"

# Lo que se versiona con el clasificador.
CANECAS = RAIZ_CLASIFICADOR / "config" / "canecas.json"
PARTICION = RAIZ_CLASIFICADOR / "datos" / "particion.csv"
INVENTARIO = RAIZ_CLASIFICADOR / "datos" / "inventario.json"

# Destino de las macros del informe.
TEX_GENERADO = RAIZ_PROYECTO.parent / "entregas" / "actividad-5" / "informe" / "latex" / "generado"

# Modelos exportados (copias locales y descargas de los alojados) y salidas de la evaluación.
MODELOS = RAIZ_PROYECTO / "modelo"
ROBUSTEZ = DATOS_PESADOS / "robustez"
FIGURAS = RAIZ_PROYECTO / "figuras"
RESULTADOS = RAIZ_CLASIFICADOR / "resultados"
MODELOS_JSON = RESULTADOS / "modelos.json"
PARIDAD_A2 = RESULTADOS / "paridad-a2.csv"

# Prueba en campo: las fotos propias no se versionan; el registro y las métricas sí.
CAMPO = DATOS_PESADOS / "campo"
REGISTRO_CAMPO = RAIZ_PROYECTO / "prueba-campo" / "registro.csv"
PROBABILIDADES_CAMPO = RESULTADOS / "probabilidades-campo-elegido.csv"
METRICAS_CAMPO = RESULTADOS / "metricas-campo.json"


def entrenamiento_json(numero: int) -> Path:
    return RESULTADOS / f"tm-entrenamiento-{numero}.json"
