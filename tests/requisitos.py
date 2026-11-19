"""Marcas para saltar las pruebas que dependen de archivos que no viajan en el repositorio público.

En la copia de trabajo esos archivos existen y las pruebas corren; en un clon limpio se saltan con el
motivo a la vista, como las que necesitan las imágenes extraídas o las figuras.
"""

import pytest

from residuos import cifras, rutas
from residuos.arnes import PARIDAD_A2, ruta_entorno

MOTIVO_PARIDAD = "la paridad con la vista previa original usa archivos locales que no se publican"

PARIDAD_REGISTRADA = all(
    ruta.is_file() for ruta in (PARIDAD_A2.probabilidades, PARIDAD_A2.salida, ruta_entorno(PARIDAD_A2.salida))
)
PARIDAD_EJECUTABLE = PARIDAD_A2.modelo.is_dir() and PARIDAD_A2.imagenes.is_dir() and PARIDAD_A2.probabilidades.is_file()

requiere_paridad = pytest.mark.skipif(not PARIDAD_REGISTRADA, reason=MOTIVO_PARIDAD)
requiere_paridad_ejecutable = pytest.mark.skipif(not PARIDAD_EJECUTABLE, reason=MOTIVO_PARIDAD)
requiere_informe = pytest.mark.skipif(
    not cifras.DESTINO.is_file(), reason="cifras.tex del informe no está generado (el informe no se publica)"
)
requiere_registro_campo = pytest.mark.skipif(
    not rutas.REGISTRO_CAMPO.is_file(), reason="el registro de la prueba en campo no se publica"
)
