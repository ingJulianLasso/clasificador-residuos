# Clasificador de residuos

Aplicación web que reconoce en una foto el material de un residuo y lo traduce a la caneca del código de colores de la Resolución 2184 de 2019: blanca para lo aprovechable, verde para lo orgánico aprovechable y negra para lo no aprovechable. El modelo es un clasificador de imágenes entrenado con Teachable Machine sobre el conjunto de datos RealWaste y alojado en la nube de Teachable Machine. La clasificación ocurre en el navegador, así que las fotos no salen del dispositivo.

**Aplicación:** <https://ingjulianlasso.github.io/clasificador-residuos/>

La aplicación tiene tres pestañas. **Clasificar** recibe una o varias fotos (en el celular, también desde la cámara), muestra el material, la caneca y sus probabilidades, y permite descargar los resultados en CSV. **Evaluar** mide el modelo con fotos que el usuario ya tiene etiquetadas, sea con un CSV `archivo,clase` o con una carpeta que tenga una subcarpeta por clase, y muestra las métricas y las matrices de confusión por material y por caneca. **Acerca de** explica la correspondencia de canecas, el modelo, los límites y las licencias.

## Cómo decide la caneca

El modelo asigna una probabilidad a cada material de RealWaste. La aplicación suma las probabilidades de los materiales que van a cada caneca y elige la caneca con la suma más alta. Por eso la caneca puede quedar más segura que el material: si el modelo duda entre plástico y vidrio, las dos opciones van a la blanca.

| Material | Etiqueta en RealWaste | Caneca |
| --- | --- | --- |
| Cartón | Cardboard | Blanca |
| Orgánicos de comida | Food Organics | Verde |
| Vidrio | Glass | Blanca |
| Metal | Metal | Blanca |
| Basura miscelánea | Miscellaneous Trash | Negra |
| Papel | Paper | Blanca |
| Plástico | Plastic | Blanca |
| Textiles | Textile Trash | Blanca |
| Vegetación | Vegetation | Verde |

La correspondencia, con la fuente normativa de cada fila, está en [`config/canecas.json`](config/canecas.json); la aplicación y la evaluación la leen de ese único archivo. La aplicación muestra además dos avisos: «caneca dudosa», cuando la probabilidad de la caneca elegida no llega a su umbral, y «foto lejos de lo aprendido», cuando la probabilidad del material más probable no llega al suyo. Los dos umbrales se fijaron antes de ver los resultados de la prueba y están en [`web/umbrales.js`](web/umbrales.js).

## Cómo se entrenó y se evaluó

**Partición.** Las 4.752 imágenes de RealWaste se dividieron por clase en un 70 % de entrenamiento y un 30 % de prueba, con semilla fija. Las fotos casi idénticas (huella perceptual con `imagehash`) forman grupos que van enteros a un solo lado, para que ninguna foto de prueba tenga una gemela en el entrenamiento. La partición está en [`datos/particion.csv`](datos/particion.csv) y el inventario de la descarga, en [`datos/inventario.json`](datos/inventario.json).

**Entrenamiento.** En Teachable Machine se entrenaron dos modelos: uno con el 70 % tal cual y otro con ese 70 % submuestreado al tamaño de la clase menor. La regla para elegir entre los dos se fijó antes de entrenar y no usa el conjunto de prueba: gana el de mayor exactitud media por clase en la validación interna de Teachable Machine. Ganó el balanceado. Los parámetros de cada entrenamiento están en [`resultados/tm-entrenamiento-1.json`](resultados/tm-entrenamiento-1.json) y [`resultados/tm-entrenamiento-2.json`](resultados/tm-entrenamiento-2.json); los enlaces de los dos modelos y la regla, en [`resultados/modelos.json`](resultados/modelos.json).

**Evaluación.** Los dos modelos se evaluaron sobre las 1.425 imágenes de prueba, en el navegador y con la misma librería que usa la aplicación. Las cifras oficiales son las del modelo elegido, cargado desde el mismo enlace público que usa la aplicación. Los intervalos son de confianza al 95 %, por bootstrap percentil con 1.000 remuestreos de las fotos de prueba.

| Métrica | Valor | IC 95 % |
| --- | --- | --- |
| F1 macro por material | 0,796 | [0,776; 0,813] |
| Exactitud balanceada por material | 0,811 | [0,793; 0,829] |
| Precisión de la caneca blanca | 0,955 | [0,943; 0,968] |

La precisión de la blanca es la fracción de las fotos enviadas a la blanca que de verdad van allí. Pesa más que las otras porque el error caro es mandar a la blanca algo orgánico o no aprovechable, que contamina el material que sí se puede recuperar.

Los resultados completos están en estos archivos:

- [`resultados/metricas-elegido.json`](resultados/metricas-elegido.json): métricas por material y por caneca, con intervalos, matrices de confusión, errores más confiados, curva de cobertura y precisión de la blanca y comparación de los dos entrenamientos con la prueba de McNemar.
- [`resultados/metricas-entrenamiento-1.json`](resultados/metricas-entrenamiento-1.json) y [`resultados/metricas-entrenamiento-2.json`](resultados/metricas-entrenamiento-2.json): cada entrenamiento sobre el conjunto de prueba.
- [`resultados/metricas-robustez.json`](resultados/metricas-robustez.json): el modelo elegido sobre seis versiones alteradas de las fotos de prueba (más oscuras, más claras, desenfocadas, giradas, acercadas y comprimidas).
- [`resultados/app-paridad.json`](resultados/app-paridad.json): la aplicación frente a la evaluación, foto por foto, sobre el mismo conjunto de prueba.
- Los CSV `resultados/probabilidades-*.csv`, con las probabilidades de cada foto, y su `.entorno.json`, con el navegador, la librería y la GPU de cada corrida.

## Cómo reproducirlo

Se necesita [uv](https://docs.astral.sh/uv/), que instala la versión de Python de `.python-version`; Node.js, para las pruebas de `web/`; y un navegador basado en Chromium (Chrome, Chromium o Brave) con WebGL, para la evaluación y las pruebas del navegador.

Las imágenes de RealWaste no se publican aquí: se descargan del repositorio de UCI y van en una carpeta `datos/` al lado del clon. Desde una carpeta de trabajo:

```bash
mkdir -p datos/descarga
git clone https://github.com/ingJulianLasso/clasificador-residuos.git clasificador
curl -L --fail -o datos/descarga/realwaste.zip https://archive.ics.uci.edu/static/public/908/realwaste.zip
cd clasificador
uv sync
uv run python -m residuos.datos
```

`residuos.datos` extrae el ZIP en `../datos/realwaste/`, revisa duplicados y escribe la partición y el inventario.

El entrenamiento se hace en la interfaz web de Teachable Machine con las fotos de entrenamiento de `datos/particion.csv` (el segundo modelo, solo con las marcadas en la columna `balanceado`). Los dos modelos ya están publicados, así que la evaluación puede repetirse directamente con sus enlaces. El arnés se conecta por CDP a un navegador abierto con `--remote-debugging-port=9333` (por ejemplo, `chromium --remote-debugging-port=9333 --user-data-dir=<carpeta-de-perfil>`) y sirve los archivos en `127.0.0.1:8765`:

```bash
MODELO=https://teachablemachine.withgoogle.com/models/9PKzru-J4/
uv run python -m residuos.arnes evaluar --modelo "$MODELO" \
    --lista datos/particion.csv --particion prueba \
    --salida resultados/probabilidades-prueba-elegido.csv
uv run python -m residuos.robustez
for p in oscura clara desenfocada girada acercada comprimida; do
    uv run python -m residuos.arnes evaluar --modelo "$MODELO" \
        --lista ../datos/robustez/$p/lista.csv --base ../datos/robustez/$p \
        --salida resultados/probabilidades-robustez-$p.csv
done
uv run python -m residuos.metricas
uv run python -m residuos.figuras
```

`residuos.robustez` escribe las versiones alteradas en `../datos/robustez/`; `residuos.metricas`, los JSON de `resultados/`; y `residuos.figuras`, las gráficas en `../figuras/`.

## Pruebas

```bash
uv run pytest -m "not navegador"
node --test web/pruebas/*.test.mjs
uv run python -m residuos.fixture_app
APP_URL=http://127.0.0.1:8766/web/ uv run pytest -s -m navegador tests/test_app.py
```

La primera línea corre las pruebas que no abren el navegador. Las que necesitan las imágenes extraídas o las figuras se saltan mientras esos archivos no existan. En un clon también se saltan, con su motivo, las de la paridad con la vista previa original de Teachable Machine, las de la prueba en campo que leen su registro y las que regeneran las cifras del informe, porque esos archivos no se publican. La segunda compara las funciones de métricas de la aplicación con valores calculados con scikit-learn. Las dos últimas preparan la lista de etiquetas de prueba y corren las pruebas del navegador, que necesitan el navegador con CDP en el puerto 9333 y se saltan si no responde. Con `APP_URL` en la dirección local, la prueba sirve esta carpeta en `127.0.0.1:8766`; con la dirección publicada, prueba la aplicación en GitHub Pages. Comprueba el origen del modelo, que la aplicación dé foto por foto las mismas probabilidades que la evaluación, el uso en el celular y la accesibilidad con axe-core, y escribe `resultados/app-paridad.json`.

## Límites

- Es un prototipo de laboratorio, en el nivel 4 de madurez tecnológica (TRL 4): se validó con las fotos de prueba del conjunto de datos, no en uso real.
- Aprendió de fotos de RealWaste, tomadas en el relleno sanitario de Whyte's Gully, en Wollongong (Australia), donde los residuos llegan mezclados y deformados. Las fotos de una cocina, con otro fondo y otra luz, pueden confundirlo; el efecto de la luz, el enfoque, el giro y la compresión está medido en `resultados/metricas-robustez.json`.
- Una foto no muestra si el objeto tiene grasa o restos de comida, si está contaminado o si mezcla materiales. Esos residuos pueden ir a la negra aunque su material sea aprovechable.
- La caneca verde sirve si el municipio recoge los orgánicos por una ruta selectiva. Si no la hay, manda la indicación local.
- La aplicación depende del modelo alojado en Teachable Machine y de las librerías que sirve jsDelivr. Si alguno deja de estar disponible, no puede clasificar.

## Licencias y cita del conjunto de datos

El código de este repositorio está bajo la licencia MIT (ver [`LICENSE`](LICENSE)). Esa licencia no cubre las imágenes de RealWaste ni el modelo entrenado con ellas.

RealWaste tiene dos licencias declaradas que no coinciden. La página de UCI lo publica con CC BY 4.0, pero el README de los autores, que viaja dentro de la misma descarga, y el artículo que lo presenta dicen CC BY-NC-SA 4.0. Ante ese conflicto se adopta la más restrictiva: uso académico no comercial, con atribución y sin redistribuir las imágenes. Por eso este repositorio no incluye fotos de RealWaste; `datos/particion.csv` solo trae el nombre, la clase y la huella SHA-256 de cada archivo.

Para citar el conjunto de datos y el artículo que lo presenta:

- Single, S., Iranmanesh, S., y Raad, R. (2023a). *RealWaste* [Conjunto de datos]. UCI Machine Learning Repository. <https://doi.org/10.24432/C5SS4G>
- Single, S., Iranmanesh, S., y Raad, R. (2023b). RealWaste: A novel real-life data set for landfill waste classification using deep learning. *Information, 14*(12), Artículo 633. <https://doi.org/10.3390/info14120633>

## Autor

Julián Andrés Lasso Figueroa, Maestría en Inteligencia Artificial, Universidad de La Salle. Proyecto final del curso de Visión por Computador.
