// Interfaz de la app: pestañas, Clasificar, Evaluar y Acerca de. La inferencia es la de inferencia.js,
// la misma de la evaluación del proyecto, con su combinación de librerías por defecto.
//
// Una sola fuente: la URL del modelo elegido sale de resultados/modelos.json y la tabla de canecas de
// config/canecas.json, los dos leídos en ejecución.

import { cargarLibrerias, cargarModelo, cargarImagen, clasificar } from './inferencia.js';
import {
  ErrorCsv,
  csvClasificacion,
  csvEvaluacion,
  emparejarConCsv,
  etiquetasDeCarpetas,
  evaluarFilas,
  leerEtiquetas,
} from './metricas.js';
import { UMBRAL_CANECA_DUDOSA, UMBRAL_LEJOS_DE_LO_APRENDIDO, avisos } from './umbrales.js';

const RUTA_MODELOS = '../resultados/modelos.json';
const RUTA_CANECAS = '../config/canecas.json';
const TIPOS_ADMITIDOS = new Set(['image/jpeg', 'image/png', 'image/webp']);
const EXTENSIONES_ADMITIDAS = /\.(jpe?g|png|webp)$/i;
const HEIC = /\.(heic|heif)$/i;
const MATERIALES_POR_TARJETA = 3;
const NOMBRES_EN_UN_AVISO = 10;
const AVISOS_DE_PROGRESO = 20;

const formatoPorcentaje = new Intl.NumberFormat('es-CO', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 });
const formatoUmbral = new Intl.NumberFormat('es-CO', { style: 'percent', maximumFractionDigits: 0 });
const formatoEntero = new Intl.NumberFormat('es-CO');
const formatoAuc = new Intl.NumberFormat('es-CO', { minimumFractionDigits: 3, maximumFractionDigits: 3 });

const $ = (selector) => document.querySelector(selector);

const estado = {
  canecas: null,
  urlModelo: null,
  modelo: null,
  etiquetas: [],
  entorno: null,
  clasificadas: [],
  urlsMiniaturas: [],
  urlsDescarga: new Map(),
  colaClasificar: Promise.resolve(),
};

// ---------------------------------------------------------------------------------------------
// Utilidades del DOM

// Crea un elemento con clase, texto y atributos; los hijos pueden ser nodos o texto. Nunca usa innerHTML,
// porque los nombres de archivo vienen del usuario.
function crear(etiqueta, { clase, texto, atributos = {} } = {}, hijos = []) {
  const nodo = document.createElement(etiqueta);
  if (clase) {
    nodo.className = clase;
  }
  if (texto !== undefined) {
    nodo.textContent = texto;
  }
  for (const [nombre, valor] of Object.entries(atributos)) {
    nodo.setAttribute(nombre, valor);
  }
  nodo.append(...hijos);
  return nodo;
}

function nombreMaterial(etiqueta) {
  return estado.canecas.materiales[etiqueta].nombre;
}

function nombreCaneca(caneca) {
  return estado.canecas.canecas[caneca].nombre;
}

// Reemplaza el archivo de un enlace de descarga y libera el anterior.
function prepararDescarga(enlace, contenido, tipo) {
  const anterior = estado.urlsDescarga.get(enlace.id);
  if (anterior) {
    URL.revokeObjectURL(anterior);
  }
  const url = URL.createObjectURL(new Blob([contenido], { type: tipo }));
  estado.urlsDescarga.set(enlace.id, url);
  enlace.href = url;
}

// Anuncia el avance sin saturar al lector de pantalla: unos veinte avisos por lote como máximo.
function avisarProgreso(nodo, verbo, hechas, total) {
  const paso = Math.max(1, Math.ceil(total / AVISOS_DE_PROGRESO));
  if (hechas === 1 || hechas === total || hechas % paso === 0) {
    nodo.textContent = `${verbo} ${formatoEntero.format(hechas)} de ${formatoEntero.format(total)}`;
  }
}

async function leerJson(ruta) {
  const respuesta = await fetch(ruta);
  if (!respuesta.ok) {
    throw new Error(`No se pudo leer ${ruta} (${respuesta.status})`);
  }
  return respuesta.json();
}

// ---------------------------------------------------------------------------------------------
// Pestañas (patrón de pestañas de WAI-ARIA, con activación al enfocar)

function configurarPestanas() {
  const pestanas = [...document.querySelectorAll('[role="tab"]')];
  const activar = (elegida, enfocar) => {
    for (const pestana of pestanas) {
      const activa = pestana === elegida;
      pestana.setAttribute('aria-selected', String(activa));
      pestana.tabIndex = activa ? 0 : -1;
      document.getElementById(pestana.getAttribute('aria-controls')).hidden = !activa;
    }
    if (enfocar) {
      elegida.focus();
    }
    history.replaceState(null, '', `#${elegida.dataset.ancla}`);
  };
  const teclas = { ArrowRight: 1, ArrowLeft: -1 };
  pestanas.forEach((pestana, i) => {
    pestana.addEventListener('click', () => activar(pestana, true));
    pestana.addEventListener('keydown', (evento) => {
      let destino = null;
      if (evento.key in teclas) {
        destino = (i + teclas[evento.key] + pestanas.length) % pestanas.length;
      } else if (evento.key === 'Home') {
        destino = 0;
      } else if (evento.key === 'End') {
        destino = pestanas.length - 1;
      }
      if (destino !== null) {
        evento.preventDefault();
        activar(pestanas[destino], true);
      }
    });
  });
  const inicial = pestanas.find((p) => `#${p.dataset.ancla}` === window.location.hash) ?? pestanas[0];
  activar(inicial, false);
}

// ---------------------------------------------------------------------------------------------
// Arranque: datos de configuración, Acerca de y modelo

function llenarAcercaDe(modelos, canecas) {
  const elegido = modelos[modelos.elegido];
  const datos = {
    materiales: `${formatoEntero.format(Object.keys(canecas.materiales).length)} materiales de RealWaste`,
    'umbral-caneca': formatoUmbral.format(UMBRAL_CANECA_DUDOSA),
    'umbral-material': formatoUmbral.format(UMBRAL_LEJOS_DE_LO_APRENDIDO),
    'fuente-canecas': canecas.fuente,
    'descripcion-modelo': elegido.descripcion,
  };
  for (const nodo of document.querySelectorAll('[data-dato]')) {
    nodo.textContent = datos[nodo.dataset.dato];
  }
  $('#enlace-modelo').href = elegido.url;

  const cuerpo = $('#tabla-canecas tbody');
  cuerpo.replaceChildren(
    ...Object.entries(canecas.materiales).map(([etiqueta, material]) =>
      crear('tr', {}, [
        crear('th', { texto: material.nombre, atributos: { scope: 'row' } }),
        crear('td', { texto: etiqueta, atributos: { lang: 'en' } }),
        crear('td', { clase: 'celda-caneca' }, [muestraCaneca(material.caneca), nombreCaneca(material.caneca)]),
        crear('td', { texto: material.fuente }),
      ]),
    ),
  );

  $('#clases-admitidas').textContent = Object.entries(canecas.materiales)
    .map(([etiqueta, material]) => `${material.nombre} (${etiqueta})`)
    .join(', ');
}

function muestraCaneca(caneca) {
  return crear('span', { clase: `muestra muestra-${caneca}`, atributos: { 'aria-hidden': 'true' } });
}

async function iniciarModelo() {
  const estadoModelo = $('#estado-modelo');
  try {
    const [modelos, canecas] = await Promise.all([leerJson(RUTA_MODELOS), leerJson(RUTA_CANECAS)]);
    estado.canecas = canecas;
    estado.urlModelo = modelos[modelos.elegido].url;
    llenarAcercaDe(modelos, canecas);
    estado.entorno = await cargarLibrerias();
    estado.modelo = await cargarModelo(estado.urlModelo);
    estado.etiquetas = estado.modelo.getClassLabels();
    const sinCaneca = estado.etiquetas.filter((e) => !(e in canecas.materiales));
    if (sinCaneca.length) {
      throw new Error(`El modelo trae clases sin caneca: ${sinCaneca.join(', ')}`);
    }
    estadoModelo.textContent = 'Modelo listo. Se descargó de la nube de Teachable Machine y clasifica en tu dispositivo.';
    document.body.dataset.modelo = 'listo';
  } catch (error) {
    estadoModelo.textContent = `No se pudo cargar el modelo: ${error.message}. Revisa la conexión y recarga la página.`;
    document.body.dataset.modelo = 'error';
    throw error;
  }
}

// ---------------------------------------------------------------------------------------------
// Clasificar

// Motivo por el que un archivo no se puede clasificar, o null si el formato es admitido.
function problemaDeFormato(archivo) {
  if (HEIC.test(archivo.name) || /image\/hei[cf]/.test(archivo.type)) {
    return 'el formato HEIC no se puede leer en el navegador. Conviértela a JPEG, PNG o WebP (en el iPhone: Ajustes, Cámara, Formatos, Más compatible).';
  }
  if (!TIPOS_ADMITIDOS.has(archivo.type) && !EXTENSIONES_ADMITIDAS.test(archivo.name)) {
    return `no es una imagen admitida (${archivo.type || 'tipo desconocido'}). Usa JPEG, PNG o WebP.`;
  }
  return null;
}

// Carga y clasifica un archivo de imagen. Devuelve la URL temporal para la miniatura, o la libera si
// `conservarUrl` es falso.
async function clasificarArchivo(archivo, conservarUrl) {
  const url = URL.createObjectURL(archivo);
  try {
    const imagen = await cargarImagen(url);
    const resultado = await clasificar(estado.modelo, imagen, estado.canecas);
    return { ...resultado, url: conservarUrl ? url : null };
  } finally {
    if (!conservarUrl) {
      URL.revokeObjectURL(url);
    }
  }
}

function barrasDeProbabilidad(probabilidades) {
  const primeras = Object.entries(probabilidades)
    .sort(([, a], [, b]) => b - a)
    .slice(0, MATERIALES_POR_TARJETA);
  return crear(
    'ol',
    { clase: 'probabilidades', atributos: { 'aria-label': 'Materiales más probables' } },
    primeras.map(([etiqueta, probabilidad]) => {
      const relleno = crear('span', { clase: 'barra-relleno' });
      relleno.style.width = `${(probabilidad * 100).toFixed(1)}%`;
      return crear('li', {}, [
        crear('span', { clase: 'probabilidad-nombre', texto: nombreMaterial(etiqueta) }),
        crear('span', { clase: 'barra', atributos: { 'aria-hidden': 'true' } }, [relleno]),
        crear('span', { clase: 'probabilidad-valor', texto: formatoPorcentaje.format(probabilidad) }),
      ]);
    }),
  );
}

function porQueDeLaCaneca(resultado) {
  const { material, caneca } = resultado;
  const canecaDelMaterial = estado.canecas.materiales[material].caneca;
  const descripcion = estado.canecas.canecas[caneca].descripcion;
  if (canecaDelMaterial === caneca) {
    return `${nombreMaterial(material)} va a la ${nombreCaneca(caneca).toLowerCase()}. ${descripcion}.`;
  }
  return `${nombreMaterial(material)} va a la ${nombreCaneca(canecaDelMaterial).toLowerCase()}, pero los materiales de la ${nombreCaneca(caneca).toLowerCase()} suman más probabilidad. ${descripcion}.`;
}

function textosDeAvisos(resultado) {
  const textos = [];
  if (resultado.avisos.canecaDudosa) {
    const confianza = formatoPorcentaje.format(resultado.probabilidadesCaneca[resultado.caneca]);
    textos.push(`Caneca dudosa: la confianza de la caneca (${confianza}) es menor que ${formatoUmbral.format(UMBRAL_CANECA_DUDOSA)}. Revisa el objeto o toma otra foto.`);
  }
  if (resultado.avisos.lejosDeLoAprendido) {
    const confianza = formatoPorcentaje.format(resultado.probabilidades[resultado.material]);
    textos.push(`Foto lejos de lo aprendido: el material más probable tiene ${confianza}, menos de ${formatoUmbral.format(UMBRAL_LEJOS_DE_LO_APRENDIDO)}. Puede ser un objeto que el modelo no conoce.`);
  }
  return textos;
}

function tarjetaDeResultado(nombre, resultado) {
  const { material, caneca, probabilidades, probabilidadesCaneca } = resultado;
  const textos = textosDeAvisos(resultado);
  const cuerpo = [
    crear('h3', { clase: 'tarjeta-titulo', texto: nombre }),
    crear('p', { clase: 'material' }, [crear('span', { clase: 'rotulo', texto: 'Material: ' }), crear('strong', { texto: nombreMaterial(material) })]),
    barrasDeProbabilidad(probabilidades),
    crear('div', { clase: `caneca caneca-${caneca}` }, [
      crear('p', { clase: 'caneca-nombre' }, [muestraCaneca(caneca), crear('strong', { texto: nombreCaneca(caneca) })]),
      crear('p', { clase: 'confianza', texto: `Confianza de la caneca: ${formatoPorcentaje.format(probabilidadesCaneca[caneca])}. Confianza del material: ${formatoPorcentaje.format(probabilidades[material])}.` }),
      crear('p', { clase: 'por-que', texto: porQueDeLaCaneca(resultado) }),
    ]),
  ];
  if (textos.length) {
    cuerpo.push(crear('ul', { clase: 'avisos', atributos: { 'aria-label': 'Avisos' } }, textos.map((texto) => crear('li', { texto }))));
  }
  const clase = textos.length ? 'tarjeta tarjeta-con-avisos' : 'tarjeta';
  return crear('li', { clase }, [
    crear('img', { clase: 'miniatura', atributos: { src: resultado.url, alt: `Foto ${nombre}` } }),
    crear('div', { clase: 'tarjeta-cuerpo' }, cuerpo),
  ]);
}

function tarjetaDeError(nombre, motivo) {
  return crear('li', { clase: 'tarjeta tarjeta-error' }, [
    crear('div', { clase: 'tarjeta-cuerpo' }, [
      crear('h3', { clase: 'tarjeta-titulo', texto: nombre }),
      crear('p', { texto: `No se pudo clasificar: ${motivo}` }),
    ]),
  ]);
}

async function clasificarLote(archivos) {
  const panel = $('#panel-clasificar');
  const progreso = $('#clasificar-progreso');
  const lista = $('#clasificar-tarjetas');
  panel.dataset.estado = 'trabajando';
  progreso.textContent = estado.modelo ? '' : 'Esperando a que cargue el modelo…';
  await modeloListo;
  let errores = 0;
  for (const [i, archivo] of archivos.entries()) {
    avisarProgreso(progreso, 'Clasificando', i + 1, archivos.length);
    const problema = problemaDeFormato(archivo);
    if (problema) {
      errores += 1;
      lista.append(tarjetaDeError(archivo.name, problema));
      continue;
    }
    try {
      const resultado = await clasificarArchivo(archivo, true);
      resultado.avisos = avisos(resultado);
      estado.urlsMiniaturas.push(resultado.url);
      estado.clasificadas.push({ archivo: archivo.name, ...resultado });
      lista.append(tarjetaDeResultado(archivo.name, resultado));
    } catch {
      errores += 1;
      lista.append(tarjetaDeError(archivo.name, 'la imagen está dañada o el navegador no la pudo leer.'));
    }
  }
  const clasificadas = archivos.length - errores;
  progreso.textContent = `Listo: ${formatoEntero.format(clasificadas)} de ${formatoEntero.format(archivos.length)} clasificadas${errores ? `; ${formatoEntero.format(errores)} no se pudieron leer` : ''}.`;
  if (estado.clasificadas.length) {
    prepararDescarga($('#clasificar-descargar-csv'), csvClasificacion(estado.clasificadas, estado.etiquetas, estado.canecas), 'text/csv;charset=utf-8');
  }
  $('#clasificar-acciones').hidden = false;
  panel.dataset.estado = 'listo';
}

function encolarClasificacion(archivos) {
  if (archivos.length === 0) {
    return;
  }
  estado.colaClasificar = estado.colaClasificar.then(() => clasificarLote(archivos)).catch(() => {
    $('#clasificar-progreso').textContent = 'No se pudo clasificar porque el modelo no cargó.';
    $('#panel-clasificar').dataset.estado = 'error';
  });
}

function limpiarClasificacion() {
  estado.urlsMiniaturas.forEach((url) => URL.revokeObjectURL(url));
  estado.urlsMiniaturas = [];
  estado.clasificadas = [];
  $('#clasificar-tarjetas').replaceChildren();
  $('#clasificar-progreso').textContent = 'Se limpiaron los resultados.';
  $('#clasificar-acciones').hidden = true;
  $('#clasificar-fotos').value = '';
  $('#panel-clasificar').dataset.estado = 'vacio';
}

function configurarClasificar() {
  const entrada = $('#clasificar-fotos');
  const zona = $('#zona-soltar');
  entrada.addEventListener('change', () => {
    encolarClasificacion([...entrada.files]);
    entrada.value = '';
  });
  zona.addEventListener('dragover', (evento) => {
    evento.preventDefault();
    zona.classList.add('zona-activa');
  });
  zona.addEventListener('dragleave', () => zona.classList.remove('zona-activa'));
  zona.addEventListener('drop', (evento) => {
    evento.preventDefault();
    zona.classList.remove('zona-activa');
    encolarClasificacion([...evento.dataTransfer.files]);
  });
  $('#clasificar-limpiar').addEventListener('click', () => estado.colaClasificar.then(limpiarClasificacion));
}

// ---------------------------------------------------------------------------------------------
// Evaluar

function listaAcotada(nombres) {
  const primeros = nombres.slice(0, NOMBRES_EN_UN_AVISO).join(', ');
  const resto = nombres.length - NOMBRES_EN_UN_AVISO;
  return resto > 0 ? `${primeros} y ${formatoEntero.format(resto)} más` : primeros;
}

function mostrarAvisos(avisosDeEvaluacion, esError = false) {
  const contenedor = $('#evaluar-avisos');
  if (avisosDeEvaluacion.length === 0) {
    contenedor.replaceChildren();
    return;
  }
  const atributos = esError ? { role: 'alert' } : {};
  contenedor.replaceChildren(
    crear('ul', { clase: esError ? 'avisos avisos-error' : 'avisos', atributos }, avisosDeEvaluacion.map((texto) => crear('li', { texto }))),
  );
}

// «1 foto no se evaluó», «3 fotos no se evaluaron»: el número con la frase que concuerda con él.
function segun(n, singular, plural) {
  return `${formatoEntero.format(n)} ${n === 1 ? singular : plural}`;
}

function avisosDelEmparejamiento(emparejamiento, ilegibles) {
  const { noAdmitidas, sinEtiqueta, desconocidas, repetidos, sinFoto } = emparejamiento;
  const textos = [];
  if (noAdmitidas.length) {
    textos.push(`${segun(noAdmitidas.length, 'archivo no es una imagen admitida y no se evaluó', 'archivos no son imágenes admitidas y no se evaluaron')}: ${listaAcotada(noAdmitidas)}.`);
  }
  if (sinEtiqueta.length) {
    textos.push(`${segun(sinEtiqueta.length, 'foto no tiene etiqueta y no se evaluó', 'fotos no tienen etiqueta y no se evaluaron')}: ${listaAcotada(sinEtiqueta)}.`);
  }
  if (desconocidas.length) {
    const clases = [...new Set(desconocidas.map((d) => d.clase))];
    textos.push(`${segun(desconocidas.length, 'foto no se evaluó porque su clase no se reconoce', 'fotos no se evaluaron porque su clase no se reconoce')}: ${listaAcotada(clases)}. Usa una etiqueta de RealWaste o su nombre en español.`);
  }
  if (repetidos.length) {
    textos.push(`${segun(repetidos.length, 'nombre de archivo aparece más de una vez en el CSV y su foto no se evaluó', 'nombres de archivo aparecen más de una vez en el CSV y sus fotos no se evaluaron')}: ${listaAcotada(repetidos)}.`);
  }
  if (sinFoto.length) {
    textos.push(`${segun(sinFoto.length, 'fila del CSV no tiene foto elegida', 'filas del CSV no tienen foto elegida')}: ${listaAcotada(sinFoto)}.`);
  }
  if (ilegibles.length) {
    textos.push(`${segun(ilegibles.length, 'foto no se pudo leer', 'fotos no se pudieron leer')}: ${listaAcotada(ilegibles)}.`);
  }
  return textos;
}

// Separa los archivos que no son imágenes admitidas antes de emparejarlos.
function separarAdmitidas(archivos) {
  const admitidas = archivos.filter((a) => problemaDeFormato(a) === null);
  const noAdmitidas = archivos.filter((a) => problemaDeFormato(a) !== null).map((a) => a.webkitRelativePath || a.name);
  return { admitidas, noAdmitidas };
}

function celdaPorcentaje(valor) {
  return crear('td', { texto: formatoPorcentaje.format(valor) });
}

function celdaAuc(valor) {
  return crear('td', { texto: valor === null ? 'n/d' : formatoAuc.format(valor) });
}

function region(id, tabla) {
  return crear('div', { clase: 'tabla-desplazable', atributos: { role: 'region', 'aria-labelledby': id, tabindex: '0' } }, [tabla]);
}

function tablaPorClase(tarea, nombres, titulo, id) {
  const { valores, soporte, clases } = tarea;
  const total = Object.values(soporte).reduce((a, b) => a + b, 0);
  const filas = clases.map((clase) =>
    crear('tr', {}, [
      crear('th', { texto: nombres(clase), atributos: { scope: 'row' } }),
      celdaPorcentaje(valores.precision[clase]),
      celdaPorcentaje(valores.recall[clase]),
      celdaPorcentaje(valores.f1[clase]),
      celdaAuc(valores.auc[clase]),
      crear('td', { texto: formatoEntero.format(soporte[clase]) }),
    ]),
  );
  const promedio = (rotulo, p, r, f, a) =>
    crear('tr', { clase: 'fila-promedio' }, [
      crear('th', { texto: rotulo, atributos: { scope: 'row' } }),
      celdaPorcentaje(p),
      celdaPorcentaje(r),
      celdaPorcentaje(f),
      a === undefined ? crear('td', { texto: 'no aplica' }) : celdaAuc(a),
      crear('td', { texto: formatoEntero.format(total) }),
    ]);
  const tabla = crear('table', { clase: 'tabla-metricas' }, [
    crear('caption', { texto: titulo, atributos: { id } }),
    crear('thead', {}, [
      crear('tr', {}, ['Clase', 'Precisión', 'Recall', 'F1', 'AUC', 'Soporte'].map((t) => crear('th', { texto: t, atributos: { scope: 'col' } }))),
    ]),
    crear('tbody', {}, [
      ...filas,
      promedio('Promedio macro', valores.precision_macro, valores.recall_macro, valores.f1_macro, valores.auc_macro),
      promedio('Promedio ponderado', valores.precision_ponderada, valores.recall_ponderado, valores.f1_ponderado),
    ]),
  ]);
  return region(id, tabla);
}

function tablaMatriz(tarea, nombres, titulo, id) {
  const { matriz, clases } = tarea;
  const encabezado = crear('tr', {}, [
    crear('td', { clase: 'esquina', texto: 'Real \\ Predicha' }),
    ...clases.map((clase) => crear('th', { texto: nombres(clase), atributos: { scope: 'col' } })),
  ]);
  const filas = matriz.map((fila, i) => {
    const totalFila = fila.reduce((a, b) => a + b, 0);
    return crear('tr', {}, [
      crear('th', { texto: nombres(clases[i]), atributos: { scope: 'row' } }),
      ...fila.map((conteo, j) =>
        crear('td', { clase: i === j ? 'acierto' : '' }, [
          crear('span', { clase: 'conteo', texto: formatoEntero.format(conteo) }),
          crear('span', { clase: 'porcentaje-fila', texto: totalFila ? formatoPorcentaje.format(conteo / totalFila) : 'n/d' }),
        ]),
      ),
    ]);
  });
  const tabla = crear('table', { clase: 'matriz' }, [
    crear('caption', { texto: titulo, atributos: { id } }),
    crear('thead', {}, [encabezado]),
    crear('tbody', {}, filas),
  ]);
  return region(id, tabla);
}

function resumenDeTarea(titulo, valores) {
  const cifra = (rotulo, valor) => crear('div', { clase: 'cifra' }, [crear('dt', { texto: rotulo }), crear('dd', { texto: formatoPorcentaje.format(valor) })]);
  return crear('div', { clase: 'resumen-tarea' }, [
    crear('h4', { texto: titulo }),
    crear('dl', { clase: 'cifras' }, [
      cifra('Exactitud', valores.exactitud),
      cifra('Exactitud balanceada', valores.exactitud_balanceada),
      cifra('F1 macro', valores.f1_macro),
    ]),
  ]);
}

function mostrarResultados(metricas) {
  const contenedor = $('#evaluar-resultados');
  const csv = crear('a', { clase: 'boton boton-secundario', texto: 'Descargar CSV', atributos: { id: 'evaluar-descargar-csv', download: 'evaluacion-por-foto.csv', href: '#' } });
  const json = crear('a', { clase: 'boton boton-secundario', texto: 'Descargar JSON', atributos: { id: 'evaluar-descargar-json', download: 'evaluacion-metricas.json', href: '#' } });
  contenedor.replaceChildren(
    crear('h3', { texto: 'Resultados' }),
    crear('p', { texto: `Se evaluaron ${formatoEntero.format(metricas.imagenes)} fotos.` }),
    crear('div', { clase: 'resumen' }, [resumenDeTarea('Por material', metricas.material.valores), resumenDeTarea('Por caneca', metricas.caneca.valores)]),
    crear('h4', { texto: 'Métricas por clase' }),
    tablaPorClase(metricas.material, nombreMaterial, 'Métricas por material', 'titulo-tabla-material'),
    tablaPorClase(metricas.caneca, nombreCaneca, 'Métricas por caneca', 'titulo-tabla-caneca'),
    crear('h4', { texto: 'Matrices de confusión' }),
    crear('p', { texto: 'Filas: clase real; columnas: clase predicha. Debajo de cada conteo va el porcentaje de su fila; la diagonal, marcada, son los aciertos.' }),
    tablaMatriz(metricas.material, nombreMaterial, 'Matriz de confusión de los materiales', 'titulo-matriz-material'),
    tablaMatriz(metricas.caneca, nombreCaneca, 'Matriz de confusión de las canecas', 'titulo-matriz-caneca'),
    crear('h4', { texto: 'Descargas' }),
    crear('p', { texto: 'El CSV trae una fila por foto con las probabilidades; el JSON, todas las métricas.' }),
    crear('div', { clase: 'acciones' }, [csv, json]),
  );
  contenedor.hidden = false;
  return { csv, json };
}

async function evaluarEmparejamiento(emparejamiento) {
  const panel = $('#panel-evaluar');
  const progreso = $('#evaluar-progreso');
  if (emparejamiento.evaluables.length === 0) {
    mostrarAvisos(['No hay fotos con una etiqueta válida para evaluar.', ...avisosDelEmparejamiento(emparejamiento, [])], true);
    progreso.textContent = '';
    panel.dataset.estado = 'error';
    return;
  }
  progreso.textContent = estado.modelo ? '' : 'Esperando a que cargue el modelo…';
  await modeloListo;
  const filas = [];
  const ilegibles = [];
  const total = emparejamiento.evaluables.length;
  for (const [i, { archivo, etiqueta, foto }] of emparejamiento.evaluables.entries()) {
    avisarProgreso(progreso, 'Evaluando', i + 1, total);
    try {
      const resultado = await clasificarArchivo(foto, false);
      filas.push({ archivo, clase_real: etiqueta, ...resultado });
    } catch {
      ilegibles.push(archivo);
    }
  }
  mostrarAvisos(avisosDelEmparejamiento(emparejamiento, ilegibles));
  if (filas.length === 0) {
    progreso.textContent = 'Ninguna foto se pudo leer.';
    panel.dataset.estado = 'error';
    return;
  }
  const metricas = evaluarFilas(filas, estado.etiquetas, estado.canecas);
  const enlaces = mostrarResultados(metricas);
  const informe = {
    modelo: estado.urlModelo,
    entorno: estado.entorno,
    fecha: new Date().toISOString(),
    ...metricas,
    sin_evaluar: {
      no_admitidas: emparejamiento.noAdmitidas,
      sin_etiqueta: emparejamiento.sinEtiqueta,
      clase_desconocida: emparejamiento.desconocidas,
      repetidas_en_el_csv: emparejamiento.repetidos,
      ilegibles,
    },
  };
  prepararDescarga(enlaces.csv, csvEvaluacion(filas, estado.etiquetas, estado.canecas), 'text/csv;charset=utf-8');
  prepararDescarga(enlaces.json, `${JSON.stringify(informe, null, 2)}\n`, 'application/json');
  progreso.textContent = `Listo: ${formatoEntero.format(filas.length)} fotos evaluadas.`;
  panel.dataset.estado = 'listo';
}

// Corre una evaluación a la vez; `preparar` arma el emparejamiento o lanza ErrorCsv.
async function correrEvaluacion(preparar) {
  const panel = $('#panel-evaluar');
  if (panel.dataset.estado === 'trabajando') {
    return;
  }
  panel.dataset.estado = 'trabajando';
  $('#evaluar-resultados').hidden = true;
  mostrarAvisos([]);
  try {
    await evaluarEmparejamiento(await preparar());
  } catch (error) {
    const mensaje = error instanceof ErrorCsv ? `El CSV no se puede leer: ${error.message}` : `No se pudo evaluar: ${error.message}`;
    mostrarAvisos([mensaje], true);
    $('#evaluar-progreso').textContent = '';
    panel.dataset.estado = 'error';
  }
}

function configurarEvaluar() {
  $('#form-csv').addEventListener('submit', (evento) => {
    evento.preventDefault();
    correrEvaluacion(async () => {
      await modeloListo;
      const texto = await $('#evaluar-csv').files[0].text();
      const lectura = leerEtiquetas(texto, estado.canecas);
      const { admitidas, noAdmitidas } = separarAdmitidas([...$('#evaluar-imagenes').files]);
      return { ...emparejarConCsv(admitidas, lectura), noAdmitidas };
    });
  });
  $('#form-carpeta').addEventListener('submit', (evento) => {
    evento.preventDefault();
    correrEvaluacion(async () => {
      await modeloListo;
      const { admitidas, noAdmitidas } = separarAdmitidas([...$('#evaluar-carpeta').files]);
      return { ...etiquetasDeCarpetas(admitidas, estado.canecas), noAdmitidas };
    });
  });
}

// ---------------------------------------------------------------------------------------------

configurarPestanas();
configurarClasificar();
configurarEvaluar();
const modeloListo = iniciarModelo();
modeloListo.catch(() => {});
