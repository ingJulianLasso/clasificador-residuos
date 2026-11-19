// Lectura de etiquetas, métricas de clasificación y archivos de descarga de la app. No toca el DOM,
// así que `node --test` lo compara con scikit-learn (web/pruebas/metricas.test.mjs).
//
// Los archivos llegan como objetos con `name` y, si vienen de una carpeta, `webkitRelativePath`.
// Las métricas siguen las definiciones de scikit-learn: matriz con filas reales y columnas predichas,
// división por cero igual a 0, promedios sobre las clases presentes en lo real o en lo predicho,
// exactitud balanceada sobre las clases con soporte y AUC de una clase contra el resto.

export class ErrorCsv extends Error {}

// ---------------------------------------------------------------------------------------------
// CSV

// Separador de la primera línea: punto y coma si solo trae punto y coma (CSV de Excel en español).
function separadorDe(texto) {
  const primera = texto.split(/\r?\n/, 1)[0];
  return primera.includes(';') && !primera.includes(',') ? ';' : ',';
}

// RFC 4180: campos entre comillas, comillas dobladas, saltos CRLF o LF y marca BOM. Omite las líneas vacías.
export function leerCsv(texto) {
  const contenido = texto.replace(/^﻿/, '');
  const separador = separadorDe(contenido);
  const filas = [];
  let fila = [];
  let campo = '';
  let entreComillas = false;
  const cerrarFila = () => {
    fila.push(campo);
    if (!(fila.length === 1 && fila[0].trim() === '')) {
      filas.push(fila);
    }
    fila = [];
    campo = '';
  };
  for (let i = 0; i < contenido.length; i += 1) {
    const caracter = contenido[i];
    if (entreComillas) {
      if (caracter === '"' && contenido[i + 1] === '"') {
        campo += '"';
        i += 1;
      } else if (caracter === '"') {
        entreComillas = false;
      } else {
        campo += caracter;
      }
    } else if (caracter === '"' && campo === '') {
      entreComillas = true;
    } else if (caracter === separador) {
      fila.push(campo);
      campo = '';
    } else if (caracter === '\n' || caracter === '\r') {
      if (caracter === '\r' && contenido[i + 1] === '\n') {
        i += 1;
      }
      cerrarFila();
    } else {
      campo += caracter;
    }
  }
  if (entreComillas) {
    throw new ErrorCsv('Hay unas comillas que abren y no cierran.');
  }
  if (campo !== '' || fila.length > 0) {
    cerrarFila();
  }
  return filas;
}

function celdaCsv(valor) {
  const texto = String(valor);
  return /[",\r\n]/.test(texto) ? `"${texto.replaceAll('"', '""')}"` : texto;
}

export function aCsv(filas) {
  return `${filas.map((fila) => fila.map(celdaCsv).join(',')).join('\n')}\n`;
}

// ---------------------------------------------------------------------------------------------
// Clases y etiquetas

// Sin tildes, sin mayúsculas, con guiones y guiones bajos como espacios: «Food_Organics» = «food organics».
export function normalizar(texto) {
  return texto
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .replace(/[_-]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();
}

// Etiqueta de RealWaste a partir de la etiqueta misma o del nombre en español de canecas.json.
export function resolverClase(texto, canecas) {
  const buscado = normalizar(texto);
  for (const [etiqueta, material] of Object.entries(canecas.materiales)) {
    if (normalizar(etiqueta) === buscado || normalizar(material.nombre) === buscado) {
      return etiqueta;
    }
  }
  return null;
}

export function nombreBase(ruta) {
  return ruta.split(/[\\/]/).pop();
}

// Lee el CSV de etiquetas (`archivo,clase`). Devuelve una entrada por nombre de archivo, sin la carpeta,
// y aparte los nombres repetidos, que no se pueden emparejar con una sola foto.
export function leerEtiquetas(texto, canecas) {
  const [encabezado, ...filas] = leerCsv(texto);
  if (!encabezado) {
    throw new ErrorCsv('El CSV está vacío.');
  }
  const columnas = encabezado.map(normalizar);
  const posArchivo = columnas.indexOf('archivo');
  const posClase = columnas.indexOf('clase');
  if (posArchivo < 0 || posClase < 0) {
    throw new ErrorCsv('La primera fila debe ser el encabezado con las columnas archivo y clase.');
  }
  if (filas.length === 0) {
    throw new ErrorCsv('El CSV tiene encabezado, pero ninguna fila de etiquetas.');
  }
  const entradas = new Map();
  const repetidos = new Set();
  filas.forEach((fila, i) => {
    const numero = i + 2;
    if (fila.length !== encabezado.length) {
      throw new ErrorCsv(`La fila ${numero} tiene ${fila.length} columnas y el encabezado tiene ${encabezado.length}.`);
    }
    const archivo = fila[posArchivo].trim();
    const clase = fila[posClase].trim();
    if (!archivo || !clase) {
      throw new ErrorCsv(`A la fila ${numero} le falta el archivo o la clase.`);
    }
    const clave = nombreBase(archivo);
    if (entradas.has(clave)) {
      repetidos.add(clave);
    }
    entradas.set(clave, { archivo, clase, etiqueta: resolverClase(clase, canecas) });
  });
  for (const clave of repetidos) {
    entradas.delete(clave);
  }
  return { entradas, repetidos: [...repetidos] };
}

// Empareja las fotos elegidas con el CSV por nombre de archivo. Las de nombre repetido en el CSV quedan
// solo en `repetidos`.
export function emparejarConCsv(archivos, lectura) {
  const resultado = { evaluables: [], sinEtiqueta: [], desconocidas: [], sinFoto: [], repetidos: lectura.repetidos };
  const usadas = new Set();
  const repetidos = new Set(lectura.repetidos);
  for (const archivo of archivos) {
    const clave = nombreBase(archivo.name);
    const entrada = lectura.entradas.get(clave);
    if (repetidos.has(clave)) {
      continue;
    }
    if (!entrada) {
      resultado.sinEtiqueta.push(archivo.name);
    } else if (entrada.etiqueta === null) {
      resultado.desconocidas.push({ archivo: entrada.archivo, clase: entrada.clase });
    } else {
      resultado.evaluables.push({ archivo: entrada.archivo, etiqueta: entrada.etiqueta, foto: archivo });
    }
    usadas.add(clave);
  }
  for (const [clave, entrada] of lectura.entradas) {
    if (!usadas.has(clave)) {
      resultado.sinFoto.push(entrada.archivo);
    }
  }
  return resultado;
}

// La clase de cada foto es el nombre de la carpeta que la contiene.
export function etiquetasDeCarpetas(archivos, canecas) {
  const resultado = { evaluables: [], sinEtiqueta: [], desconocidas: [], sinFoto: [], repetidos: [] };
  for (const archivo of archivos) {
    const ruta = archivo.webkitRelativePath || archivo.name;
    const partes = ruta.split('/');
    if (partes.length < 2) {
      resultado.sinEtiqueta.push(ruta);
      continue;
    }
    const carpeta = partes.at(-2);
    const etiqueta = resolverClase(carpeta, canecas);
    if (etiqueta === null) {
      resultado.desconocidas.push({ archivo: ruta, clase: carpeta });
    } else {
      resultado.evaluables.push({ archivo: ruta, etiqueta, foto: archivo });
    }
  }
  return resultado;
}

// ---------------------------------------------------------------------------------------------
// Métricas

const suma = (valores) => valores.reduce((total, valor) => total + valor, 0);
const media = (valores) => (valores.length ? suma(valores) / valores.length : null);
const dividir = (numerador, denominador) => (denominador > 0 ? numerador / denominador : 0);

export function matrizConfusion(reales, predichas, k) {
  const matriz = Array.from({ length: k }, () => new Array(k).fill(0));
  reales.forEach((real, i) => {
    matriz[real][predichas[i]] += 1;
  });
  return matriz;
}

export function metricasDeMatriz(matriz) {
  const k = matriz.length;
  const aciertos = matriz.map((fila, i) => fila[i]);
  const soporte = matriz.map(suma);
  const asignadas = matriz.map((_, j) => suma(matriz.map((fila) => fila[j])));
  const total = suma(soporte);
  const precision = aciertos.map((a, i) => dividir(a, asignadas[i]));
  const recall = aciertos.map((a, i) => dividir(a, soporte[i]));
  const f1 = precision.map((p, i) => dividir(2 * p * recall[i], p + recall[i]));
  const presentes = [...Array(k).keys()].filter((i) => soporte[i] > 0 || asignadas[i] > 0);
  const conSoporte = presentes.filter((i) => soporte[i] > 0);
  const promedio = (valores) => media(presentes.map((i) => valores[i]));
  const ponderado = (valores) => suma(valores.map((v, i) => (soporte[i] / total) * v));
  return {
    precision,
    recall,
    f1,
    exactitud: suma(aciertos) / total,
    exactitud_balanceada: media(conSoporte.map((i) => recall[i])),
    precision_macro: promedio(precision),
    recall_macro: promedio(recall),
    f1_macro: promedio(f1),
    precision_ponderada: ponderado(precision),
    recall_ponderado: ponderado(recall),
    f1_ponderado: ponderado(f1),
  };
}

// Área bajo la curva ROC con rangos medios (estadístico de Mann-Whitney): es el área trapezoidal que
// calcula scikit-learn, con medio punto por cada par empatado. Null si falta la clase positiva o la negativa.
export function auc(positivos, puntajes) {
  const n = puntajes.length;
  const orden = [...Array(n).keys()].sort((a, b) => puntajes[a] - puntajes[b]);
  const rangos = new Array(n);
  for (let i = 0; i < n; ) {
    let j = i;
    while (j + 1 < n && puntajes[orden[j + 1]] === puntajes[orden[i]]) {
      j += 1;
    }
    for (let t = i; t <= j; t += 1) {
      rangos[orden[t]] = (i + j) / 2 + 1;
    }
    i = j + 1;
  }
  const nPositivos = positivos.filter(Boolean).length;
  const nNegativos = n - nPositivos;
  if (nPositivos === 0 || nNegativos === 0) {
    return null;
  }
  const sumaRangos = suma(rangos.filter((_, i) => positivos[i]));
  return (sumaRangos - (nPositivos * (nPositivos + 1)) / 2) / (nPositivos * nNegativos);
}

// AUC por clase, su media sobre las clases donde existe y el micro (todas las parejas imagen y clase).
export function aucUnaContraResto(reales, probabilidades, k) {
  const porClase = [...Array(k).keys()].map((c) =>
    auc(
      reales.map((r) => r === c),
      probabilidades.map((p) => p[c]),
    ),
  );
  const definidas = porClase.filter((valor) => valor !== null);
  return {
    porClase,
    macro: media(definidas),
    micro: auc(
      reales.flatMap((r) => [...Array(k).keys()].map((c) => r === c)),
      probabilidades.flat(),
    ),
  };
}

// Una tarea (materiales o canecas): índices de la clase real y de la decidida, y una fila de
// probabilidades por imagen en el orden de `clases`. Las claves son las de metricas-elegido.json.
export function evaluarTarea(clases, reales, predichas, probabilidades) {
  const matriz = matrizConfusion(reales, predichas, clases.length);
  const valores = metricasDeMatriz(matriz);
  const areas = aucUnaContraResto(reales, probabilidades, clases.length);
  const porClase = (lista) => Object.fromEntries(clases.map((clase, i) => [clase, lista[i]]));
  return {
    clases,
    matriz,
    soporte: porClase(matriz.map(suma)),
    valores: {
      ...valores,
      precision: porClase(valores.precision),
      recall: porClase(valores.recall),
      f1: porClase(valores.f1),
      auc: porClase(areas.porClase),
      auc_macro: areas.macro,
      auc_micro: areas.micro,
    },
  };
}

// `filas`: una por imagen, con `clase_real` (etiqueta) y lo que devuelve `clasificar` con las canecas.
// `etiquetas` va en el orden del modelo; las canecas, en el de canecas.json.
export function evaluarFilas(filas, etiquetas, canecas) {
  const clasesCaneca = Object.keys(canecas.canecas);
  const posicion = (lista) => new Map(lista.map((clase, i) => [clase, i]));
  const posMaterial = posicion(etiquetas);
  const posCaneca = posicion(clasesCaneca);
  return {
    imagenes: filas.length,
    material: evaluarTarea(
      etiquetas,
      filas.map((f) => posMaterial.get(f.clase_real)),
      filas.map((f) => posMaterial.get(f.material)),
      filas.map((f) => etiquetas.map((e) => f.probabilidades[e])),
    ),
    caneca: evaluarTarea(
      clasesCaneca,
      filas.map((f) => posCaneca.get(canecas.materiales[f.clase_real].caneca)),
      filas.map((f) => posCaneca.get(f.caneca)),
      filas.map((f) => clasesCaneca.map((c) => f.probabilidadesCaneca[c])),
    ),
  };
}

// ---------------------------------------------------------------------------------------------
// Descargas

// Mismas columnas que los CSV de probabilidades del proyecto (resultados/probabilidades-*.csv).
export function csvEvaluacion(filas, etiquetas, canecas) {
  const clasesCaneca = Object.keys(canecas.canecas);
  const encabezado = [
    'archivo',
    'clase_real',
    ...etiquetas.map((e) => `prob_${e}`),
    'material',
    'caneca',
    ...clasesCaneca.map((c) => `prob_caneca_${c}`),
  ];
  const cuerpo = filas.map((f) => [
    f.archivo,
    f.clase_real,
    ...etiquetas.map((e) => f.probabilidades[e]),
    f.material,
    f.caneca,
    ...clasesCaneca.map((c) => f.probabilidadesCaneca[c]),
  ]);
  return aCsv([encabezado, ...cuerpo]);
}

// Una fila por foto clasificada, con el material en español y los dos avisos.
export function csvClasificacion(resultados, etiquetas, canecas) {
  const clasesCaneca = Object.keys(canecas.canecas);
  const siNo = (valor) => (valor ? 'sí' : 'no');
  const encabezado = [
    'archivo',
    'material',
    'etiqueta',
    'probabilidad_material',
    'caneca',
    'probabilidad_caneca',
    'caneca_dudosa',
    'lejos_de_lo_aprendido',
    ...etiquetas.map((e) => `prob_${e}`),
    ...clasesCaneca.map((c) => `prob_caneca_${c}`),
  ];
  const cuerpo = resultados.map((r) => [
    r.archivo,
    canecas.materiales[r.material].nombre,
    r.material,
    r.probabilidades[r.material],
    r.caneca,
    r.probabilidadesCaneca[r.caneca],
    siNo(r.avisos.canecaDudosa),
    siNo(r.avisos.lejosDeLoAprendido),
    ...etiquetas.map((e) => r.probabilidades[e]),
    ...clasesCaneca.map((c) => r.probabilidadesCaneca[c]),
  ]);
  return aCsv([encabezado, ...cuerpo]);
}
