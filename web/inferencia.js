// Inferencia con un modelo de imagen exportado por Teachable Machine y traducción del material a la
// caneca. La comparten el arnés de evaluación y la app, para que sus cifras coincidan por construcción.
//
// - Las librerías se cargan desde jsDelivr con versión exacta e integridad SHA-384.
// - `predict` se llama sin volteo sobre la imagen original: la librería recorta al centro y escala a 224.
// - La caneca es el argmax de la suma de probabilidades de los materiales que van a cada caneca.

const JSDELIVR = 'https://cdn.jsdelivr.net/npm';

const TFJS_1_3_1 = {
  paquete: '@tensorflow/tfjs',
  version: '1.3.1',
  url: `${JSDELIVR}/@tensorflow/tfjs@1.3.1/dist/tf.min.js`,
  integrity: 'sha384-LBtUZzDp17Y7UgB/KFromf1LnY7o5ls8Oxm4vEAoTM0U+JzaadctX2/buQiNHyQU',
};

const TFJS_4_22_0 = {
  paquete: '@tensorflow/tfjs',
  version: '4.22.0',
  url: `${JSDELIVR}/@tensorflow/tfjs@4.22.0/dist/tf.min.js`,
  integrity: 'sha384-vE8hbVJ4lezako5rlvE7bY0BVzWlFhZncPlckrqNwcUQpVtgbENTgZ8TBbnPjZre',
};

const TM_IMAGE_0_8_5 = {
  paquete: '@teachablemachine/image',
  version: '0.8.5',
  url: `${JSDELIVR}/@teachablemachine/image@0.8.5/dist/teachablemachine-image.min.js`,
  integrity: 'sha384-4Q4fzaRr/bPyBiOertaWuZ8Zn987vAWqtNLbL6h+tf/Y+flpirtyqFTd7q9ELxjS',
};

// `fijada` es la dependencia par que declara la librería; `respaldo`, las versiones del fragmento
// oficial de la interfaz. tfjs va primero porque la librería de Teachable Machine usa el global `tf`.
export const LIBRERIAS = {
  fijada: [TFJS_1_3_1, TM_IMAGE_0_8_5],
  respaldo: [TFJS_4_22_0, TM_IMAGE_0_8_5],
};

// La que pasó la prueba de paridad con la vista previa de Teachable Machine (tests/test_arnes.py).
export const LIBRERIA_POR_DEFECTO = 'fijada';

function cargarScript({ url, integrity }) {
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = url;
    script.integrity = integrity;
    script.crossOrigin = 'anonymous';
    script.onload = resolve;
    script.onerror = () => reject(new Error(`No cargó ${url}`));
    document.head.append(script);
  });
}

// Carga una combinación de librerías una sola vez por página y devuelve el entorno resultante.
export async function cargarLibrerias(nombre = LIBRERIA_POR_DEFECTO) {
  const scripts = LIBRERIAS[nombre];
  if (!scripts) {
    throw new Error(`Combinación de librerías desconocida: ${nombre}`);
  }
  for (const script of scripts) {
    await cargarScript(script);
  }
  await window.tf.ready();
  return describirEntorno(nombre);
}

function renderizadorWebGL() {
  const gl = window.tf.backend().gpgpu?.gl;
  if (!gl) {
    return null;
  }
  const extension = gl.getExtension('WEBGL_debug_renderer_info');
  return gl.getParameter(extension ? extension.UNMASKED_RENDERER_WEBGL : gl.RENDERER);
}

export function describirEntorno(nombre) {
  return {
    libreria: nombre,
    paquetes: Object.fromEntries(LIBRERIAS[nombre].map((s) => [s.paquete, s.version])),
    versiones_tfjs: { ...window.tf.version },
    version_interna_tm_image: window.tmImage.version ?? null,
    backend: window.tf.getBackend(),
    renderizador_pagina: renderizadorWebGL(),
  };
}

// `base` es la carpeta (local o alojada) que contiene model.json y metadata.json.
export function cargarModelo(base) {
  const carpeta = base.endsWith('/') ? base : `${base}/`;
  return window.tmImage.load(`${carpeta}model.json`, `${carpeta}metadata.json`);
}

export async function cargarImagen(url) {
  const imagen = new Image();
  imagen.src = url;
  await imagen.decode();
  return imagen;
}

// Clave con el valor más alto; ante un empate gana la primera, como `numpy.argmax`.
function argmax(valores) {
  let mejor = null;
  for (const [clave, valor] of Object.entries(valores)) {
    if (mejor === null || valor > valores[mejor]) {
      mejor = clave;
    }
  }
  return mejor;
}

// `probabilidades` va de etiqueta a probabilidad; `canecas` sigue el esquema de config/canecas.json.
export function traducirACaneca(probabilidades, canecas) {
  const porCaneca = Object.fromEntries(Object.keys(canecas.canecas).map((c) => [c, 0]));
  for (const [etiqueta, probabilidad] of Object.entries(probabilidades)) {
    const caneca = canecas.materiales[etiqueta]?.caneca;
    if (!(caneca in porCaneca)) {
      throw new Error(`La etiqueta ${etiqueta} no tiene una caneca válida en canecas.json`);
    }
    porCaneca[caneca] += probabilidad;
  }
  return { caneca: argmax(porCaneca), probabilidadesCaneca: porCaneca };
}

// Probabilidad por etiqueta, material más probable y, si hay canecas, la caneca y sus probabilidades.
export async function clasificar(modelo, imagen, canecas = null) {
  const clases = await modelo.predict(imagen, false);
  const probabilidades = Object.fromEntries(clases.map((c) => [c.className, c.probability]));
  const resultado = { material: argmax(probabilidades), probabilidades };
  return canecas ? { ...resultado, ...traducirACaneca(probabilidades, canecas) } : resultado;
}
