// Pruebas de metricas.js y umbrales.js sin navegador: `node --test web/pruebas/*.test.mjs`.
// Lo esperado sale de fixture-prueba.json, que escribe `uv run python -m residuos.fixture_app` con
// scikit-learn y con las métricas de la evaluación del proyecto.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';

import {
  ErrorCsv,
  auc,
  csvClasificacion,
  csvEvaluacion,
  emparejarConCsv,
  etiquetasDeCarpetas,
  evaluarFilas,
  evaluarTarea,
  leerCsv,
  leerEtiquetas,
  resolverClase,
} from '../metricas.js';
import { UMBRAL_CANECA_DUDOSA, UMBRAL_LEJOS_DE_LO_APRENDIDO, avisos } from '../umbrales.js';

const leerJson = (ruta) => JSON.parse(readFileSync(new URL(ruta, import.meta.url), 'utf8'));
const fixture = leerJson('./fixture-prueba.json');
const canecas = leerJson('../../config/canecas.json');

const TOLERANCIA = 1e-9;

// Compara dos métricas anidadas con la misma forma; los null deben coincidir.
function assertMetricas(obtenido, esperado, ruta = 'valores') {
  for (const [clave, valor] of Object.entries(esperado)) {
    const donde = `${ruta}.${clave}`;
    if (valor !== null && typeof valor === 'object') {
      assertMetricas(obtenido[clave], valor, donde);
    } else if (valor === null) {
      assert.equal(obtenido[clave], null, donde);
    } else {
      assert.ok(Math.abs(obtenido[clave] - valor) <= TOLERANCIA, `${donde}: ${obtenido[clave]} frente a ${valor}`);
    }
  }
}

function filasDelFixture() {
  const { etiquetas, clases_caneca: clasesCaneca } = fixture;
  return fixture.filas.map(([archivo, claseReal, probabilidades, material, caneca, probabilidadesCaneca]) => ({
    archivo,
    clase_real: claseReal,
    probabilidades: Object.fromEntries(etiquetas.map((e, i) => [e, probabilidades[i]])),
    material,
    caneca,
    probabilidadesCaneca: Object.fromEntries(clasesCaneca.map((c, i) => [c, probabilidadesCaneca[i]])),
  }));
}

// ---------------------------------------------------------------------------------------------
// Métricas

test('las métricas por material sobre el 30 % de prueba son las de scikit-learn', () => {
  const obtenido = evaluarFilas(filasDelFixture(), fixture.etiquetas, canecas).material;
  const esperado = fixture.esperado.material;
  assert.deepEqual(obtenido.clases, esperado.clases);
  assert.deepEqual(obtenido.matriz, esperado.matriz);
  assert.deepEqual(obtenido.soporte, esperado.soporte);
  assertMetricas(obtenido.valores, esperado.valores);
});

test('las métricas por caneca agregan con canecas.json y son las de scikit-learn', () => {
  const obtenido = evaluarFilas(filasDelFixture(), fixture.etiquetas, canecas).caneca;
  const esperado = fixture.esperado.caneca;
  assert.deepEqual(obtenido.clases, Object.keys(canecas.canecas));
  assert.deepEqual(obtenido.matriz, esperado.matriz);
  assert.deepEqual(obtenido.soporte, esperado.soporte);
  assertMetricas(obtenido.valores, esperado.valores);
});

for (const caso of fixture.casos_pequenos) {
  test(`caso pequeño con scikit-learn: ${caso.nombre}`, () => {
    const obtenido = evaluarTarea(caso.clases, caso.reales, caso.predichas, caso.probabilidades);
    assertMetricas(obtenido.valores, caso.esperado);
  });
}

test('AUC da medio punto por cada par empatado', () => {
  assert.equal(auc([false, false, true, true], [0.1, 0.4, 0.35, 0.8]), 0.75);
  assert.equal(auc([false, true, false, true], [0.5, 0.5, 0.2, 0.9]), 0.875);
});

test('AUC no existe si falta la clase positiva o la negativa', () => {
  assert.equal(auc([true, true], [0.2, 0.9]), null);
  assert.equal(auc([false, false], [0.2, 0.9]), null);
});

test('una clase que nunca se predice tiene precisión 0, no NaN', () => {
  const tarea = evaluarTarea(['a', 'b'], [0, 1], [0, 0], [[0.9, 0.1], [0.6, 0.4]]);
  assert.equal(tarea.valores.precision.b, 0);
  assert.equal(tarea.valores.f1.b, 0);
});

// ---------------------------------------------------------------------------------------------
// CSV de etiquetas

test('el CSV acepta BOM, CRLF, comillas, punto y coma y líneas vacías', () => {
  assert.deepEqual(leerCsv('﻿archivo,clase\r\n"a,1.jpg","Cartón"\r\n\r\n"b""2"".jpg",Paper\r\n'), [
    ['archivo', 'clase'],
    ['a,1.jpg', 'Cartón'],
    ['b"2".jpg', 'Paper'],
  ]);
  assert.deepEqual(leerCsv('archivo;clase\nfoto.jpg;Vidrio'), [
    ['archivo', 'clase'],
    ['foto.jpg', 'Vidrio'],
  ]);
});

test('la clase se reconoce por la etiqueta de RealWaste o por el nombre en español, sin importar tildes', () => {
  assert.equal(resolverClase('Cardboard', canecas), 'Cardboard');
  assert.equal(resolverClase('carton', canecas), 'Cardboard');
  assert.equal(resolverClase('  Orgánicos de comida ', canecas), 'Food Organics');
  assert.equal(resolverClase('food_organics', canecas), 'Food Organics');
  assert.equal(resolverClase('Vegetación', canecas), 'Vegetation');
  assert.equal(resolverClase('Icopor', canecas), null);
});

test('un CSV mal formado se rechaza con un mensaje claro', () => {
  const casos = [
    ['', /vacío/],
    ['nombre,tipo\na.jpg,Paper\n', /encabezado/],
    ['archivo,clase\n', /ninguna fila/],
    ['archivo,clase\na.jpg,Paper,extra\n', /fila 2 tiene 3 columnas/],
    ['archivo,clase\n"a.jpg,Paper\n', /comillas/],
    ['archivo,clase\na.jpg,\n', /falta el archivo o la clase/],
  ];
  for (const [texto, mensaje] of casos) {
    assert.throws(() => leerEtiquetas(texto, canecas), (error) => error instanceof ErrorCsv && mensaje.test(error.message), texto);
  }
});

test('las fotos se emparejan con el CSV por nombre, con o sin carpeta', () => {
  const lectura = leerEtiquetas(
    [
      'archivo,clase',
      'Cardboard/Cardboard_1.jpg,Cardboard',
      'papel.jpg,Papel',
      'raro.jpg,Icopor',
      'doble.jpg,Metal',
      'otra/doble.jpg,Vidrio',
      'sin-foto.jpg,Plastic',
    ].join('\n'),
    canecas,
  );
  const fotos = ['Cardboard_1.jpg', 'papel.jpg', 'raro.jpg', 'doble.jpg', 'suelta.jpg'].map((name) => ({ name }));
  const resultado = emparejarConCsv(fotos, lectura);
  assert.deepEqual(
    resultado.evaluables.map(({ archivo, etiqueta }) => [archivo, etiqueta]),
    [
      ['Cardboard/Cardboard_1.jpg', 'Cardboard'],
      ['papel.jpg', 'Paper'],
    ],
  );
  assert.deepEqual(resultado.desconocidas, [{ archivo: 'raro.jpg', clase: 'Icopor' }]);
  assert.deepEqual(resultado.repetidos, ['doble.jpg']);
  assert.deepEqual(resultado.sinEtiqueta, ['suelta.jpg']);
  assert.deepEqual(resultado.sinFoto, ['sin-foto.jpg']);
});

test('en una carpeta, la clase de cada foto es su subcarpeta', () => {
  const fotos = [
    'mis-fotos/Cartón/a.jpg',
    'mis-fotos/Glass/b.png',
    'mis-fotos/Food Organics/c.webp',
    'mis-fotos/Icopor/d.jpg',
    'suelta.jpg',
  ].map((ruta) => ({ name: ruta.split('/').pop(), webkitRelativePath: ruta.includes('/') ? ruta : '' }));
  const resultado = etiquetasDeCarpetas(fotos, canecas);
  assert.deepEqual(
    resultado.evaluables.map(({ archivo, etiqueta }) => [archivo, etiqueta]),
    [
      ['mis-fotos/Cartón/a.jpg', 'Cardboard'],
      ['mis-fotos/Glass/b.png', 'Glass'],
      ['mis-fotos/Food Organics/c.webp', 'Food Organics'],
    ],
  );
  assert.deepEqual(resultado.desconocidas, [{ archivo: 'mis-fotos/Icopor/d.jpg', clase: 'Icopor' }]);
  assert.deepEqual(resultado.sinEtiqueta, ['suelta.jpg']);
});

// ---------------------------------------------------------------------------------------------
// Descargas y umbrales

test('el CSV de la evaluación tiene las columnas de los CSV del proyecto y conserva los valores', () => {
  const filas = filasDelFixture().slice(0, 3);
  const [encabezado, ...cuerpo] = leerCsv(csvEvaluacion(filas, fixture.etiquetas, canecas));
  assert.deepEqual(encabezado, [
    'archivo',
    'clase_real',
    ...fixture.etiquetas.map((e) => `prob_${e}`),
    'material',
    'caneca',
    ...Object.keys(canecas.canecas).map((c) => `prob_caneca_${c}`),
  ]);
  cuerpo.forEach((fila, i) => {
    assert.equal(fila[0], filas[i].archivo);
    fixture.etiquetas.forEach((e, j) => assert.equal(Number(fila[2 + j]), filas[i].probabilidades[e]));
  });
});

test('el CSV de Clasificar trae el material en español, la caneca y los avisos', () => {
  const [fila] = filasDelFixture();
  const resultado = { ...fila, archivo: 'foto, con coma.jpg', avisos: avisos(fila) };
  const [encabezado, valores] = leerCsv(csvClasificacion([resultado], fixture.etiquetas, canecas));
  const registro = Object.fromEntries(encabezado.map((c, i) => [c, valores[i]]));
  assert.equal(registro.archivo, 'foto, con coma.jpg');
  assert.equal(registro.material, canecas.materiales[fila.material].nombre);
  assert.equal(registro.caneca, fila.caneca);
  assert.equal(Number(registro.probabilidad_caneca), fila.probabilidadesCaneca[fila.caneca]);
  assert.ok(['sí', 'no'].includes(registro.caneca_dudosa));
});

test('los umbrales son 0,80 para la caneca y 0,50 para el material, con desigualdad estricta', () => {
  assert.equal(UMBRAL_CANECA_DUDOSA, 0.8);
  assert.equal(UMBRAL_LEJOS_DE_LO_APRENDIDO, 0.5);
  const resultado = (pMaterial, pCaneca) => ({
    material: 'Paper',
    probabilidades: { Paper: pMaterial },
    caneca: 'blanca',
    probabilidadesCaneca: { blanca: pCaneca },
  });
  assert.deepEqual(avisos(resultado(0.5, 0.8)), { canecaDudosa: false, lejosDeLoAprendido: false });
  assert.deepEqual(avisos(resultado(0.4999, 0.7999)), { canecaDudosa: true, lejosDeLoAprendido: true });
});
