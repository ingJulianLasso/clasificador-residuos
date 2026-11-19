// Umbrales de los dos avisos de la pestaña Clasificar. Se fijaron antes de ver los resultados de la
// prueba, para que no se acomoden a ella; el informe describe su cobertura y su precisión.

// «Caneca dudosa»: la probabilidad de la caneca elegida es menor que este valor.
export const UMBRAL_CANECA_DUDOSA = 0.8;

// «Foto lejos de lo aprendido»: el material más probable no llega a este valor.
export const UMBRAL_LEJOS_DE_LO_APRENDIDO = 0.5;

// `resultado` es lo que devuelve `clasificar` de inferencia.js con las canecas.
export function avisos({ material, probabilidades, caneca, probabilidadesCaneca }) {
  return {
    canecaDudosa: probabilidadesCaneca[caneca] < UMBRAL_CANECA_DUDOSA,
    lejosDeLoAprendido: probabilidades[material] < UMBRAL_LEJOS_DE_LO_APRENDIDO,
  };
}
