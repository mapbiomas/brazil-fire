/******************************************************************************
 * MapBiomas Solo — Produtos — Paletas e Legendas
 * ----------------------------------------------------------------------------
 * Objetivo
 *   - Definir esquemas de cores, legendas e faixas de valores para os produtos
 *     de solo: estoque de carbono, PSD (areia, silte, argila), textura (L1–L3)
 *     e profundidade até pedregosidade, para padronização de visualização.
 * Saída (asset)
 *   - Objeto exportado contendo CARBON, PSD, TEXTURA_PALETTE, TEXTURA_LEGEND,
 *     SOIL_DEPTH_CLASSES, SOIL_DEPTH_LABELS e SOIL_DEPTH_PALETTE.
 * Bandas exportadas (ordem fixa ou derivada)
 *   - Não se aplica (não há bandas geradas; apenas metadados de estilo).
 * Regras de inclusão
 *   - Definição de intervalos de valores, paletas e rótulos por classe para
 *     cada produto temático.
 *   - Estruturas organizadas por níveis (L1–L3) para textura e por classes
 *     discretas para profundidade.
 * Autoria
 *   - MapBiomas Solo — contato@mapbiomas.org
 *   - Versão: 2025-12-01 Taciara
 ******************************************************************************/

/************************************************************
 * PRODUTO — ESTOQUE DE CARBONO
 ************************************************************/

var CARBON = {

  carbon: {
    min: 0,
    max: 90,
    palette: [
    '#ffffe5', '#fff7bc', '#fee391', '#fec44f',
    '#fe9929', '#ec7014', '#cc4c02', '#993404', '#662506'
    ]
  },


};


/************************************************************
 * PRODUTO — PSD (AREIA, SILTE, ARGILA)
 ************************************************************/

var PSD = {

  areia: {
    min: 0,
    max: 100,
    palette: [
      '#fef0d9','#fdd49e','#fdbb84','#fc8d59',
      '#ef6548','#d7301f','#990000'
    ]
  },

  silte: {
    min: 0,
    max: 100,
    palette: [
      '#edf8fb','#b2e2e2','#66c2a4',
      '#2ca25f','#006d2c'
    ]
  },

  argila: {
    min: 0,
    max: 100,
    palette: [
      '#fff7ec','#fee8c8','#fdd49e','#fdbb84',
      '#fc8d59','#e34a33','#b30000'
    ]
  }
};


/************************************************************
 * PRODUTO — TEXTURA (L1, L2, L3)
 ************************************************************/

var TEXTURA_PALETTE = {};

var TEXTURA_LEGEND = {};


/************************************************************
 * L1 — Grupamento Textural
 ************************************************************/

TEXTURA_PALETTE.L1 = [
  '#0080ff',
  '#707070',
  '#a83800',
  '#aa8686',
  '#b5d6ae',
  '#fffe73',
  '#d7c5a5'
];

TEXTURA_LEGEND.L1 = {
  0 : 'Não Observado | Not Observed',
  1 : 'Fragmentos de Rocha | Rock Fragments',
  2 : 'Muito Argilosa | Very Clay',
  3 : 'Argilosa | Clay',
  4 : 'Siltosa | Silt',
  5 : 'Arenosa | Sand',
  6 : 'Média | Loam'
};


/************************************************************
 * L2 — Subgrupamento Textural
 ************************************************************/

TEXTURA_PALETTE.L2 = [
  '#0080ff',
  '#707070',
  '#a83800',
  '#aa8686',
  '#f4a582',
  '#b5d6ae',
  '#d7c5a5',
  '#F8D488',
  '#E4B074',
  '#fffe73'
];

TEXTURA_LEGEND.L2 = {
  0 : 'Não Observado | Not Observed',
  1 : 'Fragmentos de Rocha | Rock Fragments',
  2 : 'Muito argilosa | Very Clay',
  3 : 'Argilosa | Clay',
  4 : 'Média-argilosa | Clay Loam',
  5 : 'Siltosa | Silt',
  6 : 'Média-siltosa | Silty Loam',
  7 : 'Média-arenosa | Sandy Loam',
  8 : 'Arenosa-média | Loamy Sand',
  9 : 'Muito Arenosa | Sand'
};


/************************************************************
 * L3 — Classe Textural Completa
 ************************************************************/

TEXTURA_PALETTE.L3 = [
  '#0080ff',
  '#707070',
  '#a83800',
  '#aa8686',
  '#3481a7',
  '#e9a9a9',
  '#80b1d3',
  '#c994c7',
  '#f4a582',
  '#d7c5a5',
  '#F8D488',
  '#E4B074',
  '#fffe73',
  '#ABBA7C',
  '#b5d6ae'
];

TEXTURA_LEGEND.L3 = {
  0 : 'Não Observado | Not Observed',
  1 : 'Fragmentos de Rocha | Rock Fragments',
  2 : 'Muito Argilosa | Very Clay',
  3 : 'Argilosa | Clay',
  4 : 'Argilo Siltosa | Silty Clay',
  5 : 'Franco Argilosa | Clay Loam',
  6 : 'Franco Argilo Siltosa | Silty Clay Loam',
  7 : 'Argilo Arenosa | Sandy Clay',
  8 : 'Franco Argilo Arenosa | Sandy Clay Loam',
  9 : 'Franca | Loam',
  10: 'Franco Arenosa | Sandy Loam',
  11: 'Areia Franca | Loamy Sand',
  12: 'Areia | Sand',
  13: 'Franco Siltosa | Silty Loam',
  14: 'Silte | Silt'
};


/************************************************************
 * PRODUTO — PROFUNDIDADE ATÉ PEDREGOSIDADE
 ************************************************************/

var SOIL_DEPTH_CLASSES = [
  0,1,2,3,4,5,6,7,8,9
];

var SOIL_DEPTH_PALETTE = [
  '#56008f',
  '#7d00a3',
  '#c300a9',
  '#ff2f9a',
  '#ff4f78',
  '#ff6e63',
  '#ff9678',
  '#ffc5a1',
  '#ffdfc6',
  '#fff8eb',
  '#fbf9f6'
];

var SOIL_DEPTH_LABEL = {
  0:'0–10 cm',
  1:'10–20 cm',
  2:'20–30 cm',
  3:'30–40 cm',
  4:'40–50 cm',
  5:'50–60 cm',
  6:'60–70 cm',
  7:'70–80 cm',
  8:'80–90 cm',
  9:'90–100 cm'
};

/************************************************************
 * EXPORTAÇÃO DO MÓDULO
 ************************************************************/
exports.CARBON             = CARBON;
exports.PSD                = PSD;
exports.TEXTURA_PALETTE    = TEXTURA_PALETTE;
exports.TEXTURA_LEGEND     = TEXTURA_LEGEND;
exports.SOIL_DEPTH_CLASSES = SOIL_DEPTH_CLASSES;
exports.SOIL_DEPTH_LABELS  = SOIL_DEPTH_LABEL;
exports.SOIL_DEPTH_PALETTE = SOIL_DEPTH_PALETTE;
