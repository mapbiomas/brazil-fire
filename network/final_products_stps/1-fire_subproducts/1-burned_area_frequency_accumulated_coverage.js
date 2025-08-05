/***************************************************************
 MAPBIOMAS FUEGO - FRECUENCIA Y ACUMULADO DE ÁREAS QUEMADAS

 📅 FECHA: 06 de mayo de 2025

 EQUIPO:
 Red de mapeo de cicatrices de fuego - MapBiomas Fuego
 - Instituto de Pesquisa Ambiental da Amazônia (IPAM)
 - Wallace Silva e Vera Arruda; wallace.silva@ipam.org.br, vera.arruda@ipam.org.br

 -------------------------------------------------------------
 📌 ¿QUÉ HACE ESTE SCRIPT?
 Genera y exporta dos productos acumulados de fuego:

 🔴 `fireFrequency`: frecuencia de quemas (0–N), sin LULC.
 🔵 `fireFrequencyCoverage`: frecuencia con codificación LULC.

 También genera:
 🟡 `fireAccumulated`: binario 0/1 indicando quemas acumuladas.
 🟢 `fireAccumulatedCoverage`: clases LULC de las áreas quemadas acumuladas.

 -------------------------------------------------------------
 🔧 ¿QUÉ DEBO MODIFICAR PARA USAR ESTE SCRIPT?
 ✅ Cambiar la variable `country` si deseas otro país.
 ✅ Verificar rutas y nombres de salida.

***************************************************************/

// 🌍 País en análisis
var country = 'paraguay'; // 'bolivia', 'chile', 'colombia', 'paraguay', 'peru'

// 🧾 Nombres de archivos de salida
var outFileNameFrequency = 'mapbiomas-fire-collection1-frequency-burned-v1'; 
var outFileNameFrequencyCoverage = 'mapbiomas-fire-collection1-frequency-burned-coverage-v1'; 
var outFileNameAccumulated = 'mapbiomas-fire-collection1-accumulated-burned-v1'; 
var outFileNameAccumulatedCoverage = 'mapbiomas-fire-collection1-accumulated-burned-coverage-v1'; 

var assetOutput = 'projects/mapbiomas-' + country + '/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/';

// 🗂️ Imagen anual de quemas
var assetFire = assetOutput + 'mapbiomas-fire-collection1-annual-burned-v1';
var annualBurned = ee.Image(assetFire);

// 🌱 Colecciones de LULC por país
var landcover = {
  bolivia: ee.Image('projects/mapbiomas-public/assets/bolivia/collection2/mapbiomas_bolivia_collection2_integration_v1')
            .slice(28).addBands(ee.Image('projects/mapbiomas-public/assets/bolivia/collection2/mapbiomas_bolivia_collection2_integration_v1').slice(-1).rename(['classification_2024'])),
  chile: ee.Image('projects/mapbiomas-public/assets/chile/collection1/mapbiomas_chile_collection1_integration_v1')
            .slice(13)
            .addBands(ee.Image('projects/mapbiomas-public/assets/chile/collection1/mapbiomas_chile_collection1_integration_v1').slice(-1).rename(['classification_2023']))
            .addBands(ee.Image('projects/mapbiomas-public/assets/chile/collection1/mapbiomas_chile_collection1_integration_v1').slice(-1).rename(['classification_2024'])),
  colombia: ee.Image('projects/mapbiomas-public/assets/colombia/collection2/mapbiomas_colombia_collection2_integration_v1')
            .slice(28).addBands(ee.Image('projects/mapbiomas-public/assets/colombia/collection2/mapbiomas_colombia_collection2_integration_v1').slice(-1).rename(['classification_2024'])),
  paraguay: ee.Image('projects/mapbiomas-public/assets/paraguay/collection2/mapbiomas_paraguay_collection2_integration_v1')
            .slice(28).addBands(ee.Image('projects/mapbiomas-public/assets/paraguay/collection2/mapbiomas_paraguay_collection2_integration_v1').slice(-1).rename(['classification_2024'])),
  peru: ee.Image('projects/mapbiomas-public/assets/peru/collection2/mapbiomas_peru_collection2_integration_v1')
            .slice(28).addBands(ee.Image('projects/mapbiomas-public/assets/peru/collection2/mapbiomas_peru_collection2_integration_v1').slice(-1).rename(['classification_2024']))
};

var lulc = landcover[country];
var geometry = lulc.geometry();

// 🧮 Generación de productos acumulados
annualBurned.bandNames().evaluate(function (bandnames) {

  var freqPrev = annualBurned.slice(0, 1).rename('fire_frequency_2013_2013');
  var freqCovPrev = annualBurned.slice(0, 1).rename('fire_frequency_2013_2013');
  
  print("bandnames",bandnames,annualBurned);
  
  bandnames.slice(1).forEach(function (bandname) {
    var year = bandname.slice(-4);
    var freq = freqPrev
      .addBands(annualBurned.select(bandname))
      .gte(1)
      .reduce('sum')
      .rename('fire_frequency_2013_' + year);

    var freqCoverage = freq
      .multiply(100)
      .add(lulc.select('classification_' + year))
      .int16();

    freqPrev = freqPrev.addBands(freq);
    freqCovPrev = freqCovPrev.addBands(freqCoverage);
  });

  var freqPost = annualBurned.slice(-1).rename('fire_frequency_2024_2024');
  var freqCovPost = annualBurned.slice(-1).rename('fire_frequency_2024_2024');

  bandnames.reverse().slice(1, -1).forEach(function (bandname) {
    var year = bandname.slice(-4);
    var freq = freqPost
      .addBands(annualBurned.select(bandname))
      .gte(1)
      .reduce('sum')
      .rename('fire_frequency_' + year + '_2024');

    var freqCoverage = freq
      .multiply(100)
      .add(lulc.select('classification_' + year))
      .int16();

    freqPost = freqPost.addBands(freq);
    freqCovPost = freqCovPost.addBands(freqCoverage);
  });

  // 🔴 Frecuencia sin y con uso del suelo
  var fireFrequency = freqPrev.addBands(freqPost).select('fire_fre.*');
  fireFrequency = fireFrequency.select(fireFrequency.bandNames().sort());
  
  var fireFrequencyCoverage = freqCovPrev.addBands(freqCovPost).select('fire_fre.*');
  fireFrequencyCoverage = fireFrequencyCoverage.select(fireFrequencyCoverage.bandNames().sort());

  // 🟡 Acumulado binario
  var fireAccumulated = fireFrequency.gte(1).rename(
    fireFrequency.bandNames().map(function (bn) {
      return ee.String(bn).replace('fire_frequency_', 'fire_accumulated_');
    })
  );

  // 🟢 Acumulado con uso del suelo
  var fireAccumulatedCoverage = fireFrequencyCoverage
    .mod(100)
    .int()
    .rename(fireAccumulated.bandNames());

  // 📤 Print de verificación
  print('🔴 fireFrequency', fireFrequency);
  print('🔵 fireFrequencyCoverage', fireFrequencyCoverage);
  print('🟡 fireAccumulated', fireAccumulated);
  print('🟢 fireAccumulatedCoverage', fireAccumulatedCoverage);

  // 📤 Exportación de resultados
  Export.image.toAsset({
    image: fireFrequency,
    description: outFileNameFrequency,
    assetId: assetOutput + outFileNameFrequency,
    pyramidingPolicy: { '.default': 'mode' },
    region: geometry,
    scale: 30,
    maxPixels: 1e13
  });

  Export.image.toAsset({
    image: fireFrequencyCoverage,
    description: outFileNameFrequencyCoverage,
    assetId: assetOutput + outFileNameFrequencyCoverage,
    pyramidingPolicy: { '.default': 'mode' },
    region: geometry,
    scale: 30,
    maxPixels: 1e13
  });

  Export.image.toAsset({
    image: fireAccumulated,
    description: outFileNameAccumulated,
    assetId: assetOutput + outFileNameAccumulated,
    pyramidingPolicy: { '.default': 'mode' },
    region: geometry,
    scale: 30,
    maxPixels: 1e13
  });

  Export.image.toAsset({
    image: fireAccumulatedCoverage,
    description: outFileNameAccumulatedCoverage,
    assetId: assetOutput + outFileNameAccumulatedCoverage,
    pyramidingPolicy: { '.default': 'mode' },
    region: geometry,
    scale: 30,
    maxPixels: 1e13
  });

  // 🧪 Visualización (opcional)
  var year = 2019;
  var fire_palettes = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Palettes.js');
  var lulc_palettes = require('users/mapbiomas/modules:Palettes.js');

  Map.addLayer(
    annualBurned,
    {
      bands: 'burned_area_' + year,
      palette: ["800000"]
    },
    'Área quemada anual - ' + year
  );

  // 🔴 Frecuencia sin LULC
  Map.addLayer(
    fireFrequency,
    {
      bands: 'fire_frequency_2013_' + year,
      min: 0,
      max: 12,
      palette: fire_palettes.get('frenquency')
    },
    '🔴 Frecuencia de fuego (sin LULC) - ' + year
  );
  
  // 🔵 Frecuencia + LULC combinados (valor total)
  Map.addLayer(
    fireFrequencyCoverage,
    {
      bands: 'fire_frequency_2013_' + year
    },
    '🔵 Frecuencia de fuego + clase LULC - valor combinado ' + year
  );
  
  // 🔵 LULC extraído da frecuencia + LULC
  Map.addLayer(
    fireFrequencyCoverage.mod(100).byte(),
    {
      bands: 'fire_frequency_2013_' + year,
      min: 0,
      max: 69,
      palette: lulc_palettes.get('classification9')
    },
    '🔵 Frecuencia de fuego - clase LULC ' + year
  );
  
  // 🔵 Solo frecuencia (extraída do valor combinado)
  Map.addLayer(
    fireFrequencyCoverage.divide(100).byte(),
    {
      bands: 'fire_frequency_2013_' + year,
      min: 0,
      max: 12,
      palette: fire_palettes.get('frenquency')
    },
    '🔵 Frecuencia de fuego - número de eventos ' + year
  );
  
  // 🟢 Acumulado + LULC combinado
  Map.addLayer(
    fireAccumulatedCoverage,
    {
      bands: 'fire_accumulated_2013_' + year
    },
    '🟢 Fuego acumulado + clase LULC - valor combinado ' + year
  );
  
  // 🟡 Solo binário do acumulado (presença de fogo)
  Map.addLayer(
    fireAccumulated,
    {
      bands: 'fire_accumulated_2013_' + year,
      palette: ['ffffff', 'ff0000']
    },
    '🟡 Fuego acumulado - presencia binaria ' + year
  );

  
});
