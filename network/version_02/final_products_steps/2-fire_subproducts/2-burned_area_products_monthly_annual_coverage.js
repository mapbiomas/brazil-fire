/***************************************************************
 MAPBIOMAS FUEGO - SUBPRODUCTOS DERIVADOS DE COBERTURA QUEIMADA

 📅 FECHA: 06 de mayo de 2025

 EQUIPO:
 Red de mapeo de cicatrices de fuego - MapBiomas Fuego
 - Instituto de Pesquisa Ambiental da Amazônia (IPAM)
 - Wallace Silva e Vera Arruda; wallace.silva@ipam.org.br, vera.arruda@ipam.org.br
 -------------------------------------------------------------
 📌 ¿QUÉ HACE ESTE SCRIPT?
 Genera y exporta 4 productos a partir de cicatrices de fuego:

 🟠 `monthlyBurnedCoverage`: codificación mes + clase LULC.
 🔵 `annualBurnedCoverage`: codificación binaria anual + clase LULC.
 🟡 `monthlyBurnedArea`: mes de ocurrencia (1 a 12), sin LULC.
 🟢 `annualBurnedArea`: fuego anual (0/1), sin LULC.

 -------------------------------------------------------------
 🔧 ¿QUÉ DEBO MODIFICAR PARA USAR ESTE SCRIPT?
 ✅ Seleccionar el país deseado (bloque de selección).
 ✅ Verificar `assetOutput`, `bounds`, nombres de salida.

***************************************************************/

// 🔧 Selección del país a trabajar (Paraguay como ejemplo)

var assetInput = 'projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/CLASSIFICATION_COLLECTIONS/collection1_fire_mask_v1';
var assetOutput = 'projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/';
var scale = 30;

// 🌎 Colección LULC por país
var landcover_bolivia   = ee.Image('projects/mapbiomas-public/assets/bolivia/collection2/mapbiomas_bolivia_collection2_integration_v1');
var landcover_chile     = ee.Image('projects/mapbiomas-public/assets/chile/collection1/mapbiomas_chile_collection1_integration_v1');
var landcover_colombia  = ee.Image('projects/mapbiomas-public/assets/colombia/collection2/mapbiomas_colombia_collection2_integration_v1');
var landcover_paraguay  = ee.Image('projects/mapbiomas-public/assets/paraguay/collection2/mapbiomas_paraguay_collection2_integration_v1');
var landcover_peru      = ee.Image('projects/mapbiomas-public/assets/peru/collection2/mapbiomas_peru_collection2_integration_v1');

// 🧭 Regiones por país
var regions_bolivia   = ee.FeatureCollection('projects/mapbiomas-bolivia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_bolivia_v1');
var regions_chile     = ee.FeatureCollection('projects/mapbiomas-chile/assets/FIRE/AUXILIARY_DATA/regiones_fuego_chile_v1');
var regions_colombia  = ee.FeatureCollection('projects/mapbiomas-colombia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_colombia_v1');
var regions_paraguay  = ee.FeatureCollection('projects/mapbiomas-paraguay/assets/FIRE/AUXILIARY_DATA/regiones_fuego_paraguay_v1');
var regions_peru      = ee.FeatureCollection('projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1');

// Selección activa: PARAGUAY
var landcover = landcover_paraguay;
var geometry = landcover.geometry();

// Añadir banda para 2024 duplicando 2023 si necesario
landcover = landcover
  .slice(28)
  .addBands(
    landcover.select('classification_2023').rename('classification_2024')
  );

// 📅 Cargar colección de cicatrices de fuego
var fireCollection = ee.ImageCollection(assetInput);

var monthlyBurnedArea = ee.Image(
  ee.List.sequence(2013, 2024, 1)
    .iterate(function(year, prev) {
      year = ee.Number(year).int();
      var image = fireCollection
        .filter(ee.Filter.eq("year", year))
        .mosaic()
        .rename(ee.String('burned_coverage_').cat(year));
      return ee.Image(prev).addBands(image);
    }, ee.Image().select())
);

// 🟠 Producto: Cobertura queimada mensual (mes + clase)
var monthlyBurnedCoverage = monthlyBurnedArea
  .multiply(100)
  .add(landcover)
  .uint16();

// 🔵 Producto: Cobertura queimada anual binaria (0/1 + clase)
var annualBurnedCoverage = monthlyBurnedArea
  .gte(1)
  .multiply(landcover)
  .uint8();

// 🟡 Producto: Área queimada mensual (solo mes 1–12)
var monthlyBandNames = monthlyBurnedArea.bandNames().map(function(bn) {
  return ee.String(bn).replace('burned_coverage_', 'burned_monthly_');
});
var monthlyBurnedAreaClean = monthlyBurnedArea
  .rename(monthlyBandNames)
  .uint8();

// 🟢 Producto: Área queimada anual binaria (0 o 1)
var annualBandNames = monthlyBandNames.map(function(bn) {
  return ee.String(bn).replace('burned_monthly_', 'burned_area_');
});
var annualBurnedArea = monthlyBurnedArea
  .gt(0)
  .rename(annualBandNames)
  .uint8();

// 📤 Exportación de resultados

print('🟠 monthlyBurnedCoverage',monthlyBurnedCoverage);

Export.image.toAsset({
  image: monthlyBurnedCoverage,
  description: 'mapbiomas-fire-collection1-monthly-burned-coverage-v1',
  assetId: assetOutput + 'mapbiomas-fire-collection1-monthly-burned-coverage-v1',
  pyramidingPolicy: { '.default': 'mode' },
  region: geometry,
  scale: scale,
  maxPixels: 1e13,
});

print('🔵 annualBurnedCoverage',annualBurnedCoverage);

Export.image.toAsset({
  image: annualBurnedCoverage,
  description: 'mapbiomas-fire-collection1-annual-burned-coverage-v1',
  assetId: assetOutput + 'mapbiomas-fire-collection1-annual-burned-coverage-v1',
  pyramidingPolicy: { '.default': 'mode' },
  region: geometry,
  scale: scale,
  maxPixels: 1e13,
});

print('🟡 monthlyBurnedArea',monthlyBurnedArea);

Export.image.toAsset({
  image: monthlyBurnedArea,
  description: 'mapbiomas-fire-collection1-monthly-burned-v1',
  assetId: assetOutput + 'mapbiomas-fire-collection1-monthly-burned-v1',
  pyramidingPolicy: { '.default': 'mode' },
  region: geometry,
  scale: scale,
  maxPixels: 1e13,
});

print('🟢 annualBurnedArea',annualBurnedArea);

Export.image.toAsset({
  image: annualBurnedArea,
  description: 'mapbiomas-fire-collection1-annual-burned-v1',
  assetId: assetOutput + 'mapbiomas-fire-collection1-annual-burned-v1',
  pyramidingPolicy: { '.default': 'mode' },
  region: geometry,
  scale: scale,
  maxPixels: 1e13,
});

// 🧪 Visualización (opcional)
var year = 2019;
var fire_palettes = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Palettes.js');
var lulc_palettes = require('users/mapbiomas/modules:Palettes.js');

Map.addLayer(
  monthlyBurnedCoverage,
  {
    bands: 'burned_coverage_' + year,
  },
  '🟠 Cobertura quemada mensual - mes y uso del suelo' + year
);

Map.addLayer(
  monthlyBurnedCoverage.mod(100).byte(),
  {
    min: 0,
    max: 69,
    bands: 'burned_coverage_' + year,
    palette: lulc_palettes.get('classification9')
  },
  '🟠 Cobertura quemada mensual - uso del suelo ' + year
);

Map.addLayer(
  monthlyBurnedCoverage.divide(100).byte(),
  {
    min: 1,
    max: 12,
    bands: 'burned_coverage_' + year,
    palette: fire_palettes.get('mensal')
  },
  '🟠 Cobertura quemada mensual - mes de ocurrencia ' + year
);

Map.addLayer(
  annualBurnedCoverage,
  {
    min: 0,
    max: 69,
    bands: 'burned_coverage_' + year,
    palette: lulc_palettes.get('classification9')
  },
  '🔵 Cobertura quemada anual - uso del suelo ' + year
);

Map.addLayer(
  monthlyBurnedAreaClean,
  {
    min: 1,
    max: 12,
    bands: 'burned_monthly_' + year,
    palette: fire_palettes.get('mensal')
  },
  '🟡 Área quemada mensual - mes de ocurrencia ' + year
);

Map.addLayer(
  annualBurnedArea,
  {
    bands: 'burned_area_' + year,
    palette: '800000'
  },
  '🟢 Área quemada anual - presencia binaria ' + year
);

