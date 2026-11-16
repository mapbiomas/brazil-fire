/* ==============================================================================
MAPBIOMAS FUEGO - MONITOR_01 - M8
Generador Universal de Estadísticas (Salida: CSV a GCS y/o Drive)

📅 FECHA: 09/2026
🏷️ VERSIÓN: 3.2 (Filtro de área quemada > 0 ha + Soporte Dual GCS/Drive)

📌 ¿QUÉ HACE ESTE SCRIPT?
1. Escanea la carpeta FILTERED/ y descubre colecciones y etapas disponibles (ft00, ft01, ft02).
2. Para cada combinación (colección, región, período), calcula la superficie de observación
   y la superficie quemada cruzada por niveles de uso y cobertura de suelo (LULC).
3. Utiliza MapBiomas Perú Colección 4 para el cruce de LULC.
4. Exporta un archivo CSV unificado filtrando únicamente los registros con área quemada mayor a cero.
============================================================================== */

// ─── CONFIGURACIÓN GENERAL ──────────────────────────────────────────────────

var COUNTRY = 'peru';
var COUNTRY_LABEL = 'Peru';
var CAMPAIGN = 'MONITOR_01';

var CATALOG_ROOT = 'projects/mapbiomas-' + COUNTRY + '/assets/FIRE/CATALOG_01';
var CLASSIFICATIONS_ROOT = CATALOG_ROOT + '/' + CAMPAIGN + '/LIBRARY_CLASSIFICATIONS/';
var FILTERED = CLASSIFICATIONS_ROOT + 'FILTERED/';

var REGIONS_ASSET = '';
var REGIONS = REGIONS_ASSET ?
    ee.FeatureCollection(REGIONS_ASSET) :
    ee.FeatureCollection('projects/mapbiomas-' + COUNTRY + '/assets/FIRE/AUXILIARY_DATA/regiones_fuego_' + COUNTRY + '_v1');
var REGION_PROPERTY = 'region_nam';

// Escala condicionada al sensor de la campaña de entrada.
var CAMPAIGN_SENSOR = {
    MONITOR_01:    { sensor: 'SENTINEL2', scale: 10 },
    COLLECTION_01: { sensor: 'LANDSAT',   scale: 30 },
    COLLECTION_02: { sensor: 'LANDSAT',   scale: 30 },
    COLLECTION_03: { sensor: 'LANDSAT',   scale: 30 },
    COLLECTION_06: { sensor: 'LANDSAT',   scale: 30 }
};
var ACTIVE_SENSOR = CAMPAIGN_SENSOR[CAMPAIGN] ||
    (CAMPAIGN.indexOf('COLLECTION') === 0 ? { sensor: 'LANDSAT', scale: 30 } : { sensor: 'SENTINEL2', scale: 10 });
var SCALE = ACTIVE_SENSOR.scale;

// Carga e integración de LULC (asset configurable por país)
var LULC_ASSET = 'projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1';
var LULC_LAST_YEAR = 2025;

function assertLulcConfig() {
    if (!LULC_ASSET) {
        print('⚠ M8: LULC_ASSET no configurado para ' + COUNTRY_LABEL + ' (' + COUNTRY + ' / ' + CAMPAIGN + '). Complete LULC_ASSET y LULC_LAST_YEAR antes de ejecutar.');
        throw new Error('LULC_ASSET no configurado para ' + COUNTRY_LABEL);
    }
    var bands;
    try {
        bands = ee.Image(LULC_ASSET).bandNames().getInfo() || [];
    } catch (e) {
        print('⚠ M8: no se pudo leer LULC_ASSET para ' + COUNTRY_LABEL + ': ' + LULC_ASSET + '.');
        throw new Error('LULC_ASSET inválido para ' + COUNTRY_LABEL);
    }
    if (bands.indexOf('classification_' + LULC_LAST_YEAR) === -1) {
        print('⚠ M8: ' + LULC_ASSET + ' no contiene classification_' + LULC_LAST_YEAR + '. Ajuste LULC_LAST_YEAR o el asset.');
        throw new Error('classification_' + LULC_LAST_YEAR + ' ausente en LULC_ASSET');
    }
}
assertLulcConfig();

var LULC = ee.Image(LULC_ASSET);
var currentYear = new Date().getFullYear();

// Proyectar la última clasificación LULC hacia años actuales/futuros
for (var year = LULC_LAST_YEAR + 1; year <= currentYear; year++) {
    LULC = LULC.addBands(LULC.select('classification_' + LULC_LAST_YEAR).rename('classification_' + year));
}

// Filtros opcionales — Dejar arrays vacíos para descubrimiento automático
var COLLECTIONS = []; 
var STAGES = ['ft00', 'ft01', 'ft02', 'ft03', 'ft04', 'ft05']; 

// ⚙️ SELECCIÓN DEL DESTINO DE EXPORTACIÓN
// Opciones válidas: 'GCS', 'DRIVE', o 'BOTH'
var EXPORT_TARGET = 'DRIVE';

// Parámetros para Google Cloud Storage (GCS)
var EXPORT_BUCKET = 'mapbiomas-fire';
var EXPORT_REGION = (COUNTRY === 'indonesia') ? 'asia' : (COUNTRY === 'drc' ? 'africa' : 'sudamerica');
var EXPORT_PREFIX = EXPORT_REGION + '/' + COUNTRY + '/CATALOG_01/' + CAMPAIGN + '/LIBRARY_STATISTICS/';

// Parámetros para Google Drive
var DRIVE_FOLDER = 'MAPBIOMAS_FIRE_STATISTICS';

// ─── DICCIONARIOS DE LEYENDA LULC ───────────────────────────────────────────

var LULC_LEVELS = {
  // Nivel 0: Agrupación macro (1: Natural, 2: Antrópico, 3: No Observado)
  0: ee.Dictionary({
    3:'Natural', 4:'Natural', 5:'Natural', 6:'Natural', 11:'Natural', 51:'Natural', 82:'Natural', 12:'Natural', 13:'Natural', 66:'Natural', 70:'Natural', 23:'Natural', 92:'Natural', 52:'Natural', 61:'Natural', 68:'Natural', 33:'Natural', 34:'Natural',
    21:'Antrópico', 9:'Antrópico', 35:'Antrópico', 40:'Antrópico', 15:'Antrópico', 24:'Antrópico', 30:'Antrópico', 25:'Antrópico', 31:'Antrópico',
    27:'No Observado'
  }),

  // Nivel 1: Clases principales
  1: ee.Dictionary({
    3:'Formación boscosa', 4:'Formación boscosa', 5:'Formación boscosa', 6:'Formación boscosa',
    11:'Formación Natural No Boscosa', 51:'Formación Natural No Boscosa', 82:'Formación Natural No Boscosa', 12:'Formación Natural No Boscosa', 13:'Formación Natural No Boscosa', 66:'Formación Natural No Boscosa', 70:'Formación Natural No Boscosa',
    21:'Área Agropecuaria', 9:'Área Agropecuaria', 35:'Área Agropecuaria', 40:'Área Agropecuaria', 15:'Área Agropecuaria',
    24:'Área Sin Vegetación', 30:'Área Sin Vegetación', 23:'Área Sin Vegetación', 92:'Área Sin Vegetación', 52:'Área Sin Vegetación', 61:'Área Sin Vegetación', 68:'Área Sin Vegetación', 25:'Área Sin Vegetación',
    33:'Cuerpo de Agua', 31:'Cuerpo de Agua', 34:'Cuerpo de Agua',
    27:'No Observado'
  }),

  // Nivel 2: Subclases detalladas (formato compacto)
  2: ee.Dictionary({
    3:'Bosque', 4:'Bosque Seco', 5:'Manglar', 6:'Bosque Inundable',
    11:'Formación Herbácea Inundable', 51:'Herbazal Inundable Tierras Bajas', 82:'Herbazal Inundable Altoandino', 12:'Formación Herbácea', 13:'Formaciones Arbustivas y Otras', 66:'Matorral y Otros Arbustales', 70:'Loma Costera (beta)',
    21:'Mosaico Agropecuario', 9:'Plantación Forestal', 35:'Palma Aceitera', 40:'Arroz (beta)', 15:'Pasto (beta)',
    24:'Infraestructura Urbana', 30:'Minería', 23:'Playa', 92:'Superficie Rocosa', 52:'Superficie Salina Costera', 61:'Superficie de Salar', 68:'Otra Área Natural Sin Veg.', 25:'Otra Área Antrópica Sin Veg.',
    33:'Río, Lago u Océano', 31:'Acuicultura', 34:'Glaciar',
    27:'No Observado'
  })
};

// ─── FUNCIONES DE APOYO (HELPERS) ───────────────────────────────────────────

function ensureFolder(name) {
    var p = name.split('/'), cur = CLASSIFICATIONS_ROOT;
    for (var i = 0; i < p.length; i++) {
        cur += p[i];
        try { ee.data.getAsset(cur); }
        catch (e) { ee.data.createAsset({ type: 'FOLDER' }, cur); }
        cur += '/';
    }
}

function extractYear(periodName) {
    var match = periodName.match(/^(\d{4})/);
    return match ? parseInt(match[1], 10) : currentYear;
}

// Construye una imagen raster con los IDs numéricos únicos de cada región
function buildTerritoryRaster(regionList) {
    var dict = ee.Dictionary.fromLists(regionList, ee.List.sequence(1, regionList.size()));
    
    var regionsWithId = REGIONS.map(function(f) {
        var name = f.get(REGION_PROPERTY);
        var id = dict.get(name);
        return f.set('territory_id', id);
    });

    var territoryRaster = ee.Image(0).paint(regionsWithId, 'territory_id').rename('territory');
    return { image: territoryRaster, dict: dict };
}

// ─── CÁLCULO DE ÁREA ─────────────────────────────────────────────────────────

function calculateArea(stageImage, territoryRaster, lulcYear) {
    var mask = stageImage.mask().reduce(ee.Reducer.max());
    var burned = stageImage.select(0).gt(0); 
    
    var fireStatus = ee.Image(0).where(mask, 1).where(burned, 2);

    var composite = lulcYear.select(0).multiply(100).add(fireStatus).rename('class');
    var pixelAreaHa = ee.Image.pixelArea().divide(10000); // m² a ha

    var reducer = ee.Reducer.sum()
        .group(1, 'class')
        .group(1, 'territory');

    var reduction = pixelAreaHa
        .addBands(territoryRaster)
        .addBands(composite)
        .reduceRegion({
            reducer: reducer,
            geometry: REGIONS.geometry().bounds(),
            scale: SCALE,
            maxPixels: 1e12,
            tileScale: 4
        });

    var groups = ee.List(reduction.get('groups'));

    return ee.FeatureCollection(groups.map(function (terrObj) {
        terrObj = ee.Dictionary(terrObj);
        var territoryId = terrObj.getNumber('territory');
        var classGroups = ee.List(terrObj.get('groups'));

        return classGroups.map(function (classObj) {
            classObj = ee.Dictionary(classObj);
            var classVal = classObj.getNumber('class');
            var lulcId = classVal.divide(100).int();
            var fireStatusVal = classVal.mod(100).int();

            var areaHa = ee.Number(classObj.get('sum')).multiply(10).round().divide(10);
            var obsArea = fireStatusVal.gte(1).multiply(areaHa);
            var burnedArea = fireStatusVal.eq(2).multiply(areaHa);

            var props = ee.Dictionary({
                'territory_id': territoryId,
                'lulc_id': lulcId,
                'area_obs_ha': obsArea,
                'area_burned_ha': burnedArea,
                'lulc_nivel_0': LULC_LEVELS[0].get(lulcId, 'Desconocido'),
                'lulc_nivel_1': LULC_LEVELS[1].get(lulcId, 'Desconocido'),
                'lulc_nivel_2': LULC_LEVELS[2].get(lulcId, 'Desconocido')
            });

            return ee.Feature(null, props);
        });
    }).flatten());
}

// ─── DESCUBRIMIENTO CLIENTE (ASSETS) ────────────────────────────────────────

function discoverCollections(cb) {
    ee.data.listAssets(FILTERED, {}, function (result) {
        if (!result || !result.assets) { cb([]); return; }
        var cols = result.assets
            .filter(function (a) { return a.type === 'FOLDER'; })
            .map(function (a) { return a.id.split('/').pop(); })
            .sort();
        cb(cols);
    });
}

function discoverStages(col, cb) {
    var path = FILTERED + col + '/';
    ee.data.listAssets(path, {}, function (result) {
        if (!result || !result.assets) { cb([]); return; }
        var stages = result.assets
            .filter(function (a) { return a.type === 'IMAGE_COLLECTION' && /^ft\d\d$/.test(a.id.split('/').pop()); })
            .map(function (a) { return a.id.split('/').pop(); })
            .sort();
        cb(stages);
    });
}

function discoverPeriodsMap(col, stages, cb) {
    var periodStageMap = {}; // { periodName: [availableStages] }
    var pending = stages.length;

    if (pending === 0) { cb(periodStageMap); return; }

    stages.forEach(function(stg) {
        var path = FILTERED + col + '/' + stg + '/';
        ee.data.listAssets(path, {}, function(result) {
            if (result && result.assets) {
                result.assets.forEach(function(a) {
                    if (a.type === 'IMAGE') {
                        var pName = a.id.split('/').pop();
                        if (!periodStageMap[pName]) periodStageMap[pName] = [];
                        periodStageMap[pName].push(stg);
                    }
                });
            }
            pending--;
            if (pending === 0) cb(periodStageMap);
        });
    });
}

// ─── EJECUCIÓN PRINCIPAL ─────────────────────────────────────────────────────

print('=== M8 — Generador Universal de Estadísticas ===');
print('País: ' + COUNTRY_LABEL + ' (' + COUNTRY + ') | Campaña: ' + CAMPAIGN);
print('Sensor: ' + ACTIVE_SENSOR.sensor + ' | Escala: ' + SCALE + ' m');
print('LULC: ' + LULC_ASSET);
print('Ruta de origen (FILTERED): ' + FILTERED);
print('Modo de exportación seleccionado: ' + EXPORT_TARGET);

Map.addLayer(REGIONS.style({ color: 'ffffff', fillColor: '00000000', width: 1 }), {}, 'Límites de Regiones');
Map.centerObject(REGIONS);

var regionList = ee.List(REGIONS.aggregate_array(REGION_PROPERTY).distinct().sort());
var territoryObj = buildTerritoryRaster(regionList);

discoverCollections(function (collections) {
    var targetCols = COLLECTIONS.length > 0
        ? collections.filter(function (c) { return COLLECTIONS.indexOf(c) !== -1; })
        : collections;

    print('Colecciones encontradas para procesar: ' + targetCols.length + ' (' + targetCols.join(', ') + ')');

    if (targetCols.length === 0) {
        print('No se encontraron colecciones en ' + FILTERED);
        return;
    }

    var totalExports = 0;

    targetCols.forEach(function (col) {
        discoverStages(col, function (stages) {
            var targetStages = STAGES.length > 0
                ? stages.filter(function (s) { return STAGES.indexOf(s) !== -1; })
                : stages;

            if (targetStages.length === 0) {
                print('  [' + col + '] No se encontraron etapas válidas (ft00, ft01, etc.).');
                return;
            }

            discoverPeriodsMap(col, targetStages, function (periodMap) {
                var periods = Object.keys(periodMap).sort();
                if (periods.length === 0) {
                    print('  [' + col + '] No se encontraron períodos cargados.');
                    return;
                }

                print('  [' + col + '] Procesando ' + periods.length + ' período(s) en etapas: ' + targetStages.join(', '));

                var lulcBands = LULC.bandNames().getInfo() || [];
                var allRowsList = [];

                periods.forEach(function (periodName) {
                    var periodYear = extractYear(periodName);
                    if (lulcBands.indexOf('classification_' + periodYear) === -1) {
                        print('  ⚠ [' + col + '] Sin LULC classification_' + periodYear + '; período omitido: ' + periodName);
                        return;
                    }
                    var lulcYear = LULC.select('classification_' + periodYear);
                    var availStages = periodMap[periodName];

                    // Extraer año y mes directamente del periodName ("YYYY_MM")
                    var periodParts = periodName.split('_');
                    var yrStr = periodParts[0];
                    var moStr = periodParts.length > 1 ? periodParts[1] : '';

                    availStages.forEach(function (stage) {
                        var assetPath = FILTERED + col + '/' + stage + '/' + periodName;
                        var img = ee.Image(assetPath);

                        var stageRows = calculateArea(img, territoryObj.image, lulcYear);

                        var enrichedRows = stageRows.map(function (f) {
                            var tId = f.get('territory_id');
                            var regName = regionList.get(ee.Number(tId).subtract(1));
                            
                            return f.set({
                                'period': periodName,
                                'year': yrStr,
                                'month': moStr,
                                'country': COUNTRY,
                                'collection': col,
                                'stage': stage,
                                'campaign': CAMPAIGN,
                                'periodicity': moStr ? 'monthly' : 'annual',
                                'region_name': regName
                            });
                        });

                        allRowsList.push(enrichedRows);
                    });
                });

                // 🎯 FILTRADO PRINCIPAL: Solo registros con área quemada > 0 ha
                var finalCollection = ee.FeatureCollection(allRowsList)
                    .flatten()
                    .filter(ee.Filter.gt('area_burned_ha', 0));

                var fileName = 'm8_stats_' + COUNTRY + '_' + CAMPAIGN + '_' + col;
                var baseFlag = ('M8_' + COUNTRY + '_' + CAMPAIGN + '_' + col).replace(/[^a-zA-Z0-9_]/g, '_').substring(0, 60);
                var selectorsList = [
                    'period', 'year', 'month', 'country', 'collection', 'stage', 'campaign', 'periodicity', 'region_name',
                    'lulc_id', 'lulc_nivel_0', 'lulc_nivel_1', 'lulc_nivel_2', 'area_burned_ha'
                ];

                // 1. Exportación a Cloud Storage (GCS)
                if (EXPORT_TARGET === 'GCS' || EXPORT_TARGET === 'BOTH') {
                    var fullPathGCS = EXPORT_PREFIX + fileName;
                    Export.table.toCloudStorage({
                        collection: finalCollection,
                        description: (baseFlag + '_GCS').substring(0, 80),
                        bucket: EXPORT_BUCKET,
                        fileNamePrefix: fullPathGCS,
                        fileFormat: 'CSV',
                        selectors: selectorsList
                    });
                    print('  🚀 Task generada (GCS): gs://' + EXPORT_BUCKET + '/' + fullPathGCS + '.csv');
                    totalExports++;
                }

                // 2. Exportación a Google Drive
                if (EXPORT_TARGET === 'DRIVE' || EXPORT_TARGET === 'BOTH') {
                    Export.table.toDrive({
                        collection: finalCollection,
                        description: (baseFlag + '_DRIVE').substring(0, 80),
                        folder: DRIVE_FOLDER,
                        fileNamePrefix: fileName,
                        fileFormat: 'CSV',
                        selectors: selectorsList
                    });
                    print('  📁 Task generada (Drive): ' + DRIVE_FOLDER + '/' + fileName + '.csv');
                    totalExports++;
                }
            });
        });
    });

    print('=== M8 preparado: ' + totalExports + ' tarea(s) enviada(s) a la pestaña Tasks ===');
});