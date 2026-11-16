// Preset: peru / MONITOR_01 / monthly
/* ==============================================================================
MAPBIOMAS FUEGO - M7 - ft02
Filtros por Uso y Cobertura del Suelo (LULC) y Píxeles Aislados

📅 FECHA: 11/2026
🏷️ VERSIÓN: 6.0 (multi-país, multi-campaña, mensual/anual)

📌 ¿QUÉ HACE ESTE SCRIPT?
1. Toma las detecciones filtradas por probabilidad de la etapa de entrada (ft01).
2. Aplica una máscara espacial para descartar detecciones sobre coberturas de suelo no permitidas
   (ej. agua, áreas urbanas) utilizando MapBiomas Uso y Cobertura del Suelo.
3. Permite aplicar un filtro LULC global o personalizar las clases por región.
4. Remueve el ruido de píxeles aislados o parches pequeños mediante 'minConnectedPixels'.
5. Genera la colección nacional filtrada (ft02) lista para la pestaña Tasks de GEE.

⚙️ PARÁMETROS A MODIFICAR POR EL USUARIO:
- COUNTRY: key usada en las rutas (ej. 'peru', 'brazil', 'drc').
- COUNTRY_LABEL: nombre para mostrar (ej. 'Peru', 'Brazil', 'DRC').
- CAMPAIGN: campaña (ej. 'MONITOR_01', 'COLLECTION_02').
- COLLECTION_BASE: propuesta de trabajo (ej. 'propuesta_a').
- LULC_ASSET: asset de cobertura y uso del suelo del país.
- LULC_LAST_YEAR: último año disponible en el asset LULC.
- REGION_PROPERTY: propiedad de región ('region', 'region_nam' o 'region_name').
- REGIONS_ASSET: asset de regiones (opcional; vacío usa el patrón estándar).
- USE_PER_REGION_LULC: true usa lulcMasksByRegion; false aplica LULC global.
- minConnectedPixels: umbral de píxeles aislados a eliminar.
- LULC_CLASSES_TO_MASK: clases LULC a excluir.
============================================================================== */

// 🌐 CONFIGURACIÓN DEL PAÍS Y RUTAS PRINCIPALES
var COUNTRY = 'peru';
var COUNTRY_LABEL = 'Peru';
var CAMPAIGN = 'MONITOR_01';
var COLLECTION_BASE = 'propuesta_a';

var CATALOG_ROOT = 'projects/mapbiomas-' + COUNTRY + '/assets/FIRE/CATALOG_01';
var CLASSIFICATIONS_ROOT = CATALOG_ROOT + '/' + CAMPAIGN + '/LIBRARY_CLASSIFICATIONS/';

// Opcional: asset de regiones (para Brazil u otro asset que no siga el patrón estándar).
var REGIONS_ASSET = '';
var REGIONS = REGIONS_ASSET ?
    ee.FeatureCollection(REGIONS_ASSET) :
    ee.FeatureCollection('projects/mapbiomas-' + COUNTRY + '/assets/FIRE/AUXILIARY_DATA/regiones_fuego_' + COUNTRY + '_v1');

// Asset de Uso y Cobertura del Suelo (LULC)
var LULC_ASSET = 'projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1';
var LULC_LAST_YEAR = 2025;
var LULC_PALETTE_ID = 'brazil';

// Aborta antes de crear/exportar si el presupuesto LULC no está configurado.
function assertLulcConfig() {
    if (!LULC_ASSET) {
        print('⚠ ft02: LULC_ASSET no configurado para ' + COUNTRY_LABEL + ' (' + COUNTRY + ' / ' + CAMPAIGN + '). Complete LULC_ASSET, LULC_LAST_YEAR y las clases antes de ejecutar.');
        throw new Error('LULC_ASSET no configurado para ' + COUNTRY_LABEL);
    }
    var bands;
    try {
        bands = ee.Image(LULC_ASSET).bandNames().getInfo() || [];
    } catch (e) {
        print('⚠ ft02: no se pudo leer LULC_ASSET para ' + COUNTRY_LABEL + ': ' + LULC_ASSET + '. Verifique el asset y LULC_LAST_YEAR.');
        throw new Error('LULC_ASSET inválido para ' + COUNTRY_LABEL);
    }
    if (bands.indexOf('classification_' + LULC_LAST_YEAR) === -1) {
        print('⚠ ft02: ' + LULC_ASSET + ' no contiene classification_' + LULC_LAST_YEAR + '. Ajuste LULC_LAST_YEAR o el asset para ' + COUNTRY_LABEL + '.');
        throw new Error('classification_' + LULC_LAST_YEAR + ' ausente en LULC_ASSET');
    }
}
assertLulcConfig();

var landcover = ee.Image(LULC_ASSET);

// 🔑 Propiedad de región
var REGION_PROPERTY = 'region'; // 'region_nam' / 'region_name' para las colecciones nuevas
var USE_PER_REGION_LULC = true; // false → aplica LULC_CLASSES_TO_MASK global
var SCALE = 10;

// ⚙️ CONFIGURACIÓN DEL USUARIO (MODIFICAR AQUÍ)
var minConnectedPixels = 2; // Umbral de píxeles aislados (<= 2 píxeles)

// 🎭 CLASES LULC GENERALES A ENMASCARAR (Base por defecto)
var LULC_CLASSES_TO_MASK = [31, 33, 34];

// 🗺️ ASIGNACIÓN POR IDENTIFICADOR DE REGIÓN (opcional, usado si USE_PER_REGION_LULC=true)
// Las claves pueden ser números (propiedad 'region') o nombres ('region_nam'/'region_name').
var lulcMasksByRegion = {
    1:  LULC_CLASSES_TO_MASK,
    2:  LULC_CLASSES_TO_MASK,
    3:  LULC_CLASSES_TO_MASK,
    4:  LULC_CLASSES_TO_MASK,
    5:  LULC_CLASSES_TO_MASK,
    6:  LULC_CLASSES_TO_MASK,
    7:  LULC_CLASSES_TO_MASK,
    8:  LULC_CLASSES_TO_MASK,
    9:  LULC_CLASSES_TO_MASK,
    10: LULC_CLASSES_TO_MASK
};

// Genera automáticamente el texto informativo para la consola e interfaz
var LULC_TEXT_BASE = '[' + LULC_CLASSES_TO_MASK.join(', ') + ']';

// ------------------------------------------------------------------------------
// CONFIGURACIÓN INTERNA Y RUTAS
// ------------------------------------------------------------------------------
var inputStage  = 'ft01';
var outputStage = 'ft02';

// Proyección del último año LULC hacia años actuales
var currentYear = new Date().getFullYear();
for (var year = LULC_LAST_YEAR + 1; year <= currentYear; year++) {
    landcover = landcover.addBands(landcover.select('classification_' + LULC_LAST_YEAR).rename('classification_' + year));
}
var LULC_PALETTE = require('users/mapbiomas/modules:Palettes.js').get(LULC_PALETTE_ID);

var filteredPath = CLASSIFICATIONS_ROOT + 'FILTERED/';
var inputCollection  = filteredPath + COLLECTION_BASE + '/' + inputStage;
var outputCollection = filteredPath + COLLECTION_BASE + '/' + outputStage;

function isYearlyPeriod(periodName) {
    return periodName.indexOf('_') === -1;
}

function periodStartEnd(periodName) {
    var parts = periodName.split('_');
    var y = parseInt(parts[0], 10);
    var m = parts.length > 1 ? parseInt(parts[1], 10) : null;
    var start = m ? ee.Date.fromYMD(y, m, 1) : ee.Date.fromYMD(y, 1, 1);
    var end = m ? start.advance(1, 'month') : start.advance(1, 'year');
    return { start: start, end: end, periodicity: m ? 'monthly' : 'yearly' };
}

function setTimeProperties(image, periodName) {
    var p = periodStartEnd(periodName);
    return image.set({
        'system:time_start': p.start.millis(),
        'system:time_end': p.end.millis()
    });
}

function ensureFolder(pathName) {
    var parts = pathName.split('/');
    var current = CLASSIFICATIONS_ROOT;
    for (var i = 0; i < parts.length; i++) {
        current += parts[i];
        try { ee.data.getAsset(current); }
        catch (e) {
            var isImageCollection = (i === parts.length - 1 && /^ft\d\d$/.test(parts[i]));
            ee.data.createAsset({ type: isImageCollection ? 'IMAGE_COLLECTION' : 'FOLDER' }, current);
        }
        current += '/';
    }
}

function loadPeriodMosaic(periodName) {
    var yearly = isYearlyPeriod(periodName);
    var sensor = yearly ? 'LANDSAT' : 'SENTINEL2';
    var periodFolder = yearly ? 'YEARLY' : 'MONTHLY';
    var bands = ['swir1', 'nir', 'red'];
    var mosaicPath = CATALOG_ROOT + '/LIBRARY_IMAGES/' + sensor + '/' + periodFolder + '/MINNBR/';
    var prefix = 'image_' + COUNTRY + '_fire_' + sensor.toLowerCase() + '_minnbr_';
    var mosaicImage = ee.Image().select();

    bands.forEach(function(b) {
        try {
            var bi = ee.ImageCollection(mosaicPath + b)
                       .filter(ee.Filter.eq('system:index', prefix + b + '_' + periodName))
                       .mosaic();
            mosaicImage = mosaicImage.addBands(ee.Image(ee.Algorithms.If(bi.bandNames().size().gt(0), bi, ee.Image(0).rename(b).updateMask(0))).select([0], [b]), null, true);
        } catch(e) {
            mosaicImage = mosaicImage.addBands(ee.Image(0).rename(b).updateMask(0), null, true);
        }
    });

    return ui.Map.Layer(mosaicImage, {
        bands: ['swir1', 'nir', 'red'],
        min: 3,
        max: 40,
        gamma: 0.85
    }, periodName + ' | Mosaico ' + sensor + ' (SWIR1, NIR, RED)');
}

// Aplica la máscara LULC global o por región y devuelve la imagen y la máscara combinada.
function applyLulcMask(lulcYearImage, sourceImage) {
    var maskedImage = sourceImage;
    var combinedLulcMask = ee.Image(0);
    var regionKeys = Object.keys(lulcMasksByRegion);

    if (USE_PER_REGION_LULC && regionKeys.length > 0) {
        regionKeys.forEach(function(regionId) {
            var regionClasses = lulcMasksByRegion[regionId];
            var key = /^\d+$/.test(regionId) ? parseInt(regionId, 10) : regionId;
            var regionGeometry = REGIONS.filter(ee.Filter.eq(REGION_PROPERTY, key));
            var regionRaster = ee.Image(0).paint(regionGeometry, 1);
            var regionLulcMask = lulcYearImage.eq(regionClasses).reduce(ee.Reducer.sum()).gte(1);
            regionLulcMask = regionLulcMask.multiply(regionRaster);
            combinedLulcMask = combinedLulcMask.max(regionLulcMask);
            maskedImage = maskedImage.updateMask(regionLulcMask.neq(1));
        });
    } else {
        combinedLulcMask = lulcYearImage.eq(LULC_CLASSES_TO_MASK).reduce(ee.Reducer.sum()).gte(1);
        maskedImage = maskedImage.updateMask(combinedLulcMask.neq(1));
    }

    return { maskedImage: maskedImage, mask: combinedLulcMask };
}

function buildLayerPanel(images, layerFactory) {
    var panel = ui.Panel({
        layout: ui.Panel.Layout.flow('vertical'),
        style: {
            position: 'bottom-left',
            maxHeight: '60%',
            width: '210px',
            padding: '4px',
            backgroundColor: 'rgba(255,255,255,0.92)',
            border: '1px solid #ccc',
            borderRadius: '4px'
        }
    });
    panel.add(ui.Label('PERÍODOS DISPONIBLES', { fontWeight: 'bold', fontSize: '11px', margin: '2px' }));
    var activeLayers = [];

    images.forEach(function (img, idx) {
        var periodName = img.id.split('/').pop();
        var checkbox = ui.Checkbox({ label: periodName, value: idx === 0, style: { fontSize: '10px', margin: '1px 2px' } });
        checkbox.onChange(function (checked) {
            if (!checked) return;
            for (var w = 0; w < panel.widgets().length(); w++) {
                var widget = panel.widgets().get(w);
                if (widget !== checkbox && widget.setValue) try { widget.setValue(false); } catch (e) {}
            }
            activeLayers.forEach(function (layer) { Map.layers().remove(layer); });
            activeLayers = [];
            var newLayers = layerFactory(img);
            newLayers.forEach(function (entry) {
                Map.layers().add(entry.layer);
                activeLayers.push(entry.layer);
            });
        });
        panel.add(checkbox);
    });

    if (images.length > 0) {
        var firstLayers = layerFactory(images[0]);
        firstLayers.forEach(function (entry) {
            Map.layers().add(entry.layer);
            activeLayers.push(entry.layer);
        });
    }
    Map.add(panel);
    return panel;
}

function extractYear(name) {
    var match = name.match(/^(\d{4})/);
    return match ? parseInt(match[1], 10) : new Date().getFullYear();
}

print('=== ft02 - Filtros de Clases LULC y Píxeles Aislados ===');
print('País: ' + COUNTRY_LABEL + ' (' + COUNTRY + ')');
print('Campaña: ' + CAMPAIGN + ' | Propuesta: ' + COLLECTION_BASE);
print('Región: ' + REGION_PROPERTY + ' | LULC por región: ' + USE_PER_REGION_LULC);
print('LULC asset: ' + LULC_ASSET);
print('Entrada (Input ft01): ' + inputCollection);
print('Salida (Output ft02): ' + outputCollection);
print('Clases LULC base a enmascarar: ' + LULC_TEXT_BASE);

ensureFolder('FILTERED/' + COLLECTION_BASE + '/' + inputStage);
ensureFolder('FILTERED/' + COLLECTION_BASE + '/' + outputStage);

ee.data.listAssets(inputCollection, {}, function (result) {
    var images = [];
    if (result && result.assets) {
        images = result.assets.filter(function (a) { return a.type === 'IMAGE'; });
    }

    if (images.length === 0) {
        print('No se encontraron imágenes en la colección de entrada (ft01).');
    } else {
        Map.setOptions('HYBRID');
        Map.addLayer(REGIONS.style({ color: 'ffffff', fillColor: '00000000', width: 1 }), {}, 'Límites de Regiones');
        Map.centerObject(REGIONS);

        buildLayerPanel(images, function (img) {
            var periodName = img.id.split('/').pop();
            var sourceImage = ee.Image(img.id);
            var imageYear = extractYear(periodName);

            var lulcYearImage = landcover.select('classification_' + imageYear);
            var lulcResult = applyLulcMask(lulcYearImage, sourceImage);
            var maskedImage = lulcResult.maskedImage;
            var combinedLulcMask = lulcResult.mask;

            // 2. Filtrado de píxeles aislados (<= minConnectedPixels)
            var connections = maskedImage.select(0).gt(0).connectedPixelCount({ maxSize: 100, eightConnected: false });
            maskedImage = maskedImage.where(connections.lte(minConnectedPixels), 0).selfMask();

            // CAPAS VISUALES
            var mosaicLayer = loadPeriodMosaic(periodName);
            var lulcLayer = ui.Map.Layer(lulcYearImage, { min: 0, max: 72, palette: LULC_PALETTE }, 'MapBiomas ' + COUNTRY_LABEL + ' LULC ' + imageYear, false);
            var lulcVisualMaskLayer = ui.Map.Layer(combinedLulcMask.selfMask(), { palette: ['00FFFF'] }, 'Máscara de Clases LULC Excluidas (Cian)');
            var beforeLayer = ui.Map.Layer(sourceImage.select(0).selfMask(), { min: 0, max: 1000, palette: ['000000'] }, periodName + ' | ANTES (ft01 - Negro)');
            var afterLayer = ui.Map.Layer(maskedImage.select(0).selfMask(), { min: 0, max: 1000, palette: ['FF0000'] }, periodName + ' | DESPUÉS (ft02 - Rojo)');

            return [
                { layer: mosaicLayer },
                { layer: lulcLayer },
                { layer: lulcVisualMaskLayer },
                { layer: beforeLayer },
                { layer: afterLayer }
            ];
        });

        // TAREAS DE EXPORTACIÓN A FT01
        var exportCount = 0;
        images.forEach(function (img) {
            var periodName = img.id.split('/').pop();
            var sourceImage = ee.Image(img.id);
            var imageYear = extractYear(periodName);

            var lulcYearImage = landcover.select('classification_' + imageYear);
            var maskedImage = applyLulcMask(lulcYearImage, sourceImage).maskedImage;

            var connections = maskedImage.select(0).gt(0).connectedPixelCount({ maxSize: 100, eightConnected: false });
            maskedImage = maskedImage.where(connections.lte(minConnectedPixels), 0).selfMask();
            maskedImage = setTimeProperties(maskedImage.copyProperties(sourceImage).set('filter_stage', 'ft02'), periodName);

            var destAsset = outputCollection + '/' + periodName;
            var periodicityTag = isYearlyPeriod(periodName) ? 'ANNUAL' : 'MONTHLY';
            var exportFlag = ('M7_ft02_' + COUNTRY.toUpperCase() + '_' + CAMPAIGN + '_' + COLLECTION_BASE + '_' + periodName + '_' + periodicityTag).replace(/[^a-zA-Z0-9_]/g, '_').substring(0, 80);

            try {
                ee.data.getAsset(destAsset);
                print('  OK (Ya existe): ' + periodName);
            } catch (e) {
                exportCount++;
                print('  Exportando a Asset ft02: ' + periodName);
                Export.image.toAsset({
                    image: ee.Image(maskedImage).toInt16(),
                    description: exportFlag,
                    assetId: destAsset,
                    pyramidingPolicy: 'mode',
                    region: REGIONS.geometry().bounds(),
                    scale: SCALE,
                    maxPixels: 1e13,
                    overwrite: true
                });
            }
        });
        print('Total de imágenes listas para exportar a ft02: ' + exportCount);
    }
});
print('=== ft02 preparado ===');