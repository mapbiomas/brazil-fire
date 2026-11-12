/* ==============================================================================
MAPBIOMAS FUEGO - MONITOR_01 - ft01_lulc_and_solitary_filters
Filtros de Cobertura de Suelo (LULC) y Píxeles Aislados (Salida: ft01)

📅 FECHA: 09/2026
🏷️ VERSIÓN: 4.8

================================================================================
¿QUÉ HACE ESTE SCRIPT? (DESCRIPCIÓN)
================================================================================
Este script es la primera etapa de filtrado espacial (Paso 1 / ft01):
1. MÁSCARA LULC (MapBiomas Perú Col. 3): Elimina detecciones de fuego exclusivamente
   en clases de AGUA [31, 33, 34] especificadas directamente para cada región.
2. PÍXELES AISLADOS (connectedPixelCount): Agrupa los píxeles contiguos y elimina
   aquellos parches pequeños o "huérfanos" cuyo tamaño total sea menor o igual al
   umbral definido (ej: <= 2 píxeles).

🎨 VISUALIZACIÓN EN MAPA:
- ANTES (ft00): Color NEGRO (#000000)
- DESPUÉS (ft01): Color ROJO (#FF0000)
- MOSAICO BASE: Composición Falso Color Sentinel-2 (SWIR1, NIR, RED)

================================================================================
⚠️ PARÁMETROS MODIFICABLES POR EL USUARIO
================================================================================
1. COLLECTION_BASE: Nombre de la colección / carpeta de trabajo (ej: 'propuesta_a').
2. minConnectedPixels: Cantidad máxima de píxeles conectados para considerar un
   grupo como "aislado" (Por defecto: 2 píxeles = 200 m²).
3. lulcMasks: Diccionario con las clases de agua asignadas directamente por región.
================================================================================ */

var CATALOG_ROOT = 'projects/mapbiomas-peru/assets/FIRE/CATALOG_01';
var CLASSIFICATIONS_ROOT = CATALOG_ROOT + '/MONITOR_01/LIBRARY_CLASSIFICATIONS/';
var REGIONS = ee.FeatureCollection('projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1');
var REGION_PROPERTY = 'region_nam';
var SCALE = 10;

// ⚙️ CONFIGURACIÓN DEL USUARIO (MODIFICAR AQUÍ)
// Nombre de su colección / variante
var COLLECTION_BASE = 'propuesta_a';
// Umbral de píxeles aislados (<= 2 píxeles)
var minConnectedPixels = 2;              

// Clases de LULC asignadas directamente a cada región
var lulcMasks = {
    'region1':  [31, 33, 34],
    'region2':  [31, 33, 34],
    'region3':  [31, 33, 34],
    'region4':  [31, 33, 34],
    'region5':  [31, 33, 34],
    'region6':  [31, 33, 34],
    'region7':  [31, 33, 34],
    'region8':  [31, 33, 34],
    'region9':  [31, 33, 34],
    'region10': [31, 33, 34],
};

// ------------------------------------------------------------------------------
// CONFIGURACIÓN INTERNA Y RUTAS (ENTRADA: ft00 -> SALIDA: ft01)
// ------------------------------------------------------------------------------
var inputStage  = 'ft00';
var outputStage = 'ft01';
var CAMPAIGN = 'MONITOR_01';

var landcover = ee.Image('projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1');
// Duplicar la banda de 2025 para años posteriores (2026+) fuera del rango de la colección
var currentYear = new Date().getFullYear();
for (var year = 2026; year <= currentYear; year++) {
    landcover = landcover.addBands(landcover.select('classification_2025').rename('classification_' + year));
}
var LULC_PALETTE = require('users/mapbiomas/modules:Palettes.js').get('brazil');

var filteredPath = CLASSIFICATIONS_ROOT + 'FILTERED/';
var inputCollection  = filteredPath + COLLECTION_BASE + '/' + inputStage;
var outputCollection = filteredPath + COLLECTION_BASE + '/' + outputStage;

function setTimeProperties(image, periodName) {
    var parts = periodName.split('_');
    var y = parseInt(parts[0], 10);
    var m = parseInt(parts[1], 10);
    var start = ee.Date.fromYMD(y, m, 1);
    var end = start.advance(1, 'month');
    return image.set({
        'system:time_start': start.millis(),
        'system:time_end': end.millis()
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

// Carga del mosaico Sentinel-2 en falso color (SWIR1, NIR, RED)
function loadSentinelMosaic(periodName) {
    var bands = ['swir1', 'nir', 'red'];
    var mosaicPath = CATALOG_ROOT + '/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR/';
    var mosaicImage = ee.Image().select();

    bands.forEach(function(b) {
        try {
            var bi = ee.ImageCollection(mosaicPath + b)
                       .filter(ee.Filter.eq('system:index', 'image_peru_fire_sentinel2_minnbr_' + b + '_' + periodName))
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
    }, periodName + ' | Mosaico S2 (SWIR1, NIR, RED)');
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

print('=== ft01 — Filtros de Agua (LULC) y Píxeles Aislados ===');
print('Entrada (Input ft00): ' + inputCollection);
print('Salida (Output ft01): ' + outputCollection);
print('Umbral de Píxeles Aislados: <= ' + minConnectedPixels + ' píxeles');

ensureFolder('FILTERED/' + COLLECTION_BASE + '/' + inputStage);
ensureFolder('FILTERED/' + COLLECTION_BASE + '/' + outputStage);

ee.data.listAssets(inputCollection, {}, function (result) {
    var images = [];
    if (result && result.assets) {
        images = result.assets.filter(function (a) { return a.type === 'IMAGE'; });
    }

    if (images.length === 0) {
        print('No se encontraron imágenes en la colección de entrada (ft00).');
    } else {
        Map.addLayer(REGIONS.style({ color: 'ffffff', fillColor: '00000000', width: 1 }), {}, 'Límites de Regiones');
        Map.centerObject(REGIONS);
        Map.addLayer(landcover.select('classification_2024').selfMask(), { min: 0, max: 72, palette: LULC_PALETTE }, 'MapBiomas Perú (LULC)', false);

        buildLayerPanel(images, function (img) {
            var periodName = img.id.split('/').pop();
            var sourceImage = ee.Image(img.id);
            var imageYear = extractYear(periodName);

            var maskedImage = sourceImage;

            // 1. Filtrado por Clases de Agua LULC
            Object.keys(lulcMasks).forEach(function (regionName) {
                var regionMaskClasses = lulcMasks[regionName];
                var regionGeometry = REGIONS.filter(ee.Filter.eq(REGION_PROPERTY, regionName));
                var regionRaster = ee.Image(0).paint(regionGeometry, 1);

                var lulcMask = landcover.select('classification_' + imageYear).eq(regionMaskClasses).reduce(ee.Reducer.sum()).gte(1);
                lulcMask = lulcMask.multiply(regionRaster);

                var finalMask = lulcMask.neq(1);
                maskedImage = maskedImage.updateMask(finalMask);
            });

            // 2. Filtrado de píxeles aislados (select(0))
            var connections = maskedImage.select(0).gt(0).connectedPixelCount({ maxSize: 100, eightConnected: false });
            maskedImage = maskedImage.where(connections.lte(minConnectedPixels), 0).selfMask();

            var mosaicLayer = loadSentinelMosaic(periodName);
            var beforeLayer = ui.Map.Layer(sourceImage.select(0).selfMask(), { min: 0, max: 1000, palette: ['000000'] }, periodName + ' | ANTES (ft00 - Negro)');
            var afterLayer = ui.Map.Layer(maskedImage.select(0).selfMask(), { min: 0, max: 1000, palette: ['FF0000'] }, periodName + ' | DESPUÉS (ft01 - Rojo)');

            return [
                { layer: mosaicLayer },
                { layer: beforeLayer },
                { layer: afterLayer }
            ];
        });

        var exportCount = 0;
        images.forEach(function (img) {
            var periodName = img.id.split('/').pop();
            var sourceImage = ee.Image(img.id);
            var imageYear = extractYear(periodName);

            var maskedImage = sourceImage;

            // 1. Filtrado por Cobertura de Suelo (Agua) por región
            Object.keys(lulcMasks).forEach(function (regionName) {
                var regionMaskClasses = lulcMasks[regionName];
                var regionGeometry = REGIONS.filter(ee.Filter.eq(REGION_PROPERTY, regionName));
                var regionRaster = ee.Image(0).paint(regionGeometry, 1);

                var lulcMask = landcover.select('classification_' + imageYear).eq(regionMaskClasses).reduce(ee.Reducer.sum()).gte(1);
                lulcMask = lulcMask.multiply(regionRaster);

                var finalMask = lulcMask.neq(1);
                maskedImage = maskedImage.updateMask(finalMask);
            });

            // 2. Filtrado de píxeles aislados (select(0))
            var connections = maskedImage.select(0).gt(0).connectedPixelCount({ maxSize: 100, eightConnected: false });
            maskedImage = maskedImage.where(connections.lte(minConnectedPixels), 0).selfMask();
            maskedImage = setTimeProperties(maskedImage.copyProperties(sourceImage).set('filter_stage', 'ft01'), periodName);

            var destAsset = outputCollection + '/' + periodName;

            try {
                ee.data.getAsset(destAsset);
                print('  OK (Ya existe): ' + periodName);
            } catch (e) {
                exportCount++;
                print('  Exportando a Asset ft01: ' + periodName);
                Export.image.toAsset({
                    image: ee.Image(maskedImage).toInt16(),
                    description: (CAMPAIGN + '_ft01_' + COLLECTION_BASE + '_' + periodName).substring(0, 80).replace(/[^a-zA-Z0-9_]/g, '_'),
                    assetId: destAsset,
                    pyramidingPolicy: 'mode',
                    region: REGIONS.geometry().bounds(),
                    scale: SCALE,
                    maxPixels: 1e13,
                    overwrite: true
                });
            }
        });
        print('Total de imágenes listas para exportar a ft01: ' + exportCount);
    }
});
print('=== ft01 finalizado ===');