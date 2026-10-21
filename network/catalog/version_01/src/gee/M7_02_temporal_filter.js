/* ==============================================================================
MAPBIOMAS FUEGO - MONITOR_01 - ft02_temporal_memory_filter
Filtro Temporal de Memoria Mensual (Salida: ft02)

📅 FECHA: 09/2026
🏷️ VERSIÓN: 5.3 (Fix UI Title Bug & Robust Data Pipeline)
============================================================================== */

var CATALOG_ROOT = 'projects/mapbiomas-peru/assets/FIRE/CATALOG_01';
var CLASSIFICATIONS_ROOT = CATALOG_ROOT + '/MONITOR_01/LIBRARY_CLASSIFICATIONS/';
var REGIONS = ee.FeatureCollection('projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1');
var SCALE = 10;

// ⚙️ CONFIGURACIÓN DEL USUARIO

// Nombre de la colección 
var COLLECTION_BASE = 'propuesta_a';     

// Cantidad de meses hacia atrás a revisar 
// para evaluar la recurrencia de la cicatriz
var monthsToLookBack = 4;                

// ------------------------------------------------------------------------------
// CONFIGURACIÓN INTERNA Y RUTAS (ENTRADA: ft01 -> SALIDA: ft02)
// ------------------------------------------------------------------------------
var inputStage   = 'ft01';
var outputStage  = 'ft02';
var CAMPAIGN     = 'MONITOR_01';

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

function buildPreviousMonthsMaskServer(sourceImg, inputCol) {
    var currentDate = ee.Date(sourceImg.get('system:time_start'));
    var startDate = currentDate.advance(-monthsToLookBack, 'month');
    
    var pastCol = inputCol
        .filter(ee.Filter.gte('system:time_start', startDate.millis()))
        .filter(ee.Filter.lt('system:time_start', currentDate.millis()));

    var pastImageCount = pastCol.size();
    
    var pastMask = ee.Image(ee.Algorithms.If(
        pastImageCount.gt(0),
        pastCol.map(function(img) { return img.select(0).unmask(0).gt(0); }).reduce(ee.Reducer.max()),
        ee.Image(0)
    ));

    return {
        mask: pastMask,
        hasPastData: pastImageCount
    };
}

function buildLayerPanel(images, layerFactory) {
    var titleLabel = ui.Label('PERÍODOS DISPONIBLES', { fontWeight: 'bold', fontSize: '11px', margin: '2px' });
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
    panel.add(titleLabel);
    
    var activeLayers = [];
    var checkboxes = [];

    images.forEach(function (img, idx) {
        var periodName = img.id.split('/').pop();
        var checkbox = ui.Checkbox({ label: periodName, value: idx === 0, style: { fontSize: '10px', margin: '1px 2px' } });
        checkboxes.push(checkbox);

        checkbox.onChange(function (checked) {
            if (!checked) return;
            
            checkboxes.forEach(function(cb) {
                if (cb !== checkbox) cb.setValue(false);
            });

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

print('=== ft02 — Filtro Temporal de Memoria Mensual ===');
print('Entrada (Input ft01): ' + inputCollection);
print('Salida (Output ft02): ' + outputCollection);
print('Meses a revisar hacia atrás: ' + monthsToLookBack + ' mes(es)');

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
        Map.addLayer(REGIONS.style({ color: 'ffffff', fillColor: '00000000', width: 1 }), {}, 'Límites de Regiones');
        Map.centerObject(REGIONS);

        var eeInputCol = ee.ImageCollection(inputCollection).map(function(img) {
            var periodName = ee.String(img.get('system:index'));
            var parts = periodName.split('_');
            var y = ee.Number.parse(parts.get(0));
            var m = ee.Number.parse(parts.get(1));
            var start = ee.Date.fromYMD(y, m, 1);
            var end = start.advance(1, 'month');
            return img.set({
                'system:time_start': start.millis(),
                'system:time_end': end.millis()
            });
        });

        buildLayerPanel(images, function (img) {
            var periodName = img.id.split('/').pop();
            var sourceImage = setTimeProperties(ee.Image(img.id), periodName);

            var pastResult = buildPreviousMonthsMaskServer(sourceImage, eeInputCol);
            
            var filteredImage = ee.Image(ee.Algorithms.If(
                pastResult.hasPastData.gt(0),
                sourceImage.updateMask(pastResult.mask.neq(1)).selfMask(),
                sourceImage
            ));

            var mosaicLayer = loadSentinelMosaic(periodName);
            var beforeLayer = ui.Map.Layer(sourceImage.select(0).selfMask(), { min: 0, max: 1000, palette: ['000000'] }, periodName + ' | ANTES (ft01 - Negro)');
            var afterLayer  = ui.Map.Layer(filteredImage.select(0).selfMask(), { min: 0, max: 1000, palette: ['FF0000'] }, periodName + ' | DESPUÉS (ft02 - Rojo)');

            return [
                { layer: mosaicLayer },
                { layer: beforeLayer },
                { layer: afterLayer }
            ];
        });

        var exportCount = 0;
        images.forEach(function (img) {
            var periodName = img.id.split('/').pop();
            var sourceImage = setTimeProperties(ee.Image(img.id), periodName);

            var pastResult = buildPreviousMonthsMaskServer(sourceImage, eeInputCol);
            
            var filteredImage = ee.Image(ee.Algorithms.If(
                pastResult.hasPastData.gt(0),
                sourceImage.updateMask(pastResult.mask.neq(1)).selfMask(),
                sourceImage
            ));
                
            filteredImage = setTimeProperties(filteredImage.copyProperties(sourceImage).set('filter_stage', 'ft02'), periodName);

            var destAsset = outputCollection + '/' + periodName;

            try {
                ee.data.getAsset(destAsset);
                print('  OK (Ya existe): ' + periodName);
            } catch (e) {
                exportCount++;
                print('  Exportando a Asset ft02: ' + periodName);
                Export.image.toAsset({
                    image: ee.Image(filteredImage).toInt16(),
                    description: (CAMPAIGN + '_ft02_' + COLLECTION_BASE + '_' + periodName).substring(0, 80).replace(/[^a-zA-Z0-9_]/g, '_'),
                    assetId: destAsset,
                    pyramidingPolicy: 'mode',
                    region: REGIONS.geometry().bounds(),
                    scale: SCALE,
                    maxPixels: 1e13,
                });
            }
        });
        print('Total de imágenes listas para exportar a ft02: ' + exportCount);
    }
});
print('=== ft02 finalizado ===');