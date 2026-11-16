// Preset: peru / MONITOR_01 / monthly
/* ==============================================================================
MAPBIOMAS FUEGO - M7 - ft03
Filtro Temporal de Memoria (Salida: ft03)

📅 FECHA: 11/2026
🏷️ VERSIÓN: 6.0 (multi-país, multi-campaña, mensual/anual)

📌 ¿QUÉ HACE ESTE SCRIPT?
1. Toma la colección filtrada por cobertura de suelo (ft02) como punto de partida.
2. Analiza el historial reciente evaluando los períodos previos configurados.
3. Identifica y elimina la sobreestimación por recurrencia/persistencia de cicatrices
   que permanecen visibles durante períodos consecutivos pero corresponden al mismo evento.
4. Genera la colección nacional filtrada por persistencia temporal (ft03) lista en Tasks.

⚙️ PARÁMETROS A MODIFICAR POR EL USUARIO:
- COUNTRY: key usada en las rutas (ej. 'peru', 'brazil', 'drc').
- COUNTRY_LABEL: nombre para mostrar.
- CAMPAIGN: campaña (ej. 'MONITOR_01', 'COLLECTION_02').
- COLLECTION_BASE: propuesta de trabajo (ej. 'propuesta_a').
- REGIONS_ASSET: asset de regiones (opcional; vacío usa el patrón estándar).
- monthsToLookBack: meses atrás para campañas mensuales.
- yearsToLookBack: años atrás para campañas anuales.
============================================================================== */

// 🌐 CONFIGURACIÓN DEL PAÍS Y RUTAS PRINCIPALES
var COUNTRY = 'peru';
var COUNTRY_LABEL = 'Peru';
var CAMPAIGN = 'MONITOR_01';
var COLLECTION_BASE = 'propuesta_a';

var CATALOG_ROOT = 'projects/mapbiomas-' + COUNTRY + '/assets/FIRE/CATALOG_01';
var CLASSIFICATIONS_ROOT = CATALOG_ROOT + '/' + CAMPAIGN + '/LIBRARY_CLASSIFICATIONS/';

var REGIONS_ASSET = '';
var REGIONS = REGIONS_ASSET ?
    ee.FeatureCollection(REGIONS_ASSET) :
    ee.FeatureCollection('projects/mapbiomas-' + COUNTRY + '/assets/FIRE/AUXILIARY_DATA/regiones_fuego_' + COUNTRY + '_v1');
var SCALE = 10;

// ⚙️ CONFIGURACIÓN DEL USUARIO (MODIFICAR AQUÍ)
var monthsToLookBack = 4; // campañas mensuales
var yearsToLookBack = 1;  // campañas anuales

// ------------------------------------------------------------------------------
// CONFIGURACIÓN INTERNA Y RUTAS (ENTRADA: ft02 -> SALIDA: ft03)
// ------------------------------------------------------------------------------
var inputStage   = 'ft02';
var outputStage  = 'ft03';

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

function buildPreviousPeriodsMaskServer(sourceImg, inputCol, periodName) {
    var currentDate = ee.Date(sourceImg.get('system:time_start'));
    var yearly = isYearlyPeriod(periodName);
    var lookback = yearly ? yearsToLookBack : monthsToLookBack;
    var unit = yearly ? 'year' : 'month';
    var startDate = currentDate.advance(-lookback, unit);

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

print('=== ft03 - Filtro Temporal de Memoria ===');
print('País: ' + COUNTRY_LABEL + ' (' + COUNTRY + ')');
print('Campaña: ' + CAMPAIGN + ' | Propuesta: ' + COLLECTION_BASE);
print('Entrada (Input ft02): ' + inputCollection);
print('Salida (Output ft03): ' + outputCollection);
print('Meses atrás (mensual): ' + monthsToLookBack + ' | Años atrás (anual): ' + yearsToLookBack);

ensureFolder('FILTERED/' + COLLECTION_BASE + '/' + inputStage);
ensureFolder('FILTERED/' + COLLECTION_BASE + '/' + outputStage);

ee.data.listAssets(inputCollection, {}, function (result) {
    var images = [];

    if (result && result.assets) {
        images = result.assets.filter(function (a) { return a.type === 'IMAGE'; });
    }

    if (images.length === 0) {
        print('No se encontraron imágenes en la colección de entrada (ft02).');
    } else {
        Map.addLayer(REGIONS.style({ color: 'ffffff', fillColor: '00000000', width: 1 }), {}, 'Límites de Regiones');
        Map.centerObject(REGIONS);

        var eeInputCol = ee.ImageCollection(inputCollection).map(function(img) {
            var periodName = ee.String(img.get('system:index'));
            var parts = periodName.split('_');
            var hasMonth = parts.size().gt(1);
            var y = ee.Number.parse(parts.get(0));
            var m = ee.Number(ee.Algorithms.If(hasMonth, ee.Number.parse(parts.get(1)), 1));
            var start = ee.Date.fromYMD(y, m, 1);
            var end = ee.Date(ee.Algorithms.If(hasMonth, start.advance(1, 'month'), start.advance(1, 'year')));
            return img.set({
                'system:time_start': start.millis(),
                'system:time_end': end.millis()
            });
        });

        buildLayerPanel(images, function (img) {
            var periodName = img.id.split('/').pop();
            var sourceImage = setTimeProperties(ee.Image(img.id), periodName);

            var pastResult = buildPreviousPeriodsMaskServer(sourceImage, eeInputCol, periodName);
            
            var filteredImage = ee.Image(ee.Algorithms.If(
                pastResult.hasPastData.gt(0),
                sourceImage.updateMask(pastResult.mask.neq(1)).selfMask(),
                sourceImage
            ));

            var mosaicLayer = loadPeriodMosaic(periodName);
            var beforeLayer = ui.Map.Layer(sourceImage.select(0).selfMask(), { min: 0, max: 1000, palette: ['000000'] }, periodName + ' | ANTES (ft02 - Negro)');
            var afterLayer  = ui.Map.Layer(filteredImage.select(0).selfMask(), { min: 0, max: 1000, palette: ['FF0000'] }, periodName + ' | DESPUÉS (ft03 - Rojo)');

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

            var pastResult = buildPreviousPeriodsMaskServer(sourceImage, eeInputCol, periodName);
            
            var filteredImage = ee.Image(ee.Algorithms.If(
                pastResult.hasPastData.gt(0),
                sourceImage.updateMask(pastResult.mask.neq(1)).selfMask(),
                sourceImage
            ));
                
            filteredImage = setTimeProperties(filteredImage.copyProperties(sourceImage).set('filter_stage', 'ft03'), periodName);

            var destAsset = outputCollection + '/' + periodName;
            var periodicityTag = isYearlyPeriod(periodName) ? 'ANNUAL' : 'MONTHLY';
            var exportFlag = ('M7_ft03_' + COUNTRY.toUpperCase() + '_' + CAMPAIGN + '_' + COLLECTION_BASE + '_' + periodName + '_' + periodicityTag).replace(/[^a-zA-Z0-9_]/g, '_').substring(0, 80);

            try {
                ee.data.getAsset(destAsset);
                print('  OK (Ya existe): ' + periodName);
            } catch (e) {
                exportCount++;
                print('  Exportando a Asset ft03: ' + periodName);
                Export.image.toAsset({
                    image: ee.Image(filteredImage).toInt16(),
                    description: exportFlag,
                    assetId: destAsset,
                    pyramidingPolicy: 'mode',
                    region: REGIONS.geometry().bounds(),
                    scale: SCALE,
                    maxPixels: 1e13,
                });
            }
        });
        print('Total de imágenes listas para exportar a ft03: ' + exportCount);
    }
});
print('=== ft03 finalizado ===');