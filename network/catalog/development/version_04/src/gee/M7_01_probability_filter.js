// Preset: peru / MONITOR_01 / monthly
/* ==============================================================================
MAPBIOMAS FUEGO - M7 - ft01
Filtro de Probabilidad de Área Quemada (Salida: ft01)

📅 FECHA: 11/2026
🏷️ VERSIÓN: 1.0 (multi-país, multi-campaña, mensual/anual)

📌 ¿QUÉ HACE ESTE SCRIPT?
1. Toma la imagen nacional de cada período de la etapa de entrada (ft00).
2. Define qué se considera área quemada: conserva solo los píxeles con
   probabilidad >= umbral (%). Los demás quedan enmascarados en las dos bandas
   ('probability' y 'dayOfYear'). Los valores de probabilidad NO se binarizan.
3. Permite ajustar el umbral con un slider y comparar ANTES / ELIMINADO / DESPUÉS
   sobre el mosaico del período.
4. "Exportar ft01" genera la colección nacional filtrada (ft01) en Tasks con el
   umbral del slider. Cada imagen guarda 'prob_threshold_pct' para trazabilidad.

📏 REGLA DEL UMBRAL (igual al slider del M3):
- La banda 'probability' es int16 0-1000 (probabilidad × 1000).
- Se conserva el píxel si probability >= max(1, umbral_% × 10).
- Umbral 0 % = regla del producto (cualquier probabilidad > 0 es quemado).

⚙️ PARÁMETROS A MODIFICAR POR EL USUARIO:
- COUNTRY: key usada en las rutas (ej. 'peru', 'brazil', 'drc').
- COUNTRY_LABEL: nombre para mostrar.
- CAMPAIGN: campaña (ej. 'MONITOR_01', 'COLLECTION_02').
- COLLECTION_BASE: propuesta de trabajo (ej. 'propuesta_a').
- REGIONS_ASSET: asset de regiones (opcional; vacío usa el patrón estándar).
- PROB_THRESHOLD_PCT: valor inicial del slider (0-100 %).
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
var PROB_THRESHOLD_PCT = 0; // valor inicial del slider (0-100 %); 0 = probability >= 1

// ------------------------------------------------------------------------------
// CONFIGURACIÓN INTERNA Y RUTAS (ENTRADA: ft00 -> SALIDA: ft01)
// ------------------------------------------------------------------------------
var inputStage  = 'ft00';
var outputStage = 'ft01';
var PROB_PALETTE = ['#ffcccc', '#ff6666', '#cc0000', '#660000'];

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
    return mosaicImage;
}

// Umbral en % -> valor de la banda int16 (0-1000). 0 % = probability >= 1.
function thresholdValue(pct) {
    return Math.max(1, Math.round(pct * 10));
}

// Conserva solo los píxeles con probabilidad >= umbral, en las dos bandas.
function applyProbabilityFilter(sourceImage, pct) {
    return sourceImage.updateMask(sourceImage.select(0).gte(thresholdValue(pct)));
}

function buildOutputImage(sourceImage, periodName, pct) {
    var out = applyProbabilityFilter(sourceImage, pct)
        .copyProperties(sourceImage)
        .set({
            'filter_stage': outputStage,
            'prob_threshold_pct': pct,
            'prob_threshold_value': thresholdValue(pct)
        });
    return setTimeProperties(out, periodName);
}

print('=== ft01 - Filtro de Probabilidad de Área Quemada ===');
print('País: ' + COUNTRY_LABEL + ' (' + COUNTRY + ')');
print('Campaña: ' + CAMPAIGN + ' | Propuesta: ' + COLLECTION_BASE);
print('Entrada (Input ft00): ' + inputCollection);
print('Salida (Output ft01): ' + outputCollection);
print('Umbral inicial: ' + PROB_THRESHOLD_PCT + ' % (probability >= ' + thresholdValue(PROB_THRESHOLD_PCT) + ')');

ensureFolder('FILTERED/' + COLLECTION_BASE + '/' + inputStage);
ensureFolder('FILTERED/' + COLLECTION_BASE + '/' + outputStage);

ee.data.listAssets(inputCollection, {}, function (result) {
    var images = [];
    if (result && result.assets) {
        images = result.assets.filter(function (a) { return a.type === 'IMAGE'; });
    }
    if (images.length === 0) {
        print('No se encontraron imágenes en la colección de entrada (ft00).');
        return;
    }

    var state = { pct: PROB_THRESHOLD_PCT, img: images[0] };
    var exported = {};

    Map.setOptions('HYBRID');
    Map.centerObject(REGIONS);

    // Capas fijas: solo se actualiza su contenido al cambiar período o umbral.
    var mosaicLayer   = ui.Map.Layer(ee.Image(), { bands: ['swir1', 'nir', 'red'], min: 3, max: 40, gamma: 0.85 }, 'Mosaico');
    var probLayer     = ui.Map.Layer(ee.Image(), { min: 0, max: 100, palette: PROB_PALETTE }, 'Probabilidad ft00 (0-100 %)', false);
    var beforeLayer   = ui.Map.Layer(ee.Image(), { palette: ['000000'] }, 'ANTES (ft00 - Negro)');
    var removedLayer  = ui.Map.Layer(ee.Image(), { palette: ['FFD600'] }, 'ELIMINADO (< umbral - Amarillo)');
    var afterLayer    = ui.Map.Layer(ee.Image(), { palette: ['FF0000'] }, 'DESPUÉS (ft01 - Rojo)');
    var regionsLayer  = ui.Map.Layer(REGIONS.style({ color: 'ffffff', fillColor: '00000000', width: 1 }), {}, 'Límites de Regiones');
    [mosaicLayer, probLayer, beforeLayer, removedLayer, afterLayer, regionsLayer].forEach(function (l) {
        Map.layers().add(l);
    });

    function redraw() {
        var periodName = state.img.id.split('/').pop();
        var source = ee.Image(state.img.id);
        var burnedBefore = source.select(0).gte(1);
        var kept = source.select(0).gte(thresholdValue(state.pct));
        var sensor = isYearlyPeriod(periodName) ? 'LANDSAT' : 'SENTINEL2';
        var suffix = ' | ' + periodName + ' | umbral ' + state.pct + ' %';

        mosaicLayer.setEeObject(loadPeriodMosaic(periodName));
        mosaicLayer.setName(periodName + ' | Mosaico ' + sensor + ' (SWIR1, NIR, RED)');
        probLayer.setEeObject(source.select(0).divide(10).toByte().selfMask());
        probLayer.setName('Probabilidad ft00 (0-100 %) | ' + periodName);
        beforeLayer.setEeObject(burnedBefore.selfMask());
        beforeLayer.setName('ANTES (ft00 - Negro) | ' + periodName);
        removedLayer.setEeObject(burnedBefore.and(kept.not()).selfMask());
        removedLayer.setName('ELIMINADO (< umbral - Amarillo)' + suffix);
        afterLayer.setEeObject(kept.selfMask());
        afterLayer.setName('DESPUÉS (ft01 - Rojo)' + suffix);
        exportButton.setLabel('Exportar ft01 (umbral ' + state.pct + ' %)');
    }

    // PANEL DE CONTROL
    var panel = ui.Panel({
        layout: ui.Panel.Layout.flow('vertical'),
        style: {
            position: 'bottom-left',
            maxHeight: '70%',
            width: '230px',
            padding: '4px',
            backgroundColor: 'rgba(255,255,255,0.92)',
            border: '1px solid #ccc',
            borderRadius: '4px'
        }
    });
    panel.add(ui.Label('UMBRAL DE PROBABILIDAD (%)', { fontWeight: 'bold', fontSize: '11px', margin: '2px' }));
    var slider = ui.Slider({
        min: 0, max: 100, step: 1, value: state.pct,
        style: { stretch: 'horizontal', margin: '0px 2px', fontSize: '10px' }
    });
    slider.onChange(ui.util.debounce(function (value) {
        state.pct = value;
        redraw();
    }, 500));
    panel.add(slider);
    panel.add(ui.Label('Conserva probability >= max(1, umbral × 10)', { fontSize: '9px', color: '#777', margin: '0px 2px 4px 2px' }));

    var exportButton = ui.Button({
        label: 'Exportar ft01',
        style: { stretch: 'horizontal', margin: '2px', fontSize: '10px' },
        onClick: function () { exportAll(state.pct); }
    });
    panel.add(exportButton);

    panel.add(ui.Label('PERÍODOS DISPONIBLES', { fontWeight: 'bold', fontSize: '11px', margin: '6px 2px 2px 2px' }));
    var periodBoxes = [];
    images.forEach(function (img, idx) {
        var periodName = img.id.split('/').pop();
        var checkbox = ui.Checkbox({ label: periodName, value: idx === 0, style: { fontSize: '10px', margin: '1px 2px' } });
        checkbox.onChange(function (checked) {
            if (!checked) {
                if (state.img === img) checkbox.setValue(true);
                return;
            }
            state.img = img;
            periodBoxes.forEach(function (other) {
                if (other !== checkbox) other.setValue(false);
            });
            redraw();
        });
        periodBoxes.push(checkbox);
        panel.add(checkbox);
    });
    Map.add(panel);

    // TAREAS DE EXPORTACIÓN A FT01 (un único umbral para todos los períodos)
    function exportAll(pct) {
        var exportCount = 0;
        print('--- Exportación ft01 | umbral ' + pct + ' % (probability >= ' + thresholdValue(pct) + ') ---');
        images.forEach(function (img) {
            var periodName = img.id.split('/').pop();
            var destAsset = outputCollection + '/' + periodName;
            try {
                ee.data.getAsset(destAsset);
                print('  OK (Ya existe): ' + periodName);
                return;
            } catch (e) {}
            if (exported[periodName] !== undefined) {
                print('  Ya enviado en esta sesión (umbral ' + exported[periodName] + ' %): ' + periodName);
                return;
            }
            var periodicityTag = isYearlyPeriod(periodName) ? 'ANNUAL' : 'MONTHLY';
            var exportFlag = ('M7_ft01_' + COUNTRY.toUpperCase() + '_' + CAMPAIGN + '_' + COLLECTION_BASE + '_' + periodName + '_' + periodicityTag).replace(/[^a-zA-Z0-9_]/g, '_').substring(0, 80);
            Export.image.toAsset({
                image: buildOutputImage(ee.Image(img.id), periodName, pct).toInt16(),
                description: exportFlag,
                assetId: destAsset,
                pyramidingPolicy: 'mode',
                region: REGIONS.geometry().bounds(),
                scale: SCALE,
                maxPixels: 1e13,
                overwrite: true
            });
            exported[periodName] = pct;
            exportCount++;
            print('  Exportando a Asset ft01: ' + periodName);
        });
        print('Total de imágenes enviadas a Tasks (ft01): ' + exportCount);
    }

    redraw();
});
print('=== ft01 preparado ===');
