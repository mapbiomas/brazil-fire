/**
 * @description Mosaicos de Qualidade MapBiomas Fire (Anual)
 * Processamento integrado de múltiplos sistemas sensores para monitoramento de cicatrizes de fogo.
 * * Sistemas e Resoluções Espaciais:
 * - Landsat (5, 7, 8, 9) : 30 metros
 * - Sentinel-2 (SR)      : 10/20 metros
 * - NASA HLS (S30/L30)   : 30 metros (Grid Harmonizado)
 * - MODIS (Terra/Aqua)   : 500 metros
 * * Nota: Inclui funções originais completas de tratamento (máscaras negativas, saturação e clip de bordas).
 * Saída: Bandas Espectrais em Byte (0-100, onde 100 = refletância 1.0) + dayOfYear em Int16.
 * * stable version: 2026 - MapBiomas Fire Monitor
 */

// ─── CONFIGURATION ───────────────────────────────────────────────────────────
var config = {
    year: 2023,
    country: 'peru',
    // geometry: ee.FeatureCollection('projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1').geometry().bounds(),
    geometry: geometry,
    vis: { bands: ['swir1', 'nir', 'red'], min: 0, max: 50 }
};

var start = ee.Date.fromYMD(config.year, 1, 1);
var end = start.advance(1, 'year');

// ─── 1. LANDSAT SENSOR LOGIC (Constelações por Ano) ──────────────────────────
var sensor_logic = {
    '1984_1998': ['L5'],
    '1999_2012': ['L5', 'L7'],
    '2013_2021': ['L7', 'L8'],
    '2022_2026': ['L8', 'L9']
};

var current_constellation = config.year < 1999 ? sensor_logic['1984_1998'] :
    config.year <= 2012 ? sensor_logic['1999_2012'] :
        config.year <= 2021 ? sensor_logic['2013_2021'] :
            sensor_logic['2022_2026'];

print('Landsat Constellation para ' + config.year + ':', current_constellation);

// ─── FUNÇÕES DE SUPORTE (AS SUAS ORIGINAIS) ──────────────────────────────────

function clipBoard_Landsat(image) {
    return image.updateMask(ee.Image().paint(image.geometry().buffer(-3000)).eq(0));
}

function bitwiseExtract(value, fromBit, toBit) {
    if (toBit === undefined) toBit = fromBit;
    var maskSize = ee.Number(1).add(toBit).subtract(fromBit);
    var mask = ee.Number(1).leftShift(maskSize).subtract(1);
    return value.rightShift(fromBit).bitwiseAnd(mask);
}

function addBand_NBR(image) {
    var exp = '( b("nir") - b("swir2") ) / ( b("nir") + b("swir2") )';
    var minimoNBR = image.expression(exp).multiply(-1).add(1).multiply(1000).int16().rename("nbr");
    return image.addBands(minimoNBR);
}

function addDOY(image) {
    var doy = ee.Image(ee.Number.parse(image.date().format('D'))).int16().rename('dayOfYear');
    return image.addBands(doy);
}

// ─── FUNÇÕES DE CORREÇÃO ORIGINAIS (MANTIDAS NA ÍNTEGRA) ─────────────────────

function corrections_LS57_col2(image) {
    var opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2);
    var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
    image = image.addBands(opticalBands, null, true).addBands(thermalBands, null, true);

    var cloudShadowBitMask = (1 << 3);
    var cloudsBitMask = (1 << 5);
    var qa = image.select('QA_PIXEL');
    var mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).and(qa.bitwiseAnd(cloudsBitMask).eq(0));

    var clear = bitwiseExtract(qa, 6);
    var water = bitwiseExtract(qa, 7);
    var radsatQA = image.select('QA_RADSAT');
    var anySaturated = bitwiseExtract(radsatQA, 0, 6);
    var mask_saturation = clear.or(water).and(anySaturated.not());

    var negative_mask = image.select(['SR_B1']).gt(0).and(image.select(['SR_B2']).gt(0)).and(
        image.select(['SR_B3']).gt(0)).and(image.select(['SR_B4']).gt(0)).and(
            image.select(['SR_B5']).gt(0)).and(image.select(['SR_B7']).gt(0));

    image = image.updateMask(mask).updateMask(mask_saturation).updateMask(negative_mask);
    var oldBands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'];
    var newBands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];
    return image.select(oldBands, newBands);
}

function corrections_LS8_col2(image) {
    var opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2);
    opticalBands = opticalBands.multiply(10000).subtract(0.0000275 * 0.2 * 1e5 * 100).round().divide(10000);
    var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
    image = image.addBands(opticalBands, null, true).addBands(thermalBands, null, true);

    var qa = image.select('QA_PIXEL');
    var cloud = qa.bitwiseAnd(1 << 3).and(qa.bitwiseAnd(1 << 9)).or(qa.bitwiseAnd(1 << 4));
    var good_pixel = qa.bitwiseAnd(1 << 6).or(qa.bitwiseAnd(1 << 7));
    var radsatQA = image.select('QA_RADSAT');
    var saturated = radsatQA.bitwiseAnd(1 << 0).or(radsatQA.bitwiseAnd(1 << 1)).or(radsatQA.bitwiseAnd(1 << 2))
        .or(radsatQA.bitwiseAnd(1 << 3)).or(radsatQA.bitwiseAnd(1 << 4)).or(radsatQA.bitwiseAnd(1 << 5)).or(radsatQA.bitwiseAnd(1 << 6));

    var negative_mask = image.select(['SR_B2']).gt(0).and(image.select(['SR_B3']).gt(0)).and(
        image.select(['SR_B4']).gt(0)).and(image.select(['SR_B5']).gt(0)).and(
            image.select(['SR_B6']).gt(0)).and(image.select(['SR_B7']).gt(0));

    image = image.updateMask(cloud.not()).updateMask(good_pixel).updateMask(saturated.not()).updateMask(negative_mask);
    var oldBands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'];
    var newBands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];
    return image.select(oldBands, newBands).float();
}

function corrections_modis(image) {
    var oldBands = ['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06', 'sur_refl_b07'];
    var newBands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];
    // Atenção: corrigi a ordem das bandas com base nos metadados originais MODIS para bater com o RGB
    return image.select(oldBands, newBands);
}

// ─── PIPELINES DE PROCESSAMENTO POR COLEÇÃO ──────────────────────────────────

var processLS57 = function (img) { return addDOY(addBand_NBR(corrections_LS57_col2(clipBoard_Landsat(img)))); };
var processLS89 = function (img) { return addDOY(addBand_NBR(corrections_LS8_col2(clipBoard_Landsat(img)))); };

var processS2 = function (img) {
    var mask = img.select('cs').gte(0.40);
    var optical = img.updateMask(mask).select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
    return addDOY(addBand_NBR(optical));
};

var processMODIS = function (img) { return addDOY(addBand_NBR(corrections_modis(img))); };

// ─── PIPELINES SEPARADOS PARA HLS S30 e L30 ──────────────────────────────────

// HLS S30 (Sentinel-2 Harmonizado)
var processHLS_S30 = function (img) {
    var mask = img.select('Fmask').bitwiseAnd(1 << 1).eq(0); // Bit 1 = Cloud
    var optical = img.updateMask(mask).select(
        ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'],  // Usa B8A (NIR Narrow)
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    );
    return addDOY(addBand_NBR(optical).set({
        'system:time_start': img.get('system:time_start'),
        'system:time_end': img.get('system:time_end')
    }));
};

// HLS L30 (Landsat 8/9 Harmonizado)
var processHLS_L30 = function (img) {
    var mask = img.select('Fmask').bitwiseAnd(1 << 1).eq(0); // Bit 1 = Cloud
    var optical = img.updateMask(mask).select(
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],   // Nomenclatura baseada no OLI
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    );
    return addDOY(addBand_NBR(optical).set({
        'system:time_start': img.get('system:time_start'),
        'system:time_end': img.get('system:time_end')
    }));
};

// ─── COLLECTION BUILDERS ─────────────────────────────────────────────────────

var ls_col = ee.ImageCollection([]);
if (current_constellation.indexOf('L9') !== -1) ls_col = ls_col.merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(start, end).filterBounds(config.geometry).map(processLS89));
if (current_constellation.indexOf('L8') !== -1) ls_col = ls_col.merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start, end).filterBounds(config.geometry).map(processLS89));
if (current_constellation.indexOf('L7') !== -1) ls_col = ls_col.merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterDate(start, end).filterBounds(config.geometry).map(processLS57));
if (current_constellation.indexOf('L5') !== -1) ls_col = ls_col.merge(ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterDate(start, end).filterBounds(config.geometry).map(processLS57));
// print('ls_col',ls_col.limit(10));

var s2_col = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start, end).filterBounds(config.geometry)
    .linkCollection(ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs']).map(processS2);
// print('s2_col',s2_col.limit(10));

var mod_col = ee.ImageCollection("MODIS/061/MOD09A1").merge(ee.ImageCollection("MODIS/061/MYD09A1"))
    .filterDate(start, end).filterBounds(config.geometry).map(processMODIS);
// print('mod_col',mod_col.limit(10));

// ─── COLLECTION BUILDER HLS ──────────────────────────────────────────────────

var s30_col = ee.ImageCollection("NASA/HLS/HLSS30/v002")
    .filterDate(start, end).filterBounds(config.geometry).map(processHLS_S30);
// print('s30_col',s30_col.limit(10))
var l30_col = ee.ImageCollection("NASA/HLS/HLSL30/v002")
    .filterDate(start, end).filterBounds(config.geometry).map(processHLS_L30);
// print('l30_col',l30_col.limit(10))
var hls_col = s30_col.merge(l30_col);

// ─── FINALIZADOR DE MOSAICOS (GERAÇÃO BYTE) ──────────────────────────────────

var finalize = function (collection, name, multiplier) {
    var mosaic = collection.qualityMosaic('nbr');

    // O multiplier é necessário pois o seu Landsat devolve escala 0-1, enquanto o S2/MODIS/HLS devolvem 0-10000.
    var spectral = mosaic.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
        .multiply(multiplier).byte();

    var doy = mosaic.select('dayOfYear').int16();
    var result = spectral.addBands(doy).clip(config.geometry);

    Map.addLayer(result, config.vis, name, false);
    return result;
};

// Como as suas funções do Landsat reduzem os valores para Refletância (0-1), multiplicamos por 100 para Byte.
// S2, HLS e MODIS rodam nativos em 0-10000, então dividimos por 100 (*0.01) para Byte.
var ls_final = finalize(ls_col, '1. Landsat Mosaic (' + current_constellation.join('+') + ')', 100);
var s2_final = finalize(s2_col, '2. Sentinel Mosaic', 0.01);
var hls_final = finalize(hls_col, '3. HLS Mosaic', 100);
var mod_final = finalize(mod_col, '4. MODIS Mosaic', 0.01);

// Map.centerObject(config.geometry, 6);