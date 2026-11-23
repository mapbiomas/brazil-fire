// =============================================================================
// MOSAICOS MENSAIS/ANUAIS HLS -> ImageCollection de Assets (MapBiomas Fire)
// Code Editor script (JavaScript ES5, sem extensão).
//
// SUPORTA MÚLTIPLOS PRODUTOS E MÉTODOS EM LISTAS:
//   - PRODUCT_TYPES : ['FUEL_MATERIAL_BYTE', 'SWIR_NIR_RED_BYTE']
//   - METHODS       : ['minnbr', 'median']
// =============================================================================

// =============================================================================
// CONFIGURAÇÃO GERAL (LISTAS DE OPÇÕES)
// =============================================================================

var PRODUCT_TYPES = ['FUEL_MATERIAL_BYTE']; // Pode conter 'FUEL_MATERIAL_BYTE',  'SWIR_NIR_RED_BYTE' ou ambos

var METHODS = ['minnbr']; // Pode conter 'minnbr', 'median' ou ambos

var PERIOD = 'monthly'; // 'monthly' | 'annual' | 'both'

var flag_export = 'FIRE_IPAM';            // Prefixo do description da tarefa
var VERSION = '1';

var YEARS = [2026];
var FIRST_MONTH = 10;                  // mes inicial, valor minimo é 1
var LAST_MONTH = 0;                   // 0 = último mês completo antecedente ao atual

var OVERWRITE = true;                   // true: reexporta assets já existentes, 
var CHECK_EMPTY_COLLECTIONS = false;     // true: ignora períodos sem cenas HLS válidas || false é mais rapido, por não fazer a verificação

var SCALE = 30;
var MAX_PIXELS = 1e13;
var PYRAMIDING_POLICY = { '.default': 'median' };

var BYTE_SCALE = 1;                     // Preserva % (0-100)
var NORMALIZE_FRACTIONS = false;        // Apenas para FUEL_MATERIAL_BYTE

// =============================================================================
// DICIONÁRIO / CONFIGURAÇÕES POR PRODUTO
// =============================================================================

function getProductConfig(productType) {
  if (productType === 'FUEL_MATERIAL_BYTE') {
    return {
      baseAsset: 'projects/mapbiomas-mosaics/assets/FIRE/HLS/fuel_material',
      bands: ['npv', 'gv', 'soil'],
      vis: { bands: ['npv', 'gv', 'soil'], min: [0, 5, 10], max: [17, 42, 90] }
    };
  } else if (productType === 'SWIR_NIR_RED_BYTE') {
    return {
      baseAsset: 'projects/mapbiomas-mosaics/assets/FIRE/HLS/swir_nir_red',
      bands: ['swir1', 'nir', 'red'],
      vis: { bands: ['swir1', 'nir', 'red'], min: 0, max: 40 }
    };
  }
}

// =============================================================================
// COLEÇÕES E FILTROS
// =============================================================================

var blockList_sentinel = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList').hls_s30();
var blockList_landsat = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList').hls_l30();

function excluirHLS(col, lista) {
  var filtros = lista.map(function(id) {
    return ee.Filter.stringStartsWith(
      'system:index',
      id.replace(/^(L30|S30)_/, '')
    );
  });

  return filtros.length
    ? ee.ImageCollection(col).filter(ee.Filter.or.apply(null, filtros).not())
    : ee.ImageCollection(col);
}

var HLS_S30 = 'NASA/HLS/HLSS30/v002';
var HLS_L30 = 'NASA/HLS/HLSL30/v002';

var CLOUD_MASK_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4); 
// HLS Fmask: bit1 = cloud, bit2 = adjacent, bit3 = shadow, bit4 = snow/ice
var OPTICAL_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];

var GV = [119.0, 475.0, 169.0, 6250.0, 2399.0, 675.0];
var NPV = [1514.0, 1597.0, 1421.0, 3053.0, 7707.0, 1975.0];
var SOIL = [1799.0, 2479.0, 3158.0, 5437.0, 7707.0, 6646.0];
var CLOUD = [4031.0, 8714.0, 7900.0, 8989.0, 7002.0, 6607.0];

var PERIODS = PERIOD === 'both' ? ['monthly', 'annual'] : [PERIOD];

// =============================================================================
// PERÍODO
// =============================================================================

function lastCompleteMonth(year) {
  var today = new Date();
  var currentYear = today.getFullYear();
  var currentMonth = today.getMonth() + 1;
  if (year < currentYear) return 12;
  if (year > currentYear) return 0;
  return Math.min(currentMonth, 12);
}

function monthsForYear(year) {
  var last = LAST_MONTH > 0 ? LAST_MONTH : lastCompleteMonth(year);
  var out = [];
  for (var m = FIRST_MONTH; m <= last; m++) out.push(m);
  return out;
}

// =============================================================================
// HLS - MÁSCARA, BANDAS ÓPTICAS E NBR
// =============================================================================

function addBand_NBR(image) {
  var exp = '( b("nir") - b("swir2") ) / ( b("nir") + b("swir2") )';
  var nbr = image.expression(exp).multiply(-1).add(1).multiply(1000).int16().rename('nbr');
  return image.addBands(nbr);
}

function cloudMaskHLS(image) {
  return image.select('Fmask').bitwiseAnd(CLOUD_MASK_BITS).eq(0);
}

// --- HLS S30 (Sentinel-2) ---
function processHLS_S30(image) {
  var optical = image.updateMask(cloudMaskHLS(image))
    .select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'], OPTICAL_BANDS)
    .float();

  // Eliminar pixels negativos/nulos
  var validMask = optical.select('blue').gt(0)
    .and(optical.select('green').gt(0))
    .and(optical.select('red').gt(0))
    .and(optical.select('nir').gt(0))
    .and(optical.select('swir1').gt(0))
    .and(optical.select('swir2').gt(0));

  optical = optical.updateMask(validMask);

  return addBand_NBR(optical);
}

// --- HLS L30 (Landsat) ---
function processHLS_L30(image) {
  var optical = image.updateMask(cloudMaskHLS(image))
    .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], OPTICAL_BANDS)
    .float();

  // Eliminar pixels negativos/nulos
  var validMask = optical.select('blue').gt(0)
    .and(optical.select('green').gt(0))
    .and(optical.select('red').gt(0))
    .and(optical.select('nir').gt(0))
    .and(optical.select('swir1').gt(0))
    .and(optical.select('swir2').gt(0));

  optical = optical.updateMask(validMask);

  return addBand_NBR(optical);
}

// =============================================================================
// SMA & PROCESSAMENTO DE IMAGEM
// =============================================================================

function fractions(image) {
  var imageSelected = image.select(OPTICAL_BANDS).multiply(10000);
  var sma = imageSelected.unmix([GV, NPV, SOIL, CLOUD]).max(0).multiply(100).int16();

  var gv = sma.select('band_0').rename('gv');
  var npv = sma.select('band_1').rename('npv');
  var soil = sma.select('band_2').rename('soil');
  var cloud = sma.select('band_3').rename('cloud');

  if (NORMALIZE_FRACTIONS) {
    var total = gv.add(npv).add(soil).add(cloud);
    gv = gv.divide(total).multiply(100);
    npv = npv.divide(total).multiply(100);
    soil = soil.divide(total).multiply(100);
    cloud = cloud.divide(total).multiply(100);
  }

  return image.addBands([gv, npv, soil, cloud]);
}

function buildMosaic(collection, method) {
  return method.toLowerCase() === 'median' ? collection.median() : collection.qualityMosaic('nbr');
}

function processProductImage(mosaic, productType) {
  var config = getProductConfig(productType);
  if (productType === 'FUEL_MATERIAL_BYTE') {
    var smaImage = fractions(mosaic);
    return smaImage.select(config.bands).multiply(BYTE_SCALE).clamp(0, 255).round().byte();
  } else {
    return mosaic.select(config.bands).multiply(100).clamp(0, 255).round().byte();
  }
}

function collectionForPeriod(bounds, start, end) {
  var s30 = excluirHLS(
    ee.ImageCollection(HLS_S30)
      .filterDate(start, end)
      .filterBounds(bounds),
    blockList_sentinel
  ).map(processHLS_S30);

  var l30 = excluirHLS(
    ee.ImageCollection(HLS_L30)
      .filterDate(start, end)
      .filterBounds(bounds),
    blockList_landsat
  ).map(processHLS_L30);

  return s30.merge(l30);
}

// =============================================================================
// ITERADOR ASSÍNCRONO LEVE
// =============================================================================

function forEachSeries(items, step, done) {
  var index = 0;
  function next() {
    if (index >= items.length) {
      done();
      return;
    }
    step(items[index++], next);
  }
  next();
}

// =============================================================================
// VERIFICAÇÃO DE ASSETS
// =============================================================================

function ensureCollection(collectionId, done) {
  ee.data.getAsset(collectionId, function (res, err) {
    if (res && !err) {
      print('Coleção destino encontrada: ' + collectionId);
      done(true);
    } else {
      print('Coleção destino não encontrada, criando nova: ' + collectionId);
      var parts = collectionId.split('/');
      try {
        ee.data.createAsset({
          type: 'IMAGE_COLLECTION',
          name: parts[parts.length - 1],
          parent: parts.slice(0, -1).join('/')
        });
        print('Coleção criada com sucesso: ' + collectionId);
        done(true);
      } catch (e) {
        print('Erro ao criar a coleção destino: ' + collectionId);
        done(false);
      }
    }
  });
}

function readInventory(collectionId, callback) {
  ee.ImageCollection(collectionId)
    .aggregate_array('system:id')
    .evaluate(
      function (ids) {
        callback(ids || []);
      },
      function () {
        callback([]);
      }
    );
}

// =============================================================================
// ESTADO & ROI
// =============================================================================

var plan = [];
var planInventories = [];

var biomas = ee.FeatureCollection('projects/ee-ipam/assets/ORIGINAIS/IBGE/limite_biomas_IBGE_2026');
var mask = ee.Image().paint(biomas).eq(0);
var geometryBounds = biomas.geometry().bounds();

var targetCollections = [];
PRODUCT_TYPES.forEach(function (productType) {
  var config = getProductConfig(productType);
  PERIODS.forEach(function (period) {
    METHODS.forEach(function (method) {
      var collectionId = config.baseAsset + '/' + period + '_' + method.toLowerCase();
      targetCollections.push({
        productType: productType,
        period: period,
        method: method,
        collectionId: collectionId
      });
    });
  });
});

// =============================================================================
// PASSO 1 - INVENTARIAR COLEÇÕES ALVO
// =============================================================================

function readInventories(done) {
  forEachSeries(targetCollections, function (entry, next) {
    ensureCollection(entry.collectionId, function (ready) {
      if (!ready) {
        next();
        return;
      }

      readInventory(entry.collectionId, function (existingIds) {
        planInventories.push({
          productType: entry.productType,
          period: entry.period,
          method: entry.method,
          collectionId: entry.collectionId,
          existingIds: existingIds
        });
        next();
      });
    });
  }, done);
}

// =============================================================================
// PASSO 2 - MONTAR PLANO DE EXPORTAÇÃO
// =============================================================================

function buildPlan() {
  planInventories.forEach(function (entry) {
    var productType = entry.productType;
    var period = entry.period;
    var method = entry.method;
    var config = getProductConfig(productType);

    YEARS.forEach(function (year) {
      var versionYear = year + '';
      var months = period === 'annual' ? [0] : monthsForYear(year);

      months.forEach(function (month) {
        var name = period === 'annual'
          ? productType + '_' + year
          : productType + '_' + year + '_' + ('0' + month).slice(-2);
        
        var assetId = entry.collectionId + '/' + name;
        var exists = entry.existingIds.indexOf(assetId) !== -1;

        if (!exists || OVERWRITE) {
          plan.push({
            productType: productType,
            period: period,
            method: method,
            assetId: assetId,
            name: assetId.split('/').pop(),
            description: flag_export + '_' + productType + '_' + method.toUpperCase() + '_' + period + '_' + year +
              (period === 'annual' ? '' : '_' + ('0' + month).slice(-2)),
            year: year,
            month: month,
            versionYear: versionYear,
            bands: config.bands,
            vis: config.vis
          });
        }
      });
    });
  });

  print(plan.length + ' tarefa(s) gerada(s) no plano acumulado.');
  runExports(plan);
}

// =============================================================================
// PASSO 3 - EXPORTAR
// =============================================================================

function intervalOf(item) {
  var start = ee.Date.fromYMD(item.year, item.month === 0 ? 1 : item.month, 1);
  var end = item.period === 'annual'
    ? ee.Date.fromYMD(item.year + 1, 1, 1)
    : start.advance(1, 'month');
  return { start: start, end: end };
}

function mosaicOf(item, bounds) {
  var interval = intervalOf(item);
  return buildMosaic(collectionForPeriod(bounds, interval.start, interval.end), item.method);
}

function exportItem(item, done) {
  var interval = intervalOf(item);
  var collection = collectionForPeriod(geometryBounds, interval.start, interval.end);

  function exportNow() {
    var mosaic = buildMosaic(collection, item.method);
    var image = processProductImage(mosaic, item.productType)
      .updateMask(mask)
      .set({
        'system:time_start': interval.start.millis(),
        'system:time_end': interval.end.millis(),
        'year': item.year,
        'month': item.month,
        'period': item.period,
        'method': item.method.toLowerCase(),
        'version': VERSION,
        'version_year': item.versionYear,
        'type': item.productType,
        'task_flag': flag_export,
        'byte_scale': BYTE_SCALE,
        'scale': SCALE,
        'bands': item.bands.join(',')
      });

    Map.addLayer(image, item.vis, item.name + '_' + item.method, false);
    
    printDiagnostics(item);

    Export.image.toAsset({
      image: image,
      description: item.description,
      assetId: item.assetId,
      region: geometryBounds,
      scale: SCALE,
      maxPixels: MAX_PIXELS,
      pyramidingPolicy: PYRAMIDING_POLICY,
      overwrite: OVERWRITE
    });

    done();
  }

  if (!CHECK_EMPTY_COLLECTIONS) {
    exportNow();
    return;
  }

  collection.size().evaluate(
    function (size) {
      if (size === 0) {
        print('AVISO: sem cenas HLS para ' + item.name + ' (' + item.method + ') - exportação ignorada.');
        done();
        return;
      }
      exportNow();
    },
    function () {
      exportNow();
    }
  );
}

function printDiagnostics(item) {
  var interval = intervalOf(item);
  var mosaic = buildMosaic(collectionForPeriod(geometryBounds, interval.start, interval.end), item.method);

  if (item.productType === 'FUEL_MATERIAL_BYTE') {
    var imageSMA = fractions(mosaic);
    var total = imageSMA.select('gv')
      .add(imageSMA.select('npv'))
      .add(imageSMA.select('soil'))
      .add(imageSMA.select('cloud'))
      .rename('soma_fracoes');

    imageSMA.updateMask(mask)
      .addBands(total)
      .select(item.bands.concat(['soma_fracoes']))
      .reduceRegion({
        reducer: ee.Reducer.percentile([1, 50, 99]),
        geometry: biomas.geometry().centroid(2000).buffer(5000),
        scale: 300,
        maxPixels: 1e7
      })
      .evaluate(function (stats) {
        print('Diagnóstico SMA % (' + item.name + ' | ' + item.method + '):', stats);
      });
  } else {
    mosaic.select(item.bands)
      .multiply(100)
      .updateMask(mask)
      .reduceRegion({
        reducer: ee.Reducer.percentile([1, 50, 99]),
        geometry: biomas.geometry().centroid(2000).buffer(5000),
        scale: 300,
        maxPixels: 1e7
      })
      .evaluate(function (stats) {
        print('Diagnóstico Refletância % (' + item.name + ' | ' + item.method + '):', stats);
      });
  }
}

function runExports(toProcess) {
  forEachSeries(toProcess, exportItem, function () {
    print('Processamento finalizado. ' + toProcess.length + ' tarefa(s) enviada(s) para a aba Tasks.');
  });
}

// =============================================================================
// MAIN
// =============================================================================

readInventories(function () {
  buildPlan();
});