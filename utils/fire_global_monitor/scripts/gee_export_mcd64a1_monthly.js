/**
 * MCD64A1 — área queimada mensal por país (ADM0) ou estado/província (ADM1)
 * ---------------------------------------------------------------------------
 * Plano B ao dataset pronto do GWIS (ver ../README.md). Use quando precisar de
 * meses mais recentes do que o GWIS publica.
 *
 * Um mês = uma task de exportação. Cada task gera um CSV com o esquema do painel:
 *   level, gid, iso3, name, parent, year, month, burned_ha, area_ha
 *
 * Estratégia: rasteriza os limites (geoBoundaries v6) e faz UMA redução agrupada
 * (Reducer.sum().repeat(2).group) na projeção nativa do MODIS. reduceRegions sobre
 * polígonos globais falha na sinusoidal ("Unable to transform edge"), e getInfo
 * estoura o tempo — por isso Export.
 *
 * Cole no Code Editor do GEE, ajuste CONFIG e rode. Depois: Tasks > Run.
 */

var CONFIG = {
  level: 0,                       // 0 = países (ADM0) | 1 = estados/províncias (ADM1)
  months: ['2026-09'],            // 'YYYY-MM' — uma task por item
  driveFolder: 'mcd64a1_global',  // pasta no Google Drive
  scale: 463.312716528,           // pixel nativo do MCD64A1
  taskTag: 'GLOBAL_ANALISES'      // nome das tasks: WS_IPAM_{taskTag}-{arquivo}
};

var BOUNDS = {
  0: 'WM/geoLab/geoBoundaries/600/ADM0',
  1: 'WM/geoLab/geoBoundaries/600/ADM1'
};

var col = ee.ImageCollection('MODIS/061/MCD64A1');
var units = ee.FeatureCollection(BOUNDS[CONFIG.level]);

// id inteiro sequencial para rasterizar
var ids = units.aggregate_array('system:index');
units = units.map(function (f) {
  return f.set('uid', ids.indexOf(f.get('system:index')).add(1));
});
var uidImg = units.reduceToImage(['uid'], ee.Reducer.first()).rename('uid').toInt();

// tabela uid -> atributos (para o join após a redução)
var adm0 = ee.FeatureCollection(BOUNDS[0]);
var lookup = units.map(function (f) {
  var iso3 = f.get('shapeGroup');
  var parent = CONFIG.level === 0 ? '' :
    ee.Feature(adm0.filter(ee.Filter.eq('shapeGroup', iso3)).first()).get('shapeName');
  return ee.Feature(null, {
    uid: f.get('uid'),
    gid: CONFIG.level === 0 ? iso3 : f.get('shapeID'),
    iso3: iso3,
    name: f.get('shapeName'),
    parent: parent
  });
});

var world = ee.Geometry.Rectangle([-180, -60, 180, 85], null, false);

function exportMonth(ym) {
  var start = ee.Date(ym + '-01');
  var img = col.filterDate(start, start.advance(1, 'month')).first();
  var proj = img.select('BurnDate').projection();
  var pxHa = ee.Image.pixelArea().divide(1e4);
  var burned = pxHa.multiply(img.select('BurnDate').gt(0).unmask(0)).rename('burned_ha');

  var stats = burned.addBands(pxHa.rename('area_ha')).addBands(uidImg)
    .reduceRegion({
      reducer: ee.Reducer.sum().repeat(2).group({groupField: 2, groupName: 'uid'}),
      geometry: world, crs: proj, scale: CONFIG.scale,
      maxPixels: 1e13, tileScale: 4
    });

  var y = parseInt(ym.slice(0, 4), 10), m = parseInt(ym.slice(5, 7), 10);
  var rows = ee.FeatureCollection(ee.List(stats.get('groups')).map(function (g) {
    g = ee.Dictionary(g);
    var sums = ee.List(g.get('sum'));
    var meta = ee.Feature(lookup.filter(ee.Filter.eq('uid', g.get('uid'))).first());
    return ee.Feature(null, {
      level: CONFIG.level, gid: meta.get('gid'), iso3: meta.get('iso3'),
      name: meta.get('name'), parent: meta.get('parent'),
      year: y, month: m,
      burned_ha: sums.get(0), area_ha: sums.get(1)
    });
  }));

  var desc = 'mcd64a1_adm' + CONFIG.level + '_' + ym.replace('-', '_');
  Export.table.toDrive({
    collection: rows, description: 'WS_IPAM_' + CONFIG.taskTag + '-' + desc, fileNamePrefix: desc,
    folder: CONFIG.driveFolder, fileFormat: 'CSV',
    selectors: ['level', 'gid', 'iso3', 'name', 'parent', 'year', 'month', 'burned_ha', 'area_ha']
  });
}

CONFIG.months.forEach(exportMonth);
print('Último mês disponível no MCD64A1:',
  ee.Date(col.aggregate_max('system:time_start')).format('YYYY-MM'));
