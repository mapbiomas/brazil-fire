/** @description  - EXPORTADOR UNIVERSAL - ESTIMATIVAS COLEÇÃO 4 (Série Temporal)
 *
 *  Este script centraliza o cálculo de áreas queimadas para diversos subprodutos
 *  e recortes territoriais. Baseado na lógica modular e assertiva (Fire * 100 + LULC).
 *
 *  by: Instituto de Pesquisa Ambiental da Amazônia (IPAM)
 *  desenvolvimento: Wallace Silva e Vera Laisa
 */

// ============================================================
// 1. O QUE VOCÊ QUER EXPORTAR? (Ative o que precisar)
// ============================================================

// --- Camadas de Dados (Datasets) ---
var SUBPRODUTOS_ATIVOS = [
  1, // Mensal (Cruzamento Fogo x LULC)
  2, // Anual (Cruzamento Fogo x LULC)
  3, // Acumulado (Cruzamento Fogo x LULC)
  4, // Frequência (Cruzamento Fogo x LULC)
  5, // Tamanho de Cicatrizes (Cruzamento Tamanho x LULC)
  6, // Mensal (Apenas Fogo)
  7, // Anual (Apenas Fogo)
  8, // Acumulado (Apenas Fogo)
  9, // Frequência (Apenas Fogo)
  10, // Tempo após fogo (Métrica temporal)
  11, // Ano do último fogo (Métrica temporal)
];

// --- Recortes Territoriais (Spatial Units) ---
var RECORTES_ATIVOS = [
  'bioma_estado',
  'bioma_municipio',
  // 'bioma_tis',         
  // 'bioma_ucs_federal', 
  // 'bioma_ucs_estadual' 
];

// ============================================================
// 2. PARÂMETROS GERAIS
// ============================================================
var driverFolder = 'MapBiomas_Fogo_Col4_Estatisticas_Universal';
var collection   = 'MapBiomas Fogo Coleção 4';
var scale        = 30;

// Opcional: Filtro de anos. Se vazio [], processa todas as bandas encontradas no asset.
var ANOS_FILTRO = []; 

// ============================================================
// 3. CARREGAMENTO DE DADOS (Assets Públicos)
// ============================================================
var lulc = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1');

lulc = lulc.addBands(lulc.slice(-1).rename('classification_2025')); // duplicando o ultimo ano para ficar com o mesmo periodo a série

var fire_monthly    = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_monthly_burned_v1');
var fire_annual     = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_v1');
var fire_acc        = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_accumulated_burned_v1').slice(0,40);
var fire_freq       = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_fire_frequency_v1').slice(0,40);
var fire_scar_size  = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_scar_size_range_v1');
var time_after_fire = ee.Image().rename('classification_1985')
  .addBands(ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_time_after_fire_v1'));
var year_last_fire  = ee.Image().rename('classification_1985')
  .addBands(ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_year_last_fire_v1'));

// ============================================================
// 4. LEGENDAS E AUXILIARES
// ============================================================
var legends      = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends.js');
var legends_terr = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends_territorios.js');

var LEGENDS = {
  biomes:    ee.Dictionary(legends.get('biomas')),
  estados:   ee.Dictionary(legends.get('estados')),
  ufs:       ee.Dictionary(legends.get('ufs')),
  situation: ee.Dictionary(legends.get('fogo_situation')),
  months:    ee.Dictionary(legends.get('fogo_months_str')),
  nivel_0:   ee.Dictionary(legends.get('lulc_mbc10_nivel_0')),
  nivel_1:   ee.Dictionary(legends.get('lulc_mbc10_nivel_1')),
  nivel_1_1: ee.Dictionary(legends.get('lulc_mbc10_nivel_1_1')),
  nivel_2:   ee.Dictionary(legends.get('lulc_mbc10_nivel_2')),
  nivel_3:   ee.Dictionary(legends.get('lulc_mbc10_nivel_3')),
  nivel_4:   ee.Dictionary(legends.get('lulc_mbc10_nivel_4')),
  mun_names: ee.Dictionary(legends_terr.get('municipios_name')),
  tis_names: ee.Dictionary(legends_terr.get('tis_name')),
  ucs_federal_pi: ee.Dictionary(legends_terr.get('federal_ucs_pi_name')),
  ucs_federal_us: ee.Dictionary(legends_terr.get('federal_ucs_us_name')),
  ucs_estadual_pi: ee.Dictionary(legends_terr.get('federal_ucs_pi_name')), 
  ucs_estadual_us: ee.Dictionary(legends_terr.get('federal_ucs_us_name')),
  scar_n1:   ee.Dictionary({ 0:'não observado',1:'< 250 ha', 2:'< 250 ha', 3:'250 - 500 ha', 4:'500 - 10.000 ha', 5:'500 - 10.000 ha', 6:'10.000 - 100.000 ha', 7:'10.000 - 100.000 ha', 8:'>= 100.000 ha' }),
  scar_n2:   ee.Dictionary({ 0:'não observado',1:'< 10 ha',  2:'10 - 250 ha', 3:'250 - 500 ha', 4:'500 - 5.000 ha', 5:'5.000 - 10.000 ha', 6:'10.000 - 50.000 ha', 7:'50.000 - 100.000 ha', 8:'>= 100.000 ha' })
};

// ============================================================
// 5. CATÁLOGO DE DATASETS (DADOS)
// ============================================================
var DATASETS = {
  1: {
    name: 'Mensal (Cobertura)', id: 'burned_cover_monthly',
    image: fire_monthly.multiply(100).unmask().add(lulc),
    decode: function(val) {
      val = ee.Number(val); var m = val.divide(100).int(); var l = val.mod(100).int();
      return ee.Dictionary({ 
        'Mês': m, 'Meses': LEGENDS.months.get(m), 'Situação': LEGENDS.situation.get(m.gte(1)), 
        'Classe': l, 'Nível 0': LEGENDS.nivel_0.get(l), 'Nível 1': LEGENDS.nivel_1.get(l), 
        'Nível 1_1': LEGENDS.nivel_1_1.get(l), 'Nível 2': LEGENDS.nivel_2.get(l), 
        'Nível 3': LEGENDS.nivel_3.get(l), 'Nível 4': LEGENDS.nivel_4.get(l) 
      });
    },
    columns: ['Mês', 'Meses', 'Situação', 'Classe', 'Nível 0', 'Nível 1', 'Nível 1_1'/*, 'Nível 2', 'Nível 3'*/, 'Nível 4']
  },
  2: {
    name: 'Anual (Cobertura)', id: 'burned_cover_annual',
    image: fire_annual.multiply(100).unmask().add(lulc),
    decode: function(val) {
      val = ee.Number(val); var f = val.divide(100).int(); var l = val.mod(100).int();
      return ee.Dictionary({ 
        'Situação': LEGENDS.situation.get(f.gte(1)), 'Classe': l, 
        'Nível 0': LEGENDS.nivel_0.get(l), 'Nível 1': LEGENDS.nivel_1.get(l), 
        'Nível 1_1': LEGENDS.nivel_1_1.get(l), 'Nível 2': LEGENDS.nivel_2.get(l), 
        'Nível 3': LEGENDS.nivel_3.get(l), 'Nível 4': LEGENDS.nivel_4.get(l) 
      });
    },
    columns: ['Situação', 'Classe', 'Nível 0', 'Nível 1', 'Nível 1_1'/*, 'Nível 2', 'Nível 3'*/, 'Nível 4']
  },
  3: {
    name: 'Acumulado (Cobertura)', id: 'burned_cover_accumulated',
    image: fire_acc.multiply(100).unmask().add(lulc),
    decode: function(val) {
      val = ee.Number(val); var a = val.divide(100).int(); var l = val.mod(100).int();
      return ee.Dictionary({ 
        'Acumulado': a, 'Classe': l, 
        'Nível 0': LEGENDS.nivel_0.get(l), 'Nível 1': LEGENDS.nivel_1.get(l), 
        'Nível 1_1': LEGENDS.nivel_1_1.get(l), 'Nível 2': LEGENDS.nivel_2.get(l), 
        'Nível 3': LEGENDS.nivel_3.get(l), 'Nível 4': LEGENDS.nivel_4.get(l) 
      });
    },
    columns: ['Acumulado', 'Classe', 'Nível 0', 'Nível 1', 'Nível 1_1'/*, 'Nível 2', 'Nível 3'*/, 'Nível 4']
  },
  4: {
    name: 'Frequência (Cobertura)', id: 'fire_frequency_coverage',
    image: fire_freq.multiply(100).unmask().add(lulc),
    decode: function(val) {
      val = ee.Number(val); var f = val.divide(100).int(); var l = val.mod(100).int();
      return ee.Dictionary({ 
        'Frequência': f, 'Classe': l, 
        'Nível 0': LEGENDS.nivel_0.get(l), 'Nível 1': LEGENDS.nivel_1.get(l), 
        'Nível 1_1': LEGENDS.nivel_1_1.get(l), 'Nível 2': LEGENDS.nivel_2.get(l), 
        'Nível 3': LEGENDS.nivel_3.get(l), 'Nível 4': LEGENDS.nivel_4.get(l) 
      });
    },
    columns: ['Frequência', 'Classe', 'Nível 0', 'Nível 1', 'Nível 1_1'/*, 'Nível 2', 'Nível 3'*/, 'Nível 4']
  },
  5: {
    name: 'Tamanho de Cicatrizes', id: 'annual_burned_scar_size_range',
    image: fire_scar_size.multiply(100).unmask().add(lulc),
    decode: function(val) {
      val = ee.Number(val); var s = val.divide(100).int(); var l = val.mod(100).int();
      return ee.Dictionary({ 
        'Tamanho N1': LEGENDS.scar_n1.get(s), 'Tamanho N2': LEGENDS.scar_n2.get(s), 'scar_range_id': s, 
        'Classe': l, 'Nível 0': LEGENDS.nivel_0.get(l), 'Nível 1': LEGENDS.nivel_1.get(l), 
        'Nível 1_1': LEGENDS.nivel_1_1.get(l), 'Nível 2': LEGENDS.nivel_2.get(l), 
        'Nível 3': LEGENDS.nivel_3.get(l), 'Nível 4': LEGENDS.nivel_4.get(l) 
      });
    },
    columns: ['Tamanho N1', 'Tamanho N2', 'scar_range_id', 'Classe', 'Nível 0', 'Nível 1', 'Nível 1_1'/*, 'Nível 2', 'Nível 3'*/, 'Nível 4']
  },
  6: {
    name: 'Mensal (Fogo)', id: 'monthly_burned',
    image: fire_monthly,
    decode: function(val) { 
      val = ee.Number(val); return ee.Dictionary({ 'Mês': val, 'Meses': LEGENDS.months.get(val), 'Situação': LEGENDS.situation.get(val.gte(1)) }); 
    },
    columns: ['Mês', 'Meses', 'Situação']
  },
  7: {
    name: 'Anual (Fogo)', id: 'burned_annual',
    image: fire_annual,
    decode: function(val) { 
      val = ee.Number(val); return ee.Dictionary({ 'Situação': LEGENDS.situation.get(val.gte(1)) }); 
    },
    columns: ['Situação']
  },
  8: {
    name: 'Acumulado (Fogo)', id: 'burned_accumulated',
    image: fire_acc,
    decode: function(val) { return ee.Dictionary({ 'Acumulado': val }); },
    columns: ['Acumulado']
  },
  9: {
    name: 'Frequência (Fogo)', id: 'fire_frequency',
    image: fire_freq,
    decode: function(val) { return ee.Dictionary({ 'Frequência': val }); },
    columns: ['Frequência']
  },
  10: {
    name: 'Tempo após fogo', id: 'time_after_fire',
    image: time_after_fire,
    decode: function(val) { return ee.Dictionary({ 'Anos após fogo': val }); },
    columns: ['Anos após fogo']
  },
  11: {
    name: 'Ano do último fogo', id: 'year_last_fire',
    image: year_last_fire,
    decode: function(val) { return ee.Dictionary({ 'Ano_Ultimo_Fogo': val }); },
    columns: ['Ano_Ultimo_Fogo']
  }
};

// ============================================================
// 6. CATÁLOGO DE RECORTES (SPATIAL UNITS)
// ============================================================
var SPATIAL_UNITS = {
  'bioma_estado': {
    name: 'Bioma_Estado',
    image: (function () {
      var bFC = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/biomas_IBGE_250mil');
      var sFC = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/estados-2017').map(function (f) { return f.set('id', ee.Number.parse(f.get('CD_GEOCUF'))); });
      return ee.Image().paint(bFC, 'CD_Bioma').multiply(100).add(ee.Image().paint(sFC, 'id'));
    })(),
    decode: function (val) {
      val = ee.Number(val); var ri = val.divide(100).int(); var si = val.mod(100).int();
      return ee.Dictionary({ 'Bioma': LEGENDS.biomes.get(ri), 'Estado': LEGENDS.estados.get(si), 'UF': LEGENDS.ufs.get(si) });
    },
    columns: ['Bioma', 'Estado', 'UF']
  },
  'bioma_municipio': {
    name: 'Bioma_Municipio',
    image: (function () {
      var bFC = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/biomas_IBGE_250mil');
      var mFC = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/city').map(function (f) { return f.set('id', ee.Number.parse(f.get('feature_id'))); });
      return ee.Image().paint(bFC, 'CD_Bioma').multiply(10000000).add(ee.Image().paint(mFC, 'id'));
    })(),
    decode: function (val) {
      val = ee.Number(val); var ri = val.divide(10000000).int(); var mi = val.mod(10000000).int();
      return ee.Dictionary({ 'Bioma': LEGENDS.biomes.get(ri), 'Município': LEGENDS.mun_names.get(mi), 'Cod Município': mi });
    },
    columns: ['Bioma', 'Município', 'Cod Município']
  },
  'municipio_puro': {
    name: 'Municipio',
    image: (function () {
      var mFC = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/city').map(function (f) { return f.set('id', ee.Number.parse(f.get('feature_id'))); });
      return ee.Image().paint(mFC, 'id');
    })(),
    decode: function (val) {
      val = ee.Number(val);
      return ee.Dictionary({ 'Município': LEGENDS.mun_names.get(val), 'Cod Município': val });
    },
    columns: ['Município', 'Cod Município']
  },
  'ti_puro': {
    name: 'TI',
    image: (function () {
      var tFC = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/indigenous_land').map(function (f) { return f.set('id', ee.Number.parse(f.get('feature_id'))); });
      return ee.Image().paint(tFC, 'id');
    })(),
    decode: function (val) {
      val = ee.Number(val);
      return ee.Dictionary({ 'Território': 'TI', 'Nome': LEGENDS.tis_names.get(val), 'Código': val });
    },
    columns: ['Território', 'Nome', 'Código']
  },
  'uc_federal_puro': {
    name: 'UC_Federal',
    image: (function () {
      var piFC = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/federal_conservation_units_integral_protection').map(function (f) { return f.set('id', ee.Number.parse(f.get('feature_id'))); });
      var usFC = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/federal_conservation_units_sustainable_use').map(function (f) { return f.set('id', ee.Number.parse(f.get('feature_id'))); });
      return ee.Image().paint(piFC, 'id').unmask(ee.Image().paint(usFC, 'id'));
    })(),
    decode: function (val) {
      var name = LEGENDS.ucs_federal_pi.get(val, LEGENDS.ucs_federal_us.get(val, 'Não identificado'));
      return ee.Dictionary({ 'Território': 'UC Federal', 'Nome': name, 'Código': val });
    },
    columns: ['Território', 'Nome', 'Código']
  }
};

// ============================================================
// 7. MOTOR DE CÁLCULO
// ============================================================
var calculateArea = function (image, territory, config, spatial_config, year) {
  var global_geometry = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/biomas_IBGE_250mil').geometry().bounds();
  var pixelAreaImg = ee.Image.pixelArea().divide(1000000); // km²

  var reducer = ee.Reducer.sum().group(1, 'class').group(1, 'territory');
  var data = pixelAreaImg.addBands(territory.rename('territory')).addBands(image.rename('class'))
    .reduceRegion({ reducer: reducer, geometry: global_geometry, scale: scale, maxPixels: 1e12 });

  var areas = ee.List(data.get('groups')).map(function (terrObj) {
    terrObj = ee.Dictionary(terrObj);
    var spatialProps = spatial_config.decode(terrObj.getNumber('territory'));

    return ee.List(terrObj.get('groups')).map(function (classObj) {
      classObj = ee.Dictionary(classObj);
      var props = config.decode(classObj.getNumber('class'));
      return ee.Feature(null, spatialProps.combine(props))
        .set({ 'Área ha': ee.Number(classObj.get('sum')).multiply(100), 'Ano': year, 'Coleção': collection });
    });
  });

  return ee.FeatureCollection(areas.flatten());
};

// ============================================================
// 8. EXECUÇÃO
// ============================================================
SUBPRODUTOS_ATIVOS.forEach(function (pid) {
  var config = DATASETS[pid];
  if (!config) return;

  config.image.bandNames().evaluate(function (bandnames) {
    
    RECORTES_ATIVOS.forEach(function (sid) {
      var spatial_config = SPATIAL_UNITS[sid];
      if (!spatial_config) return;

      var bandsToProcess = bandnames.filter(function (b) {
        if (ANOS_FILTRO.length === 0) return true;
        return ANOS_FILTRO.indexOf(b.slice(-4)) !== -1;
      });

      var serieTemporal = bandsToProcess.map(function (bandName) {
        var year = parseInt(bandName.slice(-4));
        var image = config.image.select(bandName);
        return calculateArea(image, spatial_config.image, config, spatial_config, year);
      });

      var final_fc = ee.FeatureCollection(serieTemporal).flatten();
      var fileName = 'export_' + config.id + '_' + spatial_config.name;

      print('Disparando Série:', fileName);

      Export.table.toDrive({
        collection: final_fc,
        description: fileName,
        folder: driverFolder,
        fileNamePrefix: fileName,
        fileFormat: 'CSV',
        selectors: ['Ano'].concat(spatial_config.columns).concat(config.columns).concat(['Área ha'])
      });
    });
  });
});
