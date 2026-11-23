
// Titulo: Toolkit para Geracao de Mapas de Acumulo de Combustivel — v02
// Descricao: UI refatorada (cabinets + managedLayers). Mosaicos on-the-fly com
//   SMA (GV, NPV, Soil, Cloud) para Sentinel-2, Landsat, HLS (S30/L30/merged),
//   MODIS (Terra/Aqua/merged) e VIIRS. Referencias globais: MCD64A1, FireCCI51.
// Desenvolvido por: wallace.silva@ipam.org.br e vera.arruda@ipam.org.br
// Data: 2024-06 (original) / 2026-11 

// --- LANGUAGE CONFIGURATION ---
/*
 *  Choose your language / Escolha seu idioma / Elija su idioma:
 *
 *  EN: Change APP_LANG below to 'en' for English.
 *  PT: Altere APP_LANG abaixo para 'pt' para Portugues.
 *  ES: Cambie APP_LANG abajo a 'es' para Espanol.
 */

/*disclaimer*/
print('atualização 16-11-2026, para acessar a versão anterior da ferramenta, acesse: ', ui.Label({value:'https://code.earthengine.google.com/?scriptPath=users%2Fworkspaceipam%2Fcombustible-material-accumulation%3Adev%2Fv00%2Fexplore_and_export', style:{}, targetUrl:'https://code.earthengine.google.com/?scriptPath=users%2Fworkspaceipam%2Fcombustible-material-accumulation%3Adev%2Fv00%2Fexplore_and_export'}));

var APP_LANG = 'pt';

var L = (function () {
  var dict = {
    es: {
      title: 'Acumulacion de Combustible',
      cab_config: 'TERRITORIO',
      cab_temporal: 'TEMPORAL',
      cab_layers: 'CAPAS',
      lbl_territorio: 'Territorio',
      lbl_territorio_type: 'Tipo',
      lbl_regions: 'Regiones',
      lbl_periodicity: 'Periodicidad',
      opt_annual: 'Anual',
      opt_semestral: 'Semestral',
      opt_trimestral: 'Trimestral',
      opt_monthly: 'Mensual',
      lbl_satellites: 'Mosaicos en Tiempo Real',
      lbl_reference: 'Referencia',
      lbl_vis_preset: 'Visualizacion',
      opt_fuel: 'Combustible [NPV/GV/SOIL]',
      opt_rgb_fire: '[SWIR1/NIR/RED] A',
      opt_rgb_coverage: '[SWIR1/NIR/RED] B',
      opt_rgb_false: '[NIR/SWIR1/RED]',
      opt_rgb_natural: '[RED/GREEN/BLUE]',
      opt_gray_nir: 'NIR',
      opt_gray_swir1: 'SWIR1',
      opt_gray_swir2: 'SWIR2',
      opt_gray_nbr: 'NBR',
      opt_gray_ndvi: 'NDVI',
      lbl_ref_burned: 'Area Quemada',
      ref_mcd64a1: 'MCD64A1 (MODIS)',
      ref_fire_cci: 'FireCCI51 (ESA)',
      btn_all_regions: 'Todo',
      drawer_sensors: 'Mosaicos em Tiempo Real',
      cab_mosaics: 'Mosaicos Mensuales de Combustible (HLS)',
      opt_asset_minnbr: 'HSL min NBR',
      opt_asset_median: 'HSL Mediana',
      cab_fire_data: 'Datos de fuego',
      lbl_aggregation: 'Reductores temporales',
      agg_min: 'min NBR',
      agg_max: 'max NBR',
      agg_median: 'Mediana',
      agg_mean: 'Media',
      loading: 'Cargando...',
      cab_info: 'INFORMACION',
      drawer_sources: 'Fuentes',
      drawer_tutorial: 'Tutorial',
      drawer_about: 'Acerca de',
      sources_text: 'Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED), Landsat 5-9 (USGS C2 T1 L2), HLS v002 (NASA HLSS30/HLSL30), MODIS Terra (MOD09A1), MODIS Aqua (MYD09A1), VIIRS (VNP09GA). Referencias: MCD64A1 (MODIS Burned Area), FireCCI51 (ESA). Fracciones SMA: GV (vegetacion verde), NPV (material senescente), Soil (suelo expuesto), Cloud (nube). Indices: NBR (quema), NDVI (vegetacion).',
      tutorial_text: '1. En TERRITORIO, seleccione el tipo de territorio y las regiones de interes. 2. En TEMPORAL, defina periodicidad y periodos. 3. En CAPAS, elija sensores, visualizaciones y referencias. 4. Active THUMBNAILS para ver la cuadricula de miniaturas. 5. Use el selector Grid para reorganizar bloques y celdas. 6. Active Exportar para ver botones de descarga. 7. Haga clic en las miniaturas para recargar.',
      about_text: 'Herramienta de generacion de mapas de acumulacion de material combustible mediante analisis de mezcla espectral (SMA). Desarrollado por Wallace Silva y Vera Arruda (IPAM) en el ambito de MapBiomas Fuego. Utiliza series temporales de imagenes satelitales para estimar la dinamica del combustible disponible para quema.',
    },
    pt: {
      title: 'Acumulacao de Combustivel',
      cab_config: 'TERRITORIO',
      cab_temporal: 'TEMPORAL',
      cab_layers: 'CAMADAS',
      lbl_territorio: 'Territorio',
      lbl_territorio_type: 'Tipo',
      lbl_regions: 'Regioes',
      lbl_periodicity: 'Periodicidade',
      opt_annual: 'Anual',
      opt_semestral: 'Semestral',
      opt_trimestral: 'Trimestral',
      opt_monthly: 'Mensal',
      lbl_satellites: 'Mosaicos em Tempo Real',
      lbl_reference: 'Referencia',
      lbl_vis_preset: 'Visualizacao',
      opt_fuel: 'Combustivel [NPV/GV/SOIL]',
      opt_rgb_fire: '[SWIR1/NIR/RED] A',
      opt_rgb_coverage: '[SWIR1/NIR/RED] B',
      opt_rgb_false: '[NIR/SWIR1/RED]',
      opt_rgb_natural: '[RED/GREEN/BLUE]',
      opt_gray_nir: 'NIR',
      opt_gray_swir1: 'SWIR1',
      opt_gray_swir2: 'SWIR2',
      opt_gray_nbr: 'NBR',
      opt_gray_ndvi: 'NDVI',
      lbl_ref_burned: 'Area Queimada',
      ref_mcd64a1: 'MCD64A1 (MODIS)',
      ref_fire_cci: 'FireCCI51 (ESA)',
      btn_all_regions: 'Tudo',
      drawer_sensors: 'Mosaicos em Tempo Real',
      cab_mosaics: 'Mosaicos Mensais de Combustível (HLS)',
      opt_asset_minnbr: 'HSL min NBR',
      opt_asset_median: 'HSL Mediana',
      cab_fire_data: 'Dados de fogo',
      lbl_aggregation: 'Redutores temporais',
      agg_min: 'min NBR',
      agg_max: 'max NBR',
      agg_median: 'Mediana',
      agg_mean: 'Media',
      loading: 'Carregando...',
      cab_info: 'INFORMACOES',
      drawer_sources: 'Fontes',
      drawer_tutorial: 'Tutorial',
      drawer_about: 'Sobre',
      sources_text: 'Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED), Landsat 5-9 (USGS C2 T1 L2), HLS v002 (NASA HLSS30/HLSL30), MODIS Terra (MOD09A1), MODIS Aqua (MYD09A1), VIIRS (VNP09GA). Referencias: MCD64A1 (MODIS Burned Area), FireCCI51 (ESA). Fracoes SMA: GV (vegetacao verde), NPV (material senescente), Soil (solo exposto), Cloud (nuvem). Indices: NBR (queimada), NDVI (vegetacao).',
      tutorial_text: '1. Em TERRITORIO, selecione o tipo de territorio e as regioes de interesse. 2. Em TEMPORAL, defina periodicidade e periodos. 3. Em CAMADAS, escolha sensores, visualizacoes e referencias. 4. Ative THUMBNAILS para ver o grid de miniaturas. 5. Use o seletor Grid para reorganizar blocos e celulas. 6. Ative Exportar para ver botoes de download. 7. Clique nas miniaturas para recarregar.',
      about_text: 'Ferramenta de geracao de mapas de acumulo de material combustivel via analise de mistura espectral (SMA). Desenvolvido por Wallace Silva e Vera Arruda (IPAM) no ambito do MapBiomas Fogo. Utiliza series temporais de imagens de satelite para estimar a dinamica do combustivel disponivel para queima.',
    },
    en: {
      title: 'Fuel Accumulation',
      cab_config: 'TERRITORY',
      cab_temporal: 'TEMPORAL',
      cab_layers: 'LAYERS',
      lbl_territorio: 'Territory',
      lbl_territorio_type: 'Type',
      lbl_regions: 'Regions',
      lbl_periodicity: 'Periodicity',
      opt_annual: 'Annual',
      opt_semestral: 'Semester',
      opt_trimestral: 'Quarterly',
      opt_monthly: 'Monthly',
      lbl_satellites: 'Satellite',
      lbl_reference: 'Reference',
      lbl_vis_preset: 'Visualization',
      opt_fuel: 'Fuel [NPV/GV/SOIL]',
      opt_rgb_fire: '[SWIR1/NIR/RED] A',
      opt_rgb_coverage: '[SWIR1/NIR/RED] B',
      opt_rgb_false: '[NIR/SWIR1/RED]',
      opt_rgb_natural: '[RED/GREEN/BLUE]',
      opt_gray_nir: 'NIR',
      opt_gray_swir1: 'SWIR1',
      opt_gray_swir2: 'SWIR2',
      opt_gray_nbr: 'NBR',
      opt_gray_ndvi: 'NDVI',
      lbl_ref_burned: 'Burned Area',
      ref_mcd64a1: 'MCD64A1 (MODIS)',
      ref_fire_cci: 'FireCCI51 (ESA)',
      btn_all_regions: 'All',
      drawer_sensors: 'On the fly mosaics',
      cab_mosaics: 'Monthly Fuel Mosaics (HLS)',
      opt_asset_minnbr: 'HSL min NBR',
      opt_asset_median: 'HSL Mediana',
      cab_fire_data: 'Fire data',
      lbl_aggregation: 'Temporal reducers',
      agg_min: 'min NBR',
      agg_max: 'max NBR',
      agg_median: 'Median',
      agg_mean: 'Mean',
      loading: 'Loading...',
      cab_info: 'INFORMATION',
      drawer_sources: 'Sources',
      drawer_tutorial: 'Tutorial',
      drawer_about: 'About',
      sources_text: 'Sentinel-2 (COPERNICUS/S2_SR_HARMONIZED), Landsat 5-9 (USGS C2 T1 L2), HLS v002 (NASA HLSS30/HLSL30), MODIS Terra (MOD09A1), MODIS Aqua (MYD09A1), VIIRS (VNP09GA). References: MCD64A1 (MODIS Burned Area), FireCCI51 (ESA). SMA fractions: GV (green vegetation), NPV (senescent material), Soil (bare soil), Cloud. Indices: NBR (burn), NDVI (vegetation).',
      tutorial_text: '1. In TERRITORY, select territory type and regions of interest. 2. In TEMPORAL, set periodicity and periods. 3. In LAYERS, choose sensors, visualizations and references. 4. Enable THUMBNAILS to view the thumbnail grid. 5. Use the Grid selector to rearrange blocks and cells. 6. Enable Export to see download buttons. 7. Click thumbnails to reload.',
      about_text: 'Fuel accumulation mapping tool via spectral mixture analysis (SMA). Developed by Wallace Silva and Vera Arruda (IPAM) as part of MapBiomas Fire. Uses satellite image time series to estimate available fuel dynamics for burning.',
    }
  };
  return dict[APP_LANG] || dict['en'];
})();

Map.setOptions('SATELLITE');

// --- TERRITORY CONFIGURATION ---
var GAUL_LEVEL1 = ee.FeatureCollection('FAO/GAUL/2015/level1');
var REGION_PROPERTY = 'ADM1_NAME';
var regionsCache = {};

var territoryTypes = {
  'Paises':            { id: ee.FeatureCollection('FAO/GAUL/2015/level0').filter(ee.Filter.inList('ADM0_NAME',['Brazil','Peru'])), prop: 'ADM0_NAME', filterProp: 'ADM0_NAME', filterValues: null },
  'Biomas':            { id: 'projects/ee-ipam/assets/IBGE/limite_biomas_IBGE_2026', prop: 'Bioma', filterProp: null, filterValues: null, defaultRegion: 'Pantanal' },
  'UF':                { id: 'projects/ee-ipam/assets/IBGE/IBGE_BR_UF_2025', prop: 'SIGLA_UF', filterProp: null, filterValues: null },
  'UCs':               { id: 'projects/mapbiomas-workspace/AUXILIAR/ucs-2019', prop: 'DESCRICAO', filterProp: null, filterValues: null },
  'TIs':               { id: 'users/geomapeamentoipam/AUXILIAR/territorios/tis_funai_2023', prop: 'terrai_nom', filterProp: null, filterValues: null },
  'Assentamentos':     { id: 'projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO8/VERSAO-1/settlements', prop: 'NAME', filterProp: null, filterValues: null },
  'Quilombolas':       { id: 'projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO8/VERSAO-1/quilombos', prop: 'NAME', filterProp: null, filterValues: null }
};

var ufList = ['AC','AL','AP','AM','BA','CE','DF','ES','GO','MA','MT','MS','MG','PA','PB','PR','PE','PI','RJ','RN','RS','RO','RR','SC','SP','SE','TO'];
ufList.forEach(function(uf) {
  territoryTypes['Municipios-' + uf] = {
    id: 'projects/ee-ipam/assets/IBGE/IBGE_BR_Municipios_2025',
    prop: 'NM_MUN',
    filterProp: 'SIGLA_UF',
    filterValues: [uf]
  };
});

var territoryTypeKeys = Object.keys(territoryTypes);

var current_regiones = GAUL_LEVEL1.filter(ee.Filter.eq('ADM0_NAME', 'Brazil'));

// --- VISUALIZATION SETTINGS ---
var VIS_PRESETS = {
  'fuel': { bands: ['npv', 'gv', 'soil'], min: [0, 5, 10], max: [17, 42, 90] },
  'fire': { bands: ['swir1', 'nir', 'red'], min: 0.03, max: 0.40 },
  'coverage': { bands: ['swir1', 'nir', 'red'], gain: [8, 6, 20], gamma: 0.85 },
  'false': { bands: ['nir', 'swir1', 'red'], min: 0.03, max: 0.40 },
  'natural': { bands: ['red', 'green', 'blue'], min: 0.02, max: 0.14, gamma: 2 },
  'nir': { bands: ['nir'], min: 0.08, max: 0.50 },
  'swir1': { bands: ['swir1'], min: 0.05, max: 0.40 },
  'swir2': { bands: ['swir2'], min: 0.02, max: 0.30 },
  'nbr': { bands: ['max_nbr'], min: 100, max: 1900 },
  'ndvi': { bands: ['ndvi'], min: 0.1, max: 0.8 }
};

var managedLayers = {};
var rawCheckboxes = {};
var fuelCheckboxes = {};
var fuelWarnedNotMonthly = {};
var regionCheckboxes = {};
var extraCheckboxes = {};
var aggregationCheckboxes = {};

// --- NBR REDUCTION METHOD (v06) ---
// The options are independent: each checked one yields an extra layer and an
// extra thumbnail cell. min/max are used only as the quality band inside
// qualityMosaic; median/mean reduce the whole collection and reference no band.
// The default 'min' reproduces the legacy per-pixel selection.
var AGGREGATION_KEYS = ['min', 'max', 'median', 'mean'];
var AGGREGATION_NBR_BAND = { min: 'min_nbr', max: 'max_nbr' };

// Assets mensais de combustivel exportados pelo export_mosaic/v00/fuel_toAsset.
// Colecao = {FUEL_ASSET_BASE}/monthly_<method>, com metadados year/month.
var FUEL_ASSET_BASE = 'projects/mapbiomas-mosaics/assets/FIRE/HLS/fuel_material';
var FUEL_ASSET_SOURCES = {
  'asset_minnbr': { method: 'minnbr', labelKey: 'opt_asset_minnbr' },
  'asset_median': { method: 'median', labelKey: 'opt_asset_median' }
};

// Fallback bands for an empty collection (must mirror the optical band names).
var FALLBACK_BANDS = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'min_nbr', 'max_nbr', 'ndvi', 'dayOfYear', 'gv', 'npv', 'soil', 'cloud'];

// --- REFERENCE DATASETS ---
var blockList_sentinel = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList').hls_s30();
var blockList_landsat = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList').hls_l30();

function excluirHLS(col, lista) {
  var filtros = lista.map(function(id) {
    return ee.Filter.stringStartsWith(
      'system:index', id.replace(/^(L30|S30)_/, '')
    );
  });

  return filtros.length
    ? ee.ImageCollection(col).filter(ee.Filter.or.apply(null, filtros).not())
    : ee.ImageCollection(col);
}

var extraDatasets = {
  'mcd64a1': {
    name: 'MCD64A1',
    vis: { min: 0, max: 1, palette: ['fc6000'] },
    build: function (start, end, bounds, year, month) {
      return ee.ImageCollection('MODIS/061/MCD64A1')
        .filterDate(start, end).filterBounds(bounds)
        .select('BurnDate').mean().gte(1).selfMask();
    }
  },
  'fire_cci': {
    name: 'FireCCI51',
    vis: { min: 0, max: 1, palette: ['5149ba'] },
    build: function (start, end, bounds, year, month) {
      return ee.ImageCollection('ESA/CCI/FireCCI/5_1')
        .filterDate(start, end).filterBounds(bounds)
        .select('BurnDate').mosaic().selfMask();
    }
  }
};

var layerAllRegions = ui.Map.Layer(ee.Image(), {}, 'Regions');
var layerSelectedRegion = ui.Map.Layer(ee.Image(), {}, 'Selected Region');
Map.layers().add(layerAllRegions);
Map.layers().add(layerSelectedRegion);

// --- MAIN UI ---
function user_interface() {

  function getDynamicDates(period) {
    var dates = [];
    var today = new Date();
    var currentYear = today.getFullYear();
    var currentMonth = today.getMonth() + 1;
    var maxYear = currentYear;
    var maxMonth = currentMonth;
    if (maxMonth === 0) { maxMonth = 12; maxYear--; }
    if (period === 'anual') {
      for (var y = maxYear; y >= 2000; y--) dates.push(y.toString());
    } else if (period === 'semestral') {
      for (var y = maxYear; y >= 2000; y--) {
        var maxS = (y === maxYear) ? (maxMonth >= 7 ? 2 : 1) : 2;
        for (var s = maxS; s >= 1; s--) dates.push(y + '_S' + s);
      }
    } else if (period === 'trimestral') {
      for (var y = maxYear; y >= 2000; y--) {
        var maxQ = (y === maxYear) ? Math.ceil(maxMonth / 3) : 4;
        for (var q = maxQ; q >= 1; q--) dates.push(y + '_Q' + q);
      }
    } else {
      for (var y = maxYear; y >= 2000; y--) {
        var mEnd = (y === maxYear) ? maxMonth : 12;
        for (var m = mEnd; m >= 1; m--) {
          var mm = m < 10 ? '0' + m : m.toString();
          dates.push(y + '_' + mm);
        }
      }
    }
    return dates;
  }

  function parseDateRange(dateStr) {
    var parts = dateStr.split('_');
    var year = parseInt(parts[0], 10);
    var start, end;
    if (parts.length === 1) {
      start = ee.Date.fromYMD(year, 1, 1);
      end = start.advance(1, 'year');
    } else if (parts[1].charAt(0) === 'Q') {
      var q = parseInt(parts[1].substring(1), 10);
      start = ee.Date.fromYMD(year, (q - 1) * 3 + 1, 1);
      end = start.advance(3, 'month');
    } else if (parts[1].charAt(0) === 'S') {
      var s = parseInt(parts[1].substring(1), 10);
      start = ee.Date.fromYMD(year, s === 1 ? 1 : 7, 1);
      end = start.advance(6, 'month');
    } else {
      var month = parseInt(parts[1], 10);
      start = ee.Date.fromYMD(year, month, 1);
      end = start.advance(1, 'month');
    }
    return { start: start, end: end };
  }

  var styles = {
    main_panel: { margin: '0px', padding: '1px', backgroundColor: '#ffffff', border: '1px solid #d0d0d0' },
    card: { margin: '1px', padding: '2px', border: '1px solid #e0e0e0', backgroundColor: '#fcfcfc' },
    row: { margin: '0px', padding: '0px', stretch: 'horizontal', backgroundColor: 'ffffff00' },
    title: { margin: '1px', padding: '2px', fontSize: '13px', fontWeight: 'bold', color: '#333' },
    btn_blue: { margin: '2px', padding: '0px', color: '#1a73e8', fontWeight: 'bold' }
  };

  var leftPanel = ui.Panel({ layout: ui.Panel.Layout.flow('vertical'), style: styles.main_panel });
  leftPanel.style().set({ width: '350px' });

  var panel_head = ui.Panel({ layout: ui.Panel.Layout.flow('vertical'), style: styles.row });
  var panel_body = ui.Panel({ style: { margin: '0px', padding: '0px' } });
  leftPanel.add(panel_head).add(panel_body);

  var panel_control = ui.Panel({ style: styles.card });
  panel_body.add(panel_control);

  // --- Header with IPAM brand + language flags ---
  var headerRow = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '3px', backgroundColor: '#e8f5e9', border: '1px solid #c8e6c9' } });
  // \u25CF = ● circulo preto
  var appBrand = ui.Label('Navegador \u25CF Mapas de Combustivel', { fontSize: '14px', fontWeight: 'bold', color: '#1a5c2a', margin: '1px 4px', stretch: 'horizontal' });
  headerRow.add(appBrand);
  panel_head.add(headerRow);
  var lab_loading = ui.Label({ value: '', style: { margin: '2px', color: '#888', fontSize: '10px',  minHeight: '12px'},});
  panel_head.add(lab_loading);

  // ==========================================
  // RIGHT PANEL — Thumbnail grid
  // ==========================================
  var rightPanel = ui.Panel({ layout: ui.Panel.Layout.flow('vertical'), style: styles.main_panel });
  rightPanel.style().set({ width: '460px', shown: false });
  var rightBody = ui.Panel({ style: { margin: '0px', padding: '2px' } });
  rightPanel.add(rightBody);

  var thumbnailsEnabled = false;
  var thumbDimensions = 180;
  var thumbCSSSize = 180;
  var thumbCache = {};
  var cachedThumbDim = -1;
  var mapLayersVisible = true;

  var outerDimA = 'periodo';
  var outerDimB = 'sensor';
  var innerDimA = 'regiao';
  var innerDimB = 'composicao';
  var actionsVisible = false;

  function applyThumbSize() {
    var w = thumbCSSSize + 'px';
    Object.keys(thumbCache).forEach(function (k) {
      try { if (thumbCache[k]) thumbCache[k].style().set({ width: w }); } catch (e) {}
    });
  }

  function getSelectedTerritorio() {
    var keys = Object.keys(territorioCheckboxes);
    for (var i = 0; i < keys.length; i++) {
      if (territorioCheckboxes[keys[i]].getValue()) return keys[i];
    }
    return '';
  }

  // ==========================================
  // Mosaic update logic
  // ==========================================
  var debouncePending = false;

  function debouncedUpdateMosaic() {
    if (lab_loading) lab_loading.setValue(L.loading);
    if (debouncePending) return;
    debouncePending = true;
    ee.Number(0).evaluate(function () {
      debouncePending = false;
      updateMosaic();
      if (thumbnailsEnabled) updateThumbnailGrid();
      if (lab_loading) lab_loading.setValue('');
    });
  }

  function getSelectedPeriodicity() {
    if (typeof periodicityCheckboxes === 'undefined') return 'Mensal';
    if (periodicityCheckboxes['trimestral'] && periodicityCheckboxes['trimestral'].getValue()) return 'Trimestral';
    if (periodicityCheckboxes['semestral'] && periodicityCheckboxes['semestral'].getValue()) return 'Semestral';
    if (periodicityCheckboxes['anual'] && periodicityCheckboxes['anual'].getValue()) return 'Anual';
    return 'Mensal';
  }

  function getSelectedPeriods() {
    var periods = [];
    if (typeof periodCheckboxes === 'undefined') return [];
    Object.keys(periodCheckboxes).forEach(function (k) {
      if (periodCheckboxes[k].getValue()) periods.push(k);
    });
    return periods;
  }

  function getSelectedVis() {
    var selected = [];
    if (typeof visCheckboxes === 'undefined') return [];
    Object.keys(visCheckboxes).forEach(function (k) {
      if (visCheckboxes[k].getValue()) selected.push(k);
    });
    return selected;
  }

  function getSelectedRegionNames() {
    var selected = [];
    if (typeof regionCheckboxes === 'undefined') return [];
    Object.keys(regionCheckboxes).forEach(function (k) {
      if (regionCheckboxes[k].getValue()) selected.push(k);
    });
    return selected;
  }

  function updateMosaic() {
    if (!mapLayersVisible) return;
    var checkedPeriods = getSelectedPeriods();
    var checkedVisKeys = getSelectedVis();
    var selectedNames = getSelectedRegionNames();

    var geom = current_regiones;
    if (selectedNames.length > 0) {
      geom = current_regiones.filter(ee.Filter.inList(REGION_PROPERTY, selectedNames));
    }
    var bounds = geom.geometry().bounds();
    var spatialMask = ee.Image(0).paint(geom, 1);

    var desiredLayerIds = [];

    checkedPeriods.forEach(function (dateStr) {
      var range = parseDateRange(dateStr);
      var start = range.start;
      var end = range.end;
      var year = parseInt(dateStr.split('_')[0], 10);
      // Monthly period (e.g. 2026_09) carries the month; other periodicities use 0.
      var lead = dateStr.split('_')[1];
      var month = (lead && lead.charAt(0) !== 'Q' && lead.charAt(0) !== 'S') ? parseInt(lead, 10) : 0;

      checkedVisKeys.forEach(function (visKey) {
        var vis = VIS_PRESETS[visKey];
        var requestedBands = vis.bands;
        var visSuffix = ' [' + visKey.toUpperCase() + ']';

        // --- RAW MOSAICS (optical sensors only, one layer per aggregation) ---
        var sensorIds = typeof rawCheckboxes !== 'undefined' ? Object.keys(rawCheckboxes) : [];
        sensorIds.forEach(function (sensorKey) {
          if (!rawCheckboxes[sensorKey] || !rawCheckboxes[sensorKey].getValue()) return;
          if (sensorKey.indexOf('ref_') === 0) return;

          var rawCol = buildSensorCollection(sensorKey, start, end, bounds, year);
          var sensorLabel = sensorKey.replace(/_/g, ' ');

          getSelectedAggregations().forEach(function (aggKey) {
            var layerId = 'raw_' + sensorKey + '_' + dateStr + '_' + visKey + '_' + aggKey;
            desiredLayerIds.push(layerId);

            var spectral = reduceMosaic(rawCol, aggKey, requestedBands).updateMask(spatialMask);
            var aggSuffix = (aggKey !== 'min') ? ' {' + aggKey + '}' : '';
            updateManagedLayer(layerId, spectral.select(requestedBands), vis, sensorLabel + ' - ' + dateStr + visSuffix + aggSuffix);
          });
        });
      });

      // --- REFERENCE DATA (fire products, never multiplied by aggregation) ---
      Object.keys(extraDatasets).forEach(function (key) {
        if (extraCheckboxes[key] && extraCheckboxes[key].getValue()) {
          var layerId = 'extra_' + key + '_' + dateStr;
          desiredLayerIds.push(layerId);
          var extraImg = extraDatasets[key].build(start, end, bounds, year, month);
          updateManagedLayer(layerId, extraImg.updateMask(spatialMask), extraDatasets[key].vis, extraDatasets[key].name + ' - ' + dateStr);
        }
      });

      // --- FUEL MAP (v08): assets mensais de combustivel ---
      var fuelVis = VIS_PRESETS['fuel'];
      Object.keys(FUEL_ASSET_SOURCES).forEach(function (assetKey) {
        if (!isFuelAssetEnabled(assetKey)) return;
        if (!isMonthlyPeriod(dateStr)) { warnNotMonthly(dateStr); return; }
        var assetSrc = FUEL_ASSET_SOURCES[assetKey];
        var layerId = 'fuel_' + assetKey + '_' + dateStr;
        desiredLayerIds.push(layerId);
        fuelAssetExists(dateStr, assetSrc.method, function (ok) {
          if (!ok) return;
          if (!isFuelAssetEnabled(assetKey)) return;
          if (getCheckedPeriods().indexOf(dateStr) === -1) return;
          var assetImg = fuelAssetImage(dateStr, assetSrc.method).updateMask(spatialMask);
          updateManagedLayer(layerId, assetImg, fuelVis,
            L[assetSrc.labelKey] + ' - ' + dateStr + ' [FUEL]');
        });
      });
    });

    // Remove stale layers
    Object.keys(managedLayers).forEach(function (id) {
      if (desiredLayerIds.indexOf(id) === -1) {
        Map.layers().remove(managedLayers[id]);
        delete managedLayers[id];
      }
    });
  }

  function safeQualityMosaic(col, band, requestedBands) {
    var mosaic = ee.Image(ee.Algorithms.If(
      band,
      ee.ImageCollection(col).qualityMosaic(band),
      ee.ImageCollection(col).mosaic()
    ));
    return safeImage(mosaic, requestedBands);
  }

  // Empty-collection fallback: exposes the requested bands so Image.select does
  // not throw for products absent in the requested period.
  function safeImage(image, requestedBands) {
    var names = FALLBACK_BANDS.slice();
    (requestedBands || []).forEach(function (b) {
      if (names.indexOf(b) === -1) names.push(b);
    });
    return ee.Image(ee.Algorithms.If(
      ee.Image(image).bandNames().length().eq(0),
      ee.Image.constant(ee.List.repeat(0, names.length)).rename(names).selfMask().float(),
      image
    ));
  }

  // --- NBR reduction helpers (v06) ---
  function getSelectedAggregations() {
    var selected = [];
    if (typeof aggregationCheckboxes === 'undefined') return selected;
    AGGREGATION_KEYS.forEach(function (k) {
      if (aggregationCheckboxes[k] && aggregationCheckboxes[k].getValue()) selected.push(k);
    });
    return selected;
  }

  function reduceMosaic(col, aggKey, requestedBands) {
    var ic = ee.ImageCollection(col);
    if (aggKey === 'median') return safeImage(ic.median(), requestedBands);
    if (aggKey === 'mean') return safeImage(ic.mean(), requestedBands);
    return safeQualityMosaic(col, AGGREGATION_NBR_BAND[aggKey] || 'min_nbr', requestedBands);
  }

  function updateManagedLayer(id, eeObject, vis, name) {
    if (managedLayers[id]) {
      managedLayers[id].setEeObject(eeObject);
      managedLayers[id].setVisParams(vis);
      managedLayers[id].setName(name);
    } else {
      var l = ui.Map.Layer(eeObject, vis, name);
      managedLayers[id] = l;
      var idx = Math.max(0, Map.layers().length() - 2);
      Map.layers().insert(idx, l);
    }
  }

  function updateSelectedRegionsMap() {
    var selectedNames = [];
    Object.keys(regionCheckboxes).forEach(function (key) {
      if (regionCheckboxes[key].getValue()) selectedNames.push(key);
    });
    if (selectedNames.length === 0) {
      layerSelectedRegion.setEeObject(ee.Image());
      return;
    }
    var selectedFc = current_regiones.filter(ee.Filter.inList(REGION_PROPERTY, selectedNames));
    if (mapLayersVisible) {
      try {
        layerSelectedRegion.setEeObject(selectedFc.style({ color: 'ff0000', fillColor: '00000000', width: 3 }));
        Map.centerObject(selectedFc);
      } catch (e) {}
    }
    if (thumbnailsEnabled) updateThumbnailGrid();
  }

  // ==========================================
  // COUNTRY SWITCH LOGIC
  // ==========================================
  function populateRegionCheckboxes(flowPanel, regionList, defaultRegion) {
    regionCheckboxes = {};
    regionList.forEach(function (name, i) {
      var chk = ui.Checkbox({
        label: name,
        value: defaultRegion ? name === defaultRegion : i === 0,
        onChange: function () {
          updateSelectedRegionsMap();
          debouncedUpdateMosaic();
        },
        style: { margin: '1px 3px', padding: '0px', fontSize: '10px', backgroundColor: '#f9f9f9' }
      });
      regionCheckboxes[name] = chk;
      flowPanel.add(chk);
    });
    updateSelectedRegionsMap();
    debouncedUpdateMosaic();
  }

  function loadTerritoryData(typeKey) {
    var conf = territoryTypes[typeKey];
    if (!conf) return;

    REGION_PROPERTY = conf.prop;

    var fc = ee.FeatureCollection(conf.id);
    if (conf.filterValues) {
      fc = fc.filter(ee.Filter.eq(conf.filterProp, conf.filterValues[0]));
    }
    current_regiones = fc;

    try {
      layerAllRegions.setEeObject(current_regiones.style({ color: 'ffffff', fillColor: '00000000', width: 1 }));
      layerSelectedRegion.setEeObject(ee.Image());
    } catch (e) {}

    populateTerritoryRegions(typeKey);
  }

  function populateTerritoryRegions(typeKey) {
    var conf = territoryTypes[typeKey] || {};
    var defaultRegion = conf.defaultRegion;
    drawerRegions.panel.clear();

    var header = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '0px', backgroundColor: '#f0f0f0' } });
    var titleLabel = ui.Label(L.lbl_regions, { fontSize: '11px', fontWeight: 'bold', margin: '4px', backgroundColor: '#f0f0f0' });
    var btt_all = ui.Button({
      label: L.btn_all_regions,
      style: { margin: '0px', padding: '0px', fontSize: '10px' },
      onClick: function () {
        var anyUnchecked = Object.keys(regionCheckboxes).some(function (k) { return !regionCheckboxes[k].getValue(); });
        Object.keys(regionCheckboxes).forEach(function (k) { regionCheckboxes[k].setValue(anyUnchecked, false); });
        updateSelectedRegionsMap();
        debouncedUpdateMosaic();
      }
    });
    header.add(titleLabel).add(btt_all);
    drawerRegions.panel.add(header);

    var flowPanel = ui.Panel({
      layout: ui.Panel.Layout.flow('horizontal', true),
      style: { margin: '0px', padding: '0px', backgroundColor: '#f9f9f9', maxHeight: '150px' }
    });
    drawerRegions.panel.add(flowPanel);

    if (regionsCache[typeKey]) {
      regionsCache[typeKey].sort();
      populateRegionCheckboxes(flowPanel, regionsCache[typeKey], defaultRegion);
      updatePeriodDrawer();
      return;
    }

    var loadingLbl = ui.Label('Loading...', { fontSize: '10px', color: '#888', margin: '4px' });
    flowPanel.add(loadingLbl);
    if (lab_loading) lab_loading.setValue(L.loading);

    current_regiones.aggregate_array(REGION_PROPERTY).distinct().sort().evaluate(function (list) {
      if (lab_loading) lab_loading.setValue('');
      flowPanel.remove(loadingLbl);
      if (!list || list.length === 0) {
        flowPanel.add(ui.Label('(no regions found)', { fontSize: '10px', color: '#cc0000', margin: '4px' }));
        return;
      }
      list.sort();
      regionsCache[typeKey] = list;
      populateRegionCheckboxes(flowPanel, list, defaultRegion);
    });
    updatePeriodDrawer();
  }

  // ==========================================
  // CABINET HELPERS
  // ==========================================
  function createCabinet(title, isOpen, fontSize) {
    fontSize = fontSize || '13px';
    var panel = ui.Panel({ style: { margin: '2px 0px', padding: '2px', border: '1px solid #dcdcdc', borderRadius: '4px', backgroundColor: '#fdfdfd' } });
    var content = ui.Panel({ style: { margin: '0px', padding: '0px', backgroundColor: '#fdfdfd' } });
    var header = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '2px 0px', padding: '1px', backgroundColor: '#f0f0f0', border: '1px solid #ddd' } });
    var icon = isOpen ? '\u25BC' : '\u25B6';
    var toggleBtn = ui.Button({ label: icon, style: { margin: '0px', padding: '0px 3px', backgroundColor: 'ffffff00' } });
    header.add(toggleBtn).add(ui.Label(title, { fontSize: fontSize, fontWeight: 'bold', margin: '1px 4px', color: '#444', backgroundColor: 'ffffff00' }));
    panel.add(header).add(content);
    content.style().set('shown', isOpen);
    toggleBtn.onClick(function () {
      var shown = content.style().get('shown');
      content.style().set('shown', !shown);
      toggleBtn.setLabel(shown ? '\u25B6' : '\u25BC');
    });
    return { panel: panel, content: content };
  }

  function createDrawer(title, options) {
    var panel = ui.Panel({ style: { margin: '2px 0px', padding: '1px', border: '1px solid #e0e0e0', backgroundColor: '#f9f9f9' } });
    var flowPanel = ui.Panel({
      layout: ui.Panel.Layout.flow('horizontal', true),
      style: { margin: '0px', padding: '0px', backgroundColor: '#f9f9f9', maxHeight: '100px' }
    });
    var checkboxes = {};
    options.forEach(function (opt) {
      var chk = ui.Checkbox({
        label: opt.label, value: opt.value || false,
        onChange: function (checked) {
          if (opt.onChange) opt.onChange(checked);
          debouncedUpdateMosaic();
        },
        style: { margin: '1px 3px', padding: '0px', fontSize: '10px', backgroundColor: '#f9f9f9' }
      });
      checkboxes[opt.id] = chk;
      flowPanel.add(chk);
    });
    if (title) {
      var titleLabel = ui.Label(title, { fontSize: '11px', fontWeight: 'bold', margin: '1px 2px', backgroundColor: '#f9f9f9' });
      panel.add(titleLabel);
    }
    panel.add(flowPanel);
    return { panel: panel, checkboxes: checkboxes };
  }

  // ==========================================
  // CABINET 1: CONFIGURATION
  // ==========================================
  var cabConfig = createCabinet(L.cab_config, false);
  panel_control.add(cabConfig.panel);

  var territoryOpts = territoryTypeKeys.map(function (tk) {
    return {
      id: tk, label: tk, value: tk === 'Paises',
      onChange: function (checked) {
        if (checked) {
          Object.keys(territorioCheckboxes).forEach(function (k) { if (k !== tk) territorioCheckboxes[k].setValue(false, false); });
          loadTerritoryData(tk);
          updateMetaLabel();
        }
      }
    };
  });
  var drawerTerritorio = createDrawer(L.lbl_territorio, territoryOpts);
  var territorioCheckboxes = drawerTerritorio.checkboxes;
  cabConfig.content.add(drawerTerritorio.panel);

  var drawerRegions = { panel: ui.Panel({ style: { margin: '2px 0px', padding: '2px', border: '1px solid #e0e0e0', backgroundColor: '#f9f9f9' } }) };
  cabConfig.content.add(drawerRegions.panel);

  // ==========================================
  // CABINET 2: TEMPORAL
  // ==========================================
  var cabTemporal = createCabinet(L.cab_temporal, false);
  panel_control.add(cabTemporal.panel);

  var drawerPeriodicity = createDrawer(L.lbl_periodicity, [
    { id: 'anual', label: L.opt_annual, value: false, onChange: function () { updatePeriodDrawer(); updateMetaLabel(); } },
    { id: 'semestral', label: L.opt_semestral, value: false, onChange: function () { updatePeriodDrawer(); updateMetaLabel(); } },
    { id: 'trimestral', label: L.opt_trimestral, value: false, onChange: function () { updatePeriodDrawer(); updateMetaLabel(); } },
    { id: 'mensal', label: L.opt_monthly, value: true, onChange: function () { updatePeriodDrawer(); updateMetaLabel(); } }
  ]);
  var periodicityCheckboxes = drawerPeriodicity.checkboxes;
  cabTemporal.content.add(drawerPeriodicity.panel);

  var drawerPeriodAnual = { panel: ui.Panel({ style: { margin: '1px 0px', padding: '1px' } }) };
  var drawerPeriodSemestral = { panel: ui.Panel({ style: { margin: '1px 0px', padding: '1px' } }) };
  var drawerPeriodTrimestral = { panel: ui.Panel({ style: { margin: '1px 0px', padding: '1px' } }) };
  var drawerPeriodMensal = { panel: ui.Panel({ style: { margin: '1px 0px', padding: '1px' } }) };
  cabTemporal.content.add(drawerPeriodAnual.panel).add(drawerPeriodSemestral.panel).add(drawerPeriodTrimestral.panel).add(drawerPeriodMensal.panel);

  var periodCheckboxes = {};
  function updatePeriodDrawer() {
    drawerPeriodAnual.panel.clear(); drawerPeriodSemestral.panel.clear(); drawerPeriodTrimestral.panel.clear(); drawerPeriodMensal.panel.clear();
    var showAnual = periodicityCheckboxes['anual'].getValue();
    var showSemestral = periodicityCheckboxes['semestral'].getValue();
    var showTrimestral = periodicityCheckboxes['trimestral'].getValue();
    var showMensal = periodicityCheckboxes['mensal'].getValue();
    periodCheckboxes = {};
    if (showAnual) populatePeriodPanel(drawerPeriodAnual.panel, 'anual');
    if (showSemestral) populatePeriodPanel(drawerPeriodSemestral.panel, 'semestral');
    if (showTrimestral) populatePeriodPanel(drawerPeriodTrimestral.panel, 'trimestral');
    if (showMensal) populatePeriodPanel(drawerPeriodMensal.panel, 'mensal');
    populateSensorDrawer();
    debouncedUpdateMosaic();
  }

  function populatePeriodPanel(pnl, type) {
    var dates = getDynamicDates(type);
    pnl.add(ui.Label(type.toUpperCase(), { fontSize: '10px', fontWeight: 'bold', margin: '2px' }));
    var flow = ui.Panel({
      layout: ui.Panel.Layout.flow('horizontal', true),
      style: { margin: '0px', backgroundColor: '#f9f9f9', maxHeight: '120px' }
    });
    pnl.add(flow);
    dates.forEach(function (d, i) {
      var chk = ui.Checkbox({
        label: d, value: (type === 'mensal' && i === 0),
        onChange: function () { debouncedUpdateMosaic(); },
        style: { margin: '1px 3px', fontSize: '10px', backgroundColor: '#f9f9f9' }
      });
      periodCheckboxes[d] = chk;
      flow.add(chk);
    });
  }

  // ==========================================
  // CABINET 3: LAYERS
  // ==========================================
  var cabLayers = createCabinet(L.cab_layers, true);
  panel_control.add(cabLayers.panel);

  var subCabSat = createCabinet(L.lbl_satellites, true, '12px');
  var subCabMosaic = createCabinet(L.cab_mosaics, true, '12px');
  var subCabRef = createCabinet(L.cab_fire_data, true, '12px');
  cabLayers.content.add(subCabSat.panel).add(subCabMosaic.panel).add(subCabRef.panel);

  // --- Visualization presets ---
  var drawerVis = createDrawer(L.lbl_vis_preset, [
    { id: 'fuel', label: L.opt_fuel, value: true },
    { id: 'fire', label: L.opt_rgb_fire, value: false },
    { id: 'coverage', label: L.opt_rgb_coverage, value: false },
    { id: 'false', label: L.opt_rgb_false, value: false },
    { id: 'natural', label: L.opt_rgb_natural, value: false },
    { id: 'nir', label: L.opt_gray_nir, value: false },
    { id: 'swir1', label: L.opt_gray_swir1, value: false },
    { id: 'swir2', label: L.opt_gray_swir2, value: false },
    { id: 'nbr', label: L.opt_gray_nbr, value: false },
    { id: 'ndvi', label: L.opt_gray_ndvi, value: false }
  ]);
  var visCheckboxes = drawerVis.checkboxes;
  subCabSat.content.add(drawerVis.panel);

  // --- NBR reduction method (v06): independent checkboxes, each checked one
  // --- yields an extra layer/cell. Default 'min' matches the legacy pick.
  var drawerAggregation = createDrawer(L.lbl_aggregation, [
    { id: 'min', label: L.agg_min, value: false },
    { id: 'max', label: L.agg_max, value: false },
    { id: 'median', label: L.agg_median, value: false },
    { id: 'mean', label: L.agg_mean, value: false }
  ]);
  aggregationCheckboxes = drawerAggregation.checkboxes;
  subCabSat.content.add(drawerAggregation.panel);

  // --- Mosaicos (v08): checkboxes dos assets mensais de combustivel ---
  var drawerMosaic = createDrawer('', [
    { id: 'asset_minnbr', label: L.opt_asset_minnbr, value: true },
    { id: 'asset_median', label: L.opt_asset_median, value: false }
  ]);
  fuelCheckboxes = drawerMosaic.checkboxes;
  subCabMosaic.content.add(drawerMosaic.panel);

  // --- Satellite mosaics (dynamic — rebuilt on period change) ---
  var drawerSensorsPanel = ui.Panel({ style: { margin: '2px 0px', padding: '1px' } });
  subCabSat.content.add(drawerSensorsPanel);
  rawCheckboxes = {};

  function getAvailableSensors(year) {
    var sensors = [];
    if (year >= 2015) sensors.push({ id: 'sentinel2', label: 'Sentinel-2' });
    if (year >= 1984) sensors.push({ id: 'landsat', label: 'Landsat' });
    if (year >= 2015) sensors.push({ id: 'hls', label: 'HLS' });
    if (year >= 2015) sensors.push({ id: 'hls_s30', label: 'HLS S30' });
    if (year >= 2013) sensors.push({ id: 'hls_l30', label: 'HLS L30' });
    if (year >= 2002) sensors.push({ id: 'modis', label: 'MODIS' });
    if (year >= 2000) sensors.push({ id: 'modis_terra', label: 'MODIS Terra' });
    if (year >= 2002) sensors.push({ id: 'modis_aqua', label: 'MODIS Aqua' });
    if (year >= 2012) sensors.push({ id: 'viirs', label: 'VIIRS' });
    sensors.push({ id: 'ref_mcd64a1', label: 'MCD64A1' });
    sensors.push({ id: 'ref_fire_cci', label: 'FireCCI51' });
    return sensors;
  }

  function populateSensorDrawer() {
    var existingState = {};
    Object.keys(rawCheckboxes).forEach(function (k) {
      try { existingState[k] = rawCheckboxes[k].getValue(); } catch (e) {}
    });
    if (Object.keys(existingState).length === 0) existingState['hls'] = false;
    rawCheckboxes = {};
    drawerSensorsPanel.clear();
    var checkedPeriods = getSelectedPeriods();
    var oldestYear = 2026;
    checkedPeriods.forEach(function (d) {
      var y = parseInt(d.split('_')[0], 10);
      if (y < oldestYear) oldestYear = y;
    });
    var available = getAvailableSensors(oldestYear);
    var sensorOpts = available.map(function (s) {
      return { id: s.id, label: s.label, value: existingState[s.id] || false };
    });
    var sensorDrawer = createDrawer(L.drawer_sensors, sensorOpts);
    rawCheckboxes = sensorDrawer.checkboxes;
    drawerSensorsPanel.add(sensorDrawer.panel);
  }

  // --- Reference data ---
  var drawerRef = createDrawer(L.lbl_ref_burned, [
    { id: 'mcd64a1', label: L.ref_mcd64a1, value: false },
    { id: 'fire_cci', label: L.ref_fire_cci, value: false }
  ]);
  extraCheckboxes = {};
  Object.keys(drawerRef.checkboxes).forEach(function (k) { extraCheckboxes[k] = drawerRef.checkboxes[k]; });
  subCabRef.content.add(drawerRef.panel);

  function applyToggleStyle(btn, on) {
    if (on) {
      btn.style().set({ backgroundColor: '#2e7d32', border: '1px solid #1b5e20' });
    } else {
      btn.style().set({ backgroundColor: '#c62828', border: '1px solid #b71c1c' });
    }
  }

  // --- Map toggle (always visible, left) ---
  var mapToggleBtn = ui.Button({
    label: 'MAP ON',
    style: { margin: '2px', padding: '2px 6px', fontSize: '11px', fontWeight: 'bold' },
    onClick: function () {
      mapLayersVisible = !mapLayersVisible;
      if (mapLayersVisible) {
        mapToggleBtn.setLabel('MAP ON');
        applyToggleStyle(mapToggleBtn, true);
        Object.keys(managedLayers).forEach(function (id) { managedLayers[id].setShown(true); });
        try { layerAllRegions.setShown(true); layerSelectedRegion.setShown(true); } catch (e) {}
        debouncedUpdateMosaic();
      } else {
        mapToggleBtn.setLabel('MAP OFF');
        applyToggleStyle(mapToggleBtn, false);
        Object.keys(managedLayers).forEach(function (id) { managedLayers[id].setShown(false); });
        try { layerAllRegions.setShown(false); layerSelectedRegion.setShown(false); } catch (e) {}
      }
    }
  });
  applyToggleStyle(mapToggleBtn, true);

  var thumbToggleBtn = ui.Button({
    label: 'THUMBNAILS OFF',
    style: { margin: '2px', padding: '2px 6px', fontSize: '11px', fontWeight: 'bold' },
    onClick: function () {
      thumbnailsEnabled = !thumbnailsEnabled;
      if (thumbnailsEnabled) {
        thumbToggleBtn.setLabel('THUMBNAILS ON');
        applyToggleStyle(thumbToggleBtn, true);
        try { rightPanel.style().set('shown', true); } catch (e) {}
        updateThumbnailGrid();
      } else {
        thumbToggleBtn.setLabel('THUMBNAILS OFF');
        applyToggleStyle(thumbToggleBtn, false);
        try { rightPanel.style().set('shown', false); } catch (e) {}
        thumbCache = {};
        rightBody.clear();
      }
    }
  });
  applyToggleStyle(thumbToggleBtn, false);

  var toggleRow = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '0px' } });
  toggleRow.add(mapToggleBtn).add(thumbToggleBtn);
  cabLayers.content.add(toggleRow);

  // ==========================================
  // CABINET 4: INFORMATION (commented out — uncomment to enable)
  // ==========================================
  /*
  var cabInfo = createCabinet(L.cab_info, false);
  panel_control.add(cabInfo.panel);
  var subSources = createCabinet(L.drawer_sources, false, '11px');
  subSources.content.add(ui.Label(L.sources_text, { fontSize: '10px', color: '#555', margin: '4px', whiteSpace: 'pre-wrap' }));
  cabInfo.content.add(subSources.panel);
  var subTutorial = createCabinet(L.drawer_tutorial, false, '11px');
  subTutorial.content.add(ui.Label(L.tutorial_text, { fontSize: '10px', color: '#555', margin: '4px', whiteSpace: 'pre-wrap' }));
  cabInfo.content.add(subTutorial.panel);
  var subAbout = createCabinet(L.drawer_about, false, '11px');
  subAbout.content.add(ui.Label(L.about_text, { fontSize: '10px', color: '#555', margin: '4px', whiteSpace: 'pre-wrap' }));
  cabInfo.content.add(subAbout.panel);
  */

  // --- Thumbnail controls (in rightPanel header) ---
  var thumbDimSelect = ui.Select({
    items: ['180', '360', '512', '720', '1024'],
    value: '180',
    style: { margin: '1px 3px', fontSize: '10px' },
    onChange: function (v) {
      thumbDimensions = parseInt(v, 10);
      if (thumbnailsEnabled) updateThumbnailGrid();
    }
  });

  var thumbSizeSelect = ui.Select({
    items: ['120', '180', '240', '360', '512', '720', '1024'],
    value: '180',
    style: { margin: '1px 3px', fontSize: '10px' },
    onChange: function (v) {
      thumbCSSSize = parseInt(v, 10);
      applyThumbSize();
    }
  });

  var gridOptions = [
    { label: 'Periodos \u00D7 Sensores  |  Regioes \u00B7 Composicoes',      outerA: 'periodo', outerB: 'sensor', innerA: 'regiao', innerB: 'composicao', invert: 'Sensores \u00D7 Periodos  |  Regioes \u00B7 Composicoes' },
    { label: 'Sensores \u00D7 Periodos  |  Regioes \u00B7 Composicoes',      outerA: 'sensor',  outerB: 'periodo', innerA: 'regiao', innerB: 'composicao', invert: 'Periodos \u00D7 Sensores  |  Regioes \u00B7 Composicoes' },
    { label: 'Periodos \u00D7 Regioes  |  Sensores \u00B7 Composicoes',      outerA: 'periodo', outerB: 'regiao',  innerA: 'sensor',  innerB: 'composicao', invert: 'Regioes \u00D7 Periodos  |  Sensores \u00B7 Composicoes' },
    { label: 'Regioes \u00D7 Periodos  |  Sensores \u00B7 Composicoes',      outerA: 'regiao',  outerB: 'periodo', innerA: 'sensor',  innerB: 'composicao', invert: 'Periodos \u00D7 Regioes  |  Sensores \u00B7 Composicoes' },
    { label: 'Periodos \u00D7 Composicoes  |  Sensores \u00B7 Regioes',      outerA: 'periodo', outerB: 'composicao', innerA: 'sensor', innerB: 'regiao', invert: 'Composicoes \u00D7 Periodos  |  Sensores \u00B7 Regioes' },
    { label: 'Composicoes \u00D7 Periodos  |  Sensores \u00B7 Regioes',      outerA: 'composicao', outerB: 'periodo', innerA: 'sensor', innerB: 'regiao', invert: 'Periodos \u00D7 Composicoes  |  Sensores \u00B7 Regioes' },
    { label: 'Sensores \u00D7 Regioes  |  Periodos \u00B7 Composicoes',      outerA: 'sensor',  outerB: 'regiao',  innerA: 'periodo', innerB: 'composicao', invert: 'Regioes \u00D7 Sensores  |  Periodos \u00B7 Composicoes' },
    { label: 'Regioes \u00D7 Sensores  |  Periodos \u00B7 Composicoes',      outerA: 'regiao',  outerB: 'sensor',  innerA: 'periodo', innerB: 'composicao', invert: 'Sensores \u00D7 Regioes  |  Periodos \u00B7 Composicoes' },
    { label: 'Sensores \u00D7 Composicoes  |  Periodos \u00B7 Regioes',      outerA: 'sensor',  outerB: 'composicao', innerA: 'periodo', innerB: 'regiao', invert: 'Composicoes \u00D7 Sensores  |  Periodos \u00B7 Regioes' },
    { label: 'Composicoes \u00D7 Sensores  |  Periodos \u00B7 Regioes',      outerA: 'composicao', outerB: 'sensor', innerA: 'periodo', innerB: 'regiao', invert: 'Sensores \u00D7 Composicoes  |  Periodos \u00B7 Regioes' },
    { label: 'Regioes \u00D7 Composicoes  |  Periodos \u00B7 Sensores',      outerA: 'regiao',  outerB: 'composicao', innerA: 'periodo', innerB: 'sensor', invert: 'Composicoes \u00D7 Regioes  |  Periodos \u00B7 Sensores' },
    { label: 'Composicoes \u00D7 Regioes  |  Periodos \u00B7 Sensores',      outerA: 'composicao', outerB: 'regiao', innerA: 'periodo', innerB: 'sensor', invert: 'Regioes \u00D7 Composicoes  |  Periodos \u00B7 Sensores' }
  ];

  var gridLayoutSelect = ui.Select({
    items: gridOptions.map(function (o) { return o.label; }),
    value: 'Periodos \u00D7 Sensores  |  Regioes \u00B7 Composicoes',
    style: { margin: '1px 3px', fontSize: '10px' },
    onChange: function (v) {
      var opt = gridOptions.find(function (o) { return o.label === v; });
      if (!opt) return;
      outerDimA = opt.outerA; outerDimB = opt.outerB;
      innerDimA = opt.innerA; innerDimB = opt.innerB;
      if (thumbnailsEnabled) updateThumbnailGrid();
    }
  });

  var invertOuterBtn = ui.Button({
    label: '\u27F7 Ext',
    style: { margin: '1px 3px', padding: '1px 4px', fontSize: '10px' },
    onClick: function () {
      var current = gridLayoutSelect.getValue();
      var opt = gridOptions.find(function (o) { return o.label === current; });
      if (opt) gridLayoutSelect.setValue(opt.invert, true);
    }
  });

  var invertInnerBtn = ui.Button({
    label: '\u27F7 Int',
    style: { margin: '1px 3px', padding: '1px 4px', fontSize: '10px' },
    onClick: function () {
      var tmp = innerDimA; innerDimA = innerDimB; innerDimB = tmp;
      if (thumbnailsEnabled) updateThumbnailGrid();
    }
  });

  var actionsToggleBtn = ui.Button({
    label: 'Exportar',
    style: { margin: '1px 3px', padding: '1px 4px', fontSize: '10px' },
    onClick: function () {
      actionsVisible = !actionsVisible;
      actionsToggleBtn.setLabel(actionsVisible ? 'Exportar ON' : 'Exportar');
      if (thumbnailsEnabled) updateThumbnailGrid();
    }
  });

  var clearCacheBtn = ui.Button({
    label: 'Limpar memoria',
    style: { margin: '1px 3px', padding: '1px 4px', fontSize: '10px' },
    onClick: function () { thumbCache = {}; fuelAssetCache = {}; fuelWarnedNotMonthly = {}; if (thumbnailsEnabled) updateThumbnailGrid(); }
  });

  var metaTerritorioLabel = ui.Label('', { fontSize: '9px', margin: '1px 4px', color: '#666' });
  function updateMetaLabel() {
    metaTerritorioLabel.setValue(getSelectedTerritorio() + ' \u00B7 ' + getSelectedPeriodicity());
  }

  var ctrlStyle = { margin: '1px 2px', fontSize: '9px', padding: '0px' };

  var line1 = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '1px', backgroundColor: '#f5f5f5' } });
  line1.add(ui.Label('Dimensao:', ctrlStyle)).add(thumbDimSelect).add(ui.Label('Tamanho:', ctrlStyle)).add(thumbSizeSelect);

  var line2 = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '1px', backgroundColor: '#f5f5f5' } });
  line2.add(ui.Label('Grid:', ctrlStyle)).add(gridLayoutSelect).add(invertOuterBtn).add(invertInnerBtn);

  var line3 = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '1px', backgroundColor: '#f5f5f5' } });
  line3.add(actionsToggleBtn).add(clearCacheBtn).add(metaTerritorioLabel);

  var thumbControlsRow = ui.Panel({ layout: ui.Panel.Layout.flow('vertical'), style: { margin: '0px', padding: '1px', backgroundColor: '#f5f5f5', border: '1px solid #ddd' } });
  thumbControlsRow.add(line1).add(line2).add(line3);

  rightPanel.insert(0, thumbControlsRow);

  // ==========================================
  // THUMBNAIL GRID
  // ==========================================

  function buildRegionBounds(regionName) {
    var feat = current_regiones.filter(ee.Filter.eq(REGION_PROPERTY, regionName));
    return feat.geometry().bounds();
  }

  var sensorDefinitions = {
    'sentinel2': {
      collection: "COPERNICUS/S2_SR_HARMONIZED",
      preprocess: function (col, bounds, start, end) {
        return ee.ImageCollection(col)
          .filterDate(start, end).filterBounds(bounds)
          .linkCollection(ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs'])
          .map(processS2);
      }
    },
    'landsat': {
      collection: null,
      build: function (bounds, start, end, year) {
        var rawCol = ee.ImageCollection([]);
        if (year >= 2022) rawCol = rawCol.merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(start, end).filterBounds(bounds).map(processLS89));
        if (year <= 2024) rawCol = rawCol.merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start, end).filterBounds(bounds).map(processLS89));
        if (year <= 2021) rawCol = rawCol.merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterDate(start, end).filterBounds(bounds).map(processLS57));
        if (year <= 2011) rawCol = rawCol.merge(ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterDate(start, end).filterBounds(bounds).map(processLS57));
        return rawCol;
      }
    },
    'hls_s30': {
      collection: "NASA/HLS/HLSS30/v002",
      preprocess: function(col, bounds, start, end) {
        return excluirHLS(
          ee.ImageCollection(col).filterDate(start, end).filterBounds(bounds),blockList_sentinel).map(processHLS_S30);
      }
    },
    
    'hls_l30': {
      collection: "NASA/HLS/HLSL30/v002",
      preprocess: function(col, bounds, start, end) {
        return excluirHLS(
          ee.ImageCollection(col).filterDate(start, end).filterBounds(bounds),blockList_landsat).map(processHLS_L30);
      }
    },
    'hls': {
      collection: null,
      build: function(bounds, start, end, year) {
        var s30 = sensorDefinitions['hls_s30'].preprocess("NASA/HLS/HLSS30/v002", bounds, start, end);
        var l30 = sensorDefinitions['hls_l30'].preprocess("NASA/HLS/HLSL30/v002", bounds, start, end);
        return s30.merge(l30);
      }
    },
    'modis_terra': {
      collection: "MODIS/061/MOD09A1",
      preprocess: function (col, bounds, start, end) {
        return ee.ImageCollection(col).filterDate(start, end).filterBounds(bounds).map(processMOD09);
      }
    },
    'modis_aqua': {
      collection: "MODIS/061/MYD09A1",
      preprocess: function (col, bounds, start, end) {
        return ee.ImageCollection(col).filterDate(start, end).filterBounds(bounds).map(processMYD09);
      }
    },
    'modis': {
      collection: null,
      build: function (bounds, start, end, year) {
        return sensorDefinitions['modis_terra'].preprocess("MODIS/061/MOD09A1", bounds, start, end)
          .merge(sensorDefinitions['modis_aqua'].preprocess("MODIS/061/MYD09A1", bounds, start, end));
      }
    },
    'viirs': {
      collection: "NASA/VIIRS/002/VNP09GA",
      preprocess: function (col, bounds, start, end) {
        return ee.ImageCollection(col).filterDate(start, end).filterBounds(bounds).map(processVIIRS);
      }
    },
    'ref_mcd64a1': {
      collection: null,
      build: function (bounds, start, end) {
        return ee.ImageCollection([extraDatasets['mcd64a1'].build(start, end, bounds, 0, 0)]);
      }
    },
    'ref_fire_cci': {
      collection: null,
      build: function (bounds, start, end) {
        return ee.ImageCollection([extraDatasets['fire_cci'].build(start, end, bounds, 0, 0)]);
      }
    }
  };

  function buildSensorCollection(sensorKey, start, end, bounds, year) {
    var def = sensorDefinitions[sensorKey];
    if (!def) return ee.ImageCollection([]);
    if (def.build) return def.build(bounds, start, end, year);
    return def.preprocess(def.collection, bounds, start, end);
  }

  function buildCellMosaic(regionName, dateStr, sensorKey, visKey, aggKey) {
    var range = parseDateRange(dateStr);
    var start = range.start;
    var end = range.end;
    var year = parseInt(dateStr.split('_')[0], 10);
    var bounds = buildRegionBounds(regionName);
    var geom = current_regiones.filter(ee.Filter.eq(REGION_PROPERTY, regionName));
    var spatialMask = ee.Image(0).paint(geom, 1);
    if (FUEL_ASSET_SOURCES[sensorKey]) {
      return fuelAssetImage(dateStr, FUEL_ASSET_SOURCES[sensorKey].method).updateMask(spatialMask);
    }
    if (sensorKey.indexOf('ref_') === 0) {
      var refKey = sensorKey.substring(4);
      return extraDatasets[refKey].build(start, end, bounds, year).updateMask(spatialMask);
    }
    var rawCol = buildSensorCollection(sensorKey, start, end, bounds, year);
    var vis = VIS_PRESETS[visKey];
    return reduceMosaic(rawCol, aggKey, vis ? vis.bands : null).updateMask(spatialMask);
  }

  function getCheckedRegions() {
    return Object.keys(regionCheckboxes).filter(function (k) { return regionCheckboxes[k].getValue(); });
  }
  function getCheckedPeriods() {
    var p = [];
    if (typeof periodCheckboxes !== 'undefined') {
      Object.keys(periodCheckboxes).forEach(function (k) { if (periodCheckboxes[k].getValue()) p.push(k); });
    }
    return p;
  }
  function getCheckedSensors() {
    var s = [];
    if (typeof rawCheckboxes !== 'undefined') {
      Object.keys(rawCheckboxes).forEach(function (k) {
        if (rawCheckboxes[k] && rawCheckboxes[k].getValue()) s.push(k);
      });
    }
    return s;
  }
  function getCheckedVisPresets() {
    var v = [];
    if (typeof visCheckboxes !== 'undefined') {
      Object.keys(visCheckboxes).forEach(function (k) { if (visCheckboxes[k].getValue()) v.push(k); });
    }
    return v;
  }

  // --- Fuel asset helpers (v08) ---
  function isMonthlyPeriod(dateStr) {
    return /^\d{4}_\d{2}$/.test(dateStr);
  }

  function warnNotMonthly(dateStr) {
    if (fuelWarnedNotMonthly[dateStr]) return;
    fuelWarnedNotMonthly[dateStr] = true;
    print('AVISO: gaveta Mosaicos so atende periodos mensais: ' + dateStr);
  }

  function isFuelAssetEnabled(key) {
    return !!(fuelCheckboxes[key] && fuelCheckboxes[key].getValue());
  }

  function isFuelSourceKey(sensorKey) {
    return !!FUEL_ASSET_SOURCES[sensorKey];
  }

  function isAnyFuelSourceEnabled() {
    return Object.keys(FUEL_ASSET_SOURCES).some(function (key) { return isFuelAssetEnabled(key); });
  }

  function fuelAssetKeyOf(dateStr, method) {
    return dateStr + '|' + method;
  }

  // Somente a colecao mensal (monthly_<method>), filtrada por year/month.
  function fuelAssetQuery(dateStr, method) {
    var parts = dateStr.split('_');
    var year = parseInt(parts[0], 10);
    var month = parseInt(parts[1], 10);
    var colId = FUEL_ASSET_BASE + '/monthly_' + method;
    var query = ee.ImageCollection(colId)
      .filter(ee.Filter.eq('year', year))
      .filter(ee.Filter.eq('month', month));
    return { colId: colId, query: query };
  }

  function fuelAssetImage(dateStr, method) {
    return fuelAssetQuery(dateStr, method).query.first();
  }

  var fuelAssetCache = {};
  var FUEL_ASSET_CACHE_MS = 60000;

  function fuelAssetExists(dateStr, method, callback) {
    if (!isMonthlyPeriod(dateStr)) { callback(false); return; }
    var key = fuelAssetKeyOf(dateStr, method);
    var cached = fuelAssetCache[key];
    if (cached && (new Date().getTime() - cached.t) < FUEL_ASSET_CACHE_MS) {
      callback(cached.status === 'ok');
      return;
    }
    var plan = fuelAssetQuery(dateStr, method);
    function store(status) {
      fuelAssetCache[key] = { status: status, t: new Date().getTime() };
    }
    function fail() {
      store('missing');
      print('AVISO: asset de combustivel indisponivel: ' + plan.colId + ' (' + dateStr + ')');
      callback(false);
    }
    plan.query.size().evaluate(function (size)  {
      if (!size) { fail(); return; }
      store('ok');
      callback(true);
    }, fail);
  }

  function getAxisValues(axis) {
    if (axis === 'periodo') return getCheckedPeriods();
    if (axis === 'sensor') {
      var sensors = getCheckedSensors();
      Object.keys(FUEL_ASSET_SOURCES).forEach(function (key) {
        if (isFuelAssetEnabled(key)) sensors.push(key);
      });
      return sensors;
    }
    if (axis === 'regiao') return getCheckedRegions();
    if (axis === 'composicao') {
      var visKeys = getCheckedVisPresets();
      if (isAnyFuelSourceEnabled() && visKeys.indexOf('fuel') === -1) visKeys.push('fuel');
      return visKeys;
    }
    return [];
  }

  function uncheckAxisItem(axis, value) {
    if (axis === 'periodo' && periodCheckboxes[value]) { periodCheckboxes[value].setValue(false, false); }
    if (axis === 'sensor' && fuelCheckboxes[value]) { fuelCheckboxes[value].setValue(false, false); }
    if (axis === 'sensor' && rawCheckboxes[value]) { rawCheckboxes[value].setValue(false, false); }
    if (axis === 'regiao' && regionCheckboxes[value]) { regionCheckboxes[value].setValue(false, false); }
    if (axis === 'composicao' && visCheckboxes[value]) { visCheckboxes[value].setValue(false, false); }
  }

  var thumbGridRequest = 0;

  function updateThumbnailGrid() {
    if (!thumbnailsEnabled) return;
    var request = ++thumbGridRequest;
    var needed = [];
    Object.keys(FUEL_ASSET_SOURCES).forEach(function (key) {
      if (!isFuelAssetEnabled(key)) return;
      var method = FUEL_ASSET_SOURCES[key].method;
      getCheckedPeriods().forEach(function (dateStr) {
        if (!isMonthlyPeriod(dateStr)) { warnNotMonthly(dateStr); return; }
        needed.push({ dateStr: dateStr, method: method });
      });
    });
    if (!needed.length) {
      renderThumbnailGrid({});
      return;
    }
    var pending = needed.length;
    var missing = {};
    needed.forEach(function (item) {
      fuelAssetExists(item.dateStr, item.method, function (ok) {
        if (request !== thumbGridRequest) return;
        if (!ok) missing[fuelAssetKeyOf(item.dateStr, item.method)] = true;
        pending -= 1;
        if (pending === 0) renderThumbnailGrid(missing);
      });
    });
  }

  function renderThumbnailGrid(missingAssets) {
    if (!thumbnailsEnabled) return;
    if (thumbDimensions !== cachedThumbDim) {
      thumbCache = {};
      cachedThumbDim = thumbDimensions;
    }
    rightBody.clear();
    updateMetaLabel();

    var outerLineItems = getAxisValues(outerDimA);
    var outerColItems = getAxisValues(outerDimB);
    var innerLineItems = getAxisValues(innerDimA);
    var innerColItems = getAxisValues(innerDimB);

    if (outerLineItems.length === 0 || outerColItems.length === 0 || innerLineItems.length === 0 || innerColItems.length === 0) {
      rightBody.add(ui.Label('Select at least one item in each axis group.', { fontSize: '11px', color: '#888', margin: '10px' }));
      return;
    }

    outerLineItems.forEach(function (outerLine) {
      var outerRowPanel = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '0px' } });
      outerColItems.forEach(function (outerCol) {
        var blockPanel = ui.Panel({ layout: ui.Panel.Layout.flow('vertical'), style: { margin: '3px 2px', border: '1px solid #aaa', padding: '2px', backgroundColor: '#fefefe' } });
        blockPanel.add(ui.Label(outerLine + ' \u00B7 ' + outerCol, { fontSize: '12px', fontWeight: 'bold', margin: '2px', padding: '2px', color: '#222', backgroundColor: '#e8e8e8', stretch: 'horizontal' }));

        innerLineItems.forEach(function (innerLine) {
          var innerRowPanel = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '0px' } });

          innerColItems.forEach(function (innerCol) {
            var vals = {};
            vals[outerDimA] = outerLine; vals[outerDimB] = outerCol; vals[innerDimA] = innerLine; vals[innerDimB] = innerCol;
            var regionName = vals['regiao']; var dateStr = vals['periodo']; var sensorKey = vals['sensor']; var visKey = vals['composicao'];
            var isFuelSource = isFuelSourceKey(sensorKey);
            if (isFuelSource && visKey !== 'fuel') return;
            if (!isFuelSource && visKey === 'fuel' && !(visCheckboxes['fuel'] && visCheckboxes['fuel'].getValue())) return;
            if (isFuelSource && !isMonthlyPeriod(dateStr)) { warnNotMonthly(dateStr); return; }
            var assetMethod = isFuelSource ? FUEL_ASSET_SOURCES[sensorKey].method : null;
            if (assetMethod && missingAssets[fuelAssetKeyOf(dateStr, assetMethod)]) return;
            var isRef = sensorKey.indexOf('ref_') === 0;
            var vis = isFuelSource ? VIS_PRESETS['fuel'] : (isRef ? extraDatasets[sensorKey.substring(4)].vis : VIS_PRESETS[visKey]);
            // Burned-area products have no NBR band: a single cell, no aggregation.
            var cellAggs = isRef ? ['min'] : (isFuelSource ? ['min'] : getSelectedAggregations());

            cellAggs.forEach(function (aggKey) {
              var cellKey = [regionName, dateStr, sensorKey, visKey, aggKey].join('_').replace(/ /g, '_');
              var mosaic = buildCellMosaic(regionName, dateStr, sensorKey, visKey, aggKey);
              var aggLabel = (aggKey !== 'min') ? ' \u00B7 ' + aggKey : '';

              var cellPanel = ui.Panel({ layout: ui.Panel.Layout.flow('vertical'), style: { margin: '1px', padding: '0px', border: '1px solid #ddd' } });
              cellPanel.add(ui.Label(innerLine + ' \u00B7 ' + innerCol + aggLabel, { fontSize: '9px', fontWeight: 'bold', textAlign: 'center', margin: '1px', padding: '1px', backgroundColor: '#e8e8e8', color: '#444', stretch: 'horizontal' }));

              var visImg;
              try { visImg = mosaic.visualize(vis); } catch (e) { visImg = ee.Image(0).rename(['vis-red','vis-green','vis-blue']).selfMask().int8(); }
              var thumb = ui.Thumbnail(visImg, { dimensions: thumbDimensions, region: buildRegionBounds(regionName) });
              if (thumb) {
                try { thumb.style().set({ width: thumbCSSSize + 'px' }); } catch (e) {}
                cellPanel.add(thumb);
                thumbCache[cellKey] = thumb;

                var urlLabel = ui.Label({ value: '', style: { fontSize: '7px', margin: '1px', shown: false } });
                visImg.getMapId({}, function (obj) { try { if (urlLabel) urlLabel.setValue(obj.urlFormat); } catch (e2) {} });

                thumb.onClick(function () {
                  var newMosaic = buildCellMosaic(regionName, dateStr, sensorKey, visKey, aggKey);
                  var newVisImg;
                  try { newVisImg = newMosaic.visualize(vis); } catch (e2) { newVisImg = ee.Image(0).rename(['vis-red','vis-green','vis-blue']).selfMask().int8(); }
                  try { thumb.setImage(newVisImg); } catch (e2) {}
                  newVisImg.getThumbURL({ dimensions: thumbDimensions, region: buildRegionBounds(regionName), format: 'png' }, function (newUrl) {
                    try { if (newUrl) urlLabel.setValue(newUrl); } catch (e2) {}
                  });
                });
              }

              var btnRow = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '0px', border: '1px solid #eee' } });
              var btnExport = ui.Button({
                label: '\u2193',
                style: { margin: '0px', padding: '0px 2px', fontSize: '10px' },
                onClick: function () {
                  var hasS2 = rawCheckboxes['sentinel2'] && rawCheckboxes['sentinel2'].getValue();
                  var has30m = isFuelSource || (rawCheckboxes['landsat'] && rawCheckboxes['landsat'].getValue()) || (rawCheckboxes['hls'] && rawCheckboxes['hls'].getValue()) || (rawCheckboxes['hls_s30'] && rawCheckboxes['hls_s30'].getValue()) || (rawCheckboxes['hls_l30'] && rawCheckboxes['hls_l30'].getValue());
                  var exportScale = hasS2 ? 10 : (has30m ? 30 : 500);
                  var aggSuffixExport = (cellAggs.length > 1 && aggKey !== 'min') ? '_' + aggKey : '';
                  Export.image.toDrive({ image: mosaic.select(vis.bands), description: (regionName + '_' + dateStr + '_' + sensorKey + '_' + visKey + aggSuffixExport).replace(/ /g, '_'), folder: 'export-gee', fileNamePrefix: (regionName + '_' + dateStr + '_' + sensorKey + '_' + visKey + aggSuffixExport).replace(/ /g, '_'), region: buildRegionBounds(regionName), scale: exportScale, maxPixels: 1e13 });
                }
              });
              btnRow.add(btnExport);
              var btnUrl = ui.Button({
                label: '\uD83D\uDD17',
                style: { margin: '0px', padding: '0px 2px', fontSize: '10px' },
                onClick: function () { try { if (!urlLabel) return; var shown = urlLabel.style().get('shown'); urlLabel.style().set('shown', !shown); } catch (e) {} }
              });
              btnRow.add(btnUrl);
              var btnRemove = ui.Button({
                label: '\u2715',
                style: { margin: '0px', padding: '0px 2px', fontSize: '10px', color: '#cc0000' },
                onClick: function () { uncheckAxisItem(isFuelSource ? 'sensor' : 'composicao', isFuelSource ? sensorKey : visKey); updateSelectedRegionsMap(); debouncedUpdateMosaic(); updateThumbnailGrid(); }
              });
              btnRow.add(btnRemove);
              if (actionsVisible) { cellPanel.add(btnRow); }
              cellPanel.add(urlLabel);
              innerRowPanel.add(cellPanel);
            });
          });
          blockPanel.add(innerRowPanel);
        });
        outerRowPanel.add(blockPanel);
      });
      rightBody.add(outerRowPanel);
    });
  }

  // ==========================================
  // INIT
  // ==========================================
  var mapWidget = ui.root.widgets().get(0);
  ui.root.remove(mapWidget);
  ui.root.insert(0, leftPanel);
  ui.root.insert(1, ui.SplitPanel({
    firstPanel: mapWidget,
    secondPanel: rightPanel,
    orientation: 'horizontal'
  }));
  populateSensorDrawer();
  ee.Number(0).evaluate(function () {
    loadTerritoryData('Paises');
  });
}

user_interface();

// --- PROCESSING FUNCTIONS (cloud mask + rename + NBR + SMA + DOY) ---

function addBand_NBR(image) {
  var exp = '( b("nir") - b("swir2") ) / ( b("nir") + b("swir2") )';
  var NBR = image
    .expression(exp)
    .add(1)
    .multiply(1000)
    .int16()
    .rename("nbr");

  return image
    .addBands(NBR.multiply(-1).rename("min_nbr"))
    .addBands(NBR.rename("max_nbr"));
}

function addDOY(image) {
  var doy = ee.Image(ee.Number.parse(image.date().format('D'))).int16().rename('dayOfYear');
  return image.addBands(doy);
}

function fractions(image) {
  var newBands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];
  var imageSelected = image.select(newBands).multiply(10000);
  var GV = [119.0, 475.0, 169.0, 6250.0, 2399.0, 675.0];
  var NPV = [1514.0, 1597.0, 1421.0, 3053.0, 7707.0, 1975.0];
  var Soil = [1799.0, 2479.0, 3158.0, 5437.0, 7707.0, 6646.0];
  var Cloud = [4031.0, 8714.0, 7900.0, 8989.0, 7002.0, 6607.0];
  var sma = imageSelected.unmix([GV, NPV, Soil, Cloud]).max(0).multiply(100).int16();
  sma = image
    .addBands(sma.select('band_0').rename('gv'))
    .addBands(sma.select('band_1').rename('npv'))
    .addBands(sma.select('band_2').rename('soil'))
    .addBands(sma.select('band_3').rename('cloud'));
  var summed = sma.expression('GV + NPV + SOIL + CLOUD', {
    GV: sma.select('gv'), NPV: sma.select('npv'),
    SOIL: sma.select('soil'), CLOUD: sma.select('cloud')
  });
  var shade = summed.subtract(100).abs();
  var gvs = (sma.select(['gv']).divide(summed)).multiply(100);
  var npvSoil = sma.select(['npv']).add(sma.select(['soil'])).add(sma.select('cloud'));
  return sma
    .addBands(shade.rename('shade'))
    .addBands(npvSoil.rename('npvSoil'))
    .addBands(gvs.rename('gvs'));
}

function bitwiseExtract(value, fromBit, toBit) {
  if (toBit === undefined) toBit = fromBit;
  var maskSize = ee.Number(1).add(toBit).subtract(fromBit);
  var mask = ee.Number(1).leftShift(maskSize).subtract(1);
  return value.rightShift(fromBit).bitwiseAnd(mask);
}

function clipBoard_Landsat(image) {
  return image.updateMask(ee.Image().paint(image.geometry().buffer(-3000)).eq(0));
}

// --- LANDSAT ---

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
  return image.select(['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
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
  return image.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).float();
}

function processLS57(img) {
  var optical = corrections_LS57_col2(clipBoard_Landsat(img));
  return addDOY(fractions(addBand_NBR(addNDVI(optical.set({
    'system:time_start': img.get('system:time_start')
  })))));
}
function processLS89(img) {
  var optical = corrections_LS8_col2(clipBoard_Landsat(img));
  return addDOY(fractions(addBand_NBR(addNDVI(optical.set({
    'system:time_start': img.get('system:time_start')
  })))));
}

// --- SENTINEL-2 ---

function processS2(img) {
  var mask = img.select('cs').gte(0.40);
  var optical = img.updateMask(mask).select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).float().divide(10000);
  return addDOY(fractions(addBand_NBR(addNDVI(optical.set({
    'system:time_start': img.get('system:time_start'),
    'system:time_end': img.get('system:time_end')
  })))));
}

// --- HLS S30 (Sentinel-2) ---
function processHLS_S30(img) {
  var fmaskBits = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
  var mask = img.select('Fmask').bitwiseAnd(fmaskBits).eq(0);
  
  var optical = img.updateMask(mask).select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).float();

  // Eliminar pixels negativos/nulos
  var validMask = optical.select('blue').gt(0)
    .and(optical.select('green').gt(0))
    .and(optical.select('red').gt(0))
    .and(optical.select('nir').gt(0))
    .and(optical.select('swir1').gt(0))
    .and(optical.select('swir2').gt(0));

  optical = optical.updateMask(validMask);

  return addDOY(fractions(addBand_NBR(addNDVI(optical.set({
    'system:time_start': img.get('system:time_start'),
    'system:time_end': img.get('system:time_end')
  })))));
}

// --- HLS L30 (Landsat) ---
function processHLS_L30(img) {
  var fmaskBits = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4);
  var mask = img.select('Fmask').bitwiseAnd(fmaskBits).eq(0);
  
  var optical = img.updateMask(mask).select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).float();

  // Eliminar pixels negativos/nulos
  var validMask = optical.select('blue').gt(0)
    .and(optical.select('green').gt(0))
    .and(optical.select('red').gt(0))
    .and(optical.select('nir').gt(0))
    .and(optical.select('swir1').gt(0))
    .and(optical.select('swir2').gt(0));

  optical = optical.updateMask(validMask);

  return addDOY(fractions(addBand_NBR(addNDVI(optical.set({
    'system:time_start': img.get('system:time_start'),
    'system:time_end': img.get('system:time_end')
  })))));
}

// --- MODIS Terra (MOD09A1) ---

function processMOD09(img) {
  var qa = img.select('StateQA');
  var clearMask = bitwiseExtract(qa, 0, 1).eq(0);
  var optical = img.updateMask(clearMask).select(['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06', 'sur_refl_b07'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).float().divide(10000);
  return addDOY(fractions(addBand_NBR(addNDVI(optical.set({
    'system:time_start': img.get('system:time_start')
  })))));
}

// --- MODIS Aqua (MYD09A1) ---

function processMYD09(img) {
  var qa = img.select('StateQA');
  var clearMask = bitwiseExtract(qa, 0, 1).eq(0);
  var optical = img.updateMask(clearMask).select(['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06', 'sur_refl_b07'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).float().divide(10000);
  return addDOY(fractions(addBand_NBR(addNDVI(optical.set({
    'system:time_start': img.get('system:time_start')
  })))));
}

// --- VIIRS (VNP09GA) ---

function processVIIRS(img) {
  // VNP09GA QF1 (bits 0-1): 0 = highest quality, 1 = good, 2 = poor, 3 = not produced.
  // Keep only the two best classes to drop clouds/aerosols from the mosaic.
  var qf1 = img.select('QF1');
  var clearMask = bitwiseExtract(qf1, 0, 1).lte(1);
  var optical = img.updateMask(clearMask).select(['M3', 'M4', 'M5', 'M7', 'M10', 'M11'],
    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).float();
  return addDOY(fractions(addBand_NBR(addNDVI(optical.set({
    'system:time_start': img.get('system:time_start')
  })))));
}

function addNDVI(image) {
  return image.addBands(image.normalizedDifference(['nir', 'red']).rename('ndvi'));
}
