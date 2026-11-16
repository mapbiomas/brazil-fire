/*******************************************************************************
MAPBIOMAS FUEGO - M7_00
Selección y Exportación de Predicciones Nacionales (ft00)

📅 FECHA: 10/2026
🏷️ VERSIÓN: 5.4 (Multi-país + campañas 1985 + periodo por sensor)
*******************************************************************************/

var b64 = require('users/workspaceipam/packages:mapbiomas-toolkit/utils/b64');

// ─── CONFIGURACION DEL USUARIO ──────────────────────────────────────────────

// Inicio de datos por sensor (no por coleccion).
var SENSORS = {
    LANDSAT:   { startYear: 1985 },
    SENTINEL2: { startYear: 2015 }
};

var COUNTRIES = {
    peru: {
        catalog: 'projects/mapbiomas-peru/assets/FIRE/CATALOG_01',
        regions: 'projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1',
        regionProperty: 'region_nam',
        defaultCampaign: 'MONITOR_01',
        campaigns: [
            { id: 'COLLECTION_03', periodicity: 'yearly',  sensor: 'LANDSAT' },
            { id: 'MONITOR_01',    periodicity: 'monthly', sensor: 'SENTINEL2' }
            // { id: 'COLLECTION_DEV', periodicity: 'yearly',  sensor: 'LANDSAT' },
            // { id: 'MONITOR_DEV',    periodicity: 'monthly', sensor: 'SENTINEL2' }
        ]
    },
    chile: {
        catalog: 'projects/mapbiomas-chile/assets/FIRE/CATALOG_01',
        regions: 'projects/mapbiomas-chile/assets/FIRE/AUXILIARY_DATA/regiones_fuego_chile_v1',
        regionProperty: 'region_nam',
        defaultCampaign: 'MONITOR_01',
        campaigns: [
            { id: 'COLLECTION_02', periodicity: 'yearly',  sensor: 'LANDSAT' },
            { id: 'MONITOR_01',    periodicity: 'monthly', sensor: 'SENTINEL2' }
            // { id: 'COLLECTION_DEV', periodicity: 'yearly',  sensor: 'LANDSAT' },
            // { id: 'MONITOR_DEV',    periodicity: 'monthly', sensor: 'SENTINEL2' }
        ]
    },
    bolivia: {
        catalog: 'projects/mapbiomas-bolivia/assets/FIRE/CATALOG_01',
        regions: 'projects/mapbiomas-bolivia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_bolivia_v1',
        regionProperty: 'region_nam',
        defaultCampaign: 'COLLECTION_02',
        campaigns: [
            { id: 'COLLECTION_02', periodicity: 'yearly',  sensor: 'LANDSAT' },
            { id: 'MONITOR_01',    periodicity: 'monthly', sensor: 'SENTINEL2' }
            // { id: 'COLLECTION_DEV', periodicity: 'yearly',  sensor: 'LANDSAT' },
            // { id: 'MONITOR_DEV',    periodicity: 'monthly', sensor: 'SENTINEL2' }
        ]
    },
    colombia: {
        catalog: 'projects/mapbiomas-colombia/assets/FIRE/CATALOG_01',
        regions: 'projects/mapbiomas-colombia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_colombia_v1',
        regionProperty: 'region_nam',
        defaultCampaign: 'MONITOR_01',
        campaigns: [
            { id: 'COLLECTION_02', periodicity: 'yearly',  sensor: 'LANDSAT' },
            { id: 'MONITOR_01',    periodicity: 'monthly', sensor: 'SENTINEL2' }
            // { id: 'COLLECTION_DEV', periodicity: 'yearly',  sensor: 'LANDSAT' },
            // { id: 'MONITOR_DEV',    periodicity: 'monthly', sensor: 'SENTINEL2' }
        ]
    },
    paraguay: {
        catalog: 'projects/mapbiomas-paraguay/assets/FIRE/CATALOG_01',
        regions: 'projects/mapbiomas-paraguay/assets/FIRE/AUXILIARY_DATA/regiones_fuego_paraguay_v1',
        regionProperty: 'region_nam',
        defaultCampaign: 'MONITOR_01',
        campaigns: [
            { id: 'COLLECTION_03', periodicity: 'yearly',  sensor: 'LANDSAT' },
            { id: 'MONITOR_01',    periodicity: 'monthly', sensor: 'SENTINEL2' }
            // { id: 'COLLECTION_DEV', periodicity: 'yearly',  sensor: 'LANDSAT' },
            // { id: 'MONITOR_DEV',    periodicity: 'monthly', sensor: 'SENTINEL2' }
        ]
    },
    brazil: {
        catalog: 'projects/mapbiomas-brazil/assets/FIRE/CATALOG_01',
        regions: ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2')
            .map(function(f){
                return f.set('region_name', f.getString('bioma').cat(' ').cat(f.get('region')));
            }),
        regionProperty: 'region_name',
        defaultCampaign: 'COLLECTION_06',
        campaigns: [
            { id: 'COLLECTION_06', periodicity: 'yearly',  sensor: 'LANDSAT' },
            { id: 'MONITOR_01',    periodicity: 'monthly', sensor: 'SENTINEL2' }
            // { id: 'COLLECTION_DEV', periodicity: 'yearly',  sensor: 'LANDSAT' },
            // { id: 'MONITOR_DEV',    periodicity: 'monthly', sensor: 'SENTINEL2' }
        ]
    },
    venezuela: {
        catalog: 'projects/mapbiomas-venezuela/assets/FIRE/CATALOG_01',
        regions: 'projects/mapbiomas-venezuela/assets/FIRE/AUXILIARY_DATA/regiones_fuego_venezuela_v1',
        regionProperty: 'region_nam',
        defaultCampaign: 'MONITOR_01',
        campaigns: [
            { id: 'COLLECTION_02', periodicity: 'yearly',  sensor: 'LANDSAT' },
            { id: 'MONITOR_01',    periodicity: 'monthly', sensor: 'SENTINEL2' }
            // { id: 'COLLECTION_DEV', periodicity: 'yearly',  sensor: 'LANDSAT' },
            // { id: 'MONITOR_DEV',    periodicity: 'monthly', sensor: 'SENTINEL2' }
        ]
    },
    indonesia: {
        catalog: 'projects/mapbiomas-indonesia/assets/FIRE/CATALOG_01',
        regions: 'projects/mapbiomas-indonesia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_indonesia_v1',
        regionProperty: 'region_nam',
        defaultCampaign: 'MONITOR_01',
        campaigns: [
            { id: 'COLLECTION_03', periodicity: 'yearly',  sensor: 'LANDSAT' },
            { id: 'MONITOR_01',    periodicity: 'monthly', sensor: 'SENTINEL2' }
            // { id: 'COLLECTION_DEV', periodicity: 'yearly',  sensor: 'LANDSAT' },
            // { id: 'MONITOR_DEV',    periodicity: 'monthly', sensor: 'SENTINEL2' }
        ]
    },
    DRC: {
        label: 'DRC',
        catalog: 'projects/mapbiomas-drc/assets/FIRE/CATALOG_01',
        regions: 'projects/mapbiomas-drc/assets/FIRE/AUXILIARY_DATA/regiones_fuego_drc_v1',
        regionProperty: 'region_nam',
        mosaicPrefix: 'image_drc_fire_landsat_minnbr_',
        defaultCampaign: 'COLLECTION_01',
        campaigns: [
            { id: 'COLLECTION_01', periodicity: 'yearly', sensor: 'LANDSAT' }
            // { id: 'COLLECTION_DEV', periodicity: 'yearly', sensor: 'LANDSAT' }
        ]
    }
};
var COUNTRY_ORDER = ['peru', 'chile', 'bolivia', 'colombia', 'paraguay', 'brazil', 'venezuela', 'indonesia', 'DRC'];
var SCALE = 10;
var APP_LANG = 'es';

var L = (function(){
    var d={
        pt:{title:'M7 — Selecao e Export',cfg:'CONFIGURACAO',period:'PERIODO',regions:'REGIOES',confirm:'CONFIRMAR',countries:'Pais',campaigns:'Campanhas',collections:'Colecoes',new_name:'Nome',create_btn:'Criar',placeholder:'ex: propuesta_a',loading:'Carregando...',no_data:'Sem dados.',select_all:'Selecionar todas',clear_all:'Limpar',filter_placeholder:'filtrar predicoes...',target:'Predicao',export_btn:'Exportar',pre_title:'Pre-Confirmacao',pre_body:'Sera criado/atualizado:',pre_warn:'O GEE solicitara confirmacao.',pre_ok:'OK',cancel:'Cancelar',done:'Concluido!',none:'nenhuma',assets:'GESTAO DE ASSETS',sync_btn:'Sincronizar',create_structure:'Criar estrutura',asset_target:'Destino',asset_ready:'Estrutura pronta',asset_missing:'Falta criar estrutura',asset_need_structure:'Crie a estrutura em GESTAO DE ASSETS antes de exportar.',asset_synced:'Sincronizado',asset_created:'Estrutura criada/verificada.',asset_create_warn:'As pastas serao criadas/verificadas no GEE.',view_mode:'Ver',view_class:'Classificacao',view_probability:'Probabilidade',view_doy:'Dia do ano',national_candidate:'Candidato nacional'},
        es:{title:'M7 — Selección y Exportación',cfg:'CONFIGURACIÓN',period:'PERÍODO',regions:'REGIONES',confirm:'CONFIRMAR',countries:'País',campaigns:'Campañas',collections:'Colecciones',new_name:'Nombre',create_btn:'Crear',placeholder:'ej: propuesta_a',loading:'Cargando...',no_data:'Sin datos.',select_all:'Todas',clear_all:'Limpiar',filter_placeholder:'filtrar predicciones...',target:'Predicción',export_btn:'Exportar',pre_title:'Pre-Confirmación',pre_body:'Se creará/actualizará:',pre_warn:'GEE solicitará confirmación de tarea.',pre_ok:'OK',cancel:'Cancelar',done:'¡Completado!',none:'ninguna',assets:'GESTIÓN DE ASSETS',sync_btn:'Sincronizar',create_structure:'Crear estructura',asset_target:'Destino',asset_ready:'Estructura lista',asset_missing:'Falta crear estructura',asset_need_structure:'Crea la estructura en GESTIÓN DE ASSETS antes de exportar.',asset_synced:'Sincronizado',asset_created:'Estructura creada/verificada.',asset_create_warn:'Se crearán/verificarán carpetas en GEE.',view_mode:'Ver',view_class:'Clasificación',view_probability:'Probabilidad',view_doy:'Día del año',national_candidate:'Candidato nacional'},
        en:{title:'M7 — Selection & Export',cfg:'CONFIGURATION',period:'PERIOD',regions:'REGIONS',confirm:'CONFIRM',countries:'Country',campaigns:'Campaigns',collections:'Collections',new_name:'Name',create_btn:'Create',placeholder:'e.g. propuesta_a',loading:'Loading...',no_data:'No data.',select_all:'Select all',clear_all:'Clear',filter_placeholder:'filter predictions...',target:'Prediction',export_btn:'Export',pre_title:'Pre-Confirmation',pre_body:'Will create/update:',pre_warn:'GEE will prompt for confirmation.',pre_ok:'OK',cancel:'Cancel',done:'Done!',none:'none',assets:'ASSET MANAGEMENT',sync_btn:'Sync',create_structure:'Create structure',asset_target:'Target',asset_ready:'Structure ready',asset_missing:'Structure missing',asset_need_structure:'Create the structure in ASSET MANAGEMENT before exporting.',asset_synced:'Synced',asset_created:'Structure created/verified.',asset_create_warn:'Folders will be created/verified in GEE.',view_mode:'View',view_class:'Classification',view_probability:'Probability',view_doy:'Day of year',national_candidate:'National candidate'},
        fr:{title:'M7 — Selection',cfg:'CONFIG',period:'PERIODE',regions:'REGIONS',confirm:'CONFIRMER',countries:'Pays',campaigns:'Campagnes',collections:'Collections',new_name:'Nom',create_btn:'Créer',placeholder:'ex: propuesta_a',loading:'Chargement...',no_data:'Pas de donnees.',select_all:'Toutes',clear_all:'Effacer',filter_placeholder:'filtrer predictions...',target:'Prediction',export_btn:'Exporter',pre_title:'Pre-Confirmation',pre_body:'Va creer:',pre_warn:'GEE demandera confirmation.',pre_ok:'OK',cancel:'Annuler',done:'Termine!',none:'aucune',assets:'GESTION DES ASSETS',sync_btn:'Synchroniser',create_structure:'Créer structure',asset_target:'Cible',asset_ready:'Structure prête',asset_missing:'Structure manquante',asset_need_structure:'Créez la structure dans GESTION DES ASSETS avant exporter.',asset_synced:'Synchronisé',asset_created:'Structure créée/vérifiée.',asset_create_warn:'Les dossiers seront créés/vérifiés dans GEE.',view_mode:'Vue',view_class:'Classification',view_probability:'Probabilité',view_doy:'Jour de l annee',national_candidate:'Candidat national'},
        id:{title:'M7 — Seleksi',cfg:'KONFIG',period:'PERIODE',regions:'WILAYAH',confirm:'KONFIRMASI',countries:'Negara',campaigns:'Kampanye',collections:'Koleksi',new_name:'Nama',create_btn:'Buat',placeholder:'cth: propuesta_a',loading:'Memuat...',no_data:'Tidak ada.',select_all:'Semua',clear_all:'Hapus',filter_placeholder:'filter prediksi...',target:'Prediksi',export_btn:'Ekspor',pre_title:'Pra-Konfirmasi',pre_body:'Akan dibuat:',pre_warn:'GEE akan minta konfirmasi.',pre_ok:'OK',cancel:'Batal',done:'Selesai!',none:'tidak ada',assets:'MANAJEMEN ASET',sync_btn:'Sinkron',create_structure:'Buat struktur',asset_target:'Tujuan',asset_ready:'Struktur siap',asset_missing:'Struktur belum ada',asset_need_structure:'Buat struktur di MANAJEMEN ASET sebelum ekspor.',asset_synced:'Tersinkron',asset_created:'Struktur dibuat/diverifikasi.',asset_create_warn:'Folder akan dibuat/diverifikasi di GEE.',view_mode:'Lihat',view_class:'Klasifikasi',view_probability:'Probabilitas',view_doy:'Hari dalam tahun',national_candidate:'Kandidat nasional'},
    };
    return d[APP_LANG]||d.es;
})();

var SECTION_STYLE = {
    config:  { margin:'4px', padding:'8px', backgroundColor:'#f8f9fa', border:'1px solid #e0e0e0', borderRadius:'6px' },
    period:  { margin:'4px', padding:'8px', backgroundColor:'#f0f4ff', border:'1px solid #c8d6f0', borderRadius:'6px' },
    regions: { margin:'4px', padding:'8px', backgroundColor:'#f8f9fa', border:'1px solid #e0e0e0', borderRadius:'6px' },
    confirm: { margin:'4px', padding:'8px', backgroundColor:'#e8f5e9', border:'1px solid #b8d8ba', borderRadius:'6px' },
    assets:  { margin:'4px', padding:'8px', backgroundColor:'#fff8e1', border:'1px solid #ffcc00', borderRadius:'6px' },
};

var STYLE = {
    label:        { fontSize:'11px', color:'#555', margin:'2px 0' },
    input:        { stretch:'horizontal', fontSize:'12px', margin:'2px 0' },
    btnBlue:      { margin:'2px', padding:'4px 10px', color:'#1a73e8', fontWeight:'bold' },
    btnGreen:     { margin:'2px', padding:'4px 10px', color:'#0f9d58', fontWeight:'bold' },
    btnGray:      { margin:'2px', padding:'4px 10px', color:'#70757a', fontWeight:'bold' },
    statusOk:     { color:'#0f9d58', fontWeight:'bold', fontSize:'11px' },
    statusErr:    { color:'#d32f2f', fontWeight:'bold', fontSize:'11px' },
    prePopup:     { margin:'6px 0', padding:'10px', backgroundColor:'#fff8e1', border:'1px solid #ffcc00', borderRadius:'6px' },
    sectionTitle: { fontSize:'12px', fontWeight:'bold', color:'#333', margin:'0 0 6px 0' },
    card:         { margin:'2px', padding:'4px', backgroundColor:'#fff', border:'1px solid #e0e0e0', borderRadius:'4px' },
    summaryCard:  { margin:'0 0 6px 0', padding:'6px', backgroundColor:'#fff', border:'1px solid #e0e0e0', borderRadius:'4px' },
    checkbox:     { fontSize:'10px', margin:'1px 2px' },
    gridCell:     { fontSize:'10px', margin:'1px 2px' },
};

var PALETTE = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9A6324','#fffac8','#800000'];
var PROB_PALETTE = ['#ffcccc', '#ff6666', '#cc0000', '#660000'];
var DOY_PALETTE = ['#313695', '#4575b4', '#74add1', '#abd9e9', '#e0f3f8', '#fee090', '#fdae61', '#f46d43', '#d73027'];

// ─── ESTADO DE LA APLICACIÓN ───────────────────────────────────────────────

var mapLayers = {};
var availableModels = {};
var regionModelMap = {};
var checkboxStore = {};
var modelPanels = {};
var lastClassifications = {};
var filterText = '';
var regionNames = [];

var currentYear = null;
var currentMonth = null;
var currentPeriod = '';
var collectionName = 'propuesta_a';
var viewMode = 'class'; // 'class' | 'probability' | 'doy'
var showNationalCandidate = true;
var candidateRefreshSuspended = false;
var periodRequestId = 0;
var collectionRequestId = 0;
var classificationRequestId = 0;

var enabledCountries = {};
var enabledCampaigns = {};
var enabledCollections = {};
var countryCheckboxes = {};
var campaignCheckboxes = {};
var collectionCheckboxes = {};
var periodCheckboxes = {};
var allPeriods = [];
var selectedPeriod = null;

var mapWidget = ui.Map();

// ─── REFERENCIAS A COMPONENTES DE LA INTERFAZ ────────────────────────────────

var contentRoot, regionsBox, confirmBox, summaryBox, statusLabel, exportStatus;
var countryBox, campaignBox, collectionBox, dateBox, textboxCollectionName;
var assetStatusBox, assetMessageBox;

// ─── FUNCIONES DE APOYO (HELPERS) ───────────────────────────────────────────

function getActiveCountry(){
    var keys = Object.keys(enabledCountries);
    for(var i=0; i<keys.length; i++){
        if(enabledCountries[keys[i]]) return keys[i];
    }
    return null;
}

function getActiveCampaign(){
    var keys = Object.keys(enabledCampaigns);
    for(var i=0; i<keys.length; i++){
        if(enabledCampaigns[keys[i]]) return keys[i];
    }
    return null;
}

// ─── CONFIGURACION POR PAIS / CAMPANA ────────────────────────────────────────

function countryCampaigns(country){
    return (COUNTRIES[country] && COUNTRIES[country].campaigns) || [];
}

// Resuelve la configuracion operativa de una campana, completando defaults por pais.
function campaignConf(country, campaignId){
    var countryConf = COUNTRIES[country] || {};
    var list = countryCampaigns(country);
    var meta = null;
    for(var i=0; i<list.length; i++){
        if(list[i].id === campaignId){ meta = list[i]; break; }
    }
    if(!meta){
        var fallbackId = countryConf.defaultCampaign || campaignId;
        for(var j=0; j<list.length; j++){
            if(list[j].id === fallbackId){ meta = list[j]; break; }
        }
    }
    if(!meta) meta = list[0] || { id: campaignId, periodicity: 'monthly', sensor: 'SENTINEL2' };
    var sensor = meta.sensor || 'SENTINEL2';
    var sensorConf = SENSORS[sensor] || { startYear: 2025 };
    var catalog = meta.catalog || countryConf.catalog || ('projects/mapbiomas-' + country + '/assets/FIRE/CATALOG_01');
    return {
        id: meta.id,
        periodicity: meta.periodicity === 'yearly' ? 'yearly' : 'monthly',
        sensor: sensor,
        startYear: meta.startYear || sensorConf.startYear,
        catalog: catalog,
        classRoot: meta.classRoot || (catalog + '/' + meta.id + '/LIBRARY_CLASSIFICATIONS/'),
        imagesRoot: meta.imagesRoot || (catalog + '/LIBRARY_IMAGES'),
        regions: meta.regions || countryConf.regions,
        regionProperty: meta.regionProperty || countryConf.regionProperty || 'region_nam',
        mosaicPrefix: meta.mosaicPrefix || countryConf.mosaicPrefix || ('image_' + country + '_fire_' + sensor.toLowerCase() + '_minnbr_')
    };
}

function activeCampaignConf(){
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign) return null;
    return campaignConf(country, campaign);
}

function classificationsRoot(country, campaign){
    return campaignConf(country, campaign).classRoot;
}

function regionsCollection(country, campaign){
    var regions = campaignConf(country, campaign).regions;
    return typeof regions === 'string' ? ee.FeatureCollection(regions) : regions;
}

function regionProperty(country, campaign){
    return campaignConf(country, campaign).regionProperty;
}

function activeRegionProperty(){
    var conf = activeCampaignConf();
    return conf ? conf.regionProperty : 'region_nam';
}

function isYearly(){
    var conf = activeCampaignConf();
    return !!(conf && conf.periodicity === 'yearly');
}

function ensureFolder(root, name){
    var p=name.split('/'), cur=root;
    for(var i=0;i<p.length;i++){cur+=p[i];try{ee.data.getAsset(cur)}catch(e){ee.data.createAsset({type:(i===p.length-1&&/^ft\d\d$/.test(p[i]))?'IMAGE_COLLECTION':'FOLDER'},cur)}cur+='/';}
}

function formatPeriod(y,m){
    if(isYearly() || m===null || typeof m==='undefined') return ''+y;
    return y+'_'+('0'+m).slice(-2);
}

function parsePeriod(p){
    var parts = (''+p).split('_');
    return { year: parseInt(parts[0],10), month: parts.length>1 ? parseInt(parts[1],10) : null };
}

function setCurrentPeriod(p){
    var parsed = parsePeriod(p);
    currentPeriod = p;
    currentYear = parsed.year;
    currentMonth = parsed.month;
}

function showLoading(){if(statusLabel)statusLabel.style().set('shown',true);}
function hideLoading(){if(statusLabel)statusLabel.style().set('shown',false);}

function setExportStatus(state, msg){
    if(!exportStatus)return;
    var colors={loading:{bg:'#fff8e1',icon:'⏳'},success:{bg:'#e8f5e9',icon:'✅'},error:{bg:'#ffeaea',icon:'❌'},info:{bg:'#f0f4ff',icon:'ℹ️'}};
    var c=colors[state]||colors.info;
    exportStatus.style().set('shown',true);
    exportStatus.clear();
    exportStatus.add(ui.Label(c.icon+' '+msg,{fontSize:'11px',padding:'4px 8px',backgroundColor:c.bg,borderRadius:'3px',margin:'2px 0',stretch:'horizontal'}));
}

function ft00Path(){return collectionName+'/ft00';}

// ─── GESTIÓN DE CAPAS EN EL MAPA ──────────────────────────────────────────────

function manageMapLayer(id, obj, vis, name){
    if(mapLayers[id]){
        mapLayers[id].setEeObject(obj);
        mapLayers[id].setVisParams(vis);
        mapLayers[id].setName(name);
    } else {
        mapLayers[id] = ui.Map.Layer(obj, vis, name);
        mapWidget.layers().add(mapLayers[id]);
    }
}

function removeMapLayer(id){
    if(mapLayers[id]){mapWidget.layers().remove(mapLayers[id]);delete mapLayers[id];}
}

function removeClassificationLayers(){
    Object.keys(mapLayers).forEach(function(k){if(k.indexOf('class_')===0)removeMapLayer(k);});
}

function updateRegionLayers(){
    Object.keys(mapLayers).forEach(function(k){
        if(k.indexOf('regions_boundary')===0) removeMapLayer(k);
    });
    var country = getActiveCountry();
    var conf = activeCampaignConf();
    if(!country || !conf) return;
    var fc = regionsCollection(country, getActiveCampaign());
    manageMapLayer('regions_boundary', fc.style({color:'ffffff',fillColor:'00000000',width:1}), {}, 'Regiones '+(COUNTRIES[country].label||country));
    mapWidget.centerObject(fc);
}

function resetRegionState(){
    regionModelMap = {};
    checkboxStore = {};
    filterText = '';
    removeClassificationLayers();
    removeMapLayer('national_candidate');
}

// ─── CREADOR DE CASILLAS DE VERIFICACIÓN (CHECKBOXES) ──────────────────────────

// Binary mask of a region polygon (1 inside, 0 outside).
function regionMaskImage(regionName){
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign || !regionName) return null;
    var regions = regionsCollection(country, campaign);
    return ee.Image(0).paint(regions.filter(ee.Filter.eq(regionProperty(country, campaign), regionName)), 1).eq(1);
}

// Builds the map layer for a regional model according to the active view mode.
// The region mask is applied to the preview so it matches the exported candidate.
function modelViewSpec(model, regionName){
    var img = ee.Image(model.assetId);
    var prob = img.select(0);
    var scar = prob.gt(0);
    var rmask = regionMaskImage(regionName);
    var mask = rmask ? rmask.and(scar) : scar;
    if(viewMode === 'probability'){
        return { img: prob.updateMask(mask), vis: { min: 0, max: 1000, palette: PROB_PALETTE }, label: 'prob' };
    }
    if(viewMode === 'doy'){
        return { img: img.select(1).updateMask(mask), vis: { min: 1, max: 366, palette: DOY_PALETTE }, label: 'doy' };
    }
    return { img: scar.updateMask(mask), vis: { palette: ['#ff0000'] }, label: 'class' };
}

function addModelLayer(regionName, model){
    var spec = modelViewSpec(model, regionName);
    manageMapLayer('class_'+regionName+'_'+model.modelId, spec.img, spec.vis, model.modelId+' | '+regionName+' | '+spec.label);
}

function findModel(regionName, modelId){
    var list = (availableModels[currentPeriod] && availableModels[currentPeriod][regionName]) || [];
    for(var i=0; i<list.length; i++){
        if(list[i].modelId === modelId) return list[i];
    }
    return null;
}

// Re-renders all selected models with the active view mode.
function refreshModelLayers(){
    Object.keys(regionModelMap).forEach(function(regionName){
        var modelId = regionModelMap[regionName];
        if(!modelId) return;
        var model = findModel(regionName, modelId);
        if(model) addModelLayer(regionName, model);
    });
}

// Builds the national candidate from the current region -> model selection.
// Region mask and scar mask (probability > 0) are applied here, so the same
// image feeds both the live preview and the ft00 export.
function composeNationalCandidate(){
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign) return null;
    var prob = ee.Image(0).rename('probability');
    var doy = ee.Image(0).rename('dayOfYear');
    var modelList = [];
    var selectedRegions = [];

    // Deterministic order: alphabetical, last region wins in overlaps.
    Object.keys(regionModelMap).sort().forEach(function(regionName){
        var modelId = regionModelMap[regionName];
        if(!modelId) return;
        var model = findModel(regionName, modelId);
        if(!model) return;
        selectedRegions.push(regionName);
        modelList.push(regionName+':'+modelId);
        var rmask = regionMaskImage(regionName);
        var source = ee.Image(model.assetId);
        prob = prob.where(rmask, source.select(0));
        doy = doy.where(rmask, source.select(1));
    });

    var scarMask = prob.gt(0);
    return {
        image: prob.addBands(doy).updateMask(scarMask),
        regionModels: modelList.join(','),
        regions: selectedRegions
    };
}

function nationalCandidateSpec(candidate){
    var img = candidate.image;
    if(viewMode === 'probability'){
        return { img: img.select('probability').selfMask(), vis: { min: 0, max: 1000, palette: PROB_PALETTE }, label: 'prob' };
    }
    if(viewMode === 'doy'){
        return { img: img.select('dayOfYear').selfMask(), vis: { min: 1, max: 366, palette: DOY_PALETTE }, label: 'doy' };
    }
    return { img: img.select('probability').gt(0).selfMask(), vis: { palette: ['#ff0000'] }, label: 'class' };
}

function refreshNationalCandidateLayer(){
    if(candidateRefreshSuspended) return;
    if(!showNationalCandidate){
        removeMapLayer('national_candidate');
        return;
    }
    var candidate = composeNationalCandidate();
    if(!candidate || candidate.regions.length === 0){
        removeMapLayer('national_candidate');
        return;
    }
    var spec = nationalCandidateSpec(candidate);
    manageMapLayer('national_candidate', spec.img, spec.vis, 'Candidato nacional '+currentPeriod+' | '+spec.label);
}

function createCheckboxForModel(regionName, model, containerPanel){
    if(!checkboxStore[regionName])checkboxStore[regionName]={};
    var sel = (regionModelMap[regionName] === model.modelId);
    var cb = ui.Checkbox({label:model.modelId, value:sel, style:STYLE.checkbox});
    cb.onChange(function(v){
        var key = regionName+'_'+model.modelId;
        if(v){
            regionModelMap[regionName] = model.modelId;
            Object.keys(checkboxStore[regionName]).forEach(function(k){if(k!==model.modelId)checkboxStore[regionName][k].setValue(false);});
            addModelLayer(regionName, model);
        } else {
            if(regionModelMap[regionName]===model.modelId)regionModelMap[regionName]=null;
            removeMapLayer('class_'+key);
        }
        buildConfirmPanel();
        refreshNationalCandidateLayer();
    });
    checkboxStore[regionName][model.modelId] = cb;
    containerPanel.add(cb);
    if(sel) addModelLayer(regionName, model);
}

// ─── CARGA DEL MOSAICO OFICIAL MINNBR (SENTINEL-2 MENSUAL / LANDSAT ANUAL) ────

function loadMosaic(y,m){
    var country = getActiveCountry();
    var conf = activeCampaignConf();
    if(!country || !conf) return;
    try {
        var dk = formatPeriod(y,m);
        var periodFolder = conf.periodicity === 'yearly' ? 'YEARLY' : 'MONTHLY';
        var bs = conf.imagesRoot+'/'+conf.sensor+'/'+periodFolder+'/MINNBR';
        var prefix = conf.mosaicPrefix;
        var img = ee.Image().select();
        ['blue','green','red','nir','swir1','swir2'].forEach(function(b){
            try {
                var bi = ee.ImageCollection(bs+'/'+b.toLowerCase()).filter(ee.Filter.eq('system:index',prefix+b.toLowerCase()+'_'+dk)).mosaic();
                img = img.addBands(ee.Image(ee.Algorithms.If(bi.bandNames().size().gt(0),bi,ee.Image(0).rename(b.toLowerCase()).updateMask(0))).select([0],[b.toLowerCase()]),null,true);
            } catch(e) {
                img = img.addBands(ee.Image(0).rename(b.toLowerCase()).updateMask(0),null,true);
            }
        });

        manageMapLayer('mosaic_minnbr', img, {
            bands: ['swir1', 'nir', 'red'],
            min: 5,
            max: 48,
            gamma: 1.1
        }, 'Mosaico '+conf.sensor+' MinNBR ' + dk);
    } catch (e) {
        print('ℹ️ Mosaico MinNBR no disponible para el período:', y, m);
    }
}

// ─── CARGA DE CLASIFICACIONES ────────────────────────────────────────────────

function loadClassifications(y,m,cb){
    var requestId = ++classificationRequestId;
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign){cb({});return;}
    var root = classificationsRoot(country, campaign);
    var dk = formatPeriod(y,m);
    ee.data.listAssets(root+'REGIONAL',{},function(cols){
        if(requestId !== classificationRequestId) return;
        if(!cols||!cols.assets){cb({});return}
        var dirs=cols.assets.filter(function(c){return c.type==='IMAGE_COLLECTION'}), r={}, p=dirs.length;
        if(dirs.length===0){cb({});return}
        dirs.forEach(function(c,idx){
            var modelId=c.id.split('/').pop();
            ee.data.listAssets(c.id,{},function(imgs){
                if(requestId !== classificationRequestId) return;
                if(imgs&&imgs.assets)imgs.assets.forEach(function(img){
                    if(img.type!=='IMAGE')return;
                    var name=img.id.split('/').pop(), parts=name.split('_');
                    var rp=null;for(var i=0;i<regionNames.length;i++){if(name.indexOf(regionNames[i])!==-1){rp=regionNames[i];break}}if(!rp)return;
                    var ip=null;for(var j=parts.length-1;j>=0;j--){if(/^\d{4}$/.test(parts[j])){ip=parts[j]+(j+1<parts.length&&/^\d{2}$/.test(parts[j+1])?'_'+parts[j+1]:'');break}}if(ip!==dk)return;
                    if(!r[rp])r[rp]=[];
                    if(!r[rp].some(function(x){return x.modelId===modelId}))r[rp].push({modelId:modelId,assetId:img.id,color:PALETTE[idx%15]});
                });
                p--;if(p===0)cb(r);
            });
        });
        if(p===0)cb(r);
    });
}

function loadExisting(cb){
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign){cb([]);return;}
    var root = classificationsRoot(country, campaign);
    try{
        ee.data.listAssets(root+'FILTERED/',{},function(r, err){
            if(err||!r||!r.assets){cb([]);return}
            cb(r.assets.filter(function(a){return a.type==='FOLDER'}).map(function(a){return a.id.split('/').pop()}).sort());
        });
    }catch(e){
        cb([]);
    }
}

// ─── CONTEO DE PERÍODOS ──────────────────────────────────────────────────────

function loadPeriodCounts(cb){
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign){cb({});return;}
    var root = classificationsRoot(country, campaign);
    ee.data.listAssets(root+'REGIONAL',{},function(cols){
        var counts={};
        if(!cols||!cols.assets){cb(counts);return}
        var dirs=cols.assets.filter(function(c){return c.type==='IMAGE_COLLECTION'});
        if(dirs.length===0){cb(counts);return}
        var p=dirs.length;
        dirs.forEach(function(c){
            ee.data.listAssets(c.id,{},function(imgs){
                if(imgs&&imgs.assets)imgs.assets.forEach(function(img){
                    if(img.type!=='IMAGE')return;
                    var name=img.id.split('/').pop(),parts=name.split('_'),period=null;
                    for(var j=parts.length-1;j>=0;j--){if(/^\d{4}$/.test(parts[j])){period=parts[j]+(j+1<parts.length&&/^\d{2}$/.test(parts[j+1])?'_'+parts[j+1]:'');break}}
                    if(!period)return;
                    if(!counts[period])counts[period]={total:0,regions:{}};
                    counts[period].total++;
                    var rp=null;for(var i=0;i<regionNames.length;i++){if(name.indexOf(regionNames[i])!==-1){rp=regionNames[i];break}}
                    if(rp)counts[period].regions[rp]=true;
                });
                p--;if(p===0)cb(counts);
            });
        });
    });
}

// ─── CARGA DE DATOS DE PERÍODO ──────────────────────────────────────────────

function loadPeriodData(cb){
    if(!dateBox) return;
    var requestId = ++periodRequestId;
    dateBox.clear();
    dateBox.add(ui.Label(L.loading, {fontSize:'10px',color:'#888'}));
    
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign){
        dateBox.clear();
        renderDateGrid([]);
        if(cb) cb();
        return;
    }
    
    var conf = campaignConf(country, campaign);
    var root = conf.classRoot;
    
    try{
    ee.data.listAssets(root+'FILTERED/'+collectionName+'/ft00/', {}, function(r, err){
        if(requestId !== periodRequestId) return;
        var existingPeriods = [];
        if(!err && r && r.assets){
            existingPeriods = r.assets.filter(function(a){return a.type==='IMAGE'}).map(function(a){return a.id.split('/').pop()}).sort().reverse();
        }
        
        loadPeriodCounts(function(counts){
            if(requestId !== periodRequestId) return;
            var periodKeys = [];
            if(conf.periodicity === 'yearly'){
                var yearSet = {};
                Object.keys(counts).forEach(function(k){
                    if(/^\d{4}$/.test(k) && parseInt(k,10) >= conf.startYear) yearSet[k] = true;
                });
                existingPeriods.forEach(function(k){
                    if(/^\d{4}$/.test(k) && parseInt(k,10) >= conf.startYear) yearSet[k] = true;
                });
                periodKeys = Object.keys(yearSet).sort().reverse();
            } else {
                var today = new Date();
                var my = today.getFullYear(), mm = today.getMonth();
                if(mm===0){mm=12;my--}
                for(var y=my; y>=conf.startYear; y--){
                    var me = (y===my) ? mm : 12;
                    for(var m=me; m>=1; m--){
                        periodKeys.push(y+'_'+('0'+m).slice(-2));
                    }
                }
            }
            
            var periods = periodKeys.map(function(p){
                var c = counts[p] || {total:0, regions:{}};
                return {
                    period: p,
                    regions: Object.keys(c.regions).length,
                    predictions: c.total,
                    exported: existingPeriods.indexOf(p) !== -1
                };
            });
            
            if(periods.length === 0){
                selectedPeriod = null;
                currentYear = null;
                currentMonth = null;
                allPeriods = [];
                renderDateGrid([]);
                if(cb) cb();
                return;
            }
            
            var stillValid = false;
            for(var i=0; i<periods.length; i++){
                if(periods[i].period === selectedPeriod){ stillValid = true; break; }
            }
            if(!stillValid){
                var defIdx = 0;
                for(var j=0; j<periods.length; j++){
                    if(!periods[j].exported && periods[j].predictions > 1){
                        defIdx = j; break;
                    }
                }
                selectedPeriod = periods[defIdx].period;
            }
            setCurrentPeriod(selectedPeriod);
            
            allPeriods = periods;
            renderDateGrid(periods);
            if(cb) cb();
        });
    });
    }catch(e){
        if(requestId !== periodRequestId) return;
        dateBox.clear();
        renderDateGrid([]);
        if(cb) cb();
    }
}

function renderDateGrid(periods){
    if(!dateBox) return;
    dateBox.clear();
    periodCheckboxes = {};
    var activeConf = activeCampaignConf();
    var periodSuffix = activeConf ? (' — '+activeConf.id+' ('+activeConf.periodicity+')') : '';
    dateBox.add(ui.Label(L.period+periodSuffix, STYLE.label));
    
    if(!periods || periods.length === 0){
        dateBox.add(ui.Label(L.no_data, {fontSize:'10px',color:'#aaa',margin:'2px'}));
    } else {
        periods.forEach(function(p){
            var label = p.period + ' (' + p.regions + ' reg. | ' + p.predictions + ' pred.)';
            if(p.exported) label += ' ✅';
            var cb = ui.Checkbox({
                label: label,
                value: selectedPeriod === p.period,
                style: STYLE.gridCell
            });
            periodCheckboxes[p.period] = cb;
            cb.onChange(function(v){
                if(!v) return;
                selectedPeriod = p.period;
                Object.keys(periodCheckboxes).forEach(function(k){
                    if(k !== p.period) periodCheckboxes[k].setValue(false);
                });
                setCurrentPeriod(p.period);
                resetRegionState();
                showLoading();
                loadMosaic(currentYear, currentMonth);
                var country = getActiveCountry();
                if(country){
                    mapWidget.centerObject(regionsCollection(country, getActiveCampaign()));
                }
                loadPeriodAndRepopulate();
            });
            dateBox.add(cb);
        });
    }
}

// ─── FILTRO DE BÚSQUEDA ────────────────────────────────────────────────────────

function applyFilter(){
    var data = lastClassifications||{};
    regionNames.forEach(function(rn){
        var models = (data[rn]||[]).sort(function(a,b){return a.modelId.localeCompare(b.modelId)});
        var panel = modelPanels[rn];
        if(!panel)return;
        panel.clear();
        var shown = models.filter(function(mm){return !filterText||mm.modelId.toLowerCase().indexOf(filterText)!==-1;});
        if(shown.length===0){
            panel.add(ui.Label(models.length===0?'(sin predicción)':'('+models.length+' oculto'+(models.length>1?'s':'')+')', {fontSize:'9px',color:'#aaa',margin:'1px 0'}));
        } else {
            shown.forEach(function(m){createCheckboxForModel(rn, m, panel);});
        }
    });
    buildConfirmPanel();
}

// ─── REGIONES ─────────────────────────────────────────────────────────────────

function buildRegionsLayout(callback){
    regionsBox.clear();
    regionsBox.add(ui.Label(L.loading+' '+currentPeriod, {fontSize:'10px',color:'#888'}));
    summaryBox.clear();
    summaryBox.add(ui.Label('Período: '+currentPeriod+' | Colección: '+ft00Path(), {fontSize:'9px',fontFamily:'monospace',color:'#1a73e8',margin:'1px 0'}));

    loadClassifications(currentYear, currentMonth, function(data){
        resetRegionState();
        modelPanels = {};
        regionsBox.clear();

        var headerRow = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{stretch:'horizontal',margin:'0 0 4px 0'}});
        var txtFilter = ui.Textbox({placeholder:L.filter_placeholder, style:{stretch:'horizontal',fontSize:'10px'}});
        txtFilter.onChange(function(v){filterText=v.toLowerCase();applyFilter();});
        headerRow.add(txtFilter);
        headerRow.add(ui.Button({label:L.select_all, style:STYLE.btnBlue, onClick:function(){
            candidateRefreshSuspended = true;
            regionNames.forEach(function(r){var d=lastClassifications||{};if(d[r]&&d[r].length>0){var f=d[r][0].modelId;regionModelMap[r]=f;if(checkboxStore[r])Object.keys(checkboxStore[r]).forEach(function(k){checkboxStore[r][k].setValue(k===f)});}});
            candidateRefreshSuspended = false;
            buildConfirmPanel();
            refreshNationalCandidateLayer();
        }}));
        headerRow.add(ui.Button({label:L.clear_all, style:STYLE.btnGray, onClick:function(){
            candidateRefreshSuspended = true;
            Object.keys(checkboxStore).forEach(function(r){if(checkboxStore[r])Object.keys(checkboxStore[r]).forEach(function(k){checkboxStore[r][k].setValue(false)});});
            regionModelMap={};
            candidateRefreshSuspended = false;
            buildConfirmPanel();
            refreshNationalCandidateLayer();
        }}));
        regionsBox.add(headerRow);

        var viewRow = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{stretch:'horizontal',margin:'0 0 4px 0'}});
        viewRow.add(ui.Label(L.view_mode+':', STYLE.label));
        var viewCheckboxes = {};
        [['class', L.view_class], ['probability', L.view_probability], ['doy', L.view_doy]].forEach(function(opt){
            var vcb = ui.Checkbox({label:opt[1], value:viewMode===opt[0], style:STYLE.gridCell});
            vcb.onChange(function(v){
                if(!v) return;
                viewMode = opt[0];
                Object.keys(viewCheckboxes).forEach(function(k){
                    if(k!==opt[0]) viewCheckboxes[k].setValue(false);
                });
                refreshModelLayers();
                refreshNationalCandidateLayer();
            });
            viewCheckboxes[opt[0]] = vcb;
            viewRow.add(vcb);
        });
        var natCb = ui.Checkbox({label:L.national_candidate, value:showNationalCandidate, style:STYLE.gridCell});
        natCb.onChange(function(v){
            showNationalCandidate = v;
            refreshNationalCandidateLayer();
        });
        viewRow.add(ui.Label('|', {fontSize:'11px',color:'#aaa',margin:'2px 4px'}));
        viewRow.add(natCb);
        regionsBox.add(viewRow);

        var leftColumn = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:{stretch:'horizontal'}});
        var rightColumn = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:{stretch:'horizontal'}});
        var mid = Math.ceil(regionNames.length/2);

        regionNames.forEach(function(rn, idx){
            var regionCard = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:STYLE.card});
            regionCard.add(ui.Label(rn, {fontSize:'11px',fontWeight:'bold',color:'#1a73e8',margin:'1px 0'}));

            var checkboxesPanel = ui.Panel({layout:ui.Panel.Layout.flow('vertical')});
            modelPanels[rn] = checkboxesPanel;
            regionCard.add(checkboxesPanel);
            (idx<mid ? leftColumn : rightColumn).add(regionCard);
        });

        regionsBox.add(ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{stretch:'horizontal'}, widgets:[leftColumn,rightColumn]}));

        hideLoading();
        callback(data);
    });
}

function repopulateRegionCheckboxes(data){
    availableModels = {};
    availableModels[currentPeriod] = data;
    lastClassifications = data;
    resetRegionState();

    regionNames.forEach(function(rn){
        var models = (data[rn]||[]).sort(function(a,b){return a.modelId.localeCompare(b.modelId)});
        var panel = modelPanels[rn];
        if(!panel)return;
        panel.clear();
        if(models.length===0){
            panel.add(ui.Label('(sin predicción)', {fontSize:'9px',color:'#aaa',margin:'1px 0'}));
        } else {
            if(!regionModelMap[rn])regionModelMap[rn]=models[0].modelId;
            models.forEach(function(m){createCheckboxForModel(rn, m, panel);});
        }
    });

    updateRegionSummary(data);
    buildConfirmPanel();
    refreshNationalCandidateLayer();
    hideLoading();
}

function loadPeriodAndRepopulate(){
    loadClassifications(currentYear, currentMonth, function(data){
        if(!regionNames||regionNames.length===0){buildRegionsLayout(function(data2){repopulateRegionCheckboxes(data2);});}
        else{repopulateRegionCheckboxes(data);}
    });
}

// Reloads periods and rebuilds the whole regions layout for the active country/campaign.
// Used when country or campaign changes, so no stale month/year or region list survives.
function reloadPeriodAndRegions(){
    loadPeriodData(function(){
        if(!selectedPeriod || !currentYear){
            regionsBox.clear();
            summaryBox.clear();
            return;
        }
        loadMosaic(currentYear, currentMonth);
        var country = getActiveCountry();
        if(country) mapWidget.centerObject(regionsCollection(country, getActiveCampaign()));
        buildRegionsLayout(function(data2){ repopulateRegionCheckboxes(data2); });
    });
}

function updateRegionSummary(data){
    summaryBox.clear();
    summaryBox.add(ui.Label('Período: '+currentPeriod+' | Colección: '+ft00Path(), {fontSize:'9px',fontFamily:'monospace',color:'#1a73e8',margin:'1px 0'}));
    var withData = regionNames.filter(function(r){return data[r]&&data[r].length>0;}).length;
    var totalPreds = 0; regionNames.forEach(function(r){if(data[r])totalPreds+=data[r].length;});
    summaryBox.add(ui.Label('Regiones: '+withData+'/'+regionNames.length+' con predicciones | Predicciones totales: '+totalPreds, {fontSize:'10px',color:'#333',margin:'1px 0'}));
    var selected = regionNames.filter(function(r){return!!regionModelMap[r];}).length;
    summaryBox.add(ui.Label('Predicciones seleccionadas: '+selected+'/'+withData+' regiones', {fontSize:'9px',color:selected===withData?'#0f9d58':'#e37400',margin:'1px 0'}));
}

// ─── GESTION DE ASSETS ───────────────────────────────────────────────────────

function setAssetMessage(state, msg){
    if(!assetMessageBox) return;
    var colors = {
        info:    { bg:'#f0f4ff', icon:'ℹ️' },
        success: { bg:'#e8f5e9', icon:'✅' },
        error:   { bg:'#ffeaea', icon:'❌' },
        loading: { bg:'#fff8e1', icon:'⏳' }
    };
    var c = colors[state] || colors.info;
    assetMessageBox.clear();
    assetMessageBox.add(ui.Label(c.icon+' '+msg, {fontSize:'10px',padding:'4px 6px',backgroundColor:c.bg,borderRadius:'3px',margin:'2px 0',stretch:'horizontal'}));
}

function currentCollectionName(){
    var typed = textboxCollectionName ? textboxCollectionName.getValue() : collectionName;
    return (typed || '').trim() || collectionName;
}

function assetTargetPath(){
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign) return null;
    return campaignConf(country, campaign).classRoot + 'FILTERED/' + currentCollectionName() + '/ft00';
}

function buildExportFlag(country, campaign, periodName, conf){
    var countryTag = (COUNTRIES[country].label || country).toUpperCase();
    var periodicityTag = conf.periodicity === 'yearly' ? 'ANNUAL' : 'MONTHLY';
    return ('M7_ft00_'+countryTag+'_'+campaign+'_'+currentCollectionName()+'_'+periodName+'_'+periodicityTag).replace(/[^a-zA-Z0-9_]/g,'_').substring(0,80);
}

function structureReady(){
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign) return false;
    var stageRoot = classificationsRoot(country, campaign) + 'FILTERED/' + ft00Path();
    try{ ee.data.getAsset(stageRoot); return true; }catch(e){ return false; }
}

// Read-only check: shows whether the selected output structure already exists.
function refreshAssetStatus(){
    if(!assetStatusBox) return;
    assetStatusBox.clear();
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign){
        assetStatusBox.add(ui.Label(L.no_data, {fontSize:'10px',color:'#aaa',margin:'1px 0'}));
        return;
    }
    var target = assetTargetPath();
    var exists = false;
    try{ ee.data.getAsset(target); exists = true; }catch(e){}
    assetStatusBox.add(ui.Label(L.asset_target+': '+target, {fontSize:'9px',fontFamily:'monospace',color:'#1a73e8',margin:'1px 0'}));
    assetStatusBox.add(ui.Label(exists?('✅ '+L.asset_ready):('⚠ '+L.asset_missing), exists?STYLE.statusOk:STYLE.statusErr));
}

// Read-only refresh of collections, periods and structure status.
function syncAssets(showMessage){
    collectionName = currentCollectionName();
    refreshAssetStatus();
    refreshCollectionOptions(function(){
        loadPeriodData();
        if(showMessage !== false) setAssetMessage('info', L.asset_synced);
    }, true);
}

function showCreateStructurePopup(){
    if(!assetMessageBox) return;
    var target = assetTargetPath();
    if(!target){ setAssetMessage('error', L.no_data); return; }
    assetMessageBox.clear();
    var box = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:STYLE.prePopup});
    box.add(ui.Label('⚠ '+L.create_structure, {fontSize:'12px',fontWeight:'bold',color:'#cc8800',margin:'2px'}));
    box.add(ui.Label(L.pre_body, {fontSize:'10px',color:'#333',margin:'2px'}));
    box.add(ui.Label(target, {fontSize:'10px',fontWeight:'bold',fontFamily:'monospace',color:'#1a73e8',margin:'2px',padding:'4px',backgroundColor:'#fff'}));
    box.add(ui.Label(L.asset_create_warn, {fontSize:'9px',color:'#888',margin:'4px 2px'}));
    var buttonRow = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{stretch:'horizontal',margin:'4px 0 0 0'}});
    buttonRow.add(ui.Button({label:L.cancel, style:STYLE.btnGray, onClick:function(){ assetMessageBox.clear(); refreshAssetStatus(); }}));
    buttonRow.add(ui.Button({label:L.pre_ok, style:STYLE.btnGreen, onClick:function(){ createStructure(); }}));
    box.add(buttonRow);
    assetMessageBox.add(box);
}

// Write command: creates FILTERED/{collection}/ft00 for the active country/campaign.
function createStructure(){
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    var name = currentCollectionName();
    if(!country || !campaign || !assetTargetPath()){ setAssetMessage('error', L.no_data); return; }
    collectionName = name;
    var root = classificationsRoot(country, campaign);
    setAssetMessage('loading', L.loading);
    try{
        ensureFolder(root, 'FILTERED/' + ft00Path());
        setAssetMessage('success', L.asset_created);
    }catch(e){
        setAssetMessage('error', 'Error: ' + (e && e.message ? e.message : e));
    }
    refreshAssetStatus();
    syncAssets(false);
}

function buildAssetCommandsSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.assets});
    section.add(ui.Label(L.assets, STYLE.sectionTitle));

    assetStatusBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:STYLE.summaryCard});
    section.add(assetStatusBox);

    var buttonRow = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{stretch:'horizontal'}});
    buttonRow.add(ui.Button({label:L.sync_btn, style:STYLE.btnBlue, onClick:function(){ syncAssets(true); }}));
    buttonRow.add(ui.Button({label:L.create_structure, style:STYLE.btnGreen, onClick:function(){ showCreateStructurePopup(); }}));
    section.add(buttonRow);

    assetMessageBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical')});
    section.add(assetMessageBox);

    refreshAssetStatus();
    return section;
}


function rebuildCampaignOptions(country, cb){
    if(!campaignBox) return;
    campaignBox.clear();
    campaignCheckboxes = {};
    enabledCampaigns = {};

    var list = countryCampaigns(country);
    if(list.length === 0){
        campaignBox.add(ui.Label('(sin campanas)', {fontSize:'10px',color:'#aaa',margin:'2px'}));
        if(cb) cb();
        return;
    }

    var defaultId = COUNTRIES[country].defaultCampaign;
    var defaultIdx = 0;
    for(var i=0; i<list.length; i++){
        if(list[i].id === defaultId){ defaultIdx = i; break; }
    }

    list.forEach(function(meta, idx){
        var cbx = ui.Checkbox({label:meta.id, value:idx===defaultIdx, style:STYLE.gridCell});
        enabledCampaigns[meta.id] = (idx===defaultIdx);
        cbx.onChange(function(v){
            if(!v){
                if(enabledCampaigns[meta.id]) cbx.setValue(true);
                return;
            }
            Object.keys(enabledCampaigns).forEach(function(k){enabledCampaigns[k]=false;});
            enabledCampaigns[meta.id] = true;
            Object.keys(campaignCheckboxes).forEach(function(k){
                if(k!==meta.id) campaignCheckboxes[k].setValue(false);
            });
            selectedPeriod = null;
            var countryName = getActiveCountry();
            regionNames = regionsCollection(countryName, meta.id).aggregate_array(regionProperty(countryName, meta.id)).distinct().getInfo().sort();
            resetRegionState();
            refreshCollectionOptions(function(){ reloadPeriodAndRegions(); });
        });
        campaignCheckboxes[meta.id] = cbx;
        campaignBox.add(cbx);
    });

    if(cb) cb();
}

// Reconstruye los checkboxes de colecciones para el pais/campana activos.
// keepTypedName=true preserves a name typed by the user even if it does not exist yet.
function refreshCollectionOptions(cb, keepTypedName){
    if(!collectionBox || !textboxCollectionName) return;
    var requestId = ++collectionRequestId;
    loadExisting(function(names){
        if(requestId !== collectionRequestId) return;
        collectionCheckboxes = {};
        enabledCollections = {};
        collectionBox.clear();

        if(names.length === 0){
            collectionBox.add(ui.Label('(ninguna)', {fontSize:'10px',color:'#aaa',margin:'2px'}));
            if(!keepTypedName){
                collectionName = 'propuesta_a';
                textboxCollectionName.setValue(collectionName);
            }
            refreshAssetStatus();
            if(cb) cb();
            return;
        }

        var chosen = names.indexOf(collectionName) !== -1 ? collectionName : (keepTypedName ? null : names[0]);
        names.forEach(function(n){
            var cbx = ui.Checkbox({label:n, value:n===chosen, style:STYLE.gridCell});
            enabledCollections[n] = (n===chosen);
            cbx.onChange(function(v){
                if(!v) return;
                Object.keys(enabledCollections).forEach(function(k){enabledCollections[k]=false;});
                enabledCollections[n] = true;
                Object.keys(collectionCheckboxes).forEach(function(k){
                    if(k!==n) collectionCheckboxes[k].setValue(false);
                });
                collectionName = n;
                textboxCollectionName.setValue(n);
                loadPeriodData();
            });
            collectionCheckboxes[n] = cbx;
            collectionBox.add(cbx);
        });
        if(chosen){
            collectionName = chosen;
            textboxCollectionName.setValue(collectionName);
        }
        refreshAssetStatus();
        if(cb) cb();
    });
}

// ─── SECCIÓN DE CONFIGURACIÓN ─────────────────────────────────────────────────

function buildConfigSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.config});
    section.add(ui.Label(L.cfg, STYLE.sectionTitle));

    section.add(ui.Label(L.countries, STYLE.label));
    countryBox = ui.Panel({layout:ui.Panel.Layout.flow('horizontal', true)});
    COUNTRY_ORDER.forEach(function(c, idx){
        var cb = ui.Checkbox({label:(COUNTRIES[c].label||c), value:idx===0, style:STYLE.gridCell});
        enabledCountries[c] = idx===0;
        cb.onChange(function(v){
            if(!v){
                if(enabledCountries[c]) cb.setValue(true);
                return;
            }
            Object.keys(enabledCountries).forEach(function(k){enabledCountries[k]=false;});
            enabledCountries[c] = true;
            Object.keys(countryCheckboxes).forEach(function(k){
                if(k!==c) countryCheckboxes[k].setValue(false);
            });
            selectedPeriod = null;
            rebuildCampaignOptions(c, null);
            updateRegionLayers();
            regionNames = regionsCollection(c, getActiveCampaign()).aggregate_array(regionProperty(c, getActiveCampaign())).distinct().getInfo().sort();
            resetRegionState();
            refreshCollectionOptions(function(){ reloadPeriodAndRegions(); });
        });
        countryCheckboxes[c] = cb;
        countryBox.add(cb);
    });
    section.add(countryBox);

    section.add(ui.Label(L.campaigns, STYLE.label));
    campaignBox = ui.Panel({layout:ui.Panel.Layout.flow('horizontal', true)});
    section.add(campaignBox);
    rebuildCampaignOptions(getActiveCountry(), null);

    section.add(ui.Label(L.collections, STYLE.label));
    collectionBox = ui.Panel({layout:ui.Panel.Layout.flow('horizontal')});
    section.add(collectionBox);

    var createRow = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{margin:'4px 0',stretch:'horizontal'}});
    textboxCollectionName = ui.Textbox({placeholder:L.placeholder, value:collectionName, style:STYLE.input});
    textboxCollectionName.onChange(function(v){collectionName=v; refreshAssetStatus();});
    createRow.add(textboxCollectionName);
    section.add(createRow);

    return section;
}

// ─── SECCIÓN DE PERÍODO ───────────────────────────────────────────────────────

function buildPeriodSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.period});
    dateBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical')});
    section.add(dateBox);
    return section;
}

// ─── SECCIÓN DE REGIONES ──────────────────────────────────────────────────────

function buildRegionsSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.regions});
    section.add(ui.Label(L.regions, STYLE.sectionTitle));

    summaryBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:STYLE.summaryCard});
    section.add(summaryBox);

    regionsBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical')});
    section.add(regionsBox);
    return section;
}

// ─── SECCIÓN DE CONFIRMACIÓN ──────────────────────────────────────────────────

function buildConfirmSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.confirm});
    section.add(ui.Label(L.confirm, STYLE.sectionTitle));

    confirmBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical')});
    section.add(confirmBox);
    buildConfirmPanel();
    return section;
}

function buildConfirmPanel(){
    if(!confirmBox)return;
    collectionName = currentCollectionName();
    confirmBox.clear();
    var fn = ft00Path();
    confirmBox.add(ui.Label(L.target+': '+fn+' / '+currentPeriod, {fontSize:'11px',fontFamily:'monospace',color:'#1a73e8',margin:'2px',padding:'4px',backgroundColor:'#fff',borderRadius:'3px'}));
    var names = Object.keys(regionModelMap).sort(), hasAll = true;
    names.forEach(function(r){var m=regionModelMap[r];confirmBox.add(ui.Label('  '+r+': '+(m?m:L.none), m?STYLE.statusOk:STYLE.statusErr));if(!m)hasAll=false;});
    if(names.length===0)confirmBox.add(ui.Label('Ninguna región configurada.', STYLE.statusErr));
    confirmBox.add(ui.Button({label:L.export_btn, style:STYLE.btnGreen, disabled:!hasAll||names.length===0, onClick:function(){showPrePopup();}}));
}

// ─── VENTANA POPUP PREVIA A EXPORTAR ──────────────────────────────────────────

function showPrePopup(){
    collectionName = currentCollectionName();
    if(!structureReady()){
        confirmBox.clear();
        confirmBox.add(ui.Label('⚠ '+L.asset_need_structure, STYLE.statusErr));
        confirmBox.add(ui.Button({label:L.cancel, style:STYLE.btnGray, onClick:function(){ buildConfirmPanel(); }}));
        refreshAssetStatus();
        return;
    }
    confirmBox.clear();
    var fn = ft00Path();
    var box = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:STYLE.prePopup});
    box.add(ui.Label('⚠ '+L.pre_title, {fontSize:'13px',fontWeight:'bold',color:'#cc8800',margin:'2px'}));
    box.add(ui.Label(L.pre_body, {fontSize:'11px',color:'#333',margin:'2px'}));
    box.add(ui.Label('FILTERED/'+fn+'/'+currentPeriod, {fontSize:'11px',fontWeight:'bold',fontFamily:'monospace',color:'#1a73e8',margin:'2px',padding:'4px',backgroundColor:'#fff'}));
    var popupFlag = buildExportFlag(getActiveCountry(), getActiveCampaign(), currentPeriod, activeCampaignConf());
    box.add(ui.Label('Task: '+popupFlag, {fontSize:'9px',fontFamily:'monospace',color:'#555',margin:'2px'}));
    box.add(ui.Label(L.pre_warn, {fontSize:'10px',color:'#888',margin:'4px 2px'}));
    var buttonRow = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{stretch:'horizontal',margin:'4px 0 0 0'}});
    buttonRow.add(ui.Button({label:L.cancel, style:STYLE.btnGray, onClick:function(){buildConfirmPanel();}}));
    buttonRow.add(ui.Button({label:L.pre_ok, style:STYLE.btnGreen, onClick:function(){doExport();}}));
    box.add(buttonRow);
    confirmBox.add(box);
}

// ─── PROCESO DE EXPORTACIÓN ───────────────────────────────────────────────────

function doExport(){
    confirmBox.clear();
    confirmBox.add(ui.Label(L.loading, {fontSize:'11px',color:'#1a73e8',margin:'4px'}));
    showLoading();
    
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign){
        confirmBox.add(ui.Label('Error: país o campaña no seleccionados.', STYLE.statusErr));
        return;
    }
    
    collectionName = currentCollectionName();
    var root = classificationsRoot(country, campaign);
    var conf = campaignConf(country, campaign);
    var fn = ft00Path();

    var stageRoot = root+'FILTERED/'+fn;
    var structureOk = false;
    try{ ee.data.getAsset(stageRoot); structureOk = true; }catch(e){}
    if(!structureOk){
        confirmBox.clear();
        confirmBox.add(ui.Label('⚠ '+L.asset_need_structure, STYLE.statusErr));
        hideLoading();
        refreshAssetStatus();
        return;
    }

    var candidate = composeNationalCandidate();
    if(!candidate || candidate.regions.length === 0){
        confirmBox.clear();
        confirmBox.add(ui.Label('Error: ninguna región.', STYLE.statusErr));
        hideLoading();
        return;
    }

    setExportStatus('loading', 'Exportando '+currentPeriod+' para '+fn+'...');

    var startDate = conf.periodicity === 'yearly' ? ee.Date.fromYMD(currentYear, 1, 1) : ee.Date.fromYMD(currentYear, currentMonth, 1);
    var endDate = conf.periodicity === 'yearly' ? startDate.advance(1, 'year') : startDate.advance(1, 'month');

    var nationalImage = candidate.image.set({
      'region_models': candidate.regionModels,
      'country': country,
      'campaign': campaign,
      'proposal': collectionName,
      'filter_stage': 'ft00',
      'period': currentPeriod,
      'periodicity': conf.periodicity,
      'system:time_start': startDate.millis(),
      'system:time_end': endDate.millis()
    });
    var destination = root+'FILTERED/'+fn+'/'+currentPeriod;
    var exportFlag = buildExportFlag(country, campaign, currentPeriod, conf);

    refreshNationalCandidateLayer();

    try{ee.data.getAsset(destination);setExportStatus('info', campaign+' / '+fn+'/'+currentPeriod+' ya existe');}
    catch(e){setExportStatus('success', campaign+' / '+fn+'/'+currentPeriod+' enviado');Export.image.toAsset({image:nationalImage.toInt16(),description:exportFlag,assetId:destination,pyramidingPolicy:'mode',region:regionsCollection(country, campaign).geometry().bounds(),scale:SCALE,maxPixels:1e13});}

    confirmBox.clear();
    confirmBox.add(ui.Label(L.done, {fontSize:'12px',color:'#0f9d58',fontWeight:'bold',margin:'4px'}));
    confirmBox.add(ui.Label('FILTERED/'+fn+'/'+currentPeriod, {fontSize:'10px',color:'#555',fontFamily:'monospace',margin:'2px'}));
    hideLoading();
    refreshAssetStatus();
    loadPeriodData();
}

function refreshAll(){
    loadPeriodData();
    if(selectedPeriod && currentYear){
        loadClassifications(currentYear, currentMonth, function(data){
            if(!regionNames||regionNames.length===0){buildRegionsLayout(function(data2){repopulateRegionCheckboxes(data2);});}
            else{repopulateRegionCheckboxes(data);}
        });
    }
    buildConfirmPanel();
}

// ─── CONSTRUCCIÓN DEL FORMULARIO CON SPLITPANEL ─────────────────────────────

function initConfig(){
    var country = getActiveCountry();
    var campaign = getActiveCampaign();
    if(!country || !campaign) return;
    
    regionNames = regionsCollection(country, campaign).aggregate_array(regionProperty(country, campaign)).distinct().getInfo().sort();
    
    refreshCollectionOptions(function(){
        reloadPeriodAndRegions();
    });
}

function buildForm(){
    ui.root.clear();

    var sidePanel = ui.Panel({
        layout: ui.Panel.Layout.flow('vertical'),
        style: {width: '440px', margin: '0', padding: '6px', backgroundColor: '#fff'}
    });

    sidePanel.add(ui.Label('MapBiomas-Fuego | '+L.title, {fontSize:'14px',fontWeight:'bold',color:'#d32f2f',margin:'4px'}));
    statusLabel = ui.Label({value:'', style:{fontSize:'10px',color:'#1a73e8',margin:'2px 4px',shown:false,stretch:'horizontal'}});
    sidePanel.add(statusLabel);

    sidePanel.add(buildConfigSection());
    sidePanel.add(buildAssetCommandsSection());
    sidePanel.add(buildPeriodSection());
    sidePanel.add(buildRegionsSection());

    exportStatus = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:{margin:'4px',shown:false}});
    sidePanel.add(exportStatus);

    sidePanel.add(buildConfirmSection());

    mapWidget.setOptions('HYBRID');

    var splitPanel = ui.SplitPanel({
        firstPanel: sidePanel,
        secondPanel: mapWidget,
        orientation: 'horizontal',
        wipe: false
    });

    ui.root.add(splitPanel);

    initConfig();
}

// ─── INICIALIZACIÓN ───────────────────────────────────────────────────────────

buildForm();
print('M7_00 5.4 carregado.');
