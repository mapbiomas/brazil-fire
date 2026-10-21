/*******************************************************************************
MAPBIOMAS FUEGO - MONITOR_01 - M7_00
Selección y Exportación de Predicciones Nacionales (ft00)

📅 FECHA: 09/2026
🏷️ VERSIÓN: 4.9 (Fix SplitPanel UI + Fallback de Mapa)
*******************************************************************************/

var b64 = require('users/workspaceipam/packages:mapbiomas-toolkit/utils/b64');

var CATALOG_ROOT = 'projects/mapbiomas-peru/assets/FIRE/CATALOG_01';
var CLASSIFICATIONS_ROOT = CATALOG_ROOT + '/MONITOR_01/LIBRARY_CLASSIFICATIONS/';
var REGIONS = ee.FeatureCollection('projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1');
var REGION_PROPERTY = 'region_nam';
var SCALE = 10;
var START_YEAR = 2025;
var APP_LANG = 'es';

var L = (function(){
    var d={
        pt:{title:'M7 — Selecao e Export',cfg:'CONFIGURACAO',period:'PERIODO',regions:'REGIOES',confirm:'CONFIRMAR',campaign:'Campanha',existing:'Predicoes existentes',new_title:'Criar nova',select:'Selecionar',create:'Criar e selecionar',placeholder:'ex: propuesta_a',year:'Ano',month:'Mes',load:'Carregar',loading:'Carregando...',no_data:'Sem dados.',select_all:'Selecionar todas',clear_all:'Limpar',filter_placeholder:'filtrar predicoes...',target:'Predicao',export_btn:'Exportar',pre_title:'Pre-Confirmacao',pre_body:'Sera criado/atualizado:',pre_warn:'O GEE solicitara confirmacao.',pre_ok:'OK',cancel:'Cancelar',done:'Concluido!',one:'1 predicao',none:'nenhuma'},
        es:{title:'M7 — Selección y Exportación',cfg:'CONFIGURACIÓN',period:'PERÍODO',regions:'REGIONES',confirm:'CONFIRMAR',campaign:'Campaña',existing:'Predicciones existentes',new_title:'Crear nueva colección',select:'Seleccionar',create:'Crear y seleccionar',placeholder:'ej: propuesta_a',year:'Año',month:'Mes',load:'Cargar',loading:'Cargando...',no_data:'Sin datos.',select_all:'Todas',clear_all:'Limpiar',filter_placeholder:'filtrar predicciones...',target:'Predicción',export_btn:'Exportar',pre_title:'Pre-Confirmación',pre_body:'Se creará/actualizará:',pre_warn:'GEE solicitará confirmación de tarea.',pre_ok:'OK',cancel:'Cancelar',done:'¡Completado!',one:'1 predicción',none:'ninguna'},
        en:{title:'M7 — Selection & Export',cfg:'CONFIGURATION',period:'PERIOD',regions:'REGIONS',confirm:'CONFIRM',campaign:'Campaign',existing:'Existing predictions',new_title:'Create new',select:'Select',create:'Create & select',placeholder:'e.g. propuesta_a',year:'Year',month:'Month',load:'Load',loading:'Loading...',no_data:'No data.',select_all:'Select all',clear_all:'Clear',filter_placeholder:'filter predictions...',target:'Prediction',export_btn:'Export',pre_title:'Pre-Confirmation',pre_body:'Will create/update:',pre_warn:'GEE will prompt for confirmation.',pre_ok:'OK',cancel:'Cancel',done:'Done!',one:'1 prediction',none:'none'},
        fr:{title:'M7 — Selection',cfg:'CONFIG',period:'PERIODE',regions:'REGIONS',confirm:'CONFIRMER',campaign:'Campagne',existing:'Predictions existantes',new_title:'Creer',select:'Selectionner',create:'Creer',placeholder:'ex: propuesta_a',year:'Annee',month:'Mois',load:'Charger',loading:'Chargement...',no_data:'Pas de donnees.',select_all:'Toutes',clear_all:'Effacer',filter_placeholder:'filtrer predictions...',target:'Prediction',export_btn:'Exporter',pre_title:'Pre-Confirmation',pre_body:'Va creer:',pre_warn:'GEE demandera confirmation.',pre_ok:'OK',cancel:'Annuler',done:'Termine!',one:'1 prediction',none:'aucune'},
        id:{title:'M7 — Seleksi',cfg:'KONFIG',period:'PERIODE',regions:'WILAYAH',confirm:'KONFIRMASI',campaign:'Kampanye',existing:'Prediksi yang ada',new_title:'Buat baru',select:'Pilih',create:'Buat & pilih',placeholder:'cth: propuesta_a',year:'Tahun',month:'Bulan',load:'Muat',loading:'Memuat...',no_data:'Tidak ada.',select_all:'Semua',clear_all:'Hapus',filter_placeholder:'filter prediksi...',target:'Prediksi',export_btn:'Ekspor',pre_title:'Pra-Konfirmasi',pre_body:'Akan dibuat:',pre_warn:'GEE akan minta konfirmasi.',pre_ok:'OK',cancel:'Batal',done:'Selesai!',one:'1 prediksi',none:'tidak ada'},
    };
    return d[APP_LANG]||d.es;
})();

var SECTION_STYLE = {
    config:  { margin:'4px', padding:'8px', backgroundColor:'#f8f9fa', border:'1px solid #e0e0e0', borderRadius:'6px' },
    period:  { margin:'4px', padding:'8px', backgroundColor:'#f0f4ff', border:'1px solid #c8d6f0', borderRadius:'6px' },
    regions: { margin:'4px', padding:'8px', backgroundColor:'#f8f9fa', border:'1px solid #e0e0e0', borderRadius:'6px' },
    confirm: { margin:'4px', padding:'8px', backgroundColor:'#e8f5e9', border:'1px solid #b8d8ba', borderRadius:'6px' },
    sub:     { margin:'2px 0', padding:'6px', backgroundColor:'#fff', border:'1px solid #e0e0e0', borderRadius:'4px', stretch:'horizontal' },
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
    contentsCard: { margin:'0 0 6px 0', padding:'6px', backgroundColor:'#fff', border:'1px solid #c8d6f0', borderRadius:'4px' },
    checkbox:     { fontSize:'10px', margin:'1px 2px' },
};

var PALETTE = ['#e6194b','#3cb44b','#ffe119','#4363d8','#f58231','#911eb4','#42d4f4','#f032e6','#bfef45','#fabed4','#469990','#dcbeff','#9A6324','#fffac8','#800000'];

// ─── ESTADO DE LA APLICACIÓN ───────────────────────────────────────────────────

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

// Instancia explícita del mapa para evitar la pantalla blanca
var mapWidget = ui.Map();

// ─── REFERENCIAS A COMPONENTES DE LA INTERFAZ ────────────────────────────────

var contentRoot, regionsBox, confirmBox, contentsBox, summaryBox, statusLabel, periodLoading, exportStatus;
var dropdownExisting, dropdownPeriod, dropdownCampaign, textboxCollectionName;

// ─── FUNCIONES DE APOYO (HELPERS) ───────────────────────────────────────────────

function ensureFolder(name){
    var p=name.split('/'), cur=CLASSIFICATIONS_ROOT;
    for(var i=0;i<p.length;i++){cur+=p[i];try{ee.data.getAsset(cur)}catch(e){ee.data.createAsset({type:(i===p.length-1&&/^ft\d\d$/.test(p[i]))?'IMAGE_COLLECTION':'FOLDER'},cur)}cur+='/';}
}

function formatPeriod(y,m){return m!==null?y+'_'+('0'+m).slice(-2):''+y;}

function showLoading(){if(statusLabel)statusLabel.style().set('shown',true);}
function hideLoading(){if(statusLabel)statusLabel.style().set('shown',false);}
function showPeriodLoading(){if(periodLoading)periodLoading.style().set('shown',true);}
function hidePeriodLoading(){if(periodLoading)periodLoading.style().set('shown',false);}

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

function resetRegionState(){
    regionModelMap = {};
    checkboxStore = {};
    filterText = '';
    removeClassificationLayers();
}

// ─── CREADOR DE CASILLAS DE VERIFICACIÓN (CHECKBOXES) ──────────────────────────

function createCheckboxForModel(regionName, model, containerPanel){
    if(!checkboxStore[regionName])checkboxStore[regionName]={};
    var sel = (regionModelMap[regionName] === model.modelId);
    var cb = ui.Checkbox({label:model.modelId, value:sel, style:STYLE.checkbox});
    cb.onChange(function(v){
        var key = regionName+'_'+model.modelId;
        if(v){
            regionModelMap[regionName] = model.modelId;
            Object.keys(checkboxStore[regionName]).forEach(function(k){if(k!==model.modelId)checkboxStore[regionName][k].setValue(false);});
            
            var classImg = ee.Image(model.assetId).select(0).gt(0).selfMask();
            manageMapLayer('class_'+key, classImg, {palette:['#ff0000']}, model.modelId+' | '+regionName);
        } else {
            if(regionModelMap[regionName]===model.modelId)regionModelMap[regionName]=null;
            removeMapLayer('class_'+key);
        }
        buildConfirmPanel();
    });
    checkboxStore[regionName][model.modelId] = cb;
    containerPanel.add(cb);
    if(sel){
        var classImg = ee.Image(model.assetId).select(0).gt(0).selfMask();
        manageMapLayer('class_'+regionName+'_'+model.modelId, classImg, {palette:['#ff0000']}, model.modelId+' | '+regionName);
    }
}

// ─── CARGA DEL MOSAICO SENTINEL-2 MINNBR ──────────────────────────────────────

function loadMosaic(y,m){
    try {
        var dk = formatPeriod(y,m);
        var bs = CATALOG_ROOT+'/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR';
        var img = ee.Image().select();
        ['blue','green','red','nir','swir1','swir2'].forEach(function(b){
            try {
                var bi = ee.ImageCollection(bs+'/'+b.toLowerCase()).filter(ee.Filter.eq('system:index','image_peru_fire_sentinel2_minnbr_'+b.toLowerCase()+'_'+dk)).mosaic();
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
        }, 'Mosaico Sentinel-2 MinNBR ' + dk);
    } catch (e) {
        print('ℹ️ Mosaico MinNBR no disponible para el período:', y, m);
    }
}

// ─── CARGA DE CLASIFICACIONES ────────────────────────────────────────────────

function loadClassifications(y,m,cb){
    var dk = formatPeriod(y,m);
    ee.data.listAssets(CLASSIFICATIONS_ROOT+'REGIONAL',{},function(cols){
        if(!cols||!cols.assets){cb({});return}
        var dirs=cols.assets.filter(function(c){return c.type==='IMAGE_COLLECTION'}), r={}, p=dirs.length;
        if(dirs.length===0){cb({});return}
        dirs.forEach(function(c,idx){
            var modelId=c.id.split('/').pop();
            ee.data.listAssets(c.id,{},function(imgs){
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
    ensureFolder('FILTERED');
    ee.data.listAssets(CLASSIFICATIONS_ROOT+'FILTERED/',{},function(r){
        if(!r||!r.assets){cb([]);return}
        cb(r.assets.filter(function(a){return a.type==='FOLDER'}).map(function(a){return a.id.split('/').pop()}).sort());
    });
}

// ─── CONTEO DE PERÍODOS ──────────────────────────────────────────────────────

function loadPeriodCounts(cb){
    ee.data.listAssets(CLASSIFICATIONS_ROOT+'REGIONAL',{},function(cols){
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

// ─── CONTENIDO DE LA COLECCIÓN ───────────────────────────────────────────────

function loadCollectionContents(){
    if(!contentsBox)return;
    contentsBox.clear();
    showLoading();
    var fn = ft00Path();
    var path = CLASSIFICATIONS_ROOT+'FILTERED/'+fn+'/';

    ee.data.listAssets(path,{},function(r){
        var periods=[];
        if(r&&r.assets)periods=r.assets.filter(function(a){return a.type==='IMAGE'}).map(function(a){return a.id.split('/').pop()}).sort().reverse();

        var card = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:STYLE.contentsCard});
        card.add(ui.Label(L.existing+': '+fn, {fontSize:'10px',fontWeight:'bold',color:'#1a73e8',margin:'0 0 4px 0'}));

        if(periods.length===0){
            card.add(ui.Label('(vacía)', {fontSize:'10px',color:'#aaa',margin:'2px'}));
        } else {
            var list = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:{maxHeight:'100px'}});
            periods.forEach(function(p){list.add(ui.Label(' '+p+'  ✅', {fontSize:'10px',fontFamily:'monospace',color:'#0f9d58',margin:'1px 0'}));});
            card.add(list);
            card.add(ui.Label('Total: '+periods.length, {fontSize:'10px',color:'#888',margin:'2px 0 0 0'}));
        }
        contentsBox.add(card);
        hideLoading();
        if(dropdownPeriod){populatePeriodDropdown(periods);}
    });
}

function populatePeriodDropdown(existingPeriods){
    var today=new Date();var my=today.getFullYear(),mm=today.getMonth();
    if(mm===0){mm=12;my--}
    var allPeriods=[];
    for(var y=my;y>=START_YEAR;y--){var me=(y===my)?mm:12;for(var m=me;m>=1;m--){allPeriods.push(y+'_'+('0'+m).slice(-2));}}
    var pendingPeriods = allPeriods.filter(function(p){return existingPeriods.indexOf(p)===-1;});

    loadPeriodCounts(function(counts){
        var items;
        if(pendingPeriods.length===0){
            items=['(todos completados)'];dropdownPeriod.setDisabled(true);dropdownPeriod.setValue(null);
        } else {
            items=pendingPeriods.map(function(p){
                var c=counts[p]||{total:0,regions:{}};
                var rCount=Object.keys(c.regions).length;
                return p+' ('+rCount+' reg. | '+c.total+' pred.)';
            });
            dropdownPeriod.setDisabled(false);
            dropdownPeriod.items().reset(items);
            var defIdx=0;
            for(var r=0;r<pendingPeriods.length;r++){if(((counts[pendingPeriods[r]]||{}).total||0)>1){defIdx=r;break}}
            dropdownPeriod.setValue(items[defIdx]);
            currentPeriod = pendingPeriods[defIdx];
            currentYear = parseInt(pendingPeriods[defIdx].substring(0,4),10);
            currentMonth = parseInt(pendingPeriods[defIdx].substring(5,7),10);
            loadMosaic(currentYear, currentMonth);mapWidget.centerObject(REGIONS);
            loadPeriodAndRepopulate();
        }
    });
}

var layoutBuilt = false;

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
            regionNames.forEach(function(r){var d=lastClassifications||{};if(d[r]&&d[r].length>0){var f=d[r][0].modelId;regionModelMap[r]=f;if(checkboxStore[r])Object.keys(checkboxStore[r]).forEach(function(k){checkboxStore[r][k].setValue(k===f)});}});buildConfirmPanel();
        }}));
        headerRow.add(ui.Button({label:L.clear_all, style:STYLE.btnGray, onClick:function(){
            Object.keys(checkboxStore).forEach(function(r){if(checkboxStore[r])Object.keys(checkboxStore[r]).forEach(function(k){checkboxStore[r][k].setValue(false)});});regionModelMap={};buildConfirmPanel();
        }}));
        regionsBox.add(headerRow);

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

        layoutBuilt = true;
        hideLoading();hidePeriodLoading();
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
    hideLoading();hidePeriodLoading();
}

function loadPeriodAndRepopulate(){
    loadClassifications(currentYear, currentMonth, function(data){
        if(!layoutBuilt){buildRegionsLayout(function(data2){repopulateRegionCheckboxes(data2);});}
        else{repopulateRegionCheckboxes(data);}
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

// ─── SECCIÓN DE CONFIGURACIÓN ─────────────────────────────────────────────────

function buildConfigSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.config});
    section.add(ui.Label(L.cfg, STYLE.sectionTitle));
    section.add(ui.Label(L.campaign, STYLE.label));
    dropdownCampaign = ui.Select({items:['MONITOR_01','MONITOR_DEV'], value:'MONITOR_01', style:STYLE.input});
    section.add(dropdownCampaign);

    var sideRow = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{margin:'6px 0',stretch:'horizontal'}});

    var existingPanel = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.sub});
    existingPanel.add(ui.Label(L.existing, {fontSize:'11px',fontWeight:'bold',color:'#555',margin:'0 0 2px 0'}));
    dropdownExisting = ui.Select({items:['...'], value:null, style:STYLE.input, disabled:true});
    existingPanel.add(dropdownExisting);
    existingPanel.add(ui.Button({label:L.select, style:STYLE.btnBlue, onClick:function(){
        var v = dropdownExisting.getValue(); if(!v||v==='...')return;
        collectionName = v; textboxCollectionName.setValue(collectionName); refreshAll();
    }}));

    var newPanel = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.sub});
    newPanel.add(ui.Label(L.new_title, {fontSize:'11px',fontWeight:'bold',color:'#555',margin:'0 0 2px 0'}));
    textboxCollectionName = ui.Textbox({placeholder:L.placeholder, value:collectionName, style:STYLE.input});
    textboxCollectionName.onChange(function(v){collectionName=v;});
    newPanel.add(textboxCollectionName);
    newPanel.add(ui.Button({label:L.create, style:STYLE.btnGreen, onClick:function(){
        var name = textboxCollectionName.getValue().trim()||collectionName; collectionName = name;
        ensureFolder('FILTERED/'+name+'/ft00');
        loadExisting(function(names){
            dropdownExisting.items().reset(names);dropdownExisting.setDisabled(false);
            var target = names.filter(function(n){return n===name;})[0];
            if(target){dropdownExisting.setValue(target);collectionName=name;}else{dropdownExisting.setValue(null);}
            refreshAll();
        });
        print('OK: FILTERED/'+name+'/ft00');
    }}));

    sideRow.add(existingPanel).add(newPanel);
    section.add(sideRow);
    return section;
}

// ─── SECCIÓN DE PERÍODO ───────────────────────────────────────────────────────

function buildPeriodSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.period});
    section.add(ui.Label(L.period, STYLE.sectionTitle));

    contentsBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical')});
    section.add(contentsBox);

    dropdownPeriod = ui.Select({items:['...'], value:null, style:STYLE.input, disabled:true});
    dropdownPeriod.onChange(function(v){
        if(!v||v==='...'||!/^\d/.test(v))return;
        currentPeriod = v.split(' ')[0];
        resetRegionState();
        showLoading();showPeriodLoading();
        var y = parseInt(currentPeriod.substring(0,4),10), m = parseInt(currentPeriod.substring(5,7),10);
        currentYear = y; currentMonth = m;
        loadMosaic(currentYear, currentMonth);mapWidget.centerObject(REGIONS);
        loadPeriodAndRepopulate();
    });

    var periodRow = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{stretch:'horizontal'}});
    periodRow.add(dropdownPeriod);
    periodLoading = ui.Label({value:'', style:{shown:false,margin:'2px 6px',stretch:'horizontal'}});
    periodRow.add(periodLoading);
    section.add(periodRow);
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
    confirmBox.clear();
    var fn = ft00Path();
    var box = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:STYLE.prePopup});
    box.add(ui.Label('⚠ '+L.pre_title, {fontSize:'13px',fontWeight:'bold',color:'#cc8800',margin:'2px'}));
    box.add(ui.Label(L.pre_body, {fontSize:'11px',color:'#333',margin:'2px'}));
    box.add(ui.Label('FILTERED/'+fn+'/'+currentPeriod, {fontSize:'11px',fontWeight:'bold',fontFamily:'monospace',color:'#1a73e8',margin:'2px',padding:'4px',backgroundColor:'#fff'}));
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
    var fn = ft00Path();
    var camp = dropdownCampaign.getValue();
    setExportStatus('loading', 'Exportando '+currentPeriod+' para '+fn+'...');
    ensureFolder('FILTERED/'+fn);

    var names = Object.keys(regionModelMap).filter(function(r){return!!regionModelMap[r];});
    if(names.length===0){confirmBox.add(ui.Label('Error: ninguna región.', STYLE.statusErr));return}

    var nationalImage = ee.Image(0).rename('probability');
    var doyImage = ee.Image(0).rename('dayOfYear');
    var modelList = [];

    names.forEach(function(rn){
        var modelId = regionModelMap[rn]; modelList.push(rn+':'+modelId);
        var assets = availableModels[currentPeriod]&&availableModels[currentPeriod][rn];if(!assets)return;
        var found = null; for(var i=0;i<assets.length;i++){if(assets[i].modelId===modelId){found=assets[i];break;}}if(!found)return;
        var regionMask = ee.Image(0).paint(REGIONS.filter(ee.Filter.eq(REGION_PROPERTY,rn)),1);
        var source = ee.Image(found.assetId);
        nationalImage = nationalImage.where(regionMask.eq(1),source.select(0));
        doyImage = doyImage.where(regionMask.eq(1),source.select(1));
    });

    nationalImage = nationalImage.addBands(doyImage).selfMask().set({
      'region_models': modelList.join(','),
      'campaign': camp,
      'filter_stage': 'ft00',
      'period': currentPeriod,
      'system:time_start': ee.Date.fromYMD(currentYear, currentMonth, 1).millis(),
      'system:time_end': ee.Date.fromYMD(currentYear, currentMonth, 1).advance(1, 'month').millis()
    });
    var destination = CLASSIFICATIONS_ROOT+'FILTERED/'+fn+'/'+currentPeriod;

    manageMapLayer('national_'+currentPeriod, nationalImage.select('probability').selfMask(), {min:0,max:1000,palette:['#fcc','#f00','#600']}, 'National '+currentPeriod);

    try{ee.data.getAsset(destination);setExportStatus('info', camp+' / '+fn+'/'+currentPeriod+' ya existe');}
    catch(e){setExportStatus('success', camp+' / '+fn+'/'+currentPeriod+' enviado');Export.image.toAsset({image:nationalImage.toInt16(),description:(camp+'_'+currentPeriod.replace(/_/g,'')).substring(0,80),assetId:destination,pyramidingPolicy:'mode',region:REGIONS.geometry().bounds(),scale:SCALE,maxPixels:1e13});}

    confirmBox.clear();
    confirmBox.add(ui.Label(L.done, {fontSize:'12px',color:'#0f9d58',fontWeight:'bold',margin:'4px'}));
    confirmBox.add(ui.Label('FILTERED/'+fn+'/'+currentPeriod, {fontSize:'10px',color:'#555',fontFamily:'monospace',margin:'2px'}));
    hideLoading();
    loadCollectionContents();
}

function refreshAll(){loadPeriodAndRepopulate();buildConfirmPanel();loadCollectionContents();}

// ─── CONSTRUCCIÓN DEL FORMULARIO CON SPLITPANEL ─────────────────────────────

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
    sidePanel.add(buildPeriodSection());
    sidePanel.add(buildRegionsSection());

    exportStatus = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:{margin:'4px',shown:false}});
    sidePanel.add(exportStatus);

    sidePanel.add(buildConfirmSection());

    mapWidget.setOptions('HYBRID');
    mapWidget.centerObject(REGIONS);
    
    // Adiciona o limite das regiões
    manageMapLayer('regions_boundary', REGIONS.style({color:'ffffff',fillColor:'00000000',width:1}), {}, 'Regiones');

    // Layout Dividido (Garante que o mapa fique visível na direita)
    var splitPanel = ui.SplitPanel({
        firstPanel: sidePanel,
        secondPanel: mapWidget,
        orientation: 'horizontal',
        wipe: false
    });

    ui.root.add(splitPanel);

    regionNames = REGIONS.aggregate_array(REGION_PROPERTY).distinct().getInfo().sort();

    loadExisting(function(names){
        if(names.length===0){dropdownExisting.items().reset(['(ninguna)']);dropdownExisting.setDisabled(true);}
        else {
            dropdownExisting.items().reset(names);dropdownExisting.setDisabled(false);
            dropdownExisting.setValue(names[0]);
            collectionName = names[0];
            textboxCollectionName.setValue(collectionName);
            loadCollectionContents();
        }
    });
}

// ─── INICIALIZACIÓN ───────────────────────────────────────────────────────────

buildForm();
print('M7_00 cargado.');