/********************************************
MAPBIOMAS FUEGO - MONITOR_01 - M9_00
Pre-Public Promotion (UI) — Redesign v5.4

📅 DATA: setembro 2026
🏷️ VERSAO: 5.5

📌 SEÇÕES COM FUNDO COLORIDO (ordem vertical):
  CONFIG — cinza       (paises + campanhas + fechas)
  CANDIDATOS — azul    (grids por etapa x propuesta)
  PROMOVER — verde     (nota sutil + resumo + botao)

📌 O QUE FAZ:
1. CONFIG: paises (checkboxes multi), campanhas (checkbox multi),
   fechas (grid de datas, selecao unica, promovidas marcadas)
2. Cada par pais+campanha ligado adiciona um grupo de grids
   (root: {CATALOG}/{CAMPAIGN}/LIBRARY_CLASSIFICATIONS/)
3. Etapas ftXX descobertas automaticamente (IMAGE_COLLECTION)
4. Grid por etapa: colunas = propuestas, 1 linha = fecha global
5. ✅ POR CELULA: ao promover, a origem (proposta+etapa) e gravada
   em memoria (fonte de verdade da sessao) e tambem em metadados na
   imagem via updateAsset; no load, so a celula promovida fica ✅ e
   mantem checkbox de visualizacao (nao some)
6. Sincronizacao errada (origem ausente/divergente) gera aviso
7. Botao dinamico Promover/Despromover (sem protocolo)
8. Promueve a PRE_PUBLIC via ee.data.copyAsset

📌 SIN CREACION AUTOMATICA DE ASSETS al inicializar.
   Area quemada NO se calcula aqui (papel del M8).
********************************************/

// ⚙️ CONFIGURACION DEL USUARIO
var COUNTRY_ORDER = ['peru', 'chile', 'bolivia', 'colombia', 'paraguay', 'guyana'];
var CAMPAIGNS = ['MONITOR_01', 'MONITOR_DEV'];

var COUNTRIES = {
    chile:    { catalog: 'projects/mapbiomas-chile/assets/FIRE/CATALOG_01',    regions: 'projects/mapbiomas-chile/assets/FIRE/AUXILIARY_DATA/regiones_fuego_chile_v1',    mosaic: 'image_chile_fire_sentinel2_minnbr_' },
    peru:     { catalog: 'projects/mapbiomas-peru/assets/FIRE/CATALOG_01',     regions: 'projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1',     mosaic: 'image_peru_fire_sentinel2_minnbr_' },
    bolivia:  { catalog: 'projects/mapbiomas-bolivia/assets/FIRE/CATALOG_01',  regions: 'projects/mapbiomas-bolivia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_bolivia_v1',  mosaic: 'image_bolivia_fire_sentinel2_minnbr_' },
    colombia: { catalog: 'projects/mapbiomas-colombia/assets/FIRE/CATALOG_01', regions: 'projects/mapbiomas-colombia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_colombia_v1', mosaic: 'image_colombia_fire_sentinel2_minnbr_' },
    paraguay: { catalog: 'projects/mapbiomas-paraguay/assets/FIRE/CATALOG_01', regions: 'projects/mapbiomas-paraguay/assets/FIRE/AUXILIARY_DATA/regiones_fuego_paraguay_v1', mosaic: 'image_paraguay_fire_sentinel2_minnbr_' },
    guyana:   { catalog: 'projects/mapbiomas-guyana/assets/FIRE/CATALOG_01',   regions: 'projects/mapbiomas-guyana/assets/FIRE/AUXILIARY_DATA/regiones_fuego_guyana_v1',   mosaic: 'image_guyana_fire_sentinel2_minnbr_' }
};

var SCALE = 10;
var APP_LANG = 'es';

// 🔗 Looker Studio — estadisticas de area quemada (M8)
var STATS_URL = 'https://datastudio.google.com/reporting/cc275c3b-4a5e-4b4c-97af-50191eca7698';

var L = (function(){
    var d={
        pt:{title:'M9 — Promover a PRE_PUBLIC',cfg:'CONFIGURACAO',candidates:'CANDIDATOS',promote:'PROMOVER',promote_note:'Nota: verifique a cicatriz no mapa e as estatisticas antes de promover.',country:'Paises',date:'Datas',campaign:'Campanha',proposal:'Proposta',empty_pre:'(vazio)',loading:'Carregando...',no_data:'Nenhum candidato encontrado.',done:'Promovido!',stats_link:'📊 Ver estatisticas (Looker Studio)',pre_title:'Confirmar Promocao',pre_body:'Serao copiados para PRE_PUBLIC/',pre_warn:'A pasta PRE_PUBLIC/{campanha} sera criada se necessario.',cancel:'Cancelar',ok:'Confirmar',promote_btn:'Promover para PRE_PUBLIC',target:'Destino',none_selected:'Nenhum selecionado.',selected:'Selecionados',unpromote:'Despromover',unpromote_confirm:'Despromover a(s) imagem(ns) desta data?',unpromote_done:'Despromovido!',promoted:'Promovido',sync_warn:'Atencao: imagem promovida sem origem identificada ou origem divergente das propostas/etapas atuais.'},
        es:{title:'M9 — Promover a PRE_PUBLIC',cfg:'CONFIG',candidates:'CANDIDATOS',promote:'PROMOVER',promote_note:'Nota: verifique la cicatriz en el mapa y las estadisticas antes de promover.',country:'Paises',date:'Fechas',campaign:'Campana',proposal:'Propuesta',empty_pre:'(vacio)',loading:'Cargando...',no_data:'Sin candidatos.',done:'¡Promovido!',stats_link:'📊 Ver estadisticas (Looker Studio)',pre_title:'Confirmar Promocion',pre_body:'Se copiara a PRE_PUBLIC/',pre_warn:'La carpeta PRE_PUBLIC/{campana} se creara si es necesario.',cancel:'Cancelar',ok:'Confirmar',promote_btn:'Promover a PRE_PUBLIC',target:'Destino',none_selected:'Ninguno seleccionado.',selected:'Seleccionados',unpromote:'Despromover',unpromote_confirm:'¿Despromover la(s) imagen(es) de esta fecha?',unpromote_done:'¡Despromovido!',promoted:'Promovido',sync_warn:'Atencion: imagen promovida sin origen identificado u origen divergente de las propuestas/etapas actuales.'},
        en:{title:'M9 — Promote to PRE_PUBLIC',cfg:'CONFIGURATION',candidates:'CANDIDATES',promote:'PROMOTE',promote_note:'Note: verify the scar on the map and statistics before promoting.',country:'Countries',date:'Dates',campaign:'Campaign',proposal:'Proposal',empty_pre:'(empty)',loading:'Loading...',no_data:'No candidates found.',done:'Promoted!',stats_link:'📊 View statistics (Looker Studio)',pre_title:'Confirm Promotion',pre_body:'Will copy to PRE_PUBLIC/',pre_warn:'The PRE_PUBLIC/{campaign} folder will be created if needed.',cancel:'Cancel',ok:'Confirm',promote_btn:'Promote to PRE_PUBLIC',target:'Destination',none_selected:'None selected.',selected:'Selected',unpromote:'Unpromote',unpromote_confirm:'Unpromote the image(s) for this date?',unpromote_done:'Unpromoted!',promoted:'Promoted',sync_warn:'Warning: promoted image without identified origin or origin divergent from current proposals/stages.'},
        fr:{title:'M9 — Promouvoir vers PRE_PUBLIC',cfg:'CONFIG',candidates:'CANDIDATS',promote:'PROMOUVOIR',promote_note:'Remarque : verifiez la cicatrice sur la carte et les statistiques avant de promouvoir.',country:'Pays',date:'Dates',campaign:'Campagne',proposal:'Proposition',empty_pre:'(vide)',loading:'Chargement...',no_data:'Aucun candidat.',done:'Promu!',stats_link:'📊 Voir les statistiques (Looker Studio)',pre_title:'Confirmer la Promotion',pre_body:'Va copier vers PRE_PUBLIC/',pre_warn:'Le dossier PRE_PUBLIC/{campagne} sera cree si necessaire.',cancel:'Annuler',ok:'Confirmer',promote_btn:'Promouvoir vers PRE_PUBLIC',target:'Destination',none_selected:'Aucun selectionne.',selected:'Selectionnes',unpromote:'Depromouvoir',unpromote_confirm:'Depromouvoir la/les image(s) de cette date ?',unpromote_done:'Depromu !',promoted:'Promu',sync_warn:'Attention : image promue sans origine identifiee ou origine divergente des propositions/etapes actuelles.'},
        id:{title:'M9 — Promosikan ke PRE_PUBLIC',cfg:'KONFIG',candidates:'KANDIDAT',promote:'PROMOSI',promote_note:'Catatan: periksa bekas luka di peta dan statistik sebelum mempromosikan.',country:'Negara',date:'Tanggal',campaign:'Kampanye',proposal:'Proposal',empty_pre:'(kosong)',loading:'Memuat...',no_data:'Tidak ada kandidat.',done:'Terpromosi!',stats_link:'📊 Lihat statistik (Looker Studio)',pre_title:'Konfirmasi Promosi',pre_body:'Akan disalin ke PRE_PUBLIC/',pre_warn:'Folder PRE_PUBLIC/{kampanye} akan dibuat jika perlu.',cancel:'Batal',ok:'Konfirmasi',promote_btn:'Promosikan ke PRE_PUBLIC',target:'Tujuan',none_selected:'Tidak ada yang dipilih.',selected:'Terpilih',unpromote:'Batalkan promosi',unpromote_confirm:'Batalkan promosi gambar untuk tanggal ini?',unpromote_done:'Promosi dibatalkan!',promoted:'Dipromosikan',sync_warn:'Perhatian: gambar dipromosikan tanpa asal teridentifikasi atau asal berbeda dari proposal/tahap saat ini.'},
    };
    return d[APP_LANG]||d.es;
})();

var SECTION_STYLE = {
    config:     { margin:'4px', padding:'8px', backgroundColor:'#f8f9fa', border:'1px solid #e0e0e0', borderRadius:'6px' },
    candidates: { margin:'4px', padding:'8px', backgroundColor:'#f0f4ff', border:'1px solid #c8d6f0', borderRadius:'6px' },
    promote:    { margin:'4px', padding:'8px', backgroundColor:'#e8f5e9', border:'1px solid #b8d8ba', borderRadius:'6px' },
};

var STYLE = {
    label:        { fontSize:'11px', color:'#555', margin:'2px 0' },
    input:        { stretch:'horizontal', fontSize:'12px', margin:'2px 0' },
    btnBlue:      { margin:'2px', padding:'4px 10px', color:'#1a73e8', fontWeight:'bold' },
    btnGreen:     { margin:'2px', padding:'4px 10px', color:'#0f9d58', fontWeight:'bold' },
    btnGray:      { margin:'2px', padding:'4px 10px', color:'#70757a', fontWeight:'bold' },
    btnRed:       { margin:'2px', padding:'4px 10px', color:'#d32f2f', fontWeight:'bold' },
    statusOk:     { color:'#0f9d58', fontWeight:'bold', fontSize:'11px' },
    statusErr:    { color:'#d32f2f', fontWeight:'bold', fontSize:'11px' },
    prePopup:     { margin:'6px 0', padding:'10px', backgroundColor:'#fff8e1', border:'1px solid #ffcc00', borderRadius:'6px' },
    sectionTitle: { fontSize:'12px', fontWeight:'bold', color:'#333', margin:'0 0 6px 0' },
    card:         { margin:'2px', padding:'4px', backgroundColor:'#fff', border:'1px solid #e0e0e0', borderRadius:'4px' },
    summaryCard:  { margin:'0 0 6px 0', padding:'6px', backgroundColor:'#fff', border:'1px solid #e0e0e0', borderRadius:'4px' },
    gridHeader:   { fontSize:'10px', fontWeight:'bold', color:'#1a73e8', margin:'1px 3px' },
    gridCell:     { fontSize:'10px', margin:'1px 2px' },
    gridDate:     { fontSize:'10px', fontFamily:'monospace', color:'#333', margin:'2px 4px', width:'64px' },
    gridHead:     { fontSize:'10px', fontWeight:'bold', color:'#1a73e8', margin:'2px 4px' },
    swatch:       { width:'10px', height:'10px', margin:'2px 3px', borderRadius:'2px', border:'1px solid rgba(0,0,0,0.3)' },
};

// ─── APPLICATION STATE ──────────────────────────────────────────────────────

// Paleta de cores unicas por ft/proposta (saturadas, contraste com satelite).
// As primeiras sao as mais fortes; suporta ate 40 combinacoes.
var PALETTE = [
    '#ff0055', '#00ff66', '#6600ff', '#ff9900', '#00ccff', '#ff00aa', '#aaff00', '#ff00ff',
    '#00ffaa', '#ff5500', '#5500ff', '#00ffdd', '#ffcc00', '#ff0066', '#33ff00', '#cc00ff',
    '#00aaff', '#ffaa00', '#aa00ff', '#00ff33', '#ff0044', '#66ff00', '#ff00cc', '#00ffee',
    '#ff8800', '#4400ff', '#00ff88', '#ff2200', '#88ff00', '#ff00ee', '#00ddff', '#ffee00',
    '#ff0066', '#00ffcc', '#ff00aa', '#22ff00', '#cc00aa', '#00ccff', '#ffaa44', '#ff44cc'
];
var colorIndex = {};               // chave stage__proposal -> indice na paleta

function colorFor(stage, proposal){
    var key = stage+'__'+proposal;
    if(!(key in colorIndex))colorIndex[key] = Object.keys(colorIndex).length % PALETTE.length;
    return PALETTE[colorIndex[key]];
}

var selectedCells = {};          // key: pais__campanha__etapa__propuesta__fecha
var countryData = {};            // { pais: { CAMPAIGN: { proposals, stages, stageData, promoted } } }
var promotedOrigin = {};         // { pais: { CAMPAIGN: { period: {proposal, stage} } } } — origem das promovidas (metadados)
var viewChecks = {};             // cellKey -> bool — visualizacao de celulas promovidas
var enabledCountries = {};       // { pais: bool }
var enabledCampaigns = {};       // { CAMPAIGN: bool }
var allDates = [];
var selectedDate = null;
var actionMode = 'promote';   // 'promote' | 'unpromote' (botao principal)
var cellCheckboxes = {};         // key -> ui.Checkbox (radio por linea)
var dateCheckboxes = {};         // fecha -> ui.Checkbox
var mapLayers = {};
var mapWidget = ui.Map();
var countryBox, campaignBox, dateBox, candidatesBox, promoteBox, statusLabel, exportStatus, promoteButton;

// ─── HELPERS ─────────────────────────────────────────────────────────────────

function classificationsRoot(country, camp){return COUNTRIES[country].catalog+'/'+camp+'/LIBRARY_CLASSIFICATIONS/';}

function prePublicPath(country, camp){return classificationsRoot(country, camp)+'PRE_PUBLIC/'+camp.toLowerCase()+'/';}

function cellKey(country, camp, stage, proposal, period){return country+'__'+camp+'__'+stage+'__'+proposal+'__'+period;}

function promotedKey(country, camp, period){return country+'__'+camp+'__'+period;}

function activePairs(){
    var pairs = [];
    Object.keys(enabledCountries).forEach(function(c){
        if(!enabledCountries[c])return;
        Object.keys(enabledCampaigns).forEach(function(camp){
            if(enabledCampaigns[camp])pairs.push({country:c, camp:camp});
        });
    });
    return pairs;
}

function isDatePromotedGlobal(date){
    return activePairs().some(function(pair){
        return countryData[pair.country] && countryData[pair.country][pair.camp] && countryData[pair.country][pair.camp].promoted[date];
    });
}

function promotedKeysForDate(date){
    var keys = [];
    activePairs().forEach(function(pair){
        var d = countryData[pair.country] && countryData[pair.country][pair.camp];
        if(d && d.promoted[date])keys.push(promotedKey(pair.country, pair.camp, date));
    });
    return keys;
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

// ─── MAP LAYERS ─────────────────────────────────────────────────────────────

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

function clearCandidateLayers(){
    Object.keys(mapLayers).forEach(function(k){
        if(k.indexOf('cand_')===0||k.indexOf('mosaic_')===0||k.indexOf('promoted_')===0)removeMapLayer(k);
    });
}

function updateRegionLayers(){
    Object.keys(mapLayers).forEach(function(k){
        if(k.indexOf('regions_boundary_')===0)removeMapLayer(k);
    });
    var first = null;
    Object.keys(enabledCountries).forEach(function(c){
        if(!enabledCountries[c])return;
        if(!first)first = c;
        var fc = ee.FeatureCollection(COUNTRIES[c].regions);
        manageMapLayer('regions_boundary_'+c, fc.style({color:'ffffff',fillColor:'00000000',width:1}), {}, 'Regiones '+c);
    });
    if(first)mapWidget.centerObject(ee.FeatureCollection(COUNTRIES[first].regions));
}

function loadMosaic(country, period){
    var bs = COUNTRIES[country].catalog+'/LIBRARY_IMAGES/SENTINEL2/MONTHLY/MINNBR';
    var prefix = COUNTRIES[country].mosaic || 'image_peru_fire_sentinel2_minnbr_';
    var img = ee.Image().select();
    ['blue','green','red','nir','swir1','swir2'].forEach(function(b){
        try {
            var bi = ee.ImageCollection(bs+'/'+b.toLowerCase()).filter(ee.Filter.eq('system:index',prefix+b.toLowerCase()+'_'+period)).mosaic();
            img = img.addBands(ee.Image(ee.Algorithms.If(bi.bandNames().size().gt(0),bi,ee.Image(0).rename(b.toLowerCase()).updateMask(0))).select([0],[b.toLowerCase()]),null,true);
        } catch(e) {
            img = img.addBands(ee.Image(0).rename(b.toLowerCase()).updateMask(0),null,true);
        }
    });
    manageMapLayer('mosaic_'+country+'_'+period, img, {
        bands: ['swir1', 'nir', 'red'],
        min: 5,
        max: 48,
        gamma: 1.1
    }, 'Mosaico Sentinel-2 MinNBR '+period+' | '+country);
}

// ─── SELECT / DESELECT CELL ─────────────────────────────────────────────────

function setCellSelected(country, camp, stage, proposal, period, v){
    var key = cellKey(country, camp, stage, proposal, period);
    selectedCells[key] = v;
    if(v){
        loadMosaic(country, period);
        var img = ee.Image(classificationsRoot(country, camp)+'FILTERED/'+proposal+'/'+stage+'/'+period).select('probability').selfMask();
        manageMapLayer('cand_'+key, img, {min:0,max:1000,palette:[colorFor(stage, proposal)]}, 'Cand '+period+' | '+proposal+' | '+stage+' | '+camp+' | '+country);
        // radio por linea: desmarcar otras propuestas de la misma fecha en este grid
        countryData[country][camp].proposals.forEach(function(p2){
            if(p2!==proposal){
                var k2 = cellKey(country, camp, stage, p2, period);
                if(selectedCells[k2]){
                    selectedCells[k2] = false;
                    if(cellCheckboxes[k2])cellCheckboxes[k2].setValue(false);
                    removeMapLayer('cand_'+k2);
                }
            }
        });
    } else {
        removeMapLayer('cand_'+key);
        var any = Object.keys(selectedCells).some(function(k){
            return selectedCells[k] && k.split('__')[4]===period;
        });
        if(!any)removeMapLayer('mosaic_'+country+'_'+period);
    }
    updatePromoteSummary();
}

// ─── LOAD ALL ───────────────────────────────────────────────────────────────

function loadAll(){
    clearCandidateLayers();
    selectedCells = {};
    cellCheckboxes = {};
    viewChecks = {};
    candidatesBox.clear();
    candidatesBox.add(ui.Label(L.loading,{fontSize:'10px',color:'#888'}));

    var pairs = activePairs();
    if(pairs.length===0){
        countryData = {};
        allDates = [];
        selectedDate = null;
        renderDateGrid();
        renderCandidates();
        updatePromoteSummary();
        hideLoading();
        return;
    }

    countryData = {};
    var pending = pairs.length;
    pairs.forEach(function(pair){loadPair(pair, function(){pending--;if(pending===0)finishLoad();});});
}

function loadPair(pair, cb){
    var c = pair.country, camp = pair.camp;
    if(!countryData[c])countryData[c] = {};
    countryData[c][camp] = {proposals:[], stages:[], stageData:{}, promoted:{}};
    var root = classificationsRoot(c, camp);
    ee.data.listAssets(root+'FILTERED/',{},function(r,err){
        var props=[];
        if(!err&&r&&r.assets)props=r.assets.filter(function(a){return a.type==='FOLDER'}).map(function(a){return a.id.split('/').pop()}).sort();
        countryData[c][camp].proposals = props;
        if(props.length===0){loadPromoted(c, camp, cb);return;}

        var pendingStages = props.length;
        var stageUnion = {};
        props.forEach(function(prop){
            ee.data.listAssets(root+'FILTERED/'+prop+'/',{},function(r2,err2){
                if(!err2&&r2&&r2.assets)r2.assets.forEach(function(a){
                    var name = a.id.split('/').pop();
                    // etapas ftXX sao IMAGE_COLLECTION no GEE (ver ensureFolder do M7)
                    if((a.type==='IMAGE_COLLECTION'||a.type==='FOLDER')&&/^ft\d\d$/.test(name))stageUnion[name]=true;
                });
                pendingStages--;
                if(pendingStages===0){
                    countryData[c][camp].stages = Object.keys(stageUnion).sort();
                    var stg = countryData[c][camp].stages;
                    if(stg.length===0){loadPromoted(c, camp, cb);return;}
                    stg.forEach(function(s){countryData[c][camp].stageData[s]={};props.forEach(function(p){countryData[c][camp].stageData[s][p]=[];});});
                    var pendingImgs = stg.length*props.length;
                    stg.forEach(function(s){
                        props.forEach(function(p){
                            ee.data.listAssets(root+'FILTERED/'+p+'/'+s+'/',{},function(r3,err3){
                                if(!err3&&r3&&r3.assets)r3.assets.forEach(function(a){
                                    if(a.type==='IMAGE')countryData[c][camp].stageData[s][p].push(a.id.split('/').pop());
                                });
                                pendingImgs--;
                                if(pendingImgs===0)loadPromoted(c, camp, cb);
                            });
                        });
                    });
                }
            });
        });
    });
}

function loadPromoted(country, camp, cb){
    ee.data.listAssets(prePublicPath(country, camp),{},function(r,err){
        var props = [];
        if(!err&&r&&r.assets)props = r.assets.filter(function(a){return a.type==='IMAGE';});
        if(!promotedOrigin[country])promotedOrigin[country] = {};
        if(!promotedOrigin[country][camp])promotedOrigin[country][camp] = {};
        if(props.length===0){cb();return;}
        var pending = props.length;
        props.forEach(function(a){
            var period = a.id.split('/').pop();
            countryData[country][camp].promoted[period] = true;
            // Le properties de ambas as fontes (metadata do asset e properties da imagem)
            var p = {};
            try{
                var info = ee.data.getAsset(a.id);
                if(info && info.properties)p = info.properties;
            }catch(e1){}
            if(!(p['source_proposal'] && p['source_stage'])){
                try{
                    var g = ee.Image(a.id).getInfo();
                    if(g && g.properties){
                        p['source_proposal'] = g.properties['source_proposal'];
                        p['source_stage'] = g.properties['source_stage'];
                    }
                }catch(e2){}
            }
            var prev = promotedOrigin[country][camp][period];
            if(p['source_proposal'] && p['source_stage']){
                promotedOrigin[country][camp][period] = {proposal:p['source_proposal'], stage:p['source_stage']};
            } else if(prev){
                // Preserva origem conhecida em memoria (promovida nesta sessao)
                promotedOrigin[country][camp][period] = prev;
            } else {
                promotedOrigin[country][camp][period] = null;
            }
            pending--;
            if(pending===0)cb();
        });
    });
}

function finishLoad(){
    // Limpa origens de pares nao ativos (evita dados orfaos ao desmarcar campanha)
    Object.keys(promotedOrigin).forEach(function(c){
        Object.keys(promotedOrigin[c]).forEach(function(camp){
            var active = enabledCountries[c] && enabledCampaigns[camp];
            if(!active)delete promotedOrigin[c][camp];
        });
        if(Object.keys(promotedOrigin[c]).length===0)delete promotedOrigin[c];
    });

    var dateSet = {};
    activePairs().forEach(function(pair){
        var d = countryData[pair.country][pair.camp];
        d.stages.forEach(function(s){
            d.proposals.forEach(function(p){
                (d.stageData[s][p]||[]).forEach(function(dt){dateSet[dt]=true;});
            });
        });
        Object.keys(d.promoted).forEach(function(dt){dateSet[dt]=true;});
    });
    allDates = Object.keys(dateSet).sort().reverse();
    if(!selectedDate || allDates.indexOf(selectedDate)===-1){
        selectedDate = allDates[0]||null;
    }
    checkPromotedSync();
    renderDateGrid();
    renderCandidates();
    updatePromoteSummary();
    hideLoading();
}

function checkPromotedSync(){
    // Metadados de origem divergentes ou ausentes: avisa mas nao bloqueia
    activePairs().forEach(function(pair){
        var c = pair.country, camp = pair.camp;
        var d = countryData[c] && countryData[c][camp];
        if(!d)return;
        var origins = promotedOrigin[c] && promotedOrigin[c][camp] ? promotedOrigin[c][camp] : {};
        Object.keys(origins).forEach(function(period){
            var o = origins[period];
            if(!o)return;
            var okStage = d.stages.indexOf(o.stage)!==-1;
            var okProposal = d.proposals.indexOf(o.proposal)!==-1;
            if(!okStage || !okProposal){
                setExportStatus('error', L.sync_warn+' ('+c+' | '+camp+' | '+period+' | '+o.proposal+'/'+o.stage+')');
                print('⚠️ '+L.sync_warn+' '+c+' | '+camp+' | '+period+' | '+o.proposal+'/'+o.stage);
            }
        });
    });
}

// ─── DATE GRID ──────────────────────────────────────────────────────────────

function renderDateGrid(){
    if(!dateBox)return;
    dateBox.clear();
    dateBox.add(ui.Label(L.date, STYLE.label));
    var row = ui.Panel({layout:ui.Panel.Layout.flow('horizontal')});
    if(allDates.length===0){
        row.add(ui.Label(L.empty_pre, {fontSize:'10px',color:'#aaa',margin:'2px'}));
    } else {
        allDates.forEach(function(d){
            var anyPromoted = isDatePromotedGlobal(d);
            var cb = ui.Checkbox({
                label: d + (anyPromoted?'  ✅':''),
                value: selectedDate===d,
                style: STYLE.gridCell
            });
            dateCheckboxes[d] = cb;
            cb.onChange(function(v){
                if(!v)return;
                selectedDate = d;
                Object.keys(dateCheckboxes).forEach(function(d2){
                    if(d2!==d&&dateCheckboxes[d2].getValue())dateCheckboxes[d2].setValue(false);
                });
                selectedCells = {};
                cellCheckboxes = {};
                clearCandidateLayers();
                renderCandidates();
                updatePromoteSummary();
            });
            row.add(cb);
        });
    }
    dateBox.add(row);
}

// ─── RENDER CANDIDATES (groups per country+campaign) ────────────────────────

function renderCandidates(){
    candidatesBox.clear();
    actionMode = computeActionMode();
    var pairs = activePairs();
    if(pairs.length===0){
        candidatesBox.add(ui.Label(L.no_data,{fontSize:'11px',color:'#d32f2f',margin:'4px'}));
        return;
    }
    if(!selectedDate){
        candidatesBox.add(ui.Label(L.empty_pre,{fontSize:'11px',color:'#888',margin:'4px'}));
        return;
    }

    pairs.forEach(function(pair){
        var c = pair.country, camp = pair.camp;
        var d = countryData[c] && countryData[c][camp];
        if(!d)return;
        var group = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:STYLE.card});
        group.add(ui.Label(c+' | '+camp+' | '+classificationsRoot(c, camp), {fontSize:'10px',fontWeight:'bold',color:'#1a73e8',margin:'2px'}));

        var stages = d.stages.filter(function(s){
            return d.proposals.some(function(p){return (d.stageData[s][p]||[]).indexOf(selectedDate)!==-1;});
        });

        if(stages.length===0){
            group.add(ui.Label('('+L.empty_pre+')', {fontSize:'9px',color:'#aaa',margin:'1px 3px'}));
        } else {
            stages.forEach(function(stg){
                group.add(buildStageGrid(c, camp, stg, d));
            });
        }
        candidatesBox.add(group);
    });
}

function buildStageGrid(country, camp, stage, d){
    var grid = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:{margin:'2px 0'}});
    var origin = (promotedOrigin[country] && promotedOrigin[country][camp]) ? promotedOrigin[country][camp][selectedDate] : null;
    var head = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{backgroundColor:'#eef2fa',margin:'0 0 2px 0'}});
    head.add(ui.Label(stage, STYLE.gridHeader));
    d.proposals.forEach(function(prop){
        var headPromoted = origin && origin.proposal===prop && origin.stage===stage;
        var sw = ui.Label('', {backgroundColor:colorFor(stage, prop), ...STYLE.swatch});
        head.add(sw);
        head.add(ui.Label(prop + (headPromoted?'  ✅':''), {fontSize:'10px',fontWeight:'bold',color:headPromoted?'#0f9d58':'#1a73e8',margin:'2px 4px'}));
    });
    grid.add(head);

    var row = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{margin:'1px 0'}});
    row.add(ui.Label(selectedDate, STYLE.gridDate));
    d.proposals.forEach(function(prop){
        var available = (d.stageData[stage][prop]||[]).indexOf(selectedDate)!==-1;
        if(!available){
            row.add(ui.Label('·', {fontSize:'9px',color:'#ccc',margin:'2px 6px'}));
            return;
        }
        var cellPromoted = origin && origin.proposal===prop && origin.stage===stage;
        var key = cellKey(country, camp, stage, prop, selectedDate);
        var sw = ui.Label('', {backgroundColor:colorFor(stage, prop), ...STYLE.swatch});
        row.add(sw);
        if(cellPromoted){
            var viewCb = ui.Checkbox({label:'✅', value:!!viewChecks[key], style:STYLE.gridCell});
            viewCb.onChange(function(v){toggleViewPromoted(country, camp, stage, prop, selectedDate, key, v);});
            row.add(viewCb);
            return;
        }
        var cb = ui.Checkbox({label:'', value:!!selectedCells[key], style:STYLE.gridCell});
        cellCheckboxes[key] = cb;
        cb.onChange(function(v){setCellSelected(country, camp, stage, prop, selectedDate, v);});
        row.add(cb);
    });
    grid.add(row);
    return grid;
}

function toggleViewPromoted(country, camp, stage, proposal, period, key, v){
    viewChecks[key] = v;
    if(v){
        loadMosaic(country, period);
        var img = ee.Image(prePublicPath(country, camp)+period).select('probability').selfMask();
        manageMapLayer('promoted_'+key, img, {min:0,max:1000,palette:[colorFor(stage, proposal)]}, 'Promovido '+period+' | '+proposal+' | '+stage+' | '+camp+' | '+country);
    } else {
        removeMapLayer('promoted_'+key);
    }
}

function computeActionMode(){
    if(!selectedDate)return 'promote';
    // Data selecionada ja promovida => sempre Despromover (para poder trocar/reverter)
    return promotedKeysForDate(selectedDate).length>0 ? 'unpromote' : 'promote';
}

function updatePromoteSummary(){
    if(!promoteBox)return;
    var sel = Object.keys(selectedCells).filter(function(k){return selectedCells[k];});
    actionMode = computeActionMode();

    promoteBox.clear();
    if(actionMode==='unpromote'){
        var pk = promotedKeysForDate(selectedDate);
        promoteBox.add(ui.Label(L.target+': PRE_PUBLIC/{campana}', {fontSize:'11px',fontFamily:'monospace',color:'#d32f2f',margin:'2px',padding:'4px',backgroundColor:'#fff',borderRadius:'3px'}));
        promoteBox.add(ui.Label(L.unpromote+': '+pk.length+' imagen'+(pk.length===1?'':'es')+' ('+selectedDate+')', pk.length>0?STYLE.statusOk:STYLE.statusErr));
    } else {
        promoteBox.add(ui.Label(L.target+': PRE_PUBLIC/{campana}', {fontSize:'11px',fontFamily:'monospace',color:'#1a73e8',margin:'2px',padding:'4px',backgroundColor:'#fff',borderRadius:'3px'}));
        promoteBox.add(ui.Label(L.selected+': '+sel.length+' imagen'+(sel.length===1?'':'es'), sel.length>0?STYLE.statusOk:STYLE.statusErr));
    }
    if(promoteButton){
        promoteButton.setLabel(actionMode==='unpromote' ? L.unpromote : L.promote_btn);
        promoteButton.style().set('color', actionMode==='unpromote' ? '#d32f2f' : '#0f9d58');
        if(actionMode==='unpromote'){
            promoteButton.setDisabled(promotedKeysForDate(selectedDate).length===0);
        } else {
            promoteButton.setDisabled(sel.length===0);
        }
    }
}

// ─── UNPROMOTE (via botao principal, data selecionada) ──────────────────────

function showUnpromoteForDate(){
    promoteBox.clear();
    var keys = promotedKeysForDate(selectedDate);
    if(keys.length===0){
        promoteBox.add(ui.Label(L.none_selected,STYLE.statusErr));
        return;
    }
    var box = ui.Panel({layout:ui.Panel.Layout.flow('vertical'),style:STYLE.prePopup});
    box.add(ui.Label('⚠ '+L.unpromote_confirm,{fontSize:'13px',fontWeight:'bold',color:'#cc8800',margin:'2px'}));
    box.add(ui.Label(selectedDate, {fontSize:'11px',fontWeight:'bold',fontFamily:'monospace',color:'#1a73e8',margin:'2px',padding:'4px',backgroundColor:'#fff'}));
    keys.forEach(function(k){
        var parts = k.split('__');
        box.add(ui.Label('  '+parts[0]+' | '+parts[1], {fontSize:'11px',fontFamily:'monospace',color:'#d32f2f',margin:'1px'}));
    });
    var br = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'),style:{stretch:'horizontal',margin:'4px 0 0 0'}});
    br.add(ui.Button({label:L.cancel,style:STYLE.btnGray,onClick:function(){updatePromoteSummary();}}));
    br.add(ui.Button({label:L.ok,style:STYLE.btnRed,onClick:function(){doUnpromote(keys);}}));
    box.add(br);
    promoteBox.add(box);
}

function doUnpromote(checked){
    setExportStatus('loading','Despromovendo...');
    var total = 0;
    checked.forEach(function(k){
        var parts = k.split('__');
        var dest = prePublicPath(parts[0], parts[1])+parts[2];
        try{
            ee.data.deleteAsset(dest);
            print('Despromovido: '+dest);
            total++;
            if(promotedOrigin[parts[0]] && promotedOrigin[parts[0]][parts[1]]){
                delete promotedOrigin[parts[0]][parts[1]][parts[2]];
            }
        }catch(e){
            print('Error: '+dest+' -> '+e);
        }
    });
    setExportStatus('success',L.unpromote_done+' ('+total+')');
    clearCandidateLayers();
    loadAll();
}

// ─── PRE-POPUP ──────────────────────────────────────────────────────────────

function showPrePopup(){
    promoteBox.clear();
    var sel = Object.keys(selectedCells).filter(function(k){return selectedCells[k];});
    if(sel.length===0){promoteBox.add(ui.Label(L.none_selected,STYLE.statusErr));return;}

    var box = ui.Panel({layout:ui.Panel.Layout.flow('vertical'),style:STYLE.prePopup});
    box.add(ui.Label('⚠ '+L.pre_title,{fontSize:'13px',fontWeight:'bold',color:'#cc8800',margin:'2px'}));
    box.add(ui.Label(L.pre_body,{fontSize:'11px',color:'#333',margin:'2px'}));
    sel.forEach(function(k){
        var parts = k.split('__');
        box.add(ui.Label('  '+parts[4]+'  |  '+parts[0]+'  |  '+parts[1]+'  |  '+parts[2]+'  |  '+parts[3],{fontSize:'11px',fontFamily:'monospace',color:'#1a73e8',margin:'1px'}));
    });
    box.add(ui.Label(L.pre_warn,{fontSize:'10px',color:'#888',margin:'4px 2px'}));
    var br = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'),style:{stretch:'horizontal',margin:'4px 0 0 0'}});
    br.add(ui.Button({label:L.cancel,style:STYLE.btnGray,onClick:function(){updatePromoteSummary();}}));
    br.add(ui.Button({label:L.ok,style:STYLE.btnGreen,onClick:function(){doPromote(sel);}}));
    box.add(br);
    promoteBox.add(box);
}

// ─── CREATE PRE_PUBLIC STRUCTURE (only here, inside a button) ───────────────

function getAssetInfo(path){
    try{return ee.data.getAsset(path);}catch(e){return null;}
}

function ensurePrePublicStructure(country, camp){
    var root = classificationsRoot(country, camp);
    var base = root+'PRE_PUBLIC';
    var info = getAssetInfo(base);
    var t = (info && info.type) ? String(info.type).toLowerCase() : '';

    if(info===null){
        ee.data.createAsset({type:'FOLDER'}, base);
    } else if(t !== 'folder'){
        // Legado da v2.0: raiz criada como IMAGE_COLLECTION (nao permite aninhar
        // sub-colecoes). Recria como FOLDER somente se estiver vazia. Se contiver
        // assets (dados reais), aborta pedindo reestruturacao manual.
        var listing = ee.data.listAssets(base, {});
        if(!listing || typeof listing.assets === 'undefined'){
            setExportStatus('error','Nao foi possivel inspecionar PRE_PUBLIC. Recrie como FOLDER manualmente no GEE.');
            return false;
        }
        var hasAssets = listing.assets.length>0;
        if(!hasAssets){
            ee.data.deleteAsset(base);
            ee.data.createAsset({type:'FOLDER'}, base);
            print('ℹ️ PRE_PUBLIC recriado como FOLDER (era '+info.type+').');
        } else {
            setExportStatus('error','PRE_PUBLIC e '+info.type+' e contem assets. Reestruture para FOLDER manualmente.');
            return false;
        }
    }

    var camppath = base + '/' + camp.toLowerCase();
    if(getAssetInfo(camppath)===null){
        ee.data.createAsset({type:'IMAGE_COLLECTION'}, camppath);
    }
    return true;
}

// ─── PROMOTE ────────────────────────────────────────────────────────────────

function doPromote(cells){
    promoteBox.clear();
    promoteBox.add(ui.Label(L.loading,{fontSize:'11px',color:'#1a73e8',margin:'4px'}));
    setExportStatus('loading','Promovendo '+cells.length+' imagen'+(cells.length===1?'':'es')+'...');
    showLoading();

    var total = 0;
    cells.forEach(function(k){
        var parts = k.split('__');
        var c = parts[0], camp = parts[1], stage = parts[2], prop = parts[3], period = parts[4];
        if(!ensurePrePublicStructure(c, camp))return;
        var src = classificationsRoot(c, camp)+'FILTERED/'+prop+'/'+stage+'/'+period;
        var dest = prePublicPath(c, camp)+period;
        try{ee.data.getAsset(dest);setExportStatus('info','Ja existe: '+period+' ('+camp+')');}
        catch(e){
            total++;
            print('Copiando: '+src+' -> '+dest);
            ee.data.copyAsset(src,dest);
            // Fonte de verdade em memoria: garante o ✅ na sessao, independente da persistencia
            if(!promotedOrigin[c])promotedOrigin[c] = {};
            if(!promotedOrigin[c][camp])promotedOrigin[c][camp] = {};
            promotedOrigin[c][camp][period] = {proposal:prop, stage:stage};
            // Grava metadados de origem na imagem promovida (persistencia entre execucoes)
            var originProps = {
                'source_proposal': prop,
                'source_stage': stage,
                'source_country': c,
                'source_campaign': camp,
                'promoted_date': new Date().toISOString().split('T')[0]
            };
            var ok = false;
            if(typeof ee.data.setAssetProperties === 'function'){
                try{
                    ee.data.setAssetProperties(dest, originProps);
                    print('Metadados de origem gravados (setAssetProperties): '+dest+' -> '+prop+'/'+stage);
                    ok = true;
                }catch(se){
                    print('Aviso setAssetProperties: '+dest+' -> '+se);
                }
            }
            if(!ok){
                try{
                    ee.data.updateAsset(dest, {properties: originProps}, ['properties']);
                    print('Metadados de origem gravados (updateAsset): '+dest+' -> '+prop+'/'+stage);
                }catch(ue){
                    print('Aviso: nao foi possivel gravar metadados de origem em '+dest+': '+ue);
                }
            }
        }
    });

    promoteBox.clear();
    promoteBox.add(ui.Label(L.done+' ('+total+' imagen'+(total===1?'':'es')+')',{fontSize:'12px',color:'#0f9d58',fontWeight:'bold',margin:'4px'}));
    hideLoading();
    loadAll();
}

// ─── SECTIONS ───────────────────────────────────────────────────────────────

function buildConfigSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.config});
    section.add(ui.Label(L.cfg, STYLE.sectionTitle));

    section.add(ui.Label(L.country, STYLE.label));
    countryBox = ui.Panel({layout:ui.Panel.Layout.flow('horizontal')});
    COUNTRY_ORDER.forEach(function(c, idx){
        var cb = ui.Checkbox({label:c, value:idx===0, style:STYLE.gridCell});
        enabledCountries[c] = idx===0;
        cb.onChange(function(v){
            enabledCountries[c]=v;
            selectedDate=null;
            updateRegionLayers();
            loadAll();
        });
        countryBox.add(cb);
    });
    section.add(countryBox);

    section.add(ui.Label(L.campaign, STYLE.label));
    campaignBox = ui.Panel({layout:ui.Panel.Layout.flow('horizontal')});
    CAMPAIGNS.forEach(function(c, idx){
        var cb = ui.Checkbox({label:c, value:idx===0, style:STYLE.gridCell});
        enabledCampaigns[c] = idx===0;
        cb.onChange(function(v){enabledCampaigns[c]=v;selectedDate=null;loadAll();});
        campaignBox.add(cb);
    });
    section.add(campaignBox);

    dateBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical')});
    section.add(dateBox);
    renderDateGrid();
    return section;
}

function buildCandidatesSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.candidates});
    section.add(ui.Label(L.candidates, STYLE.sectionTitle));
    candidatesBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical')});
    section.add(candidatesBox);
    return section;
}

function buildPromoteSection(){
    var section = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:SECTION_STYLE.promote});
    section.add(ui.Label(L.promote, STYLE.sectionTitle));

    var note = ui.Label('ℹ️ '+L.promote_note, {fontSize:'10px',color:'#888',margin:'0 0 4px 0'});
    section.add(note);
    var link = ui.Label(L.stats_link, {fontSize:'10px',color:'#1a73e8',margin:'0 0 4px 18px',textDecoration:'underline'});
    link.setUrl(STATS_URL);
    section.add(link);

    promoteBox = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:STYLE.summaryCard});
    section.add(promoteBox);

    var actionsRow = ui.Panel({layout:ui.Panel.Layout.flow('horizontal'), style:{stretch:'horizontal',margin:'4px 0'}});
    promoteButton = ui.Button({label:L.promote_btn, style:STYLE.btnGreen, disabled:true, onClick:function(){
        if(actionMode==='unpromote'){showUnpromoteForDate();}
        else{showPrePopup();}
    }});
    actionsRow.add(promoteButton);
    section.add(actionsRow);
    updatePromoteSummary();
    return section;
}

// ─── FORM ───────────────────────────────────────────────────────────────────

function buildForm(){
    ui.root.clear();

    var sidePanel = ui.Panel({
        layout: ui.Panel.Layout.flow('vertical'),
        style: {width: '560px', margin: '0', padding: '6px', backgroundColor: '#fff'}
    });

    sidePanel.add(ui.Label('MapBiomas-Fuego | '+L.title,{fontSize:'14px',fontWeight:'bold',color:'#d32f2f',margin:'4px'}));
    statusLabel = ui.Label({value:'', style:{fontSize:'10px',color:'#1a73e8',margin:'2px 4px',shown:false,stretch:'horizontal'}});
    sidePanel.add(statusLabel);

    sidePanel.add(buildConfigSection());
    sidePanel.add(buildCandidatesSection());

    exportStatus = ui.Panel({layout:ui.Panel.Layout.flow('vertical'), style:{margin:'4px',shown:false}});
    sidePanel.add(exportStatus);

    sidePanel.add(buildPromoteSection());

    mapWidget.setOptions('SATELLITE');
    updateRegionLayers();

    var splitPanel = ui.SplitPanel({
        firstPanel: sidePanel,
        secondPanel: mapWidget,
        orientation: 'horizontal',
        wipe: false
    });

    ui.root.add(splitPanel);

    loadAll();
}

// ─── INIT ───────────────────────────────────────────────────────────────────

buildForm();
print('M9_00 5.5 carregado.');