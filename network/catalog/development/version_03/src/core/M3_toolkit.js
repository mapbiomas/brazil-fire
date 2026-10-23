// MapBiomas Fire | M3 Sample Toolkit | v08
// Multi-country sample collection, rescue and export
/*
 * Purpose: territory campaign selector with layer presets + consolidated result drawers.
 * Runtime: Google Earth Engine Code Editor.
 * Scope: Peru, Brazil, Bolivia, Colombia, Chile, Paraguay, Venezuela, Indonesia and DRC;
 *        historical sample rescue; Asset + GCS export.
 * Status: development version.
 * Based on: M3_Toolkit_Monitor_Fuego_v07.
 */

/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var geometry = /* color: #98ff00 */ee.Geometry.MultiPoint(),
    fire = /* color: #ff0000 */ee.Geometry.MultiPoint(),
    notFire = /* color: #0000ff */ee.Geometry.MultiPoint();
/***** End of imports. If edited, may not auto-convert in the playground. *****/
/********************************************
v10: Exclusive Tool for Sample Collection and Export (Vectors)
- Ultra-Compact UI and Margins (0~1px)
- Dynamic Multi-Country (Chile, Peru...)
- Automatic region search via .aggregate_array().distinct().sort()
- Dynamic update of Destination/Source Assets
********************************************/

// =============================================================================
// 01. APPLICATION CONFIGURATION
// =============================================================================
var PERSONAL_TASK_FLAG = 'CATALOG';
var CATALOG_ROOT = 'projects/mapbiomas-peru/assets/FIRE/CATALOG_01';

var APP_CONFIG = {
    language: 'es',
    startYear: 1985,
    monthlyStartYear: 2019,
    defaultCountry: 'Peru'
};

// =============================================================================
// 02. LANGUAGE DICTIONARY
// =============================================================================
// Add or edit translations here; interface logic should use L keys only.
var L = (function () {
    var dict = {
        es: {
            title: 'MapBiomas Fuego',
            cab_config: 'TERRITORIO',
            cab_temporal: 'TEMPORAL',
            cab_layers: 'CAPAS',
            cab_samples: 'MUESTRAS',
            lbl_country: 'Pais',
            lbl_regions: 'Regiones',
            lbl_periodicity: 'Periodicidad',
            opt_annual: 'Anual',
            opt_monthly: 'Mensual',
            lbl_asset_mosaics: 'Mosaicos classificados',
            lbl_raw_mosaics: 'Mosaicos on the fly',
            lbl_ref_cats: 'Categorias de Referencia',
            lbl_burned: 'Area Quemada',
            lbl_hotspots: 'Focos / Buffers',
            cab_instructions: 'Como usar este modulo',
            instructions: '1. Use las herramientas de dibujo en el mapa.\n' +
                '2. Capa "fire" (rojo) = area quemada.\n' +
                '3. Capa "notFire" (azul) = area no quemada.\n' +
                '4. Complete los metadatos en la pestana EXPORTAR.\n' +
                '5. Haga clic en EXPORTAR para guardar en Asset y GCS.\n' +
                '6. Vaya a la pestana Tasks en GEE para ejecutar.',
            tab_import: 'IMPORTAR',
            tab_export: 'EXPORTAR',
            btn_load: 'Cargar',
            lbl_version: 'Sample id:',
            lbl_shortname: 'Nombre corto (opcional):',
            lbl_comment: 'Comentario (metadato):',
            lbl_date: 'Fecha representada:',
            lbl_satellites: 'Valido para satelites (opcional):',
            placeholder_date: 'Seleccionar fecha...',
            placeholder_asset: 'Elegir muestra existente...',
            placeholder_sn: 'apodo (ej: norte_amazonia)',
            placeholder_cmt: 'comentario (metadato libre)',
            btn_export: 'EXPORTAR (Asset + GCS)',
            ver_checking: 'Verificando...',
            ver_suggested: 'sugerida: ',
            ver_empty: 'asset vacio',
            task_created: 'Tareas creadas: ',
            task_hint: ' (vaya a la pestana Tasks para ejecutar)',
            err_layers: 'Faltan capas "fire" o "notFire" en el mapa.',
            err_no_sat: 'Seleccione al menos un satelite como metadato.',
            err_no_date: 'Seleccione la fecha representada.',
            lbl_stats: 'Resumen de Muestras:',
            lbl_refresh_stats: 'Actualizar',
            status_no_samples: 'Sin muestras dibujadas',
            lbl_fire: 'Fuego',
            lbl_not_fire: 'No Fuego',
            lbl_total: 'Total',
            lbl_area: 'Area (ha)',
            lbl_count: 'Cant.',
            lbl_satellites_toggle: 'Satelites',
            lbl_vis_preset: 'Visualizacion',
            opt_rgb_fire: '[SWIR1/NIR/RED] A',
            opt_rgb_coverage: '[SWIR1/NIR/RED] B',
            opt_rgb_false: '[NIR/SWIR1/RED]',
            opt_rgb_natural: '[RED/GREEN/BLUE]',
            opt_gray_nir: 'NIR',
            opt_gray_swir1: 'SWIR1',
            opt_gray_swir2: 'SWIR2',
            opt_gray_nbr: 'NBR',
            opt_gray_ndvi: 'NDVI',
            opt_rgb_swir_alt: '[SWIR2/SWIR1/NIR]',
            cab_satellites: 'Satelite',
            cab_reference: 'Referencia',
            cab_classifications: 'Clasificaciones',
            cab_class_regional: 'Regionales',
            cab_class_filtered: 'FILTERED',
            cab_class_candidates: 'CANDIDATES',
            cab_class_prepublic: 'PRE_PUBLIC',
            lbl_proposal: 'Propuesta',
            lbl_stage: 'Etapa',
            lbl_show: 'Mostrar',
            lbl_images: 'imagenes',
            status_no_results: 'Sin resultados',
            short_guide: 'Guia',
            short_import: 'Importar',
            short_export: 'Exportar',
            btn_fire: 'Fuego',
            btn_not_fire: 'No Fuego',
            btn_hand: 'Mover',
            btn_delete: 'Borrar',
            btn_center: 'Centrar',
            loading: '',
            ref_name_modis_ba: 'MODIS Burned Area',
            ref_name_gabam: 'GABAM',
            ref_name_firms: 'FIRMS',
            ref_name_fire_cci: 'FIRE_CCI',
            ref_cicatrizes_peru: 'Cicatrizes Peru',
            ref_name_buffer_inpe: 'Buffer Focos INPE',
            ref_name_hotspots_inpe: 'Hotspots INPE (Puntos)',
            lbl_all_regions: 'Todas las Regiones',
            lbl_selected_region: 'Region Seleccionada',
            status_no_models_region: '(sin modelos en la region)',
            lbl_qty_pct: 'Cant%',
            lbl_area_pct: 'Area%',
            lbl_class_col: 'CLASE',
            btn_all_regions: 'Todos',
            btn_sel_single: 'Única',
            btn_sel_multi: 'Varias',
            placeholder_region_filter: 'filtre regiones (ej. amazonia -02)',
            placeholder_date_filter: 'filtre fechas (ej. 2023 -01)',
            ref_modis_mcd64a1: 'MODIS MCD64A1',
            ref_buffer_focos: 'Buffer Focos',
            ref_hotspots_inpe: 'Hotspots INPE',
            sat_sentinel2_buffer: 'Sentinel2 Buffer',
            lbl_bands: 'Bandas',
            lbl_band_classification: 'Clasificacion',
            lbl_band_probability: 'Probabilidad',
            lbl_band_dayofyear: 'Dia del Ano',
            lbl_dashboard: 'Panel',
            placeholder_campaign: 'Seleccionar campana...',
            lbl_campaign: 'Campana',
            lbl_select_campaign: 'Seleccionar Campana'
        },
        pt: {
            title: 'MapBiomas Fogo',
            cab_config: 'TERRITÓRIO',
            cab_temporal: 'TEMPORAL',
            cab_layers: 'CAMADAS',
            cab_samples: 'AMOSTRAS',
            lbl_country: 'Pais',
            lbl_regions: 'Regioes',
            lbl_periodicity: 'Periodicidade',
            opt_annual: 'Anual',
            opt_monthly: 'Mensal',
            lbl_asset_mosaics: 'Mosaicos classificados',
            lbl_raw_mosaics: 'Mosaicos on the fly',
            lbl_ref_cats: 'Categorias de Referencia',
            lbl_burned: 'Area Queimada',
            lbl_hotspots: 'Focos / Buffers',
            cab_instructions: 'Como usar este modulo',
            instructions: '1. Use as ferramentas de desenho no mapa.\n' +
                '2. Camada "fire" (vermelho) = area queimada.\n' +
                '3. Camada "notFire" (azul) = area nao queimada.\n' +
                '4. Preencha os metadados na aba EXPORTAR.\n' +
                '5. Clique em EXPORTAR para salvar no Asset e GCS.\n' +
                '6. Acesse a aba Tasks no GEE para executar.',
            tab_import: 'IMPORTAR',
            tab_export: 'EXPORTAR',
            btn_load: 'Carregar',
            lbl_version: 'Sample id:',
            lbl_shortname: 'Shortname (opcional):',
            lbl_comment: 'Comentario (metadado):',
            lbl_date: 'Data representada:',
            lbl_satellites: 'Valido para os satelites (opcional):',
            placeholder_date: 'Selecionar data...',
            placeholder_asset: 'Escolher amostra existente...',
            placeholder_sn: 'apelido (ex: norte_amazonia)',
            placeholder_cmt: 'comentario (metadado livre)',
            btn_export: 'EXPORTAR (Asset + GCS)',
            ver_checking: 'Verificando...',
            ver_suggested: 'sugerida: ',
            ver_empty: 'asset vazio',
            task_created: 'Tasks criadas: ',
            task_hint: ' (va a aba Tasks para executar)',
            err_layers: 'Faltam camadas "fire" ou "notFire" no mapa.',
            err_no_sat: 'Selecione ao menos um satelite como metadato.',
            err_no_date: 'Selecione a data representada.',
            lbl_stats: 'Resumo das Amostras:',
            lbl_refresh_stats: 'Atualizar',
            status_no_samples: 'Sem amostras desenhadas',
            lbl_fire: 'Fogo',
            lbl_not_fire: 'Nao Fogo',
            lbl_total: 'Total',
            lbl_area: 'Area (ha)',
            lbl_count: 'Qtd.',
            lbl_satellites_toggle: 'Satelites',
            lbl_vis_preset: 'Visualizacao',
            opt_rgb_fire: '[SWIR1/NIR/RED] A',
            opt_rgb_coverage: '[SWIR1/NIR/RED] B',
            opt_rgb_false: '[NIR/SWIR1/RED]',
            opt_rgb_natural: '[RED/GREEN/BLUE]',
            opt_gray_nir: 'NIR',
            opt_gray_swir1: 'SWIR1',
            opt_gray_swir2: 'SWIR2',
            opt_gray_nbr: 'NBR',
            opt_gray_ndvi: 'NDVI',
            opt_rgb_swir_alt: '[SWIR2/SWIR1/NIR]',
            cab_satellites: 'Satelite',
            cab_reference: 'Referencia',
            cab_classifications: 'Classificacoes',
            cab_class_regional: 'Regionais',
            cab_class_filtered: 'FILTERED',
            cab_class_candidates: 'CANDIDATES',
            cab_class_prepublic: 'PRE_PUBLIC',
            lbl_proposal: 'Proposta',
            lbl_stage: 'Etapa',
            lbl_show: 'Mostrar',
            lbl_images: 'imagens',
            status_no_results: 'Sem resultados',
            short_guide: 'Guia',
            short_import: 'Importar',
            short_export: 'Exportar',
            btn_fire: 'Fogo',
            btn_not_fire: 'Nao Fogo',
            btn_hand: 'Mover',
            btn_delete: 'Deletar',
            btn_center: 'Centralizar',
            loading: '',
            ref_name_modis_ba: 'MODIS Burned Area',
            ref_name_gabam: 'GABAM',
            ref_name_firms: 'FIRMS',
            ref_name_fire_cci: 'FIRE_CCI',
            ref_cicatrizes_peru: 'Cicatrizes Peru',
            ref_name_buffer_inpe: 'Buffer Focos INPE',
            ref_name_hotspots_inpe: 'Hotspots INPE (Pontos)',
            lbl_all_regions: 'Todas as Regioes',
            lbl_selected_region: 'Regiao Selecionada',
            status_no_models_region: '(sem modelos na regiao)',
            lbl_qty_pct: 'Qtd%',
            lbl_area_pct: 'Area%',
            lbl_class_col: 'CLASSE',
            btn_all_regions: 'Todos',
            btn_sel_single: 'Única',
            btn_sel_multi: 'Várias',
            placeholder_region_filter: 'filtre regiões (ex.: amazonia -02)',
            placeholder_date_filter: 'filtre datas (ex.: 2023 -01)',
            ref_modis_mcd64a1: 'MODIS MCD64A1',
            ref_buffer_focos: 'Buffer Focos',
            ref_hotspots_inpe: 'Hotspots INPE',
            sat_sentinel2_buffer: 'Sentinel2 Buffer',
            lbl_bands: 'Bandas',
            lbl_band_classification: 'Classificacao',
            lbl_band_probability: 'Probabilidade',
            lbl_band_dayofyear: 'Dia do Ano',
            lbl_dashboard: 'Painel',
            placeholder_campaign: 'Selecionar campanha...',
            lbl_campaign: 'Campanha',
            lbl_select_campaign: 'Selecionar Campanha'
        },
        en: {
            title: 'MapBiomas Fire',
            cab_config: 'TERRITORY',
            cab_temporal: 'TEMPORAL',
            cab_layers: 'LAYERS',
            cab_samples: 'SAMPLES',
            lbl_country: 'Country',
            lbl_regions: 'Regions',
            lbl_periodicity: 'Periodicity',
            opt_annual: 'Annual',
            opt_monthly: 'Monthly',
            lbl_asset_mosaics: 'Mosaicos classificados',
            lbl_raw_mosaics: 'Mosaicos on the fly',
            lbl_ref_cats: 'Reference Categories',
            lbl_burned: 'Burned Area',
            lbl_hotspots: 'Hotspots / Buffers',
            cab_instructions: 'How to use this module',
            instructions: '1. Use the map drawing tools.\n' +
                '2. Layer "fire" (red) = burned area.\n' +
                '3. Layer "notFire" (blue) = unburned area.\n' +
                '4. Fill in metadata in the EXPORT tab.\n' +
                '5. Click EXPORT to save to Asset and GCS.\n' +
                '6. Go to the Tasks tab in GEE to run.',
            tab_import: 'IMPORT',
            tab_export: 'EXPORT',
            btn_load: 'Load',
            lbl_version: 'Sample id:',
            lbl_shortname: 'Short name (optional):',
            lbl_comment: 'Comment (metadata):',
            lbl_date: 'Represented date:',
            lbl_satellites: 'Valid for satellites (optional):',
            placeholder_date: 'Select date...',
            placeholder_asset: 'Choose existing sample...',
            placeholder_sn: 'alias (e.g. north_amazonia)',
            placeholder_cmt: 'comment (free metadata)',
            btn_export: 'EXPORT (Asset + GCS)',
            ver_checking: 'Checking...',
            ver_suggested: 'suggested: ',
            ver_empty: 'empty asset',
            task_created: 'Tasks created: ',
            task_hint: ' (go to Tasks tab to run)',
            err_layers: 'Missing "fire" or "notFire" layers on map.',
            err_no_sat: 'Select at least one satellite as metadata.',
            err_no_date: 'Select the represented date.',
            lbl_stats: 'Sample Summary:',
            lbl_refresh_stats: 'Refresh',
            status_no_samples: 'No drawn samples',
            lbl_fire: 'Fire',
            lbl_not_fire: 'Not Fire',
            lbl_total: 'Total',
            lbl_area: 'Area (ha)',
            lbl_count: 'Qty.',
            lbl_satellites_toggle: 'Satellites',
            lbl_vis_preset: 'Visualization',
            opt_rgb_fire: '[SWIR1/NIR/RED] A',
            opt_rgb_coverage: '[SWIR1/NIR/RED] B',
            opt_rgb_false: '[NIR/SWIR1/RED]',
            opt_rgb_natural: '[RED/GREEN/BLUE]',
            opt_gray_nir: 'NIR',
            opt_gray_swir1: 'SWIR1',
            opt_gray_swir2: 'SWIR2',
            opt_gray_nbr: 'NBR',
            opt_gray_ndvi: 'NDVI',
            opt_rgb_swir_alt: '[SWIR2/SWIR1/NIR]',
            cab_satellites: 'Satellite',
            cab_reference: 'Reference',
            cab_classifications: 'Classifications',
            cab_class_regional: 'Regional',
            cab_class_filtered: 'FILTERED',
            cab_class_candidates: 'CANDIDATES',
            cab_class_prepublic: 'PRE_PUBLIC',
            lbl_proposal: 'Proposal',
            lbl_stage: 'Stage',
            lbl_show: 'Show',
            lbl_images: 'images',
            status_no_results: 'No results',
            short_guide: 'Guide',
            short_import: 'Import',
            short_export: 'Export',
            btn_fire: 'Fire',
            btn_not_fire: 'Not Fire',
            btn_hand: 'Move',
            btn_delete: 'Delete',
            btn_center: 'Center',
            loading: '',
            ref_name_modis_ba: 'MODIS Burned Area',
            ref_name_gabam: 'GABAM',
            ref_name_firms: 'FIRMS',
            ref_name_fire_cci: 'FIRE_CCI',
            ref_cicatrizes_peru: 'Peru Fire Scars',
            ref_name_buffer_inpe: 'INPE Hotspot Buffer',
            ref_name_hotspots_inpe: 'INPE Hotspots (Points)',
            lbl_all_regions: 'All Regions',
            lbl_selected_region: 'Selected Region',
            status_no_models_region: '(no models in region)',
            lbl_qty_pct: 'Qty%',
            lbl_area_pct: 'Area%',
            lbl_class_col: 'CLASS',
            btn_all_regions: 'All',
            btn_sel_single: 'Single',
            btn_sel_multi: 'Multiple',
            placeholder_region_filter: 'filter regions (e.g. amazonia -02)',
            placeholder_date_filter: 'filter dates (e.g. 2023 -01)',
            ref_modis_mcd64a1: 'MODIS MCD64A1',
            ref_buffer_focos: 'Buffer Focos',
            ref_hotspots_inpe: 'Hotspots INPE',
            sat_sentinel2_buffer: 'Sentinel2 Buffer',
            lbl_bands: 'Bands',
            lbl_band_classification: 'Classification',
            lbl_band_probability: 'Probability',
            lbl_band_dayofyear: 'Day of Year',
            lbl_dashboard: 'Dashboard',
            placeholder_campaign: 'Select campaign...',
            lbl_campaign: 'Campaign',
            lbl_select_campaign: 'Select Campaign'
        }
    };
    return dict[APP_CONFIG.language] || dict['es'];
})();

// =============================================================================
// 03. UI ICONS
// =============================================================================
// Using base64 strings for simple, self-contained UI icons
var ICONS = {
    load: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAdUlEQVR4nGNgGAWjYBSMglEwCkbBSAMMDA0Nf//+Zf7//z8DIyNjOxBnoig8DMUuKPLBUPyGog8NixMQp4EIFIDiNBB9vBSNdMAwNDTEAwMDA6Dof0SGYuC/BCh+Q5EvAMVpIPrpUDTSAEPDQ0M8MNAAoPANAOfdJAmhH1mMAAAAAElFTkSuQmCC',
    download: "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABgAAAAYCAYAAADgdz34AAAACXBIWXMAAAsTAAALEwEAmpwYAAAAgklEQVR4nO2UOQqAMBQF5xpWFhY2FlpYeXuDHkLQwkKwEI8QCViI+5IgaAYeBAJvyA8J/IkMkGOECYGcxQoWSCv4nqBaKd1KeUcQA92J8h5I7p4iAtqdcrUX8hAfqFfKGyBAEy5QTMrV/XhoxgHy8VdVa4t+xIWHJQ8iTAtSAwN4iQH5pnQ8NtSv6QAAAABJRU5ErkJggg==",

};

Map.setOptions('SATELLITE');

// =============================================================================
// 04. COUNTRY, CAMPAIGN AND SAMPLE SOURCE CATALOG
// =============================================================================
var countryConfigs = {
    'Peru': {
        asset_regions: 'projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1',
        // Monitor campaigns keep their own auxiliary folder; collection
        // campaigns share the standard country regions (M7 ft01 default).
        getAssetRegions: function(campaign) {
            if (campaign === 'COLLECTION_03') return countryConfigs['Peru'].asset_regions;
            return CATALOG_ROOT + '/' + campaign + '/AUXILIARY/regiones_fuego_peru_v1';
        },
        countryCode: 'peru',
        getImagesRoot: function() { return CATALOG_ROOT + '/LIBRARY_IMAGES'; },
        bucket: 'mapbiomas-fire',
        property: 'region_nam',
        campaigns: ['MONITOR_01', 'MONITOR_DEV', 'COLLECTION_03'],
        defaultCampaign: 'MONITOR_01',
        periodicityByCampaign: { MONITOR_01: 'monthly', MONITOR_DEV: 'monthly', COLLECTION_03: 'yearly' },
        getAssetSamples: function(campaign) { return CATALOG_ROOT + '/' + campaign + '/LIBRARY_SAMPLES'; },
        getGcsSamples: function(campaign) { return 'sudamerica/peru/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getRegionalFolder: function(campaign) { return CATALOG_ROOT + '/' + campaign + '/LIBRARY_CLASSIFICATIONS/REGIONAL'; },
        // Latest official collection published under projects/mapbiomas-public.
        mbfireCollection: {
            asset: 'projects/mapbiomas-public/assets/peru/fire/collection2/mapbiomas_peru_fire_collection2_annual_burned_v1',
            bandPrefix: 'burned_area_'
        },
        mbfireMonitor: {
            asset: 'projects/mapbiomas-public/assets/peru/fire/monitor/mapbiomas_fire_monthly_burned_v1'
        },
        historicalSampleFolders: [
            { label: 'COLLECTION1', path: 'projects/mapbiomas-peru/assets/FIRE/COLLECTION1/SAMPLES' },
            { label: 'COLLECTION2', path: 'projects/mapbiomas-peru/assets/FIRE/COLLECTION2/SAMPLES' }
        ]
    },
    'Brazil': {
        asset_regions: ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2')
          .map(function(f){return f.set('region_name', f.getString('bioma').cat(' ').cat(f.get('region')))}),
        getAssetRegions: function(campaign) { return countryConfigs['Brazil'].asset_regions; },
        countryCode: 'brazil',
        getImagesRoot: function() { return 'projects/mapbiomas-brazil/assets/FIRE/CATALOG_01/LIBRARY_IMAGES'; },
        bucket: 'mapbiomas-fire',
        property: 'region_name',
        campaigns: ['COLLECTION_06', 'COLLECTION_06_DEV', 'MONITOR_01'],
        defaultCampaign: 'COLLECTION_06',
        periodicityByCampaign: { COLLECTION_06: 'yearly', COLLECTION_06_DEV: 'yearly', MONITOR_01: 'monthly' },
        getAssetSamples: function(campaign) { return 'projects/mapbiomas-brazil/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getGcsSamples: function(campaign) { return 'sudamerica/brazil/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getRegionalFolder: function(campaign) { return 'projects/mapbiomas-brazil/assets/FIRE/' + campaign + '/LIBRARY_CLASSIFICATIONS/REGIONAL'; },
        // collection5_1 is declared in the monitor catalog but not published yet,
        // so collection4_1 is the latest one actually available in mapbiomas-public.
        mbfireCollection: {
            asset: 'projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_v1',
            bandPrefix: 'burned_area_'
        },
        mbfireMonitor: {
            asset: 'projects/mapbiomas-public/assets/brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1'
        },
        historicalSampleFolders: [
            { label: 'AMOSTRAS_COLECAO1', path: 'projects/mapbiomas-workspace/FOGO/AMOSTRAS_COLECAO1' },
            { label: 'AMOSTRAS_COLECAO2', path: 'projects/mapbiomas-workspace/FOGO/AMOSTRAS_COLECAO2' },
            { label: 'AMOSTRAS_COLECAO3', path: 'projects/mapbiomas-workspace/FOGO/AMOSTRAS_COLECAO3' },
            { label: 'AMOSTRAS_COLECAO4', path: 'projects/mapbiomas-workspace/FOGO/AMOSTRAS_COLECAO4' }
        ]
    },
    'Bolivia': {
        asset_regions: 'projects/mapbiomas-bolivia/assets/FIRE/CATALOG_01/COLLECTION_02/AUXILIARY/regiones_fuego_bolivia_v1',
        getAssetRegions: function(campaign) { return countryConfigs['Bolivia'].asset_regions; },
        countryCode: 'bolivia',
        getImagesRoot: function() { return 'projects/mapbiomas-bolivia/assets/FIRE/CATALOG_01/LIBRARY_IMAGES'; },
        bucket: 'mapbiomas-fire',
        property: 'region_nam',
        campaigns: ['COLLECTION_02', 'MONITOR_01', 'MONITOR_DEV'],
        defaultCampaign: 'COLLECTION_02',
        periodicityByCampaign: { COLLECTION_02: 'yearly', MONITOR_01: 'monthly', MONITOR_DEV: 'monthly' },
        getAssetSamples: function(campaign) { return 'projects/mapbiomas-bolivia/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getGcsSamples: function(campaign) { return 'sudamerica/bolivia/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getRegionalFolder: function(campaign) { return 'projects/mapbiomas-bolivia/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_CLASSIFICATIONS/REGIONAL'; },
        mbfireCollection: {
            asset: 'projects/mapbiomas-public/assets/bolivia/fire/collection1/mapbiomas_bolivia_fire_collection1_annual_burned_v1',
            bandPrefix: 'burned_area_'
        },
        historicalSampleFolders: [
            { label: 'COLLECTION1', path: 'projects/mapbiomas-bolivia/assets/FIRE/COLLECTION1/SAMPLES' }
        ]
    },
    'Colombia': {
        asset_regions: 'projects/mapbiomas-colombia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_colombia_v1',
        getAssetRegions: function(campaign) { return countryConfigs['Colombia'].asset_regions; },
        countryCode: 'colombia',
        getImagesRoot: function() { return 'projects/mapbiomas-colombia/assets/FIRE/CATALOG_01/LIBRARY_IMAGES'; },
        bucket: 'mapbiomas-fire',
        property: 'region_nam',
        campaigns: ['MONITOR_01', 'COLLECTION_02'],
        defaultCampaign: 'MONITOR_01',
        periodicityByCampaign: { MONITOR_01: 'monthly', COLLECTION_02: 'yearly' },
        getAssetSamples: function(campaign) { return 'projects/mapbiomas-colombia/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getGcsSamples: function(campaign) { return 'sudamerica/colombia/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getRegionalFolder: function(campaign) { return 'projects/mapbiomas-colombia/assets/FIRE/' + campaign + '/LIBRARY_CLASSIFICATIONS/REGIONAL'; },
        mbfireCollection: {
            asset: 'projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_annual_burned_v1',
            bandPrefix: 'burned_area_'
        },
        historicalSampleFolders: [
            { label: 'COLLECTION1', path: 'projects/mapbiomas-colombia/assets/FIRE/COLLECTION1/SAMPLES' }
        ]
    },
    'Chile': {
        asset_regions: 'projects/mapbiomas-chile/assets/FIRE/AUXILIARY_DATA/regiones_fuego_chile_v1',
        getAssetRegions: function(campaign) { return countryConfigs['Chile'].asset_regions; },
        countryCode: 'chile',
        getImagesRoot: function() { return 'projects/mapbiomas-chile/assets/FIRE/CATALOG_01/LIBRARY_IMAGES'; },
        bucket: 'mapbiomas-fire',
        property: 'region_nam',
        campaigns: ['COLLECTION_02', 'MONITOR_01'],
        defaultCampaign: 'MONITOR_01',
        periodicityByCampaign: { COLLECTION_02: 'yearly', MONITOR_01: 'monthly' },
        getAssetSamples: function(campaign) { return 'projects/mapbiomas-chile/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getGcsSamples: function(campaign) { return 'sudamerica/chile/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getRegionalFolder: function(campaign) { return 'projects/mapbiomas-chile/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_CLASSIFICATIONS/REGIONAL'; },
        mbfireCollection: {
            asset: 'projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_annual_burned_v1',
            bandPrefix: 'burned_area_'
        },
        historicalSampleFolders: [
            { label: 'COLLECTION1', path: 'projects/mapbiomas-chile/assets/FIRE/COLLECTION1/SAMPLES' }
        ]
    },
    'Paraguay': {
        asset_regions: 'projects/mapbiomas-paraguay/assets/FIRE/AUXILIARY_DATA/regiones_fuego_paraguay_v1',
        getAssetRegions: function(campaign) { return countryConfigs['Paraguay'].asset_regions; },
        countryCode: 'paraguay',
        getImagesRoot: function() { return 'projects/mapbiomas-paraguay/assets/FIRE/CATALOG_01/LIBRARY_IMAGES'; },
        bucket: 'mapbiomas-fire',
        property: 'region_nam',
        campaigns: ['COLLECTION_03', 'MONITOR_01'],
        defaultCampaign: 'MONITOR_01',
        periodicityByCampaign: { COLLECTION_03: 'yearly', MONITOR_01: 'monthly' },
        getAssetSamples: function(campaign) { return 'projects/mapbiomas-paraguay/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getGcsSamples: function(campaign) { return 'sudamerica/paraguay/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getRegionalFolder: function(campaign) { return 'projects/mapbiomas-paraguay/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_CLASSIFICATIONS/REGIONAL'; },
        mbfireCollection: {
            asset: 'projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_annual_burned_v1',
            bandPrefix: 'burned_area_'
        },
        historicalSampleFolders: [
            { label: 'COLLECTION1', path: 'projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/SAMPLES' },
            { label: 'COLLECTION2', path: 'projects/mapbiomas-paraguay/assets/FIRE/COLLECTION2/SAMPLES' }
        ]
    },
    'Venezuela': {
        asset_regions: 'projects/mapbiomas-venezuela/assets/FIRE/AUXILIARY_DATA/regiones_fuego_venezuela_v1',
        getAssetRegions: function(campaign) { return countryConfigs['Venezuela'].asset_regions; },
        countryCode: 'venezuela',
        getImagesRoot: function() { return 'projects/mapbiomas-venezuela/assets/FIRE/CATALOG_01/LIBRARY_IMAGES'; },
        bucket: 'mapbiomas-fire',
        property: 'region_nam',
        campaigns: ['COLLECTION_02', 'MONITOR_01'],
        defaultCampaign: 'MONITOR_01',
        periodicityByCampaign: { COLLECTION_02: 'yearly', MONITOR_01: 'monthly' },
        getAssetSamples: function(campaign) { return 'projects/mapbiomas-venezuela/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getGcsSamples: function(campaign) { return 'sudamerica/venezuela/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getRegionalFolder: function(campaign) { return 'projects/mapbiomas-venezuela/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_CLASSIFICATIONS/REGIONAL'; },
        historicalSampleFolders: [
            { label: 'COLLECTION1', path: 'projects/mapbiomas-venezuela/assets/FIRE/COLLECTION1/SAMPLES' }
        ]
    },
    'Indonesia': {
        asset_regions: 'projects/mapbiomas-indonesia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_indonesia_v1',
        getAssetRegions: function(campaign) { return countryConfigs['Indonesia'].asset_regions; },
        countryCode: 'indonesia',
        getImagesRoot: function() { return 'projects/mapbiomas-indonesia/assets/FIRE/CATALOG_01/LIBRARY_IMAGES'; },
        bucket: 'mapbiomas-fire',
        property: 'region_nam',
        campaigns: ['COLLECTION_03', 'MONITOR_01'],
        defaultCampaign: 'MONITOR_01',
        periodicityByCampaign: { COLLECTION_03: 'yearly', MONITOR_01: 'monthly' },
        getAssetSamples: function(campaign) { return 'projects/mapbiomas-indonesia/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getGcsSamples: function(campaign) { return 'asia/indonesia/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getRegionalFolder: function(campaign) { return 'projects/mapbiomas-indonesia/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_CLASSIFICATIONS/REGIONAL'; },
        mbfireCollection: {
            asset: 'projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_annual_burned_v1',
            bandPrefix: 'burned_area_'
        },
        historicalSampleFolders: []
    },
    'DRC': {
        asset_regions: 'projects/mapbiomas-drc/assets/FIRE/AUXILIARY_DATA/regiones_fuego_drc_v1',
        getAssetRegions: function(campaign) { return countryConfigs['DRC'].asset_regions; },
        countryCode: 'drc',
        getImagesRoot: function() { return 'projects/mapbiomas-drc/assets/FIRE/CATALOG_01/LIBRARY_IMAGES'; },
        bucket: 'mapbiomas-fire',
        property: 'region_nam',
        campaigns: ['COLLECTION_01'],
        defaultCampaign: 'COLLECTION_01',
        periodicityByCampaign: { COLLECTION_01: 'yearly' },
        getAssetSamples: function(campaign) { return 'projects/mapbiomas-drc/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getGcsSamples: function(campaign) { return 'africa/drc/CATALOG_01/' + campaign + '/LIBRARY_SAMPLES'; },
        getRegionalFolder: function(campaign) { return 'projects/mapbiomas-drc/assets/FIRE/CATALOG_01/' + campaign + '/LIBRARY_CLASSIFICATIONS/REGIONAL'; },
        historicalSampleFolders: []
    }
};

// Builds the official mosaic layer catalog from the country's image root.
// Keeps all supported sensors and avoids hardcoded paths per country.
function buildMosaicLayers(conf) {
    var base = conf.getImagesRoot();
    var prefix = 'image_' + conf.countryCode + '_fire_';
    var defs = [
        ['sentinel2', 'Sentinel2', 'SENTINEL2', 'sentinel2', false],
        ['sentinel2_buffer', 'Sentinel2 Buffer', 'SENTINEL2', 'sentinel2', true],
        ['landsat', 'Landsat', 'LANDSAT', 'landsat', false],
        ['landsat_buffer', 'Landsat Buffer', 'LANDSAT', 'landsat', true],
        ['hls', 'HLS', 'HLS', 'hls', false],
        ['hls_buffer', 'HLS Buffer', 'HLS', 'hls', true],
        ['modis', 'MODIS', 'MODIS', 'modis', false],
        ['modis_buffer', 'MODIS Buffer', 'MODIS', 'modis', true]
    ];
    var layers = {};
    defs.forEach(function(d) {
        layers[d[0]] = {
            label: d[1],
            sensor: d[2],
            base: base,
            prefix: prefix + d[3],
            buffer: d[4]
        };
    });
    return layers;
}

Object.keys(countryConfigs).forEach(function(c) {
    countryConfigs[c].mosaicLayers = buildMosaicLayers(countryConfigs[c]);
});

// Suggested layer/periodicity presets applied when the territory campaign changes.
var CAMPAIGN_PRESETS = {
    Peru: {
        MONITOR_01: { periodicity: 'monthly', assets: ['sentinel2'] },
        MONITOR_DEV: { periodicity: 'monthly', assets: ['sentinel2'] },
        COLLECTION_03: { periodicity: 'yearly', assets: ['landsat'] }
    },
    Brazil: {
        COLLECTION_06: { periodicity: 'yearly', assets: ['landsat'] },
        COLLECTION_06_DEV: { periodicity: 'yearly', assets: ['landsat'] },
        MONITOR_01: { periodicity: 'monthly', assets: ['sentinel2'] }
    },
    Bolivia: {
        COLLECTION_02: { periodicity: 'yearly', assets: ['landsat'] },
        MONITOR_01: { periodicity: 'monthly', assets: ['sentinel2'] },
        MONITOR_DEV: { periodicity: 'monthly', assets: ['sentinel2'] }
    },
    Colombia: {
        MONITOR_01: { periodicity: 'monthly', assets: ['sentinel2'] },
        COLLECTION_02: { periodicity: 'yearly', assets: ['landsat'] }
    },
    Chile: {
        COLLECTION_02: { periodicity: 'yearly', assets: ['landsat'] },
        MONITOR_01: { periodicity: 'monthly', assets: ['sentinel2'] }
    },
    Paraguay: {
        COLLECTION_03: { periodicity: 'yearly', assets: ['landsat'] },
        MONITOR_01: { periodicity: 'monthly', assets: ['sentinel2'] }
    },
    Venezuela: {
        COLLECTION_02: { periodicity: 'yearly', assets: ['landsat'] },
        MONITOR_01: { periodicity: 'monthly', assets: ['sentinel2'] }
    },
    Indonesia: {
        COLLECTION_03: { periodicity: 'yearly', assets: ['landsat'] },
        MONITOR_01: { periodicity: 'monthly', assets: ['sentinel2'] }
    },
    DRC: {
        COLLECTION_01: { periodicity: 'yearly', assets: ['landsat'] }
    }
};

// Returns a safe preset for a country/campaign pair.
function getCampaignPreset(countryName, campaign) {
    var conf = countryConfigs[countryName];
    var byCountry = CAMPAIGN_PRESETS[countryName] || {};
    var preset = byCountry[campaign] || {};
    var mode = preset.periodicity
        || (conf.periodicityByCampaign && conf.periodicityByCampaign[campaign])
        || conf.defaultPeriodicity
        || 'monthly';
    var assets = (preset.assets && preset.assets.length) ? preset.assets : ['sentinel2'];
    return { periodicity: mode === 'yearly' ? 'yearly' : 'monthly', assets: assets };
}

// =============================================================================
// 05. ASSET HELPERS AND RUNTIME STATE
// =============================================================================
// Active regions collection initialized from APP_CONFIG and updated by the UI.
var current_regiones = ee.FeatureCollection(
    countryConfigs[APP_CONFIG.defaultCountry].getAssetRegions
        ? countryConfigs[APP_CONFIG.defaultCountry].getAssetRegions(countryConfigs[APP_CONFIG.defaultCountry].defaultCampaign)
        : countryConfigs[APP_CONFIG.defaultCountry].asset_regions
);
var regionNamesCache = {};
var regionNamesPending = {};
var assetLayersCache = {};
var assetPresetPending = null;
var pendingOps = 0;
var lab_activity = null;

// Builds the fire emoji string client-side (avoids EE intercepting '🔥'.repeat).
function fireEmojis(n) {
    var s = '';
    for (var i = 0; i < n; i++) s += '🔥';
    return s;
}

// Shows one fire emoji next to the title for each in-flight server operation.
function trackOpStart() {
    pendingOps++;
    if (lab_activity) {
        lab_activity.setValue(fireEmojis(pendingOps));
        lab_activity.style().set('shown', true);
    }
}

// Removes one fire emoji as an operation returns.
function trackOpEnd() {
    if (pendingOps > 0) pendingOps--;
    if (lab_activity) {
        lab_activity.setValue(fireEmojis(pendingOps));
        lab_activity.style().set('shown', pendingOps > 0);
    }
}

// Wraps an async callback so the activity indicator follows each in-flight operation.
function trackEvaluate(fn) {
    trackOpStart();
    return function () {
        trackOpEnd();
        return fn.apply(null, arguments);
    };
}

var ASYNC_STATE = {
    regions: { token: 0 },
    classifications: { token: 0 },
    results: { token: 0 },
    samples: { token: 0 },
    import: { token: 0 },
    stats: { token: 0 },
    assets: { token: 0 },
    version: { token: 0 }
};

// Starts an operation and invalidates any previous callback in its domain.
function beginAsync(operation) {
    operation.token += 1;
    return operation.token;
}

// Returns false when a callback belongs to an obsolete request.
function isCurrentAsync(operation, token) {
    return operation.token === token;
}

// Creates missing folders below the existing Earth Engine asset namespace.
function ensureAssetFolder(path, callback) {
    var parts = path.split('/');
    var namespace = parts.slice(0, 4).join('/');
    var current = namespace;
    var index = 4;

    function step() {
        if (index >= parts.length) {
            callback();
            return;
        }
        current += '/' + parts[index++];
        try {
            ee.data.getAsset(current, function (asset, err) {
                if (asset) {
                    step();
                    return;
                }
                try {
                    ee.data.createAsset({ type: 'FOLDER' }, current, function () { step(); });
                } catch (e) {
                    step();
                }
            });
        } catch (e) {
            try {
                ee.data.createAsset({ type: 'FOLDER' }, current, function () { step(); });
            } catch (err) {
                step();
            }
        }
    }

    step();
}

// Lists every asset below a folder, following pagination tokens.
function listAssetsAsync(parent, callback, pageToken, accumulated) {
    var params = pageToken ? { pageToken: pageToken } : {};
    var acc = accumulated || [];
    try {
        ee.data.listAssets(parent, params, function(result, error) {
            if (error) { callback(null, error); return; }
            if (!result) { callback(acc, null); return; }
            var assets = acc.concat(result.assets || []);
            if (result.nextPageToken) listAssetsAsync(parent, callback, result.nextPageToken, assets);
            else callback(assets, null);
        });
    } catch (e) {
        callback(null, e);
    }
}

// Returns sorted region names, caching one server lookup per country/campaign.
// Concurrent requests for the same country/campaign are queued until the lookup returns.
function getCountryRegionNames(countryName, campaign, callback) {
    var conf = countryConfigs[countryName];
    var selectedCampaign = campaign || conf.defaultCampaign || conf.campaigns[0];
    var cacheKey = countryName + '::' + selectedCampaign;
    if (regionNamesCache[cacheKey]) {
        callback(regionNamesCache[cacheKey]);
        return;
    }
    if (regionNamesPending[cacheKey]) {
        regionNamesPending[cacheKey].push(callback);
        return;
    }
    regionNamesPending[cacheKey] = [callback];
    var regions = conf.getAssetRegions ? conf.getAssetRegions(selectedCampaign) : conf.asset_regions;
    ee.FeatureCollection(regions)
        .aggregate_array(conf.property)
        .distinct()
        .sort()
        .evaluate(trackEvaluate(function (names) {
            regionNamesCache[cacheKey] = names || [];
            var cbs = regionNamesPending[cacheKey] || [];
            regionNamesPending[cacheKey] = null;
            cbs.forEach(function (cb) { cb(regionNamesCache[cacheKey]); });
        }));
}

// Warms the region-name cache for every country's default campaign.
function preloadAllRegionNames() {
    Object.keys(countryConfigs).forEach(function (c) {
        var conf = countryConfigs[c];
        getCountryRegionNames(c, conf.defaultCampaign || conf.campaigns[0], function () {});
    });
}

// Loads the static hotspot asset list once; the layer build must never block the UI.
function preloadHotspotAssets() {
    if (hotspotAssetsCache !== null || hotspotAssetsPending) return;
    hotspotAssetsPending = true;
    listAssetsAsync('projects/mapbiomas-fire-485203/assets/DATABASE/monthly-focus-sul-america', function (assets, err) {
        hotspotAssetsCache = err ? [] : (assets || []);
        hotspotAssetsPending = false;
    });
}

// Ensures the selected campaign and its sample library exist before export.
function ensureCampaignFolders(conf, campaign, callback) {
    var samplePath = conf.getAssetSamples(campaign);
    var libraryPath = samplePath.split('/').slice(0, -1).join('/');
    ensureAssetFolder(libraryPath, function () {
        ensureAssetFolder(samplePath, function () {
            callback();
        });
    });
}

// =============================================================================
// 06. VISUALIZATION PRESETS
// =============================================================================
var VIS_PRESETS = {
    'fire': { bands: ['swir1', 'nir', 'red'], min: 3, max: 40 },
    'coverage': { 'bands': ['swir1', 'nir', 'red'], 'gain': [8, 6, 20], 'gamma': 0.85 },
    'false': { bands: ['nir', 'swir1', 'red'], min: 3, max: 40 },
    'swir_alt': { bands: ['swir2', 'swir1', 'nir'], min: 3, max: 40 },
    'natural': { bands: ['red', 'green', 'blue'], min: 4, max: 14, gamma: 2 },
    'nir': { bands: ['nir'], min: 16, max: 48 },
    'swir1': { bands: ['swir1'], min: 16, max: 48 },
    'swir2': { bands: ['swir2'], min: 16, max: 48 },
    'nbr': { bands: ['nbr'], min: -0.1, max: 0.5 },
    'ndvi': { bands: ['ndvi'], min: 0.1, max: 0.8 }
};

var currentVisMode = 'fire';
var visAsset = VIS_PRESETS[currentVisMode];
var visRaw = VIS_PRESETS[currentVisMode];

var managedLayers = {};
var rawCheckboxes = {};

var extraDatasets = {
    'mcd64a1': {
        name: L.ref_name_modis_ba,
        vis: { min: 0, max: 1, palette: ['fc6000'] },
        build: function (start, end, bounds, year, month) {
            return ee.ImageCollection('MODIS/061/MCD64A1')
                .filterDate(start, end).filterBounds(bounds)
                .select('BurnDate').mean().gte(1).selfMask();
        }
    },
    'gabam': {
        name: L.ref_name_gabam,
        vis: { min: 0, max: 1, palette: ['b200ac'] },
        build: function (start, end, bounds, year, month) {
            return ee.ImageCollection('projects/sat-io/open-datasets/GABAM')
                .filterDate(start, end).filterBounds(bounds)
                .mosaic().selfMask();
        }
    },
    'firms': {
        name: L.ref_name_firms,
        vis: { min: 0, max: 1, palette: ['823b15'] },
        build: function (start, end, bounds, year, month) {
            return ee.ImageCollection('FIRMS')
                .filterDate(start, end).filterBounds(bounds)
                .select('T21').mosaic().selfMask();
        }
    },
    'fire_cci': {
        name: L.ref_name_fire_cci,
        vis: { min: 0, max: 1, palette: ['5149ba'] },
        build: function (start, end, bounds, year, month) {
            return ee.ImageCollection('ESA/CCI/FireCCI/5_1')
                .filterDate(start, end).filterBounds(bounds)
                .select('BurnDate').mosaic().selfMask();
        }
    },
    'peru_ref': {
        name: L.ref_cicatrizes_peru,
        vis: { min: 0, max: 1, palette: ['804000'] },
        build: function (start, end, bounds, year, month) {
            try {
                var img = ee.Image("projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/REFERENCES/cicatrizes_fuego_reference_2016_2024/cicatriz_fuego_" + year);
                return img.selfMask();
            } catch (e) {
                return ee.Image().select();
            }
        }
    },
    'buffer_inpe': {
        name: L.ref_name_buffer_inpe,
        vis: { min: 0, max: 1, palette: ['ff0000'] },
        build: function (start, end, bounds, year, month) {
            var buffer = ee.ImageCollection("projects/workspace-ipam/assets/BUFFER-DOUBLE-MONTHLY-FOCUS-OF-INPE-SULAMERICA")
                .filter(ee.Filter.eq('year', year));
            if (month) buffer = buffer.filter(ee.Filter.eq('month', month));
            return buffer.mean().selfMask();
        }
    },
    // 'buffer_inpe_inv': {
    //     name: 'Buffer Focos INPE (Invertido)',
    //     vis: { min: 0, max: 1, palette: ['000000'] },
    //     build: function (start, end, bounds, year, month) {
    //         var buffer = ee.ImageCollection("projects/workspace-ipam/assets/BUFFER-DOUBLE-MONTHLY-FOCUS-OF-INPE-SULAMERICA")
    //             .filter(ee.Filter.eq('year', year));
    //         if (month) buffer = buffer.filter(ee.Filter.eq('month', month));
    //         var bImg = buffer.mean().unmask(0);
    //         // Inverted: 1 where there is no buffer, masked where there is
    //         return ee.Image(1).updateMask(bImg.not());
    //     }
    // },
    'hotspots_inpe': {
        name: L.ref_name_hotspots_inpe,
        vis: { color: '0f03fc' },
        build: function (start, end, bounds, year, month) {
            if (!month) return ee.Image().select();
            if (hotspotAssetsCache === null) preloadHotspotAssets();
            var assets = hotspotAssetsCache || [];
            for (var i = 0; i < assets.length; i++) {
                var filename = assets[i].id.split('/').slice(-1)[0];
                if (filename.indexOf('_') > -1) {
                    var split = filename.split('_')[1].split('-');
                    if (split.length >= 2) {
                        var aYear = parseInt(split[0], 10);
                        var aMonth = parseInt(split[1], 10);
                        if (aYear === year && aMonth === month) {
                            return ee.FeatureCollection(assets[i].id).style({ color: '0f03fc', pointSize: 2 });
                        }
                    }
                }
            }
            return ee.Image().select();
        }
    }
};

// =============================================================================
// 07. CLASSIFICATION LAYERS
// =============================================================================
var REGIONAL_FOLDER = countryConfigs[APP_CONFIG.defaultCountry].getRegionalFolder(countryConfigs[APP_CONFIG.defaultCountry].defaultCampaign);

var CLASS_PALETTE = [
    '#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231',
    '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
    '#469990', '#dcbeff', '#9A6324', '#fffAC8', '#800000',
    '#aaffc3', '#808000', '#ffd8b1', '#000075', '#a9a9a9',
    '#e6beff', '#d2f53c', '#ff46b7', '#008080', '#e6beff',
    '#aa6e28', '#ffcce0', '#808080', '#ffe119', '#911eb4',
    '#46f0f0', '#f032e6', '#d2f53c', '#fabebe', '#008080',
    '#e6beff', '#aa6e28', '#fffAC8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#e6194b',
    '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4'
];

var PROB_PALETTE = ['#ffcccc', '#ff6666', '#cc0000', '#660000'];

var START_YEAR = APP_CONFIG.startYear;
var MONTHLY_START_YEAR = APP_CONFIG.monthlyStartYear || APP_CONFIG.startYear;

var classCheckboxes = {};
var classBandCheckboxes = {};
var hotspotAssetsCache = null;
var hotspotAssetsPending = false;
var resultSourceCheckboxes = {};
var resultStatusLabels = {};
var filteredRows = {};
var filteredStageColumns = ['ft00', 'ft01', 'ft02', 'ft03', 'ft04'];
var resultState = {
    filtered: {},
    prepublic: { periods: [] }
};
// One distinct color per classification layer. Colors are assigned in display
// order and kept for the whole session so the map layer always matches its
// swatch. Text stays black; the ■ label carries the color.
var classificationColorOrder = [];
var classificationColors = {};
function classificationColor(key) {
    if (!classificationColors[key]) {
        classificationColors[key] = CLASS_PALETTE[classificationColorOrder.length % CLASS_PALETTE.length];
        classificationColorOrder.push(key);
    }
    return classificationColors[key];
}

// Small colored square used as the layer legend next to the label.
function colorSwatch(color) {
    return ui.Label('\u25A0', {
        color: color,
        fontSize: '11px',
        margin: '0px 3px 0px 0px',
        padding: '0px'
    });
}

var regionCheckboxes = {};
var layerAllRegions = ui.Map.Layer(ee.Image(), {}, L.lbl_all_regions);
var layerSelectedRegion = ui.Map.Layer(ee.Image(), {}, L.lbl_selected_region);

// Layers are now dynamically managed via managedLayers and Map.layers().add/remove
Map.layers().add(layerAllRegions);
Map.layers().add(layerSelectedRegion);

// Builds the application panels and wires user interactions.
function user_interface() {
    var appReady = false;
    var regionSelectMode = 'multi';
    var layerDateMode = 'multi';
    var btt_all_regions = null;
    var btt_region_mode = null;
    var btt_center_icon = null;
    var panelFiltered = null;
    var btt_date_mode = null;
    var txt_region_filter = null;
    var txt_date_filter = null;
    var regionFilterGroups = [];
    var dateFilterGroups = [];
    var selectedCountry = 'Peru';
    var selectedCampaignCfg = null;
    var periodicityActive = { anual: false, mensal: true };

    function getDynamicDates(period) {
        var dates = [];
        var today = new Date();
        var currentYear = today.getFullYear();
        var currentMonth = today.getMonth() + 1;

        var maxYear = currentYear;
        var maxMonth = currentMonth - 1;
        if (maxMonth === 0) { maxMonth = 12; maxYear--; }

        if (period === 'anual') {
            for (var y = START_YEAR; y <= maxYear; y++) dates.push(y.toString());
        } else {
            for (var y = MONTHLY_START_YEAR; y <= maxYear; y++) {
                var mEnd = (y === maxYear) ? maxMonth : 12;
                for (var m = 1; m <= mEnd; m++) {
                    var mm = m < 10 ? '0' + m : m.toString();
                    dates.push(y + '_' + mm);
                }
            }
        }
        return dates.reverse();
    }

    // --- STYLES DICTIONARY (0~1px) ---
    var styles = {
        main_panel: { margin: '0px', padding: '1px', backgroundColor: '#ffffff', border: '1px solid #d0d0d0' },
        card: { margin: '1px', padding: '2px', border: '1px solid #e0e0e0', backgroundColor: '#fcfcfc' },
        row: { margin: '0px', padding: '0px', stretch: 'horizontal', backgroundColor: 'ffffff00' },
        title: { margin: '1px', padding: '2px', fontSize: '13px', fontWeight: 'bold', color: '#333' },
        label: { margin: '1px', padding: '0px', fontSize: '12px', color: '#555' },
        input: { margin: '1px', padding: '0px', stretch: 'horizontal' },
        btn_blue: { margin: '1px', padding: '0px', color: '#1a73e8', fontWeight: 'bold' },
        btn_green: { margin: '1px', padding: '0px', color: '#0f9d58', fontWeight: 'bold' },
        btn_red: { margin: '1px', padding: '0px', color: '#d32f2f', fontWeight: 'bold' },
        tab_active: { margin: '0px', padding: '1px', border: '1px solid #1a73e8', color: '#1a73e8', fontWeight: 'bold', backgroundColor: '#e8f0fe', stretch: 'horizontal' },
        tab_inactive: { margin: '0px', padding: '1px', border: '1px solid #d3d3d3', color: '#70757a', backgroundColor: '#f1f3f4', stretch: 'horizontal' }
    };

    var b64 = require('users/workspaceipam/packages:mapbiomas-toolkit/utils/b64');

    // --- BASE STRUCTURE ---
    var panel = ui.Panel({ layout: ui.Panel.Layout.flow('vertical'), style: styles.main_panel });
    panel.style().set({ width: '350px' });
    ui.root.insert(0, panel);

    var panel_head = ui.Panel({ layout: ui.Panel.Layout.flow('vertical'), style: styles.row });
    var panel_body = ui.Panel({ style: { margin: '0px', padding: '0px' } });
    panel.add(panel_head).add(panel_body);

    var logo = ui.Button({ imageUrl: b64.get('logo_mapbiomas_fuego'), style: { margin: '1px', padding: '0px' } });
    var title = ui.Label(L.title, { fontSize: '15px', fontWeight: 'bold', color: '#222', margin: '1px', stretch: 'horizontal' });
    lab_activity = ui.Label('', { fontSize: '13px', margin: '1px', shown: false });
    var titleRow = ui.Panel({
        layout: ui.Panel.Layout.flow('horizontal'),
        style: { margin: '0px', padding: '0px', stretch: 'horizontal', backgroundColor: 'ffffff00' }
    });
    titleRow.add(title).add(lab_activity);
    panel_head.add(logo).add(titleRow);

    var panel_control = ui.Panel({ style: styles.card });
    var panel_samples = ui.Panel({ style: styles.card });
    panel_body.add(panel_control);

    // ==========================================
    // GENERAL MAP / DRAWING FUNCTIONS
    // ==========================================
    var updateTimer = null;
    var lab_loading = ui.Label({ imageUrl: b64.get('load_gif'), style: { margin: '4px', shown: false } });

    function debouncedUpdateMosaic() {
        lab_loading.style().set('shown', true);
        if (updateTimer) {
            ui.util.clearTimeout(updateTimer);
        }
        updateTimer = ui.util.setTimeout(function () {
            updateMosaic();
            lab_loading.style().set('shown', false);
        }, 1000);
    }

    function getSelectedCountry() {
        return (typeof selectedCountry !== 'undefined' && selectedCountry) ? selectedCountry : 'Peru';
    }

    function getSelectedCampaign() {
        if (selectedCampaignCfg) {
            return selectedCampaignCfg;
        }
        if (typeof select_campaign_exp !== 'undefined' && select_campaign_exp && select_campaign_exp.getValue()) {
            return select_campaign_exp.getValue();
        }
        var conf = countryConfigs[getSelectedCountry()];
        return conf.defaultCampaign || conf.campaigns[0];
    }

    function getSelectedPeriodicity() {
        return periodicityActive['mensal'] ? 'Mensal' : 'Anual';
    }

    function getSelectedPeriods() {
        var periods = [];
        if (typeof periodCheckboxes === 'undefined') return [];
        Object.keys(periodCheckboxes).forEach(function (k) {
            if (periodCheckboxes[k].getValue()) periods.push(k);
        });
        return periods;
    }

    // Periods that actually generate map layers, respecting the date layer mode.
    function getMapPeriods() {
        var all = getSelectedPeriods();
        if (layerDateMode === 'single' && all.length > 1) return [all[all.length - 1]];
        return all;
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

    // Rebuilds visible map layers from the current temporal and layer state.
    function updateMosaic() {
        var checkedPeriods = getMapPeriods();
        var checkedVisKeys = getSelectedVis();
        var pais = getSelectedCountry().toLowerCase();

        // Base bands needed for loading
        var baseBands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'dayOfYear'];

        // Lógica de Regiões Selecionadas
        var selectedNames = getSelectedRegionNames();

        var geom = current_regiones;
        if (selectedNames.length > 0) {
            geom = current_regiones.filter(ee.Filter.inList(countryConfigs[getSelectedCountry()].property, selectedNames));
        }
        var bounds = geom.geometry().bounds();
        var spatialMask = ee.Image(0).paint(geom, 1);

        var desiredLayerIds = [];

        checkedPeriods.forEach(function (dateStr) {
            var year = parseInt(dateStr.split('_')[0], 10);
            var month = dateStr.indexOf('_') !== -1 ? parseInt(dateStr.split('_')[1], 10) : null;
            var start = month ? ee.Date.fromYMD(year, month, 1) : ee.Date.fromYMD(year, 1, 1);
            var end = month ? start.advance(1, 'month') : start.advance(1, 'year');
            var folderPeriod = month ? 'MONTHLY' : 'YEARLY';
            var periodicity = month ? 'monthly' : 'yearly';

            checkedVisKeys.forEach(function (visKey) {
                var vis = VIS_PRESETS[visKey];
                var requestedBands = vis.bands;
                var visSuffix = ' [' + visKey.toUpperCase() + ']';

                // --- GESTÃO DE ASSETS (Mosaicos Oficiais) ---
                var mosaicLayers = countryConfigs[getSelectedCountry()].mosaicLayers || {};
                Object.keys(mosaicLayers).forEach(function (s) {
                    var layer = mosaicLayers[s];
                    var layerChk = assetCheckboxes[s + '__' + folderPeriod];
                    if (layerChk && layerChk.getValue()) {
                        var layerId = 'asset_' + s + '_' + dateStr + '_' + visKey;
                        desiredLayerIds.push(layerId);

                        var mosaicType = layer.buffer ? 'MINNBR_BUFFER' : 'MINNBR';
                        var sensorFolder = layer.sensor;

                        // Skip layers whose mosaic collection is not available for this period.
                        var periodCacheKey = getSelectedCountry() + '::' + s + '::' + folderPeriod;
                        if (assetLayersCache[periodCacheKey] === false) return;

                        var assetBase = layer.base;
                        var imgAsset = ee.Image().select();

                        try {
                            baseBands.forEach(function (b) {
                                // Estrutura Corrigida: {base}/SENSOR/PERIOD/MOSAIC/band (lowercase)
                                var colId = [assetBase, sensorFolder, folderPeriod, mosaicType, b.toLowerCase()].join('/');
                                var col = ee.ImageCollection(colId);

                                // Filtra a imagem pelo nome institucional
                                var dateKey = periodicity === 'monthly' ? year + '_' + ('0' + month).slice(-2) : year;
                                var imgName = layer.prefix + '_' + mosaicType.toLowerCase() + '_' + b.toLowerCase() + '_' + dateKey;

                                // Tenta carregar a imagem (safe-load via mosaic)
                                var bandImg = col.filter(ee.Filter.eq('system:index', imgName)).mosaic();
                                
                                // Garante que a banda tenha o nome correto mesmo que esteja vazia
                                var safeImg = ee.Image(ee.Algorithms.If(
                                    bandImg.bandNames().size().gt(0),
                                    bandImg,
                                    ee.Image(0).rename(b.toLowerCase()).updateMask(0)
                                ));
                                
                                imgAsset = imgAsset.addBands(safeImg.select([0], [b.toLowerCase()]), null, true);
                            });

                            imgAsset = addIndices(imgAsset);
                            var label = 'Asset ' + layer.label + ' - ' + dateStr + visSuffix;
                            updateManagedLayer(layerId, imgAsset.select(requestedBands).updateMask(spatialMask), vis, label);
                        } catch (e) {
                            updateManagedLayer(layerId, ee.Image().select(), {}, 'Asset ' + s + ' - ' + dateStr + ' [Err]');
                        }
                    }
                });

                // --- GESTÃO DE RAW ---
                ['sentinel2', 'landsat', 'modis', 'hls', 'planet'].forEach(function (s) {
                    if (rawCheckboxes[s] && rawCheckboxes[s].getValue()) {
                        var layerId = 'raw_' + s + '_' + dateStr + '_' + visKey;
                        desiredLayerIds.push(layerId);

                        if (s === 'planet') {
                            var planetCol = ee.ImageCollection('projects/planet-nicfi/assets/basemaps/americas')
                                .filterDate(start, end).filterBounds(bounds)
                                .map(function (img) { return img.select(['B', 'G', 'R', 'N'], ['blue', 'green', 'red', 'nir']); });

                            var planetImg = planetCol.mosaic();
                            var planetVis = { "opacity": 1, "bands": ["red", "green", "blue"], "min": 125, "max": 1858, "gamma": 1 };
                            var planetBands = ["red", "green", "blue"];

                            // Fallback for Planet if SWIR is requested
                            if (requestedBands.indexOf('swir1') === -1 && requestedBands.indexOf('swir2') === -1 && requestedBands.indexOf('nbr') === -1) {
                                planetImg = addIndices(planetImg);
                                planetVis = vis;
                                planetBands = requestedBands;
                            }

                            updateManagedLayer(layerId, planetImg.select(planetBands).updateMask(spatialMask), planetVis, 'Planet - ' + dateStr + visSuffix);
                            return;
                        }

                        var rawCol = ee.ImageCollection([]);
                        var multiplier = 100;
                        if (s === 'landsat') {
                            var sensor_logic = { '1984_1998': ['L5'], '1999_2012': ['L5', 'L7'], '2013_2021': ['L7', 'L8'], '2022_2026': ['L8', 'L9'] };
                            var current_constellation = year < 1999 ? sensor_logic['1984_1998'] : year <= 2012 ? sensor_logic['1999_2012'] : year <= 2021 ? sensor_logic['2013_2021'] : sensor_logic['2022_2026'];
                            if (current_constellation.indexOf('L9') !== -1) rawCol = rawCol.merge(ee.ImageCollection("LANDSAT/LC09/C02/T1_L2").filterDate(start, end).filterBounds(bounds).map(processLS89));
                            if (current_constellation.indexOf('L8') !== -1) rawCol = rawCol.merge(ee.ImageCollection("LANDSAT/LC08/C02/T1_L2").filterDate(start, end).filterBounds(bounds).map(processLS89));
                            if (current_constellation.indexOf('L7') !== -1) rawCol = rawCol.merge(ee.ImageCollection("LANDSAT/LE07/C02/T1_L2").filterDate(start, end).filterBounds(bounds).map(processLS57));
                            if (current_constellation.indexOf('L5') !== -1) rawCol = rawCol.merge(ee.ImageCollection("LANDSAT/LT05/C02/T1_L2").filterDate(start, end).filterBounds(bounds).map(processLS57));
                        } else if (s === 'sentinel2') {
                            rawCol = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start, end).filterBounds(bounds).linkCollection(ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs']).map(processS2);
                            multiplier = 0.01;
                        } else if (s === 'modis') {
                            rawCol = ee.ImageCollection("MODIS/061/MOD09A1") // Terra
                                // .merge(ee.ImageCollection("MODIS/061/MYD09A1")) // Aqua (comentado)
                                .filterDate(start, end).filterBounds(bounds).map(processMODIS);
                            multiplier = 0.01;
                        } else if (s === 'hls') {
                            rawCol = ee.ImageCollection("NASA/HLS/HLSS30/v002").filterDate(start, end).filterBounds(bounds).map(processHLS_S30).merge(ee.ImageCollection("NASA/HLS/HLSL30/v002").filterDate(start, end).filterBounds(bounds).map(processHLS_L30));
                        }
                        var spectral = rawCol.qualityMosaic('nbr').select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).multiply(multiplier);
                        spectral = addIndices(spectral);
                        updateManagedLayer(layerId, spectral.select(requestedBands).updateMask(spatialMask), vis, 'Raw ' + s + ' - ' + dateStr + visSuffix);
                    }
                });
            });

            // --- GESTÃO DE REFERÊNCIA ---
            Object.keys(extraDatasets).forEach(function (key) {
                if (extraCheckboxes[key] && extraCheckboxes[key].getValue()) {
                    var layerId = 'extra_' + key + '_' + dateStr;
                    desiredLayerIds.push(layerId);
                    var extraImg = extraDatasets[key].build(start, end, bounds, year, month);
                    updateManagedLayer(layerId, extraImg.updateMask(spatialMask), extraDatasets[key].vis, extraDatasets[key].name + ' - ' + dateStr);
                }
            });

            // --- CLASSIFICAÇÕES REGIONAIS ---
            var classBandCls = classBandCheckboxes['cls'] && classBandCheckboxes['cls'].getValue();
            var classBandPrb = classBandCheckboxes['prb'] && classBandCheckboxes['prb'].getValue();
            var classBandDoy = classBandCheckboxes['doy'] && classBandCheckboxes['doy'].getValue();
            if (classBandCls || classBandPrb || classBandDoy) {
                var selectedRegions = getSelectedRegionNames();
                Object.keys(classCheckboxes).forEach(function (modelId) {
                    if (!classCheckboxes[modelId].getValue()) return;
                    var color = classificationColor('regional:' + modelId);
                    var col = ee.ImageCollection(REGIONAL_FOLDER + '/' + modelId);

                    // Mosaic todas as regioes selecionadas em uma unica imagem
                    var regionImages = selectedRegions.map(function (regionName) {
                        return col.filter(ee.Filter.stringContains('system:index', regionName))
                            .filter(ee.Filter.stringContains('system:index', dateStr)).mosaic();
                    });
                    var img = ee.ImageCollection(regionImages).mosaic();
                    var prbImg = img.select([0], ['probability']).divide(10).toByte().selfMask().updateMask(spatialMask);
                    var clsImg = img.select([0], ['probability']).gte(1).selfMask().updateMask(spatialMask);
                    var doyImg = img.select([1], ['dayOfYear']).selfMask().updateMask(spatialMask);

                    // Ordem inversa: doy (base) → prb → cls (topo)
                    if (classBandDoy) {
                        var lId = 'class_' + modelId + '_' + dateStr + '_doy';
                        desiredLayerIds.push(lId);
                        updateManagedLayer(lId,
                            doyImg,
                            { min: 1, max: 366, palette: ['313695', '4575b4', '74add1', 'abd9e9', 'e0f3f8', 'fee090', 'fdae61', 'f46d43', 'd73027'] },
                            modelId + ' doy ' + dateStr);
                    }
                    if (classBandPrb) {
                        var lId = 'class_' + modelId + '_' + dateStr + '_prb';
                        desiredLayerIds.push(lId);
                        updateManagedLayer(lId,
                            prbImg,
                            { min: 0, max: 100, palette: PROB_PALETTE },
                            modelId + ' prob ' + dateStr);
                    }
                    if (classBandCls) {
                        var lId = 'class_' + modelId + '_' + dateStr + '_cls';
                        desiredLayerIds.push(lId);
                        updateManagedLayer(lId,
                            clsImg,
                            { min: 0, max: 1, palette: ['00000000', color] },
                            modelId + ' ' + dateStr);
                    }
                });
            }

            // --- CONSOLIDATED RESULTS (FILTERED / PRE_PUBLIC) ---
            function drawResult(basePath, key, label, color) {
                var resultImg = ee.Image(basePath + dateStr);
                var resultColor = color || '#e6194b';
                var prbImg = resultImg.select([0], ['probability']).divide(10).toByte().selfMask().updateMask(spatialMask);
                var clsImg = resultImg.select([0], ['probability']).gte(1).selfMask().updateMask(spatialMask);
                var doyImg = resultImg.select([1], ['dayOfYear']).selfMask().updateMask(spatialMask);

                // Same three visualizations used by the regional classifications:
                // day of year (base) -> probability -> binary classification (top).
                if (classBandDoy) {
                    var lIdDoy = 'result_' + key + '_' + dateStr + '_doy';
                    desiredLayerIds.push(lIdDoy);
                    updateManagedLayer(lIdDoy,
                        doyImg,
                        { min: 1, max: 366, palette: ['313695', '4575b4', '74add1', 'abd9e9', 'e0f3f8', 'fee090', 'fdae61', 'f46d43', 'd73027'] },
                        label + ' doy');
                }
                if (classBandPrb) {
                    var lIdPrb = 'result_' + key + '_' + dateStr + '_prb';
                    desiredLayerIds.push(lIdPrb);
                    updateManagedLayer(lIdPrb,
                        prbImg,
                        { min: 0, max: 100, palette: PROB_PALETTE },
                        label + ' prob');
                }
                if (classBandCls) {
                    var lIdCls = 'result_' + key + '_' + dateStr + '_cls';
                    desiredLayerIds.push(lIdCls);
                    updateManagedLayer(lIdCls,
                        clsImg,
                        { min: 0, max: 1, palette: ['00000000', resultColor] },
                        label);
                }
            }

            Object.keys(resultState.filtered).forEach(function (key) {
                var selection = resultState.filtered[key];
                if (!selection.periods || selection.periods.indexOf(dateStr) === -1) return;
                drawResult(
                    filteredPath(selection.proposal, selection.stage),
                    'filtered_' + selection.proposal + '_' + selection.stage,
                    'filtered ' + selection.proposal + ' ' + selection.stage + ' ' + dateStr,
                    classificationColor('filtered:' + selection.proposal + ':' + selection.stage));
            });

            if (resultSourceCheckboxes.prepublic && resultSourceCheckboxes.prepublic.getValue()
                && resultState.prepublic.periods.indexOf(dateStr) !== -1) {
                drawResult(getResultPath('prepublic'), 'prepublic', 'prepublic ' + dateStr, classificationColor('prepublic'));
            }
        });

        // Remover camadas que não estão mais marcadas
        Object.keys(managedLayers).forEach(function (id) {
            if (desiredLayerIds.indexOf(id) === -1) {
                Map.layers().remove(managedLayers[id]);
                delete managedLayers[id];
            }
        });
    }

    // Lists regional classification models compatible with selected regions.
function updateClassificationsDrawer() {
        drawerClassModels.panel.clear();
        classCheckboxes = {};
        var selectedNames = getSelectedRegionNames();
        if (selectedNames.length === 0) return;

        var classificationToken = beginAsync(ASYNC_STATE.classifications);
        var labelLoading = ui.Label(L.loading, { fontSize: '10px', color: '#888' });
        drawerClassModels.panel.add(labelLoading);

        listAssetsAsync(REGIONAL_FOLDER, function (modelAssets, err) {
            if (!isCurrentAsync(ASYNC_STATE.classifications, classificationToken)) return;
            if (err || !modelAssets || modelAssets.length === 0) {
                labelLoading.setValue(L.status_no_models_region);
                return;
            }

            var pending = modelAssets.length;
            var totalAdded = 0;

            modelAssets.forEach(function (c) {
                var modelId = c.id.split('/').pop();
                var colPath = REGIONAL_FOLDER + '/' + modelId;
                listAssetsAsync(colPath, function (assets, imageErr) {
                    if (!isCurrentAsync(ASYNC_STATE.classifications, classificationToken)) return;
                    pending--;
                    if (!imageErr && assets && assets.length) {
                        var imageNames = assets.map(function (i) { return i.id.split('/').pop(); });
                        var hasData = selectedNames.some(function (r) {
                            return imageNames.some(function (img) { return img.indexOf(r) !== -1; });
                        });

                        if (hasData) {
                            totalAdded++;
                            if (totalAdded === 1) drawerClassModels.panel.remove(labelLoading);
                            var classRow = ui.Panel({
                                layout: ui.Panel.Layout.flow('horizontal'),
                                style: { margin: '1px 0px', padding: '0px', stretch: 'horizontal' }
                            });
                            var chk = ui.Checkbox({
                                label: modelId,
                                value: false,
                                onChange: function () { syncCheckBorder(chk); debouncedUpdateMosaic(); },
                                style: { margin: '0px', padding: '0px', fontSize: '10px', color: '#000' }
                            });
                            styleCheck(chk, false);
                            classRow.add(colorSwatch(classificationColor('regional:' + modelId))).add(chk);
                            classCheckboxes[modelId] = chk;
                            drawerClassModels.panel.add(classRow);
                        }
                    }
                    if (pending === 0 && totalAdded === 0) {
                        labelLoading.setValue(L.status_no_models_region);
                    }
                });
            });
        });
    }

    // Creates or updates one managed map layer without duplicating it.
    function updateManagedLayer(id, eeObject, vis, name) {
        if (managedLayers[id]) {
            managedLayers[id].setEeObject(eeObject);
            managedLayers[id].setVisParams(vis);
            managedLayers[id].setName(name);
        } else {
            var l = ui.Map.Layer(eeObject, vis, name);
            managedLayers[id] = l;
            // Insere antes das regiões (que estão no topo)
            var idx = Math.max(0, Map.layers().length() - 2);
            Map.layers().insert(idx, l);
        }
    }

    // Updates the selected-region outline and recenters the map.
    function updateSelectedRegionsMap() {
        var selectedNames = [];
        Object.keys(regionCheckboxes).forEach(function (key) {
            if (regionCheckboxes[key].getValue()) selectedNames.push(key);
        });
        if (selectedNames.length === 0) {
            layerSelectedRegion.setEeObject(ee.Image());
            return;
        }
        var selectedFc = current_regiones.filter(ee.Filter.inList(countryConfigs[getSelectedCountry()].property, selectedNames));
        layerSelectedRegion.setEeObject(selectedFc.style({ color: 'ff0000', fillColor: '00000000', width: 3 }));
        Map.centerObject(selectedFc);
    }

    // Removes current fire/not-fire drawings and recreates empty layers.
    function clearDrawingTools() {
        var layers = Map.drawingTools().layers();
        // Remove existing fire/notFire layers
        for (var i = layers.length() - 1; i >= 0; i--) {
            var l = layers.get(i);
            if (l.getName() === 'fire' || l.getName() === 'notFire') {
                layers.remove(l);
            }
        }
        // Create new empty geometry layers for collection
        Map.drawingTools().addLayer({ geometries: [], name: 'fire', color: 'ff0000', shown: true });
        Map.drawingTools().addLayer({ geometries: [], name: 'notFire', color: '0000ff', shown: true });
    }

    // Ensures the fire and not-fire drawing layers exist.
    function initDrawingTools() {
        var layers = Map.drawingTools().layers();
        var hasFire = false;
        var hasNotFire = false;
        for (var i = 0; i < layers.length(); i++) {
            var name = layers.get(i).getName();
            if (name === 'fire') hasFire = true;
            if (name === 'notFire') hasNotFire = true;
        }
        if (!hasFire) Map.drawingTools().addLayer({ geometries: [], name: 'fire', color: 'ff0000', shown: true });
        if (!hasNotFire) Map.drawingTools().addLayer({ geometries: [], name: 'notFire', color: '0000ff', shown: true });
    }

    // Centers the map on the current drawn sample geometries.
    function centerDrawingTools() {
        var fireLayer = Map.drawingTools().layers().filter(function (ly) { return ly.getName() === 'fire'; })[0];
        var notFireLayer = Map.drawingTools().layers().filter(function (ly) { return ly.getName() === 'notFire'; })[0];
        var fireGeom = fireLayer ? fireLayer.getEeObject().geometries() : ee.List([]);
        var notFireGeom = notFireLayer ? notFireLayer.getEeObject().geometries() : ee.List([]);
        var totalFc = ee.FeatureCollection(fireGeom.map(function (g) { return ee.Feature(ee.Geometry(g)); }))
            .merge(ee.FeatureCollection(notFireGeom.map(function (g) { return ee.Feature(ee.Geometry(g)); })));

        totalFc.size().evaluate(trackEvaluate(function (size) {
            if (size > 0) Map.centerObject(totalFc);
        }));
    }

    // --- Sample Stats Panel ---

    // --- Sample Stats Panel ---
    var panel_stats = ui.Panel({
        style: { margin: '4px 0px', padding: '4px', border: '1px solid #e0e0e0', backgroundColor: '#f9f9f9' }
    });

    // Recalculates counts and area for the current drawn samples.
    var statsTimer = null;

    function updateStats() {
        var statsToken = beginAsync(ASYNC_STATE.stats);
        panel_stats.clear();
        panel_stats.add(ui.Label(L.lbl_stats, { fontSize: '11px', fontWeight: 'bold', margin: '2px 0px' }));

        var layers = Map.drawingTools().layers();
        var fireLayer = null;
        var notFireLayer = null;
        for (var i = 0; i < layers.length(); i++) {
            var ly = layers.get(i);
            if (ly.getName() === 'fire') fireLayer = ly;
            if (ly.getName() === 'notFire') notFireLayer = ly;
        }

        if (!fireLayer || !notFireLayer) {
            panel_stats.add(ui.Label(L.status_no_samples, { fontSize: '10px', color: '#888', margin: '2px 0px' }));
            return;
        }

        var fireGeoms = fireLayer.getEeObject().geometries();
        var notFireGeoms = notFireLayer.getEeObject().geometries();

        var stats = ee.Dictionary({
            'fire_count': fireGeoms.length(),
            'fire_area': ee.Number(fireGeoms.map(function (g) { return ee.Geometry(g).area(); }).reduce(ee.Reducer.sum())).divide(10000),
            'notFire_count': notFireGeoms.length(),
            'notFire_area': ee.Number(notFireGeoms.map(function (g) { return ee.Geometry(g).area(); }).reduce(ee.Reducer.sum())).divide(10000)
        });

        panel_stats.add(ui.Label({ imageUrl: b64.get('load_gif'), style: { margin: '4px' } }));

        stats.evaluate(trackEvaluate(function (s, err) {
            if (!isCurrentAsync(ASYNC_STATE.stats, statsToken)) return;
            if (err) {
                panel_stats.clear();
                panel_stats.add(ui.Label('Error recalculating stats', { fontSize: '10px', color: 'red' }));
                return;
            }
            if (!s) return;
            panel_stats.clear();
            panel_stats.add(ui.Label(L.lbl_stats, { fontSize: '11px', fontWeight: 'bold', margin: '2px 0px' }));

            var totalCount = s.fire_count + s.notFire_count;
            if (totalCount === 0) {
                panel_stats.add(ui.Label(L.status_no_samples, { fontSize: '10px', color: '#888', margin: '2px 0px' }));
                return;
            }
            var totalArea = s.fire_area + s.notFire_area;

            function makeRow(label, count, area, color, isHeader) {
                var pctArea = totalArea > 0 ? (area / totalArea * 100).toFixed(1) : '0.0';
                var pctCount = totalCount > 0 ? (count / totalCount * 100).toFixed(1) : '0.0';
                var fSize = '9px';
                var fw = isHeader ? 'bold' : 'normal';

                return ui.Panel({
                    layout: ui.Panel.Layout.flow('horizontal'),
                    style: { margin: '0px', padding: '0px', backgroundColor: 'ffffff00' },
                    widgets: [
                        ui.Label(label, { width: '55px', color: color, fontSize: fSize, fontWeight: 'bold' }),
                        ui.Label(isHeader ? count : count.toString(), { width: '30px', textAlign: 'right', fontSize: fSize, fontWeight: fw }),
                        ui.Label(isHeader ? L.lbl_qty_pct : pctCount + '%', { width: '40px', textAlign: 'right', color: '#888', fontSize: fSize, fontWeight: fw }),
                        ui.Label(isHeader ? area : area.toFixed(1), { width: '60px', textAlign: 'right', fontSize: fSize, fontWeight: fw }),
                        ui.Label(isHeader ? L.lbl_area_pct : pctArea + '%', { width: '40px', textAlign: 'right', color: '#888', fontSize: fSize, fontWeight: fw })
                    ]
                });
            }

            panel_stats.add(makeRow(L.lbl_class_col, L.lbl_count, L.lbl_area, '#333', true));
            panel_stats.add(makeRow(L.lbl_fire, s.fire_count, s.fire_area, '#d32f2f', false));
            panel_stats.add(makeRow(L.lbl_not_fire, s.notFire_count, s.notFire_area, '#1a73e8', false));
            panel_stats.add(ui.Label('──────────────────────────────', { margin: '0px', color: '#ccc' }));
            panel_stats.add(makeRow(L.lbl_total, totalCount, totalArea, '#333', false));
        }));
    }

    // Debounces stats recalculation while the user is drawing or editing.
    function scheduleStatsUpdate() {
        if (statsTimer) ui.util.clearTimeout(statsTimer);
        statsTimer = ui.util.setTimeout(function () {
            statsTimer = null;
            updateStats();
        }, 350);
    }

    Map.drawingTools().onEdit(scheduleStatsUpdate);
    Map.drawingTools().onLayerAdd(scheduleStatsUpdate);
    Map.drawingTools().onLayerRemove(scheduleStatsUpdate);

    // Styles a checkbox and keeps its border in sync with its state.
    function styleCheck(chk, initial) {
        chk.style().set('margin', '2px 6px 2px 0px');
        chk.style().set('padding', '0px');
        chk.style().set('fontSize', '10px');
        if (initial !== undefined) chk.setValue(!!initial, false);
        syncCheckBorder(chk);
    }

    // Outlines the checkbox when checked, transparent when unchecked.
    function syncCheckBorder(chk) {
        var active = chk.getValue();
        chk.style().set('border', active ? '1px solid rgba(21,101,192,0.4)' : '1px solid rgba(0,0,0,0)');
    }

    // Sets a checkbox value programmatically and keeps the border in sync.
    function setCheckValue(chk, v) {
        chk.setValue(v, false);
        syncCheckBorder(chk);
    }

    // Discreet style shared by all toggle buttons; outline when active.
    function styleToggleButton(btn, active) {
        btn.style().set('margin', '0px');
        btn.style().set('padding', '0px 2px');
        btn.style().set('fontSize', '10px');
        btn.style().set('border', active ? '1px solid rgba(21,101,192,0.4)' : '1px solid rgba(0,0,0,0)');
    }

    // Switches between single ('single') and multiple ('multi') region selection.
    function setRegionSelectMode(mode) {
        regionSelectMode = mode;
        if (btt_region_mode) {
            btt_region_mode.setLabel(mode === 'single' ? L.btn_sel_single : L.btn_sel_multi);
            styleToggleButton(btt_region_mode, mode === 'multi');
        }
        if (mode === 'single') {
            var first = null;
            Object.keys(regionCheckboxes).forEach(function (k) {
                if (regionCheckboxes[k].getValue()) {
                    if (first === null) { first = k; } else { setCheckValue(regionCheckboxes[k], false); }
                }
            });
        }
        updateAllRegionsButtonState();
        updateSelectedRegionsMap();
        updateExportLabel();
        debouncedUpdateMosaic();
        updateShortnameSuggestion();
        updateClassificationsDrawer();
    }

    // Highlights the 'Todos' button when every visible (filtered) region is selected.
    function updateAllRegionsButtonState() {
        if (!btt_all_regions) return;
        var names = Object.keys(regionCheckboxes).filter(isRegionVisible);
        var allOn = names.length > 0 && names.every(function (k) { return regionCheckboxes[k].getValue(); });
        styleToggleButton(btt_all_regions, allOn);
    }

    // Toggles between showing all selected dates or only the latest one on the map.
    function toggleDateMode() {
        layerDateMode = (layerDateMode === 'single') ? 'multi' : 'single';
        if (btt_date_mode) {
            btt_date_mode.setLabel(layerDateMode === 'single' ? L.btn_sel_single : L.btn_sel_multi);
            styleToggleButton(btt_date_mode, layerDateMode === 'multi');
        }
        debouncedUpdateMosaic();
    }

    // ==========================================
    // SMART FILTER (busca com operadores implícitos)
    // ',' = OU entre blocos | ' ' = E dentro do bloco | '-palavra' = NÃO (exclusão)
    // Precedência: OU -> E -> NÃO
    // ==========================================

    // Normaliza texto: minúsculas e remove espaços duplicados.
    function smartNormalize(s) {
        return (s || '').toLowerCase().replace(/\s+/g, ' ').trim();
    }

    // Converte a query em uma lista de grupos de condições.
    // Cada grupo = { pos: [termos positivos], neg: [termos negativos] }.
    function smartParseQuery(q) {
        return smartNormalize(q).split(',').map(function (bloco) {
            var pos = [], neg = [];
            bloco.split(' ').forEach(function (termo) {
                if (termo === '') return;
                if (termo.charAt(0) === '-') neg.push(termo.slice(1));
                else pos.push(termo);
            });
            return { pos: pos, neg: neg };
        }).filter(function (g) { return g.pos.length > 0 || g.neg.length > 0; });
    }

    // Testa se o texto do item satisfaz um único grupo de condições.
    function smartMatchGroup(texto, g) {
        for (var i = 0; i < g.pos.length; i++) {
            if (texto.indexOf(g.pos[i]) === -1) return false;
        }
        for (var j = 0; j < g.neg.length; j++) {
            if (texto.indexOf(g.neg[j]) !== -1) return false;
        }
        return true;
    }

    // Testa se o item passa no filtro: passa em pelo menos um dos grupos (OU).
    function smartFilterMatch(texto, groups) {
        var t = smartNormalize(texto);
        if (groups.length === 0) return true;
        for (var i = 0; i < groups.length; i++) {
            if (smartMatchGroup(t, groups[i])) return true;
        }
        return false;
    }

    // Region filter: a region is visible when it matches the typed query.
    function isRegionVisible(name) {
        return smartFilterMatch(name, regionFilterGroups);
    }

    // Hides region checkboxes that do not match the typed filter.
    function applyRegionFilter(text) {
        regionFilterGroups = smartParseQuery(text);
        Object.keys(regionCheckboxes).forEach(function (name) {
            regionCheckboxes[name].style().set('shown', isRegionVisible(name));
        });
        updateAllRegionsButtonState();
    }

    // Hides date checkboxes that do not match the typed filter.
    function applyDateFilter(text) {
        dateFilterGroups = smartParseQuery(text);
        Object.keys(periodCheckboxes).forEach(function (d) {
            periodCheckboxes[d].style().set('shown', smartFilterMatch(d, dateFilterGroups));
        });
    }

    // ==========================================
    // LÓGICA DE TROCA DE PAÍS / CAMPANHA
    // ==========================================
    // Applies the campaign's suggested periodicity (monthly/yearly).
    function applyCampaignPeriodicity(conf, campaign, mode) {
        var selectedMode = mode || (conf.periodicityByCampaign && conf.periodicityByCampaign[campaign]) || conf.defaultPeriodicity || 'monthly';
        periodicityActive['anual'] = (selectedMode === 'yearly');
        periodicityActive['mensal'] = (selectedMode === 'monthly');
        if (periodicityButtons) {
            styleToggleButton(periodicityButtons['anual'], periodicityActive['anual']);
            styleToggleButton(periodicityButtons['mensal'], periodicityActive['mensal']);
        }
        updatePeriodDrawer();
    }

    // Selects the most recent period of the active periodicity.
    function selectLatestPeriod() {
        var keys = Object.keys(periodCheckboxes);
        if (!keys.length) return;
        keys.forEach(function (k, i) { setCheckValue(periodCheckboxes[k], i === 0); });
        updateExportLabel();
        debouncedUpdateMosaic();
    }

    // Loads a country/campaign territory and rebuilds every dependent drawer.
    function loadTerritory(countryName, campaign) {
        var conf = countryConfigs[countryName];
        var selectedCampaign = campaign || conf.defaultCampaign || conf.campaigns[0];

        current_regiones = ee.FeatureCollection(
            conf.getAssetRegions ? conf.getAssetRegions(selectedCampaign) : conf.asset_regions
        );
        if (conf.getRegionalFolder) REGIONAL_FOLDER = conf.getRegionalFolder(selectedCampaign);

        // Atualiza bordas de todas as regiões no mapa
        layerAllRegions.setEeObject(current_regiones.style({ color: 'ffffff', fillColor: '00000000', width: 1 }));
        layerSelectedRegion.setEeObject(ee.Image()); // Limpa o vermelho

        // Build the header and flow panel once; only the region checkboxes change.
        if (!drawerRegions.flowPanel) {
            drawerRegions.panel.add(ui.Label(L.lbl_regions, { fontSize: '11px', fontWeight: 'bold', margin: '1px 2px', backgroundColor: '#f9f9f9' }));

            var header = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '0px', backgroundColor: '#f9f9f9' } });

            btt_all_regions = ui.Button({
                label: L.btn_all_regions,
                style: { margin: '0px', padding: '0px 2px', fontSize: '10px', border: '1px solid rgba(0,0,0,0)' },
                onClick: function () {
                    var visible = Object.keys(regionCheckboxes).filter(isRegionVisible);
                    var anyUnchecked = visible.some(function (k) { return !regionCheckboxes[k].getValue(); });
                    if (anyUnchecked) {
                        if (regionSelectMode === 'single') setRegionSelectMode('multi');
                        visible.forEach(function (k) { setCheckValue(regionCheckboxes[k], true); });
                    } else {
                        visible.forEach(function (k) { setCheckValue(regionCheckboxes[k], false); });
                    }
                    updateAllRegionsButtonState();
                    updateSelectedRegionsMap();
                    updateExportLabel();
                    debouncedUpdateMosaic();
                    updateShortnameSuggestion();
                }
            });

            btt_region_mode = ui.Button({
                label: L.btn_sel_multi,
                style: { margin: '0px', padding: '0px 2px', fontSize: '10px', border: '1px solid rgba(21,101,192,0.4)' },
                onClick: function () { setRegionSelectMode(regionSelectMode === 'single' ? 'multi' : 'single'); }
            });

            btt_center_icon = ui.Button({
                label: L.btn_center,
                style: { margin: '0px', padding: '0px 2px', fontSize: '10px', border: '1px solid rgba(0,0,0,0)' },
                onClick: function () {
                    var selectedNames = [];
                    Object.keys(regionCheckboxes).forEach(function (key) {
                        if (regionCheckboxes[key].getValue()) selectedNames.push(key);
                    });
                    if (selectedNames.length > 0) {
                        Map.centerObject(current_regiones.filter(ee.Filter.inList(countryConfigs[getSelectedCountry()].property, selectedNames)));
                    } else {
                        Map.centerObject(current_regiones);
                    }
                }
            });

            var spacer = ui.Panel({ style: { stretch: 'horizontal' } });
            header.add(btt_all_regions).add(btt_center_icon).add(lab_loading).add(spacer).add(btt_region_mode);
            drawerRegions.panel.add(header);
            drawerRegions.flowPanel = ui.Panel({
                layout: ui.Panel.Layout.flow('horizontal', true),
                style: { margin: '0px', padding: '0px', backgroundColor: '#f9f9f9', maxHeight: '100px' }
            });
            drawerRegions.panel.add(drawerRegions.flowPanel);
        }

        drawerRegions.flowPanel.clear();
        lab_loading.style().set('shown', true);

        var regionToken = beginAsync(ASYNC_STATE.regions);
        getCountryRegionNames(countryName, selectedCampaign, function (list) {
            if (!isCurrentAsync(ASYNC_STATE.regions, regionToken)) return;
            lab_loading.style().set('shown', false);
            regionCheckboxes = {};
            list.forEach(function (name, i) {
                var chk = ui.Checkbox({
                    label: name,
                    value: i === 0, // Apenas a primeira região ligada por default
                    onChange: function () {
                        if (regionSelectMode === 'single' && chk.getValue()) {
                            Object.keys(regionCheckboxes).forEach(function (k) {
                                if (k !== name) setCheckValue(regionCheckboxes[k], false);
                            });
                        }
                        syncCheckBorder(chk);
                        updateAllRegionsButtonState();
                        updateSelectedRegionsMap();
                        updateExportLabel();
                        debouncedUpdateMosaic();
                        updateShortnameSuggestion();
                        updateClassificationsDrawer();
                    },
                    style: { margin: '1px 3px', padding: '0px', fontSize: '10px', backgroundColor: '#f9f9f9' }
                });
                styleCheck(chk, i === 0);
                regionCheckboxes[name] = chk;
                drawerRegions.flowPanel.add(chk);
            });
            applyRegionFilter(txt_region_filter ? txt_region_filter.getValue() : '');
            updateAllRegionsButtonState();
            updateSelectedRegionsMap();
            updateExportLabel();
            debouncedUpdateMosaic();
            updateShortnameSuggestion();
            updateClassificationsDrawer();
        });

        // Sincroniza as campanhas de amostragem (importação/exportação).
        var campaigns = conf.campaigns;
        select_campaign_imp.items().reset(campaigns);
        select_campaign_exp.items().reset(campaigns);
        select_campaign_imp.setValue(selectedCampaign, false);
        select_campaign_exp.setValue(selectedCampaign, false);
        selectedCampaignCfg = selectedCampaign;
        if (typeof drawerCampaign !== 'undefined' && drawerCampaign) {
            Object.keys(drawerCampaign.buttons).forEach(function (key) {
                styleToggleButton(drawerCampaign.buttons[key], key === selectedCampaign);
            });
        }
        updateImportList(conf, selectedCampaign);

        var preset = getCampaignPreset(countryName, selectedCampaign);
        var periodKey = preset.periodicity === 'yearly' ? 'YEARLY' : 'MONTHLY';
        assetPresetPending = preset.assets.map(function (id) { return id + '__' + periodKey; });
        applyCampaignPeriodicity(conf, selectedCampaign, preset.periodicity);
        selectLatestPeriod();
        updateClassificationsDrawer();
        updateAssetDrawer();
        updateMbfireDrawer();
        refreshResultSources();
    }

    // Rebuilds the campaign buttons for the active country only.
    function rebuildCampaignButtons(countryName, activeCampaign) {
        if (typeof drawerCampaign === 'undefined' || !drawerCampaign) return;
        var conf = countryConfigs[countryName];
        var active = activeCampaign || conf.defaultCampaign || conf.campaigns[0];
        drawerCampaign.flowPanel.clear();
        drawerCampaign.buttons = {};
        conf.campaigns.forEach(function (campaign) {
            var btn = ui.Button({
                label: campaign,
                style: { margin: '0px', padding: '0px 2px', fontSize: '10px', border: '1px solid rgba(0,0,0,0)' },
                onClick: function () {
                    if (campaign === getSelectedCampaign()) return;
                    setRegionCampaign(conf, campaign);
                }
            });
            styleToggleButton(btn, campaign === active);
            drawerCampaign.buttons[campaign] = btn;
            drawerCampaign.flowPanel.add(btn);
        });
    }

    // Keeps the country-level entry point used by the country buttons.
    function loadCountryData(countryName) {
        var conf = countryConfigs[countryName];
        var defaultCampaign = conf.defaultCampaign || conf.campaigns[0];
        rebuildCampaignButtons(countryName, defaultCampaign);
        loadTerritory(countryName, defaultCampaign);
    }

    // Rebuilds the territory when the active campaign changes.
    function setRegionCampaign(conf, campaign) {
        loadTerritory(getSelectedCountry(), campaign);
    }

    // Lists campaign and historical sample sources for the active country.
    function updateImportList(conf, campaign) {
        // Campanhas sao fontes nativas; historicos ficam disponiveis para resgate.
        try {
            var folders = [{ label: campaign, path: conf.getAssetSamples(campaign), native: true }];
            (conf.historicalSampleFolders || []).forEach(function (folder) {
                folders.push({ label: folder.label, path: folder.path, native: false });
            });

            select_campaign_imp.items().reset(folders.map(function (folder) {
                return { label: folder.label, value: folder.native ? folder.label : folder.path };
            }));
            select_campaign_imp.setValue(campaign, false);
            loadSampleAssets(folders[0].path);
        } catch (e) {
            select_campaign_imp.items().reset([]);
            select_address.items().reset([]);
            select_campaign_imp.setPlaceholder('Sample folder unavailable');
            select_address.setPlaceholder('No vector samples found');
        }
    }

    // Lists direct vector assets; nested folders and collections are ignored.
function loadSampleAssets(folderPath) {
        var sampleToken = beginAsync(ASYNC_STATE.samples);
        select_address.items().reset([]);
        select_address.setPlaceholder(L.loading);
        listAssetsAsync(folderPath, function (assets, err) {
            if (!isCurrentAsync(ASYNC_STATE.samples, sampleToken)) return;
            if (err) {
                select_address.setPlaceholder('Folder unavailable or has no vectors');
                return;
            }
            var items = (assets || []).filter(function (obj) {
                return obj.type === 'TABLE';
            }).map(function (obj) {
                return { label: obj.id.split('/').slice(-1)[0], value: obj.id };
            });
            select_address.items().reset(items);
            select_address.setPlaceholder(items.length > 0 ? L.placeholder_asset : 'No vector samples in folder');
            if (items.length > 0) select_address.setValue(items[0].value, false);
        });
    }

    // ==========================================
    // CARD 1: FILTROS
    // ==========================================
    function createCabinet(title, isOpen, customMargin) {
        var panel = ui.Panel({ style: { margin: customMargin || '4px 0px', padding: '2px', border: '1px solid #dcdcdc', borderRadius: '4px', backgroundColor: '#fdfdfd' } });
        var content = ui.Panel({ style: { margin: '0px', padding: '0px', backgroundColor: '#fdfdfd' } });
        var header = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '2px 0px', padding: '1px', backgroundColor: '#f0f0f0', border: '1px solid #ddd' } });

        var iconKey = isOpen ? 'arraw_v' : 'arraw_>';
        var toggleBtn = ui.Button({ imageUrl: b64.get(iconKey), style: { margin: '0px', padding: '0px', backgroundColor: 'ffffff00' } });

        var fontSize = customMargin ? '15px' : '18px';
        header.add(toggleBtn).add(ui.Label(title, { fontSize: fontSize, fontWeight: 'bold', margin: '1px 4px', color: '#444', backgroundColor: 'ffffff00' }));
        panel.add(header).add(content);
        content.style().set('shown', isOpen);
        toggleBtn.onClick(function () {
            var isShown = content.style().get('shown');
            content.style().set('shown', !isShown);
            toggleBtn.setImageUrl(!isShown ? b64.get('arraw_v') : b64.get('arraw_>'));
        });
        return { panel: panel, content: content };
    }

    function createLayerDrawer(title, options) {
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
                    syncCheckBorder(chk);
                    if (opt.onChange) opt.onChange(checked);
                    debouncedUpdateMosaic();
                },
                style: { margin: '1px 3px', padding: '0px', fontSize: '10px', backgroundColor: '#f9f9f9' }
            });
            styleCheck(chk, opt.value || false);
            checkboxes[opt.id] = chk;
            flowPanel.add(chk);
        });
        var titleLabel = ui.Label(title, { fontSize: '11px', fontWeight: 'bold', margin: '1px 2px', backgroundColor: '#f9f9f9' });
        panel.add(titleLabel).add(flowPanel);
        return { panel: panel, checkboxes: checkboxes, flowPanel: flowPanel };
    }

    function createButtonDrawer(title, options) {
        var panel = ui.Panel({ style: { margin: '2px 0px', padding: '1px', border: '1px solid #e0e0e0', backgroundColor: '#f9f9f9' } });
        var flowPanel = ui.Panel({
            layout: ui.Panel.Layout.flow('horizontal', true),
            style: { margin: '0px', padding: '0px', backgroundColor: '#f9f9f9', maxHeight: '100px', stretch: 'horizontal' }
        });
        var buttons = {};
        options.forEach(function (opt) {
            var btn = ui.Button({
                label: opt.label,
                style: { margin: '0px', padding: '0px 2px', fontSize: '10px', border: '1px solid rgba(0,0,0,0)' },
                onClick: function () {
                    if (opt.onClick) opt.onClick(opt.id, btn);
                }
            });
            styleToggleButton(btn, opt.active || false);
            buttons[opt.id] = btn;
            flowPanel.add(btn);
        });
        var titleLabel = ui.Label(title, { fontSize: '11px', fontWeight: 'bold', margin: '1px 2px', backgroundColor: '#f9f9f9' });
        panel.add(titleLabel).add(flowPanel);
        return { panel: panel, buttons: buttons, flowPanel: flowPanel };
    }

    // --- CABINET 1: CONFIGURATION ---
    var cabConfig = createCabinet(L.cab_config, false);
    panel_control.add(cabConfig.panel);

    var drawerCountry = createButtonDrawer(L.lbl_country, Object.keys(countryConfigs).map(function (c) {
        return {
            id: c, label: c, active: c === 'Peru', onClick: function (id) {
                if (id === selectedCountry) return;
                selectedCountry = id;
                Object.keys(drawerCountry.buttons).forEach(function (k) { styleToggleButton(drawerCountry.buttons[k], k === id); });
                if (appReady) loadCountryData(id);
            }
        };
    }));
    var countryButtons = drawerCountry.buttons;
    cabConfig.content.add(drawerCountry.panel);

    // Territory campaign selector: buttons for the active country only.
    var drawerCampaign = createButtonDrawer(L.lbl_campaign, []);
    cabConfig.content.add(drawerCampaign.panel);

    var drawerRegions = { panel: ui.Panel({ style: { margin: '2px 0px', padding: '2px', border: '1px solid #e0e0e0', backgroundColor: '#f9f9f9' } }), flowPanel: null };
    cabConfig.content.add(drawerRegions.panel);

    txt_region_filter = ui.Textbox({
        placeholder: L.placeholder_region_filter,
        onChange: function () { applyRegionFilter(txt_region_filter.getValue()); },
        style: { margin: '3px 2px', padding: '2px', fontSize: '10px', stretch: 'horizontal' }
    });
    cabConfig.content.add(txt_region_filter);

    // --- CABINET 2: TEMPORAL ---
    var cabTemporal = createCabinet(L.cab_temporal, false);
    panel_control.add(cabTemporal.panel);

    var drawerPeriodicity = createButtonDrawer(L.lbl_periodicity, [
        { id: 'anual', label: L.opt_annual, active: false, onClick: function (id) { setPeriodicity(id); } },
        { id: 'mensal', label: L.opt_monthly, active: true, onClick: function (id) { setPeriodicity(id); } }
    ]);
    var periodicityButtons = drawerPeriodicity.buttons;
    cabTemporal.content.add(drawerPeriodicity.panel);

    btt_date_mode = ui.Button({
        label: L.btn_sel_multi,
        style: { margin: '0px', padding: '0px 2px', fontSize: '10px', border: '1px solid rgba(0,0,0,0)' },
        onClick: function () { toggleDateMode(); }
    });
    styleToggleButton(btt_date_mode, layerDateMode === 'multi');
    var dateSpacer = ui.Panel({ style: { stretch: 'horizontal' } });
    drawerPeriodicity.flowPanel.add(dateSpacer).add(btt_date_mode);

    var drawerPeriodAnual = { panel: ui.Panel({ style: { margin: '1px 0px', padding: '1px' } }) };
    var drawerPeriodMensal = { panel: ui.Panel({ style: { margin: '1px 0px', padding: '1px' } }) };
    cabTemporal.content.add(drawerPeriodAnual.panel).add(drawerPeriodMensal.panel);

    txt_date_filter = ui.Textbox({
        placeholder: L.placeholder_date_filter,
        onChange: function () { applyDateFilter(txt_date_filter.getValue()); },
        style: { margin: '3px 2px', padding: '2px', fontSize: '10px', stretch: 'horizontal' }
    });
    cabTemporal.content.add(txt_date_filter);

    var periodCheckboxes = {};
    function setPeriodicity(id) {
        periodicityActive[id] = !periodicityActive[id];
        styleToggleButton(periodicityButtons[id], periodicityActive[id]);
        updatePeriodDrawer();
        updateAssetDrawer();
    }

    function updatePeriodDrawer() {
        drawerPeriodAnual.panel.clear(); drawerPeriodMensal.panel.clear();
        var showAnual = periodicityActive['anual'];
        var showMensal = periodicityActive['mensal'];

        periodCheckboxes = {}; // Reinicia para não acumular períodos de categorias desmarcadas

        if (showAnual) populatePeriodPanel(drawerPeriodAnual.panel, 'anual');
        if (showMensal) populatePeriodPanel(drawerPeriodMensal.panel, 'mensal');
        applyDateFilter(txt_date_filter ? txt_date_filter.getValue() : '');
        updateExportLabel(); debouncedUpdateMosaic();
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
                label: d, value: (type === 'mensal' && i === 0), // O primeiro agora é o mais recente
                onChange: function () { syncCheckBorder(chk); updateExportLabel(); debouncedUpdateMosaic(); },
                style: { margin: '1px 3px', fontSize: '10px', backgroundColor: '#f9f9f9' }
            });
            styleCheck(chk, type === 'mensal' && i === 0);
            periodCheckboxes[d] = chk;
            flow.add(chk);
        });
    }

    function getSelectedPeriods() {
        var selected = [];
        Object.keys(periodCheckboxes).forEach(function (k) {
            if (periodCheckboxes[k].getValue()) selected.push(k);
        });
        return selected;
    }

    // --- CABINET: LAYERS (Satellites + Reference) ---
    var cabLayers = createCabinet(L.cab_layers, true);
    panel_control.add(cabLayers.panel);

    var subCabSat = createCabinet(L.cab_satellites, true, '2px 2px 2px 10px');
    var subCabRef = createCabinet(L.cab_reference, false, '2px 2px 2px 10px');
    cabLayers.content.add(subCabSat.panel).add(subCabRef.panel);

    // 1. Define sub-drawers for Reference
    var extraCheckboxes = {};
    // Official MapBiomas Fire products published under projects/mapbiomas-public.
    // Both layers follow the selected period: the collection layer serves the
    // annual scar of the chosen year, the monitor layer sums the chosen months.
    var mbfireBandCache = {};

    // Reads the years actually published in the asset so the app never selects
    // a band that does not exist. Result is cached per asset.
    function mbfireYears(cfg) {
        if (!cfg || !cfg.asset) return [];
        var entry = mbfireBandCache[cfg.asset];
        if (!entry) {
            entry = { years: [] };
            mbfireBandCache[cfg.asset] = entry;
            ee.Image(cfg.asset).bandNames().evaluate(function (names, err) {
                if (err || !names) {
                    entry.years = [];
                    print('MBFIRE: no bands found for ' + cfg.asset);
                    return;
                }
                entry.years = names.map(function (name) {
                    var match = /(\d{4})$/.exec(name);
                    return match ? parseInt(match[1], 10) : null;
                }).filter(function (y) { return y !== null; });
                debouncedUpdateMosaic();
            });
        }
        return entry.years;
    }

    function mbfireConf(key) {
        var conf = countryConfigs[getSelectedCountry()] || {};
        return conf[key] || null;
    }

    function emptyBurned() {
        return ee.Image(0).rename('burned').selfMask();
    }

    extraDatasets['mbfire_collection'] = {
        name: 'MBFIRE Collection',
        vis: { min: 0, max: 1, palette: ['000000'] },
        build: function (start, end, bounds, year, month) {
            var cfg = mbfireConf('mbfireCollection');
            if (!cfg) return emptyBurned();
            var numericYear = parseInt(year, 10);
            if (mbfireYears(cfg).indexOf(numericYear) === -1) return emptyBurned();
            return ee.Image(cfg.asset)
                .select(cfg.bandPrefix + numericYear).gt(0)
                .rename('burned').selfMask();
        }
    };

    extraDatasets['mbfire_monitor'] = {
        name: 'MBFIRE Monitor',
        vis: { min: 0, max: 1, palette: ['000000'] },
        build: function (start, end, bounds, year, month) {
            var cfg = mbfireConf('mbfireMonitor');
            if (!cfg) return emptyBurned();
            var col = ee.ImageCollection(cfg.asset)
                .filter(ee.Filter.eq('year', ee.Number(year)));
            if (month) col = col.filter(ee.Filter.eq('month', ee.Number(month)));
            // No rename: an empty selection yields a bandless image, which is
            // still a valid (invisible) layer instead of a server error.
            return col.sum().gt(0).selfMask();
        }
    };

    var drawerAQ = createLayerDrawer(L.lbl_burned, [
        { id: 'mcd64a1', label: L.ref_modis_mcd64a1, value: false },
        { id: 'gabam', label: 'GABAM', value: false },
        { id: 'fire_cci', label: 'FIRE_CCI', value: false },
        { id: 'peru_ref', label: L.ref_cicatrizes_peru, value: false },
        { id: 'mbfire_collection', label: 'MBFIRE Collection', value: false },
        { id: 'mbfire_monitor', label: 'MBFIRE Monitor', value: false }
    ]);

    // Hides the MBFIRE layers when the active country has no public product and
    // clears the checkbox so no stale layer survives a country switch.
    function updateMbfireDrawer() {
        var conf = countryConfigs[getSelectedCountry()] || {};
        var chkCollection = drawerAQ.checkboxes['mbfire_collection'];
        var chkMonitor = drawerAQ.checkboxes['mbfire_monitor'];
        if (chkCollection) {
            var hasCollection = !!conf.mbfireCollection;
            chkCollection.style().set('shown', hasCollection);
            if (!hasCollection && chkCollection.getValue()) setCheckValue(chkCollection, false);
        }
        if (chkMonitor) {
            var hasMonitor = !!conf.mbfireMonitor;
            chkMonitor.style().set('shown', hasMonitor);
            if (!hasMonitor && chkMonitor.getValue()) setCheckValue(chkMonitor, false);
        }
    }

    var drawerFocos = createLayerDrawer(L.lbl_hotspots, [
        { id: 'buffer_inpe', label: L.ref_buffer_focos, value: false },
        { id: 'hotspots_inpe', label: L.ref_hotspots_inpe, value: false }
    ]);
    drawerAQ.panel.style().set('shown', true); drawerFocos.panel.style().set('shown', true);
    subCabRef.content.add(drawerAQ.panel).add(drawerFocos.panel);

    // 2. Define sub-drawers for Satellites
    var drawerVis = createLayerDrawer(L.lbl_vis_preset, [
        { id: 'fire', label: L.opt_rgb_fire, value: true },
        { id: 'coverage', label: L.opt_rgb_coverage, value: false },
        { id: 'false', label: L.opt_rgb_false, value: false },
        { id: 'swir_alt', label: L.opt_rgb_swir_alt, value: false },
        { id: 'natural', label: L.opt_rgb_natural, value: false },
        { id: 'nir', label: L.opt_gray_nir, value: false },
        { id: 'swir1', label: L.opt_gray_swir1, value: false },
        { id: 'swir2', label: L.opt_gray_swir2, value: false },
        { id: 'nbr', label: L.opt_gray_nbr, value: false },
        { id: 'ndvi', label: L.opt_gray_ndvi, value: false }
    ]);
    var visCheckboxes = drawerVis.checkboxes;

    subCabSat.content.add(drawerVis.panel);

    var drawerAsset = createLayerDrawer(L.lbl_asset_mosaics, []);
    var assetCheckboxes = drawerAsset.checkboxes;

    // Rebuilds the official mosaic drawer from the selected country's mosaicLayers,
    // showing one checkbox per (layer x period) whose mosaic collection exists
    // (checked via the 'blue' band having at least one file) and whose
    // periodicity (anual/mensal) is currently active.
    function updateAssetDrawer() {
        var conf = countryConfigs[getSelectedCountry()];
        var layers = conf.mosaicLayers || {};
        var layerIds = Object.keys(layers);
        var periods = ['MONTHLY', 'YEARLY'];
        var preset = assetPresetPending;
        var assetToken = beginAsync(ASYNC_STATE.assets);

        var previous = {};
        Object.keys(drawerAsset.checkboxes).forEach(function (k) {
            previous[k] = drawerAsset.checkboxes[k].getValue();
        });

        var missing = [];
        layerIds.forEach(function (id) {
            var layer = layers[id];
            var mosaicType = layer.buffer ? 'MINNBR_BUFFER' : 'MINNBR';
            periods.forEach(function (period) {
                var cacheKey = getSelectedCountry() + '::' + id + '::' + period;
                if (cacheKey in assetLayersCache) return;
                missing.push({
                    cacheKey: cacheKey,
                    colId: [layer.base, layer.sensor, period, mosaicType, 'blue'].join('/')
                });
            });
        });

        function renderAssetDrawer() {
            if (!isCurrentAsync(ASYNC_STATE.assets, assetToken)) return;
            drawerAsset.flowPanel.clear();
            drawerAsset.checkboxes = {};
            var anyVisible = false;
            layerIds.forEach(function (id) {
                var layer = layers[id];
                periods.forEach(function (period) {
                    var cacheKey = getSelectedCountry() + '::' + id + '::' + period;
                    var periodActive = period === 'MONTHLY' ? periodicityActive['mensal'] : periodicityActive['anual'];
                    if (assetLayersCache[cacheKey] !== true || !periodActive) return;

                    var chkId = id + '__' + period;
                    var periodLabel = period === 'MONTHLY' ? L.opt_monthly : L.opt_annual;
                    var defaultValue = preset !== null
                        ? preset.indexOf(chkId) !== -1
                        : (chkId in previous ? previous[chkId] : (chkId === 'sentinel2_buffer__MONTHLY'));
                    var chk = ui.Checkbox({
                        label: layer.label + ' (' + periodLabel + ')',
                        value: defaultValue,
                        onChange: function (checked) {
                            syncCheckBorder(chk);
                            debouncedUpdateMosaic();
                        },
                        style: { margin: '1px 3px', padding: '0px', fontSize: '10px', backgroundColor: '#f9f9f9' }
                    });
                    styleCheck(chk, defaultValue);
                    drawerAsset.checkboxes[chkId] = chk;
                    assetCheckboxes = drawerAsset.checkboxes;
                    drawerAsset.flowPanel.add(chk);
                    anyVisible = true;
                });
            });
            drawerAsset.panel.style().set('shown', anyVisible);
            assetPresetPending = null;
            if (appReady) debouncedUpdateMosaic();
        }

        if (!missing.length) {
            renderAssetDrawer();
            return;
        }

        var pending = missing.length;
        missing.forEach(function (item) {
            listAssetsAsync(item.colId, function (assets, err) {
                if (!isCurrentAsync(ASYNC_STATE.assets, assetToken)) return;
                assetLayersCache[item.cacheKey] = !err && assets && assets.length > 0;
                pending--;
                if (pending === 0) renderAssetDrawer();
            });
        });
    }

    var drawerRaw = createLayerDrawer(L.lbl_raw_mosaics, [
        { id: 'sentinel2', label: 'Sentinel2', value: false },
        { id: 'landsat', label: 'Landsat', value: false },
        { id: 'modis', label: 'MODIS', value: false },
        { id: 'hls', label: 'HLS', value: false },
        { id: 'planet', label: 'Planet', value: false }
    ]);
    rawCheckboxes = drawerRaw.checkboxes;

    subCabSat.content.add(drawerAsset.panel).add(drawerRaw.panel);

    // Mesclar os checkboxes para o gerenciamento global
    Object.keys(drawerAQ.checkboxes).forEach(function (k) { extraCheckboxes[k] = drawerAQ.checkboxes[k]; });
    Object.keys(drawerFocos.checkboxes).forEach(function (k) { extraCheckboxes[k] = drawerFocos.checkboxes[k]; });

    // --- SUB-CABINET: CLASSIFICAÇÕES (REGIONAL + CONSOLIDATED RESULTS) ---
    var subCabClass = createCabinet(L.cab_classifications, false, '2px 2px 2px 10px');
    cabLayers.content.add(subCabClass.panel);

    var drawerClassToggle = createLayerDrawer(L.lbl_bands, [
        { id: 'cls', label: L.lbl_band_classification, value: true, onChange: function () { debouncedUpdateMosaic(); } },
        { id: 'prb', label: L.lbl_band_probability, value: false, onChange: function () { debouncedUpdateMosaic(); } },
        { id: 'doy', label: L.lbl_band_dayofyear, value: false, onChange: function () { debouncedUpdateMosaic(); } }
    ]);
    classBandCheckboxes = drawerClassToggle.checkboxes;
    subCabClass.content.add(drawerClassToggle.panel);

    // A. Regional classification models (M6).
    var subCabRegional = createCabinet(L.cab_class_regional, true, '2px 2px 2px 10px');
    subCabClass.content.add(subCabRegional.panel);
    var drawerClassModels = { panel: ui.Panel({ style: { margin: '2px 0px', padding: '1px', backgroundColor: '#f9f9f9' } }) };
    subCabRegional.content.add(drawerClassModels.panel);

    // B. Consolidated results (M7 FILTERED, M9 PRE_PUBLIC).
    function classificationResultsRoot() {
        var conf = countryConfigs[getSelectedCountry()];
        return conf.getRegionalFolder(getSelectedCampaign()).replace(/\/REGIONAL\/?$/, '/');
    }

    function currentResultContext() {
        return getSelectedCountry() + '::' + getSelectedCampaign();
    }

    function setResultStatus(source, text) {
        if (resultStatusLabels[source]) resultStatusLabels[source].setValue(text);
    }

    function resultEntries(assets, allowedTypes, validateName) {
        return (assets || []).filter(function (a) {
            return allowedTypes.indexOf(a.type) !== -1;
        }).map(function (a) {
            return a.id.split('/').pop();
        }).filter(function (name) {
            return !validateName || validateName(name);
        });
    }

    function filteredPath(proposal, stage) {
        return classificationResultsRoot() + 'FILTERED/' + proposal + '/' + stage + '/';
    }

    function getResultPath(source) {
        if (source === 'prepublic') {
            return classificationResultsRoot() + 'PRE_PUBLIC/' + getSelectedCampaign().toLowerCase() + '/';
        }
        return null;
    }

    function filteredSelectionKey(proposal, stage) {
        return proposal + '::' + stage;
    }

    function updateFilteredStatus() {
        var proposalCount = Object.keys(filteredRows).length;
        var selectedCount = Object.keys(resultState.filtered).length;
        if (!proposalCount) {
            setResultStatus('filtered', L.status_no_results);
            return;
        }
        setResultStatus('filtered', L.lbl_proposal + ': ' + proposalCount + ' | ' + L.lbl_stage + ': ' + selectedCount);
    }

    function loadFilteredSelection(proposal, stage) {
        var key = filteredSelectionKey(proposal, stage);
        var ctx = currentResultContext();
        listAssetsAsync(filteredPath(proposal, stage), function (assets, err) {
            if (ctx !== currentResultContext()) return;
            var selection = resultState.filtered[key];
            if (!selection) return;
            selection.periods = err ? [] : resultEntries(assets, ['IMAGE']);
            debouncedUpdateMosaic();
        });
    }

    function toggleFilteredSelection(proposal, stage, checked) {
        var key = filteredSelectionKey(proposal, stage);
        if (checked) {
            resultState.filtered[key] = { proposal: proposal, stage: stage, periods: [] };
            loadFilteredSelection(proposal, stage);
        } else {
            delete resultState.filtered[key];
            debouncedUpdateMosaic();
        }
        updateFilteredStatus();
    }

    // Each proposal uses two lines: the name (flexible, truncated with an
    // ellipsis) and a wrapping row of stage checkboxes. Nothing has a fixed
    // pixel width, so the drawer never grows a horizontal scrollbar.
    function rebuildFilteredRow(proposal, stages) {
        var row = filteredRows[proposal];
        if (!row) return;
        row.stages = stages || [];
        row.stagesPanel.clear();
        filteredStageColumns.forEach(function (stage) {
            if (row.stages.indexOf(stage) === -1) return;
            var selected = !!resultState.filtered[filteredSelectionKey(proposal, stage)];
            var chk = ui.Checkbox({
                label: stage,
                value: selected,
                onChange: function () {
                    syncCheckBorder(chk);
                    toggleFilteredSelection(proposal, stage, chk.getValue());
                },
                style: { margin: '0px', padding: '0px', fontSize: '9px', color: '#000' }
            });
            styleCheck(chk, selected);
            var stageCell = ui.Panel({
                layout: ui.Panel.Layout.flow('horizontal'),
                style: { margin: '0px 4px 0px 0px', padding: '0px' }
            });
            stageCell.add(colorSwatch(classificationColor('filtered:' + proposal + ':' + stage))).add(chk);
            row.checkboxes[stage] = chk;
            row.stagesPanel.add(stageCell);
        });
    }

    // GEE style keys are whitelisted (no overflow/ellipsis), so long proposal
    // names are shortened in JS instead of relying on CSS truncation.
    function filteredProposalLabel(name) {
        return name.length > 30 ? name.substring(0, 29) + '...' : name;
    }

    function addFilteredRow(proposal) {
        var panel = ui.Panel({
            layout: ui.Panel.Layout.flow('vertical'),
            style: { margin: '2px 0px', padding: '0px', stretch: 'horizontal' }
        });
        var label = ui.Label(filteredProposalLabel(proposal), {
            stretch: 'horizontal',
            fontSize: '9px',
            fontWeight: 'bold',
            color: '#333',
            margin: '0px',
            padding: '0px',
            minWidth: '0',
            whiteSpace: 'nowrap'
        });
        var stagesPanel = ui.Panel({
            layout: ui.Panel.Layout.flow('horizontal', true),
            style: { margin: '1px 0px 0px 0px', padding: '0px', stretch: 'horizontal' }
        });
        panel.add(label).add(stagesPanel);
        filteredRows[proposal] = {
            panel: panel,
            label: label,
            stagesPanel: stagesPanel,
            stages: [],
            checkboxes: {}
        };
        panelFiltered.add(panel);
    }

    function refreshFilteredSources(root, ctx) {
        if (!panelFiltered) return;
        resultState.filtered = {};
        filteredRows = {};
        panelFiltered.clear();
        setResultStatus('filtered', L.loading);

        listAssetsAsync(root + 'FILTERED/', function (assets, err) {
            if (ctx !== currentResultContext()) return;
            if (err) {
                updateFilteredStatus();
                return;
            }
            var proposals = resultEntries(assets, ['FOLDER', 'IMAGE_COLLECTION']).sort();
            if (!proposals.length) {
                updateFilteredStatus();
                return;
            }
            proposals.forEach(function (proposal) {
                addFilteredRow(proposal);
            });
            var pending = proposals.length;
            proposals.forEach(function (proposal) {
                listAssetsAsync(root + 'FILTERED/' + proposal + '/', function (stageAssets, stageErr) {
                    if (ctx !== currentResultContext()) return;
                    var stages = stageErr ? [] : resultEntries(stageAssets, ['FOLDER', 'IMAGE_COLLECTION'], function (name) {
                        return /^ft\d\d$/.test(name);
                    }).sort();
                    rebuildFilteredRow(proposal, stages);
                    pending--;
                    if (pending === 0) updateFilteredStatus();
                });
            });
        });
    }

    function refreshResultSources() {
        var campaign = getSelectedCampaign();
        var root = classificationResultsRoot();
        var ctx = currentResultContext();

        resultState.filtered = {};
        resultState.prepublic.periods = [];
        filteredRows = {};
        setResultStatus('filtered', L.loading);
        setResultStatus('prepublic', L.loading);

        refreshFilteredSources(root, ctx);

        // Official promoted result (M9).
        listAssetsAsync(root + 'PRE_PUBLIC/' + campaign.toLowerCase() + '/', function (assets, err) {
            if (ctx !== currentResultContext()) return;
            if (err) {
                setResultStatus('prepublic', L.status_no_results);
                return;
            }
            var periods = resultEntries(assets, ['IMAGE']);
            resultState.prepublic.periods = periods;
            setResultStatus('prepublic', periods.length ? (periods.length + ' ' + L.lbl_images) : L.status_no_results);
        });
    }

    function makeFilteredDrawer() {
        var cab = createCabinet(L.cab_class_filtered, false, '2px 2px 2px 10px');
        subCabClass.content.add(cab.panel);
        panelFiltered = ui.Panel({
            style: { margin: '1px 0px', padding: '0px', stretch: 'horizontal', backgroundColor: '#f9f9f9' }
        });
        cab.content.add(panelFiltered);
        var status = ui.Label('', { fontSize: '9px', color: '#777', margin: '1px' });
        resultStatusLabels.filtered = status;
        cab.content.add(status);
    }

    function makePrepublicDrawer() {
        var cab = createCabinet(L.cab_class_prepublic, false, '2px 2px 2px 10px');
        subCabClass.content.add(cab.panel);
        var row = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '0px', padding: '0px' } });
        var chk = ui.Checkbox({
            label: L.lbl_show,
            value: false,
            onChange: function () { syncCheckBorder(chk); debouncedUpdateMosaic(); },
            style: { margin: '1px 3px', padding: '0px', fontSize: '10px', backgroundColor: '#f9f9f9' }
        });
        styleCheck(chk, false);
        row.add(colorSwatch(classificationColor('prepublic'))).add(chk);
        var status = ui.Label('', { fontSize: '9px', color: '#777', margin: '1px' });
        resultStatusLabels.prepublic = status;
        resultSourceCheckboxes.prepublic = chk;
        cab.content.add(row).add(status);
    }

    makeFilteredDrawer();
    makePrepublicDrawer();

    // --- CABINET 4: SAMPLES ---
    var cabSamples = createCabinet(L.cab_samples, true);
    panel_control.add(cabSamples.panel);

    // Sub-drawers panels
    var drawerInst = { panel: ui.Panel({ style: { shown: false, border: '1px solid #eee', margin: '2px' } }) };
    var drawerImp = { panel: ui.Panel({ style: { shown: false, border: '1px solid #eee', margin: '2px' } }) };
    var drawerAmo = { panel: ui.Panel({ style: { shown: false, border: '1px solid #eee', margin: '2px' } }) };
    var drawerExp = { panel: ui.Panel({ style: { shown: false, border: '1px solid #eee', margin: '2px' } }) };

    var drawerControl = createLayerDrawer(L.lbl_dashboard, [
        { id: 'inst', label: L.short_guide, value: false, onChange: function (v) { drawerInst.panel.style().set('shown', v); } },
        { id: 'imp', label: L.short_import, value: false, onChange: function (v) { drawerImp.panel.style().set('shown', v); } },
        { id: 'amo', label: L.cab_samples, value: true, onChange: function (v) { drawerAmo.panel.style().set('shown', v); if (v) updateStats(); } },
        { id: 'exp', label: L.short_export, value: true, onChange: function (v) { drawerExp.panel.style().set('shown', v); if (v) { suggestNextVersion(); syncDateSelect(); } } }
    ]);
    drawerExp.panel.style().set('shown', true);
    drawerAmo.panel.style().set('shown', true);

    // --- CAMPAIGN SELECTORS (Synced) ---
    function makeCampaignSelector() {
        return ui.Select({
            items: [],
            placeholder: L.placeholder_campaign,
            value: null,
            style: { margin: '2px', stretch: 'horizontal' }
        });
    }

    var select_campaign_imp = makeCampaignSelector();
    var select_campaign_exp = makeCampaignSelector();
    select_campaign_exp.onChange(function (campaign) {
        if (!campaign) return;
        setRegionCampaign(countryConfigs[getSelectedCountry()], campaign);
    });

    cabSamples.content.add(drawerControl.panel);
    cabSamples.content.add(drawerInst.panel);
    cabSamples.content.add(drawerImp.panel);
    cabSamples.content.add(drawerAmo.panel);
    cabSamples.content.add(drawerExp.panel);

    // Populate Sub-drawers
    drawerInst.panel.add(ui.Label(L.instructions, { fontSize: '10px', color: '#555', margin: '2px', whiteSpace: 'pre' }));

    // Drawing control buttons row
    var rowDraw = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: { margin: '2px 0px', stretch: 'horizontal' } });

    function setLayerMode(name) {
        var layers = Map.drawingTools().layers();
        for (var i = 0; i < layers.length(); i++) {
            if (layers.get(i).getName() === name) {
                Map.drawingTools().setShape('polygon');
                Map.drawingTools().setSelected(layers.get(i));
                return;
            }
        }
    }

    var btnFire = ui.Button({ label: L.btn_fire, style: { color: '#d32f2f', margin: '1px', padding: '0px', stretch: 'horizontal' }, onClick: function () { setLayerMode('fire'); } });
    var btnNotFire = ui.Button({ label: L.btn_not_fire, style: { color: '#1a73e8', margin: '1px', padding: '0px', stretch: 'horizontal' }, onClick: function () { setLayerMode('notFire'); } });
    var btnHand = ui.Button({ label: L.btn_hand, style: { margin: '1px', padding: '0px', stretch: 'horizontal' }, onClick: function () { Map.drawingTools().setShape(null); } });
    var btnClear = ui.Button({ label: L.btn_delete, style: { margin: '1px', padding: '0px', stretch: 'horizontal' }, onClick: clearDrawingTools });
    var btnCenter = ui.Button({ label: L.btn_center, style: { margin: '1px', padding: '0px', stretch: 'horizontal' }, onClick: centerDrawingTools });

    var btt_refresh_stats = ui.Button({
        label: '↻',
        style: { margin: '1px', padding: '0px', width: '28px', fontSize: '12px' },
        onClick: function () { updateStats(); }
    });

    rowDraw.add(btnFire).add(btnNotFire).add(btnHand).add(btnClear).add(btnCenter).add(btt_refresh_stats);

    drawerAmo.panel.add(rowDraw);
    drawerAmo.panel.add(panel_stats);
    // Export and Import will be populated below...

    // ==========================================
    // ABA: IMPORTAR
    // ==========================================
    var select_address = ui.Select({ style: styles.input });
    select_campaign_imp.onChange(function (folderPath) {
        if (!folderPath) return;
        var conf = countryConfigs[getSelectedCountry()];
        var isCampaign = (conf.campaigns || []).indexOf(folderPath) !== -1;
        if (isCampaign) {
            setRegionCampaign(conf, folderPath);
        } else {
            loadSampleAssets(folderPath);
        }
    });

    var lab_import_loading = ui.Label({ imageUrl: b64.get('load_gif'), style: { margin: '4px', shown: false } });
    var lab_import_status = ui.Label('', { fontSize: '10px', color: 'red', margin: '2px', shown: false });

    var btt_import_action = ui.Button({
        imageUrl: ICONS.download, style: styles.btn_blue,
        onClick: function () {
            var id = select_address.getValue();
            if (!id) return;
            var importToken = beginAsync(ASYNC_STATE.import);
            lab_import_status.style().set('shown', false);
            lab_import_loading.style().set('shown', true);

            var parts = id.split('_');
            var lastPart = parts[parts.length - 1], penultPart = parts[parts.length - 2];
            var recoveredPeriod = (!isNaN(penultPart) && !isNaN(lastPart)) ? (penultPart + '_' + lastPart) : lastPart;
            periodicityActive['mensal'] = (!isNaN(penultPart) && !isNaN(lastPart));
            periodicityActive['anual'] = !periodicityActive['mensal'];
            styleToggleButton(periodicityButtons['mensal'], periodicityActive['mensal']);
            styleToggleButton(periodicityButtons['anual'], periodicityActive['anual']);
            updatePeriodDrawer();
            Object.keys(periodCheckboxes).forEach(function (k) {
                if (periodCheckboxes[k]) periodCheckboxes[k].setValue(k === recoveredPeriod, false);
            });
            var matchedRegion = null;
            var regionNames = Object.keys(regionCheckboxes);
            for (var i = 0; i < regionNames.length; i++) {
                if (id.indexOf(regionNames[i]) !== -1) { matchedRegion = regionNames[i]; break; }
            }
            if (matchedRegion) {
                Object.keys(regionCheckboxes).forEach(function (k) { regionCheckboxes[k].setValue(k === matchedRegion, false); });
                updateSelectedRegionsMap();
            }
            updateExportPreview();

            var pending = 2;
            var results = {};
            function onImportResult(which, list, err) {
                if (!isCurrentAsync(ASYNC_STATE.import, importToken)) return;
                if (err) {
                    lab_import_loading.style().set('shown', false);
                    lab_import_status.setValue('Error importing samples').style().set('shown', true);
                    return;
                }
                results[which] = list;
                pending--;
                if (pending > 0) return;
                lab_import_loading.style().set('shown', false);
                // Remove existing fire/notFire drawing layers
                var layers = Map.drawingTools().layers();
                for (var i = layers.length() - 1; i >= 0; i--) {
                    var l = layers.get(i);
                    if (l.getName() === 'fire' || l.getName() === 'notFire') {
                        layers.remove(l);
                    }
                }
                Map.drawingTools().addLayer({ geometries: results.fire, name: 'fire', color: 'ff0000', shown: true });
                Map.drawingTools().addLayer({ geometries: results.notFire, name: 'notFire', color: '0000ff', shown: true });
                updateStats();
            }

            ee.FeatureCollection(id).filter(ee.Filter.eq('fire', 1)).geometry().coordinates().map(function (list) { return ee.Geometry.Polygon(list); })
                .evaluate(trackEvaluate(function (list, err) { onImportResult('fire', list, err); }));
            ee.FeatureCollection(id).filter(ee.Filter.eq('fire', 0)).geometry().coordinates().map(function (list) { return ee.Geometry.Polygon(list); })
                .evaluate(trackEvaluate(function (list, err) { onImportResult('notFire', list, err); }));
        }
    });

    var row_imp = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: styles.row });
    row_imp.add(select_campaign_imp);
    drawerImp.panel.add(ui.Label('Vector sample', { fontSize: '10px', margin: '3px 2px' }));
    row_imp.add(select_address).add(btt_import_action).add(lab_import_loading).add(lab_import_status);
    drawerImp.panel.add(row_imp);

    // ==========================================
    // ABA: EXPORTAR
    // ==========================================

    // --- Versão automática ---
    var lab_version_status = ui.Label(L.ver_checking, { fontSize: '10px', color: '#888', margin: '1px' });
    var txt_version = ui.Textbox({ placeholder: '0001', value: 'v0001', onChange: updateExportPreview, style: { margin: '1px', padding: '0px', width: '55px' } });

    function suggestNextVersion() {
        lab_version_status.setValue(L.ver_checking).style().set('color', '#888');
        var conf = countryConfigs[getSelectedCountry()];
        var campaign = select_campaign_exp.getValue();
        var versionToken = beginAsync(ASYNC_STATE.version);

        if (!campaign) {
            txt_version.setValue('0001');
            lab_version_status.setValue(L.ver_empty).style().set('color', '#e65100');
            updateExportPreview();
            return;
        }

        var campaignPath = conf.getAssetSamples(campaign);
        listAssetsAsync(campaignPath, function (assets, err) {
            if (!isCurrentAsync(ASYNC_STATE.version, versionToken)) return;
            if (err || !assets || assets.length === 0) {
                txt_version.setValue('0001');
                lab_version_status.setValue(L.ver_empty).style().set('color', '#e65100');
                updateExportPreview();
                return;
            }

            var maxV = 0;
            assets.forEach(function (a) {
                var fname = a.id.split('/').slice(-1)[0];
                var m = fname.match(/^samples_(\d{4})/);
                if (m) {
                    var n = parseInt(m[1], 10);
                    if (n > maxV) maxV = n;
                }
            });

            var nextNumber = ('000' + (maxV + 1)).slice(-4);
            var next = 'samples_' + nextNumber;

            txt_version.setValue(nextNumber);
            lab_version_status
                .setValue(L.ver_suggested + next)
                .style().set('color', '#0f9d58');
            updateExportPreview();
        });
    }

    // --- Shortname (opcional) ---
    var txt_shortname = ui.Textbox({ placeholder: L.placeholder_sn, onChange: updateExportPreview, style: { margin: '1px', padding: '0px', stretch: 'horizontal' } });
    var txt_comment = ui.Textbox({ placeholder: L.placeholder_cmt, style: { margin: '1px', padding: '0px', stretch: 'horizontal' } });
    var select_date_export = ui.Select({ placeholder: L.placeholder_date, onChange: updateExportPreview, style: { margin: '1px', stretch: 'horizontal' } });

    function buildDateSelectItems() {
        var items = [];
        var today = new Date();
        // Limitamos ao ano atual real, não permitindo datas futuras inconsistentes
        var maxYear = today.getFullYear();
        var maxMonth = today.getMonth();
        if (maxMonth === 0) { maxMonth = 12; maxYear--; }

        // Se estivermos em 2026 no sistema, mas os dados reais param antes, 
        // vamos forçar um limite conservador (ex: 2025_12) se necessário,
        // ou deixar o usuário escolher. Aqui vamos apenas garantir que a ordem é inversa.

        // Meses
        for (var y = maxYear; y >= MONTHLY_START_YEAR; y--) {
            var mEnd = (y === maxYear) ? Math.min(maxMonth, 12) : 12;
            for (var m = mEnd; m >= 1; m--) {
                var mm = m < 10 ? '0' + m : m.toString();
                items.push({ label: y + '_' + mm, value: y + '_' + mm });
            }
        }
        // Anos
        for (var yr = maxYear; yr >= START_YEAR; yr--) {
            items.push({ label: '' + yr, value: '' + yr });
        }
        return items;
    }

    function syncDateSelect() {
        select_date_export.items().reset(buildDateSelectItems());
        var periods = getSelectedPeriods();
        // PRIORIDADE: Se houver um período selecionado na aba TEMPORAL, usa ele!
        if (periods.length > 0) {
            select_date_export.setValue(periods[0]);
        }
        updateExportPreview();
    }

    // --- Satélites (apenas metadado) ---
    var satLabels = ['Sentinel2', 'Landsat', 'MODIS', 'HLS', 'Planet'];
    var satValues = ['sentinel2', 'landsat', 'modis', 'hls', 'planet'];
    var satCheckboxes = {};
    var satRow = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal', true), style: { margin: '0px', padding: '0px', maxHeight: '50px' } });
    satValues.forEach(function (sv, i) {
        var chk = ui.Checkbox({ label: satLabels[i], value: sv === 'sentinel2', onChange: function () { syncCheckBorder(chk); updateExportPreview(); }, style: { margin: '1px 3px', fontSize: '10px' } });
        styleCheck(chk, sv === 'sentinel2');
        satCheckboxes[sv] = chk;
        satRow.add(chk);
    });

    // --- Preview do nome ---
    var lab_export_preview = ui.Label({ value: '', style: { margin: '1px', fontSize: '10px', fontWeight: 'bold', color: '#1a73e8', stretch: 'horizontal' } });
    var lab_date_error = ui.Label(L.err_no_date, { fontSize: '10px', color: 'red', margin: '0px 2px', shown: false });

    function getSelectedSatellites() {
        var sats = [];
        satValues.forEach(function (sv) { if (satCheckboxes[sv].getValue()) sats.push(sv); });
        return sats;
    }
    function getSampleName() {
        var num = txt_version.getValue() || '0001';
        num = num.replace(/^samples_/, '');
        return 'samples_' + num;
    }
    function updateExportPreview() {
        var ver = getSampleName();
        var sn = txt_shortname.getValue() ? redundanceReplace(txt_shortname.getValue()) : '';
        var dt = select_date_export.getValue();

        lab_date_error.style().set('shown', !dt);

        var dt_label = dt || 'YYYY';
        var parts = [ver];

        if (sn) parts.push(sn);
        parts.push(redundanceReplace(dt_label));

        lab_export_preview.setValue(parts.join('_'));
    }

    // Alias para compatibilidade com outros módulos que chamam updateExportLabel
    function updateExportLabel() { updateExportPreview(); }

    function updateShortnameSuggestion() {
        var selected = [];
        Object.keys(regionCheckboxes).forEach(function (k) {
            if (regionCheckboxes[k].getValue()) selected.push(k);
        });
        if (selected.length === 0) return;
        var suggestion = selected.length === 1 ? selected[0] : 'multi_' + selected.length + '_regiones';
        txt_shortname.setValue(redundanceReplace(suggestion));
        updateExportPreview();
    }

    // --- getLandcoverVector com metadados ricos ---
    function getLandcoverVector(satellite) {
        var fireLayer = Map.drawingTools().layers().filter(function (ly) { return ly.getName() === 'fire'; });
        var notFireLayer = Map.drawingTools().layers().filter(function (ly) { return ly.getName() === 'notFire'; });
        if (fireLayer.length === 0 || notFireLayer.length === 0) {
            print(L.err_layers);
            return null;
        }
        var selectedNames = [];
        Object.keys(regionCheckboxes).forEach(function (k) { if (regionCheckboxes[k].getValue()) selectedNames.push(k); });
        var safeRegion = selectedNames.length === 1 ? selectedNames[0] : (selectedNames.length > 1 ? 'multi' : 'none');
        var datePeriod = select_date_export.getValue() || 'YYYY';
        var ver = getSampleName();
        var sn = txt_shortname.getValue() || '';
        var comment = txt_comment.getValue() || '';
        var now = new Date();
        var createdAt = now.getFullYear() + '-' + (now.getMonth() + 1) + '-' + now.getDate();

        function makeProps(fireVal) {
            return {
                'fire': fireVal, 'region': safeRegion, 'period': datePeriod,
                'satellite': satellite || 'unspecified', 'version': ver,
                'shortname': sn, 'comment': comment, 'created_at': createdAt
            };
        }

        var fire = ee.FeatureCollection(
            fireLayer[0].getEeObject().geometries().map(function (g) {
                return ee.Feature(ee.Geometry(g), makeProps(1));
            })
        );
        var notFire = ee.FeatureCollection(
            notFireLayer[0].getEeObject().geometries().map(function (g) {
                return ee.Feature(ee.Geometry(g), makeProps(0));
            })
        );
        return fire.merge(notFire);
    }

    // --- Botão unificado de exportação ---
    var btt_export = ui.Button({
        label: L.btn_export,
        style: { margin: '1px 0px', fontWeight: 'bold', color: '#0f9d58', stretch: 'horizontal' },
        onClick: function () {
            var dt = select_date_export.getValue();
            if (!dt) {
                print(L.err_no_date);
                return;
            }

            var sats = getSelectedSatellites();
            if (sats.length === 0) { print(L.err_no_sat); return; }
            var conf = countryConfigs[getSelectedCountry()];
            var selectedCampaign = select_campaign_exp.getValue();
            var ver = getSampleName();
            var sn = txt_shortname.getValue() ? redundanceReplace(txt_shortname.getValue()) : '';
            var dt_formatted = redundanceReplace(dt || 'YYYY');

            // Join all selected satellites into a single string for metadata
            var satelliteStr = sats.join(',');

            var vec = getLandcoverVector(satelliteStr);
            if (!vec) return;

            var parts = [ver];
            if (sn) parts.push(sn);
            parts.push(dt);

            var desc = parts.join('_');

            try {
                ensureCampaignFolders(conf, selectedCampaign, function () {
                    Export.table.toAsset({
                        collection: vec,
                        description: PERSONAL_TASK_FLAG + '_ASSET_' + desc,
                        assetId: conf.getAssetSamples(selectedCampaign) + '/' + desc
                    });
                    Export.table.toCloudStorage({
                        collection: vec,
                        description: PERSONAL_TASK_FLAG + '_GCS_' + desc,
                        bucket: conf.bucket,
                        fileNamePrefix: conf.getGcsSamples(selectedCampaign) + '/' + desc,
                        fileFormat: 'CSV'
                    });
                    print(L.task_created + desc + L.task_hint);
                });
            } catch (e) {
                print('Nao foi possivel criar a pasta da campanha: ' + e);
            }
        }
    });

    // --- Montar o panel_export ---
    drawerExp.panel.add(ui.Label(L.lbl_select_campaign, { fontSize: '11px', fontWeight: 'bold', margin: '4px 2px 2px 2px' }));
    drawerExp.panel.add(select_campaign_exp);
    drawerExp.panel.add(ui.Label(L.lbl_version, styles.label));
    var rowVer = ui.Panel({ layout: ui.Panel.Layout.flow('horizontal'), style: styles.row });
    rowVer.add(txt_version).add(lab_version_status);
    drawerExp.panel.add(rowVer);

    drawerExp.panel.add(ui.Label(L.lbl_shortname, styles.label));
    drawerExp.panel.add(txt_shortname);

    drawerExp.panel.add(ui.Label(L.lbl_comment, styles.label));
    drawerExp.panel.add(txt_comment);

    drawerExp.panel.add(ui.Label(L.lbl_date, styles.label));
    drawerExp.panel.add(select_date_export);
    drawerExp.panel.add(lab_date_error);

    drawerExp.panel.add(ui.Label(L.lbl_satellites, styles.label));
    drawerExp.panel.add(satRow);

    drawerExp.panel.add(lab_export_preview);
    drawerExp.panel.add(btt_export);


    // INITIALIZE APP (Initial trigger)
    preloadAllRegionNames();
    preloadHotspotAssets();
    loadCountryData(APP_CONFIG.defaultCountry);
    appReady = true;
    updateImportList(countryConfigs[getSelectedCountry()], countryConfigs[getSelectedCountry()].defaultCampaign);
    updateMosaic();
    initDrawingTools();
    syncDateSelect();
    suggestNextVersion();
    updateStats();
}

user_interface();

function redundanceReplace(string) {
    return string.toLowerCase().replace(/ /gi, '_').replace(/-/gi, '_')
        .replace(/â|á|à|ã|ä/gi, 'a').replace(/ê|é|è|ẽ/gi, 'e')
        .replace(/î|í|ì!ĩ/gi, 'i').replace(/ô|ó|ò|õ|ö/gi, 'o')
        .replace(/û|ú|ù|ũ|ü/gi, 'u').replace(/ç/gi, 'c');
}

// ─── FUNÇÕES DE SUPORTE E CORREÇÕES ORIGINAIS (RAW MOSAICS) ─────────────────
function clipBoard_Landsat(image) {
    return image.updateMask(ee.Image().paint(image.geometry().buffer(-3000)).eq(0));
}

function bitwiseExtract(value, fromBit, toBit) {
    if (toBit === undefined) toBit = fromBit;
    var maskSize = ee.Number(1).add(toBit).subtract(fromBit);
    var mask = ee.Number(1).leftShift(maskSize).subtract(1);
    return value.rightShift(fromBit).bitwiseAnd(mask);
}

function addIndices(image) {
    var ndvi = image.normalizedDifference(['nir', 'red']).rename('ndvi');
    var nbr = image.normalizedDifference(['nir', 'swir2']).rename('nbr');
    return image.addBands([ndvi, nbr]);
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
    return image.select(oldBands, newBands);
}

var processLS57 = function (img) { return addDOY(addBand_NBR(corrections_LS57_col2(clipBoard_Landsat(img)))); };
var processLS89 = function (img) { return addDOY(addBand_NBR(corrections_LS8_col2(clipBoard_Landsat(img)))); };

var processS2 = function (img) {
    var mask = img.select('cs').gte(0.40);
    var optical = img.updateMask(mask).select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']);
    return addDOY(addBand_NBR(optical));
};

var processMODIS = function (img) { return addDOY(addBand_NBR(corrections_modis(img))); };

var processHLS_S30 = function (img) {
    var mask = img.select('Fmask').bitwiseAnd(1 << 1).eq(0);
    var optical = img.updateMask(mask).select(
        ['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    );
    return addDOY(addBand_NBR(optical).set({
        'system:time_start': img.get('system:time_start'),
        'system:time_end': img.get('system:time_end')
    }));
};

var processHLS_L30 = function (img) {
    var mask = img.select('Fmask').bitwiseAnd(1 << 1).eq(0);
    var optical = img.updateMask(mask).select(
        ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
        ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    );
    return addDOY(addBand_NBR(optical).set({
        'system:time_start': img.get('system:time_start'),
        'system:time_end': img.get('system:time_end')
    }));
};
