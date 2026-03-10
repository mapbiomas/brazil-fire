/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var aoi_test = 
    /* color: #d63000 */
    /* shown: false */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-51.4755904127785, -11.089163175148474],
          [-51.4755904127785, -12.980166056389645],
          [-49.9814497877785, -12.980166056389645],
          [-49.9814497877785, -11.089163175148474]]], null, false);
/***** End of imports. If edited, may not auto-convert in the playground. *****/
/** @description  - calculate scar fire area
 *  v1 https://code.earthengine.google.com/2d77ef40df2fbd6b8116703e4ee4561e
 * 
 *    by: Instituto de Pesquisa Ambiental da Amazônia
 *    desenvolvimento: Wallace Silva e Vera Laisa;
 * 
 */

var driverFolder = 'colecao4';
var collection = 'MapBiomas Fogo Coleção 4';
var subproduct = 'Acumulado';
var scale = 30;
var description = 'csv-col4-burned_cover_acumullated_unplished_fund-area';
var mapbiomas = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-accumulated-burned-coverage-v1').int();
print('mapbiomas',mapbiomas);
var unpublished_fire = mapbiomas.gte(1).slice(0,1).addBands(ee.Image('projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-time-after-fire-v1').eq(0));
print('unpublished_fire',unpublished_fire);

mapbiomas = mapbiomas.slice(0,40).add(unpublished_fire.multiply(100).unmask());
print('mapbiomas',mapbiomas);



// Define a list of years to export
var years = [
    1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
    1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 
    2017, 2018, 2019, 2020, 2021, 2022, 
    2023, 2024
];

print(mapbiomas);
// Image area in km2
var pixelArea = ee.Image.pixelArea().divide(1000000);

var regions = ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2');
var regions_img = ee.Image().paint(regions,'region');

var states = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/estados-2017')
  .map(function(feat){return feat.set('CD_GEOCUF',ee.Number.parse(feat.get('CD_GEOCUF')))});
  print('states',states,states.aggregate_array('CD_GEOCUF').distinct()
.iterate(function(current,previous){return ee.Dictionary(previous).set(ee.Number(current).int(),states.filter(ee.Filter.eq('CD_GEOCUF',ee.Number(current))).first().get('NM_ESTADO'))},{}));
var states_img = ee.Image().paint(states,'CD_GEOCUF');

var territory = regions_img//.multiply(100).add(states_img);
Map.addLayer(territory.randomVisualizer());

//////////////////////// fundiario
var fundiario = ee.Image('projects/mapbiomas-workspace/AUXILIAR/IMAFLORA2025/malhafundiaria_br_Imaflora_abril2025');

var classIdIn =  [  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 16, 17, 18, 19, 20, 99, 13, 15, 14, 101,  102,  103,  104,  105,  106,  107,  108,  109,  110,  111,  112,  116,  117,  118,  119,  120,  199,  113,  115,  114 ];
var classIdOut = [  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 12, 1,  2,  2,  8,  99, 13, 15, 14, 1,    2,    3,    4,    5,    6,    7,    8,    9,    10,   11,   12,   12,   1,    2,    2,    8,    99,   13,   15,   14  ];
// https://ipamamazonia-my.sharepoint.com/:x:/r/personal/vera_arruda_ipam_org_br/Documents/Arquivos%20de%20Chat%20do%20Microsoft%20Teams/tabela_legenda_Fundiario.xlsx?d=w2e26d1a2bb654e6486e374c9eff97f06&csf=1&web=1&e=Db9GEV

fundiario = fundiario.remap(classIdIn,classIdOut);

Map.addLayer(fundiario.randomVisualizer(),{},'fundiario');

territory = territory.multiply(100).add(fundiario);

Map.addLayer(territory.randomVisualizer());

////////////////////////

var geometry = regions.geometry().bounds();
// var geometry = aoi_test

//legendas
var legends = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends.js');

var  regions = ee.Dictionary(legends.get('fogo_regions_col2')),
     biomes = ee.Dictionary(legends.get('biomas')),
     estados = ee.Dictionary(legends.get('estados')),
     ufs = ee.Dictionary(legends.get('ufs')),
    situation = ee.Dictionary({
      0:'Já foi queimado',
      1:'1ª vez queimando',
    }),
    // situation = ee.Dictionary(legends.get('fogo_situation')),
    // months_legend = ee.Dictionary(legends.get('fogo_months_str')),
    // months_int_legend = ee.Dictionary(legends.get('fogo_months_int')),
     niveis = ee.Dictionary(legends.get('lulc_mbc09_niveis')),
     nivel_0 = ee.Dictionary(legends.get('lulc_mbc09_nivel_0')),
     nivel_1 = ee.Dictionary(legends.get('lulc_mbc09_nivel_1')),
     nivel_1_1 = ee.Dictionary(legends.get('lulc_mbc09_nivel_1_1')),
     nivel_2 = ee.Dictionary(legends.get('lulc_mbc09_nivel_2')),
     nivel_3 = ee.Dictionary(legends.get('lulc_mbc09_nivel_3')),
     nivel_4 = ee.Dictionary(legends.get('lulc_mbc09_nivel_4')),
     fundiario_mapb = ee.Dictionary({
      1:'Terras Indígena Homologada',
      2:'Terras Indígena não Homologada',
      3:'Áreas militares',
      4:'Glebas públicas',
      5:'Glebas públicas – FPND',
      6:'Território Quilombola Declarado',
      7:'Território Quilombola não Declarado',
      8:'Imóvel rural privado',
      9:'Assentamento-A',
      10:'Assentamento-B',
      11:'UCUS',
      12:'UCPI',
      16:'Imóvel Rural Privado e UCPI',
      17:'Terra Indígena Homologada e UCPI',
      18:'Terra Indígena não Homologada e UCUS',
      19:'Terra Indígena não Homologada e UCPI',
      20:'Imóvel Rural Privado e UCUS',
      99:'Outras sobreposições',
      13:'ASRFG',
      15:'Massas d’água',
      14:'Áreas Urbanas',
    });



/**
 * Calculate area crossing a cover map (deforestation, mapbiomas)
 * and a region map (states, biomes, municipalites)
 * @param image 
 * @param territory 
 * @param geometry
 */
var calculateArea = function (image, territory, geometry) {

    var reducer = ee.Reducer.sum().group(1, 'class').group(1, 'territory');

    var territotiesData = pixelArea.addBands(territory).addBands(image)
        .reduceRegion({
            reducer: reducer,
            // geometry: geometry_2,
            geometry: geometry,
            scale:scale,
            maxPixels: 1e12
        });

    territotiesData = ee.List(territotiesData.get('groups'));

    var areas = territotiesData.map(
      // * Convert a complex ob to feature collection
      function (obj) {
  
        obj = ee.Dictionary(obj);
    
        var territory_int = obj.getNumber('territory').int();
        
        var fundiario_int = territory_int.mod(100).int();
        
        var regionsId = territory_int.divide(100).int();
        // var regionsId = territory_int.divide(10000).int();
        // var stateId = territory_int.divide(100).int().mod(100).int();
        var classesAndAreas = ee.List(obj.get('groups'));
    
        var tableRows = classesAndAreas.map(
            function (classAndArea) {
                classAndArea = ee.Dictionary(classAndArea);
    
                var classId = classAndArea.getNumber('class');
                var situation_fire = classId.divide(100).int();
                var lulcId = classId.mod(100).int();
                
                var area_km2 = classAndArea.get('sum');
                var area_ha = ee.Number(area_km2).multiply(100); // Convertendo km² para hectares
    
                var tableColumns = ee.Feature(null)
                    // .set('territory', territory_int)
                    // .set('class_int', classId)
                    .set('Área km²', area_km2)
                    .set('Área ha', area_ha)
                    .set('Região', regions.get(regionsId))
                    .set('Bioma', biomes.get(regionsId.divide(10).int()))
                    .set('Estados', estados.get(stateId))
                    .set('UFs', ufs.get(stateId))
                    .set('Fogo', lulcId.gte(1))
                    .set('Fogo inédito', situation.get(situation_fire))
                    .set('Classe', lulcId)
                    .set('Subproduto',subproduct)
                    .set('Níveis', niveis.get(lulcId))
                    .set('Nível 0', nivel_0.get(lulcId))
                    .set('Nível 1', nivel_1.get(lulcId))
                    .set('Nível 1_1', nivel_1_1.get(lulcId))
                    .set('Nível 2', nivel_2.get(lulcId))
                    .set('Nível 3', nivel_3.get(lulcId))
                    .set('Nível 4', nivel_4.get(lulcId))
                    // .set('fundiario_int', fundiario_int)
                    .set('Fundiário', fundiario_mapb.get(fundiario_int));
    
                return tableColumns;
            }
        );
    
        return ee.FeatureCollection(ee.List(tableRows));
      }
    );

    areas = ee.FeatureCollection(areas).flatten();

    return areas;
};

var areas = years.map(
    function (year) {
        var image = mapbiomas.select('fire_accumulated_1985_' + year);

        var areas = calculateArea(image, territory, geometry);

        // set additional properties
        areas = areas.map(
            function (feature) {
                return feature
                  .set({
                    'Ano':year,
                    'Coleção':collection,
                  });
            }
        );

        return areas;
    }
);

areas = ee.FeatureCollection(areas).flatten();

print(areas.limit(10));

Export.table.toDrive({
    collection: areas,
    description: description,
    folder: driverFolder,
    fileNamePrefix: description,
    fileFormat: 'CSV'
});