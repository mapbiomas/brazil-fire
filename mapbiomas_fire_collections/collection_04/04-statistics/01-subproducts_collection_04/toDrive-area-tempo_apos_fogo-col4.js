/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var geometry_2 = 
    /* color: #d63000 */
    /* displayProperties: [
      {
        "type": "rectangle"
      }
    ] */
    ee.Geometry.Polygon(
        [[[-54.23600464199918, -15.835370754946123],
          [-54.23600464199918, -16.847435806636867],
          [-53.35709839199918, -16.847435806636867],
          [-53.35709839199918, -15.835370754946123]]], null, false);
/***** End of imports. If edited, may not auto-convert in the playground. *****/
/** @description  - calculate time after fire
 *  15 de maio - estatisticas_Colecao2/subprodutos
 *  
 *    by: Instituto de Pesquisa Ambiental da Amazônia
 *    desenvolvimento: Wallace Silva e Vera Laisa;
 * 
 */

var collection = 'MapBiomas Fogo Coleção 3.1';
var description = 'csv-col4-time_after_fire-area';
var driverFolder = 'colecao4';
var scale = 30;

//var burned_coverage = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_monthly_burned_coverage_v1').addBands(ee.Image().rename('burned_coverage_2024'));
var burned_coverage = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-monthly-burned-coverage-v1');
// Seleciona todas as bandas, exceto a primeira (bandas são indexadas a partir de 0)
var burned_coverage1 = burned_coverage.select(ee.List.sequence(1, burned_coverage.bandNames().length().subtract(1)));
var taf = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-time-after-fire-v1');

var mapbiomas = burned_coverage1.gte(1).unmask().multiply(100).add(taf);


// Define a list of years to export
var years = [
    // 1985,
    1986, 1987, 1988, 1989, 1990, 1991, 1992,
    1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 
    2017, 2018, 2019, 2020, 2021,2022, 2023, 2024
];


// Image area in km2
var pixelArea = ee.Image.pixelArea().divide(1000000);

var regions = ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2');
var regions_img = ee.Image().paint(regions,'region').divide(10).int();

var states = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/estados-2017')
  .map(function(feat){return feat.set('CD_GEOCUF',ee.Number.parse(feat.get('CD_GEOCUF')))});
  print('states',states,states.aggregate_array('CD_GEOCUF').distinct()
.iterate(function(current,previous){return ee.Dictionary(previous).set(ee.Number(current).int(),states.filter(ee.Filter.eq('CD_GEOCUF',ee.Number(current))).first().get('NM_ESTADO'))},{}));
var states_img = ee.Image().paint(states,'CD_GEOCUF');

var territory = regions_img.multiply(100).add(states_img);
Map.addLayer(territory.randomVisualizer());

var geometry = regions.geometry().bounds();

//legendas
var legends = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends.js');

var  regions = ee.Dictionary(legends.get('fogo_regions_col2')),
     biomes = ee.Dictionary(legends.get('biomas')),
     estados = ee.Dictionary(legends.get('estados')),
     ufs = ee.Dictionary(legends.get('ufs')),
     situation = ee.Dictionary(legends.get('fogo_situation')),
     months_legend = ee.Dictionary(legends.get('fogo_months_str')),
     niveis = ee.Dictionary(legends.get('lulc_mbc08_niveis')),
     nivel_0 = ee.Dictionary(legends.get('lulc_mbc08_nivel_0')),
     nivel_1 = ee.Dictionary(legends.get('lulc_mbc08_nivel_1')),
     nivel_1_1 = ee.Dictionary(legends.get('lulc_mbc08_nivel_1_1')),
     nivel_2 = ee.Dictionary(legends.get('lulc_mbc08_nivel_2')),
     nivel_3 = ee.Dictionary(legends.get('lulc_mbc08_nivel_3')),
     nivel_4 = ee.Dictionary(legends.get('lulc_mbc08_nivel_4'));
/**
 * Calculate area crossing a cover map (mapbiomas fogo)
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
            scale: scale,
            maxPixels: 1e12
        });

    territotiesData = ee.List(territotiesData.get('groups'));

    var areas = territotiesData.map(
      // * Convert a complex ob to feature collection
      function (obj) {
  
        obj = ee.Dictionary(obj);
    
        var territory = obj.getNumber('territory').int();
        var regionsId = territory.divide(100).int();
        var stateId = territory.mod(100).int();
        var classesAndAreas = ee.List(obj.get('groups'));
    
        var tableRows = classesAndAreas.map(
            function (classAndArea) {
                classAndArea = ee.Dictionary(classAndArea);
    
                var classId = classAndArea.getNumber('class');
                var fire_age = classId.mod(100).int();
                var fire_int = classId.divide(100).int();
                
                var area_km2 = classAndArea.get('sum');
                var area_ha = ee.Number(area_km2).multiply(100);
    
                var tableColumns = ee.Feature(null)
                    //Ano Corrente e Acumulado
                    .set('Área km²', area_km2)
                    .set('Área ha', area_ha)
                    .set('Bioma', biomes.get(regionsId.int()))
                    .set('Estados', estados.get(stateId))
                    // .set('classId',classId)
                    .set('Idade do fogo',fire_age)
                    .set('fire_int',fire_int)
                    .set('Situação',situation.get(fire_int));


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
        var image = mapbiomas.select('burned_coverage_' + year);

        var areas = calculateArea(image, territory, geometry);

        // set additional properties
        areas = areas.map(
            function (feature) {
                return feature
                  .set({
                    'Ano':year,
                    'Ultima ano que pegou fogo':ee.Number(year).subtract(feature.getNumber('Idade do fogo')).int()
                    // 'asset':data,
                    // 'Coleção':collection,
                    // 'Coleção_ano':collection_year,
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
