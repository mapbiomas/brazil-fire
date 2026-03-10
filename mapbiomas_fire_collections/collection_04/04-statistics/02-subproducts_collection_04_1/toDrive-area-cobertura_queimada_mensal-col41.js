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
/** @description  - calculate scar fire area 
 *  
 *    by: Instituto de Pesquisa Ambiental da Amazônia
 *    desenvolvimento: Wallace Silva e Vera Laisa;
 * 
 */

var collection = 'MapBiomas Fogo Coleção 4.1';
var description = 'csv-col41-burned_cover_monthly-area';
var mapbiomas = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/2_Subprodutos_col41/mapbiomas-fire-collection41-monthly-burned-coverage-v1');
var driverFolder = 'colecao41';
var scale = 30;

// Define a list of years to export
var years = [
    1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
    1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000,
    2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 
    2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 
    2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024
];


// Image area in km2
var pixelArea = ee.Image.pixelArea().divide(1000000);

var regions = ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2');
var regions_img = ee.Image().paint(regions,'region');

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
     niveis = ee.Dictionary(legends.get('lulc_mbc10_niveis')),
     nivel_0 = ee.Dictionary(legends.get('lulc_mbc10_nivel_0')),
     nivel_1 = ee.Dictionary(legends.get('lulc_mbc10_nivel_1')),
     nivel_1_1 = ee.Dictionary(legends.get('lulc_mbc10_nivel_1_1')),
     nivel_2 = ee.Dictionary(legends.get('lulc_mbc10_nivel_2')),
     nivel_3 = ee.Dictionary(legends.get('lulc_mbc10_nivel_3')),
     nivel_4 = ee.Dictionary(legends.get('lulc_mbc10_nivel_4'));


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
                var monthId = classId.divide(100).int();
                var lulcId = classId.mod(100).int();
                
                var area_km2 = classAndArea.get('sum');
                var area_ha = ee.Number(area_km2).multiply(100);
                
    
                var tableColumns = ee.Feature(null)
                    .set('Área km²', area_km2)
                    .set('Área ha', area_ha)
                    .set('Região', regions.get(regionsId))
                    .set('Bioma', biomes.get(regionsId.divide(10).int()))
                    .set('Estados', estados.get(stateId))
                    .set('UFs', ufs.get(stateId))
                    .set('Situação', situation.get(monthId.gte(1)))
                    .set('Meses', months_legend.get(monthId))
                    .set('Mês', monthId)
                    .set('Classe', lulcId)
                    .set('Níveis', niveis.get(lulcId))
                    .set('Nível 0', nivel_0.get(lulcId))
                    .set('Nível 1', nivel_1.get(lulcId))
                    .set('Nível 1_1', nivel_1_1.get(lulcId))
                    .set('Nível 2', nivel_2.get(lulcId))
                    .set('Nível 3', nivel_3.get(lulcId))
                    .set('Nível 4', nivel_4.get(lulcId));
                    
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
