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

var collection = 'MapBiomas Fogo Coleção 4.1';
var description = 'csv-col41-year-last-fire';
var driverFolder = 'colecao41';
var scale = 30;

var year_last_fire = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/2_Subprodutos_col41/mapbiomas-fire-collection41-year-last-fire-v1');

// Define a list of years to export
var years = [2024];


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
     situation = ee.Dictionary(legends.get('fogo_situation'));
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
                var year_last_fire = classId;
                
                var area_km2 = classAndArea.get('sum');
                var area_ha = ee.Number(area_km2).multiply(100);
    
                var tableColumns = ee.Feature(null)
                    //Ano Corrente e Acumulado
                    .set('Área km²', area_km2)
                    .set('Área ha', area_ha)
                    .set('Bioma', biomes.get(regionsId.int()))
                    .set('Estados', estados.get(stateId))
                    .set('Ano Ultimo fogo',year_last_fire)
                    .set('fire_int',classId)
                    //.set('Situação',situation.get(classId));


                return tableColumns;
            }
        );
    
        return ee.FeatureCollection(ee.List(tableRows));
      }
    );

    areas = ee.FeatureCollection(areas).flatten();

    return areas;
};

var areas = years.map(function (year) {
    // Seleciona a banda referente ao ano
    var year_band = 'classification_' + year;
    var image = year_last_fire.select(year_band);
    
    // Aplica máscara para considerar apenas onde houve fogo
    var masked_image = image.selfMask();
    
    var areas = calculateArea(masked_image, territory, geometry);

    areas = areas.map(function (feature) {
        return feature.set({'Ano': year});
    });

    return areas;
});



areas = ee.FeatureCollection(areas).flatten();

print(areas.limit(10));

Export.table.toDrive({
    collection: areas,
    description: description,
    folder: driverFolder,
    fileNamePrefix: description,
    fileFormat: 'CSV'
});
