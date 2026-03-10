/** @description  - calculate scar fire area for municipalities
 *  
 *    by: Instituto de Pesquisa Ambiental da Amazônia
 *    desenvolvimento: Wallace Silva e Vera Laisa;
 * 
 */

var driverFolder = 'colecao4';
var collection = 'MapBiomas Fogo Coleção 4';
var subproduct = 'Acumulado';
var scale = 30;
var data = 'col4_municipios';
var description = 'csv-col4_acumullated-area-municipios';
var mapbiomas = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-accumulated-burned-v1').int(); // burned cover accumulated area

// Define a list of years to export
var years = [2024];


print(mapbiomas)
// Image area in km2
var pixelArea = ee.Image.pixelArea().divide(1000000);

var regions = ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2');
var regions_img = ee.Image().paint(regions,'region');

var municipios = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/city')
  //.limit(10)
  .map(function(feat){
    return feat.set('feature_id',ee.Number.parse(feat.get('feature_id')))
  });
/*print('municipios',municipios,municipios.aggregate_array('feature_id').distinct()
  .iterate(function(current,previous){
    return ee.Dictionary(previous)
      .set(ee.Number(current).int(),municipios.filter(ee.Filter.eq('feature_id',ee.Number(current))).first().get('name_pt_br'))
  },{}));*/
var municipios_img = ee.Image().paint(municipios,'feature_id');

/*var states = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/estados-2017')
  .map(function(feat){
    return feat.set('CD_GEOCUF',ee.Number.parse(feat.get('CD_GEOCUF')))
  });
print('states',states,states.aggregate_array('CD_GEOCUF').distinct()
  .iterate(function(current,previous){
    return ee.Dictionary(previous)
      .set(ee.Number(current).int(),states.filter(ee.Filter.eq('CD_GEOCUF',ee.Number(current))).first().get('NM_ESTADO'))
  },{}));
var states_img = ee.Image().paint(states,'CD_GEOCUF');*/

var territory = regions_img.multiply(10000000).add(municipios_img);
Map.addLayer(territory.randomVisualizer(),{},'territory');
//Map.addLayer(regions, {}, 'Regiões');
Map.addLayer(municipios, {}, 'municipios');

var geometry = regions.geometry().bounds();

//legendas
var legends = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends.js');

var  regions = ee.Dictionary(legends.get('fogo_regions_col2')),
     biomes = ee.Dictionary(legends.get('biomas')),
     estados = ee.Dictionary(legends.get('estados')),
     ufs = ee.Dictionary(legends.get('ufs')),
    // situation = ee.Dictionary(legends.get('fogo_situation')),
    // months_legend = ee.Dictionary(legends.get('fogo_months_str')),
    // months_int_legend = ee.Dictionary(legends.get('fogo_months_int')),
     niveis = ee.Dictionary(legends.get('lulc_mbc09_niveis')),
     nivel_0 = ee.Dictionary(legends.get('lulc_mbc09_nivel_0')),
     nivel_1 = ee.Dictionary(legends.get('lulc_mbc09_nivel_1')),
     nivel_1_1 = ee.Dictionary(legends.get('lulc_mbc09_nivel_1_1')),
     nivel_2 = ee.Dictionary(legends.get('lulc_mbc09_nivel_2')),
     nivel_3 = ee.Dictionary(legends.get('lulc_mbc09_nivel_3')),
     nivel_4 = ee.Dictionary(legends.get('lulc_mbc09_nivel_4'));
     
var legends_municipios = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends_territorios.js');

var municipiosNM = ee.Dictionary(legends_municipios.get('municipios_name'));

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
    
        var territory = obj.getNumber('territory').int();
        var regionsId = territory.divide(10000000).int();
        //var stateId = territory.mod(100).int();
        var municipioId = territory.mod(10000000).int();
        var classesAndAreas = ee.List(obj.get('groups'));
    
        var tableRows = classesAndAreas.map(
            function (classAndArea) {
                classAndArea = ee.Dictionary(classAndArea);
    
                var classId = classAndArea.getNumber('class');


                var lulcId = classId;
                
                var area_km2 = classAndArea.get('sum');
                var area_ha = ee.Number(area_km2).multiply(100); // Convertendo km² para hectares

    
                var tableColumns = ee.Feature(null)
                    // .set('territory', territory)
                    // .set('class_int', classId)
                    .set('Área km²', area_km2)
                    .set('Área ha', area_ha)
                    //.set('Região', regions.get(regionsId))
                    .set('Bioma', biomes.get(regionsId.divide(10).int()))
                    .set('Municípios', municipiosNM.get(municipioId))
                    .set('Cod Município', municipioId)
                    //.set('Estados', estados.get(stateId))
                    //.set('UFs', ufs.get(stateId))
                    //.set('Fogo', lulcId.gte(1))
                    //.set('Classe', lulcId)
                    .set('Subproduto',subproduct)
/*                    .set('Níveis', niveis.get(lulcId))
                    .set('Nível 0', nivel_0.get(lulcId))
                    .set('Nível 1', nivel_1.get(lulcId))
                    .set('Nível 1_1', nivel_1_1.get(lulcId))
                    .set('Nível 2', nivel_2.get(lulcId))
                    .set('Nível 3', nivel_3.get(lulcId))
                    .set('Nível 4', nivel_4.get(lulcId));*/
    
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
