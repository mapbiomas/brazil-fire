/** @description  - calculate scar fire area for conservation units
 *   
 *  
 *    by: Instituto de Pesquisa Ambiental da Amazônia
 *    desenvolvimento: Wallace Silva e Vera Laisa;
 * 
 */

var driverFolder = 'colecao4';
var collection = 'MapBiomas Fogo Coleção 4';
var subproduct = 'Acumulado';
var scale = 30;
////Território de Áreas Protegidas - TI///// MUDAR LINHA 120
/*var data = 'tis';
var territorio = 'Terras Indígenas'
var asset_area_protegida = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/indigenous_land');
*/
////Território de Áreas Protegidas - estadual UCs Proteção Integral///// MUDAR LINHA 122
/*var data = 'estadual_ucs_pi';
var territorio = 'UC de Proteção Integral'
var asset_area_protegida = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/state_conservation_units_integral_protection');
*/
////Território de Áreas Protegidas - estadual UCs Uso Sustentável///// MUDAR LINHA 124
/*var data = 'estadual_ucs_us';
var territorio = 'UC de Uso Sustentável'
var asset_area_protegida = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/state_conservation_units_sustainable_use');
*/
////Território de Áreas Protegidas - federal UCs Proteção Integral///// MUDAR LINHA 126
/*var data = 'federal_ucs_pi';
var territorio = 'UC de Proteção Integral'
var asset_area_protegida = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/federal_conservation_units_integral_protection');
*/
////Território de Áreas Protegidas - federal UCs Uso Sustentável///// MUDAR LINHA 128
/*var data = 'federal_ucs_us';
var territorio = 'UC de Uso Sustentável'
var asset_area_protegida = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/federal_conservation_units_sustainable_use');
*/


var description = 'csv-col4-acumullated-area-24-' + data;
var mapbiomas = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-accumulated-burned-v1').int(); //  queimada acumulada

// Define a list of years to export
var years = [2024];
print(mapbiomas)

// Image area in km2
var pixelArea = ee.Image.pixelArea().divide(1000000);

var regions = ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2');
var regions_img = ee.Image().paint(regions,'region');

var area_protegida = asset_area_protegida
  .map(function(feat){
    return feat.set('feature_id',ee.Number.parse(feat.get('feature_id')))
  });
var area_protegida_img = ee.Image().paint(area_protegida,'feature_id');

var territory = regions_img.multiply(10000).add(area_protegida_img);
Map.addLayer(territory.randomVisualizer(),{},'territory');
//Map.addLayer(regions, {}, 'Regiões');
Map.addLayer(area_protegida, {}, 'area_protegida');

var geometry = regions.geometry().bounds();

//legendas
var legends = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends.js');
var legends_territorios = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends_territorios.js');

var  regions = ee.Dictionary(legends.get('fogo_regions_col2')),
     biomes = ee.Dictionary(legends.get('biomas')),
     estados = ee.Dictionary(legends.get('estados')),
     ufs = ee.Dictionary(legends.get('ufs'));
     
var  tisNM = ee.Dictionary(legends_territorios.get('tis_name')),
     estadual_ucs_pi_NM = ee.Dictionary(legends_territorios.get('estadual_ucs_pi_name')),
     estadual_ucs_us_NM = ee.Dictionary(legends_territorios.get('estadual_ucs_us_name')),
     federal_ucs_pi_NM = ee.Dictionary(legends_territorios.get('federal_ucs_pi_name')),
     federal_ucs_us_NM = ee.Dictionary(legends_territorios.get('federal_ucs_us_name'));

/*** Calculate area crossing a cover map and a region map  */

var calculateArea = function (image, territory, geometry) {

    var reducer = ee.Reducer.sum().group(1, 'class').group(1, 'territory');
    var territotiesData = pixelArea.addBands(territory).addBands(image)
        .reduceRegion({
            reducer: reducer,
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
        var regionsId = territory.divide(10000).int();
        var territoryId = territory.mod(10000).int();
        var classesAndAreas = ee.List(obj.get('groups'));
    
        var tableRows = classesAndAreas.map(
            function (classAndArea) {
                classAndArea = ee.Dictionary(classAndArea);
    
                var classId = classAndArea.getNumber('class');
                var lulcId = classId;
                var area_km2 = classAndArea.get('sum');
                var area_ha = ee.Number(area_km2).multiply(100); // Convertendo km² para hectares
                var tableColumns = ee.Feature(null)
                    .set('Área km²', area_km2)
                    .set('Área ha', area_ha)
                    //.set('Região', regions.get(regionsId))
                    .set('Bioma', biomes.get(regionsId.divide(10).int()))
                    //.set('Nome', tisNM.get(territoryId))
                    
                    //.set('Nome', estadual_ucs_pi_NM.get(territoryId))

                    //.set('Nome', estadual_ucs_us_NM.get(territoryId))
                    
                    //.set('Nome', federal_ucs_pi_NM.get(territoryId))

                    //.set('Nome', federal_ucs_us_NM.get(territoryId))
                    
                    .set('Codigo', territoryId)
                    .set('Subproduto',subproduct)
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
                    'Área Protegida': territorio,
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
