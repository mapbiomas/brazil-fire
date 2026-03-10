/** @description  - calculate scar fire area for conservation units
 *   
 *  
 *    by: Instituto de Pesquisa Ambiental da Amazônia
 *    desenvolvimento: Wallace Silva e Vera Laisa;
 * 
 */

// Define the folder, collection, subproduct and scale
var driverFolder = 'colecao4';
var collection = 'MapBiomas Fogo Coleção 4';
var scale = 30;
var territorio = 'Base Fundiária';
var description = 'csv-col4-acumullated-area-base-fundiaria24';

// Load the MapBiomas Fire Collection 4
var mapbiomas = ee.Image('projects/earthengine-legacy/assets/projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-accumulated-burned-v1').int(); // burned cover accumulated area

// Define a list of years to export
var years = [2024];

// Compute the pixel area in km2
var pixelArea = ee.Image.pixelArea().divide(1000000);

// Load the regions feature collection
var regions = ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2');
var geometry = regions.geometry().bounds();
print(regions, 'regions');
var regions_img = ee.Image().paint(regions,'region');
Map.addLayer(regions, {}, 'Regiões');
Map.addLayer(regions_img.randomVisualizer(), {}, 'regions_img');

// Create an image with the same value as band 'b1' from the asset_base_fundiaria
var base_fundiaria = ee.Image('users/geomapeamentoipam/AUXILIAR/territorios/base_fundiaria_ipam_2024')
  .unmask(); // ocupando vazios mascarados como 0 "sem informação"
Map.addLayer(base_fundiaria.randomVisualizer(),{},'base_fundiaria');
print(base_fundiaria.randomVisualizer(), {}, 'base_fundiaria');

var territory = regions_img.multiply(1000).add(base_fundiaria);
print(territory, 'territory');
Map.addLayer(territory.randomVisualizer(), {}, 'territory');

//legendas
var legends = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends.js');

var  regions = ee.Dictionary(legends.get('fogo_regions_col2')),
     biomes = ee.Dictionary(legends.get('biomas')),
     estados = ee.Dictionary(legends.get('estados')),
     ufs = ee.Dictionary(legends.get('ufs'));
     
//base fundiaria
var legends_territory = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends_base_fundiaria_2024');

var base_fundiaria_nivel_1 = legends_territory.get('base_fundiaria_nivel_1'),
    base_fundiaria_nivel_2 = legends_territory.get('base_fundiaria_nivel_2');
    
print('biomes:', biomes); // Adicionado para verificar se legends_territory está definido corretamente
print('Base Fundiaria Nivel 1:', base_fundiaria_nivel_1); // Adicionado para verificar se base_fundiaria_nivel_1 está definido corretamente
print('Base Fundiaria Nivel 2:', base_fundiaria_nivel_2); // Adicionado para verificar se base_fundiaria_nivel_2 está definido corretamente

/*** Calculate area crossing a cover map and a region map  */

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
            var regionsId = territory.divide(1000).int();
            var territoryId = territory.mod(1000).int();
            var classesAndAreas = ee.List(obj.get('groups'));
        
            var tableRows = classesAndAreas.map(
                function (classAndArea) {
                    classAndArea = ee.Dictionary(classAndArea);
        
                    var classId = classAndArea.getNumber('class');
                    var area_km2 = classAndArea.get('sum');
                    var area_ha = ee.Number(area_km2).multiply(100);
                    var tableColumns = ee.Feature(null)
                        .set('Área km²', area_km2)
                        .set('Área ha', area_ha)
                        //.set('Região', regions.get(regionsId))
                        .set('Bioma', biomes.get(regionsId.divide(10).int()))
                        .set('Base fundiaria nível 1', base_fundiaria_nivel_1.get(territoryId))
                        .set('Base fundiaria nível 2', base_fundiaria_nivel_2.get(territoryId))
                        .set('Codigo', territoryId);
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
                return feature.set({
                    'Ano': year,
                    'Coleção': collection,
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
