/** @description  - calculate scar fire area for municipalities
 *  
 *    by: Instituto de Pesquisa Ambiental da Amazônia
 *    desenvolvimento: Wallace Silva e Vera Laisa;
 * 
 */

var driverFolder = 'colecao4';
var collection = 'MapBiomas Fogo Coleção 3.1';
var subproduct = 'Qualidade da pastagem por area queimada acumulada';
var scale = 30;
var data = 'col31';
var description = 'csv-col31_acumullated_pasture_vigor_biome';
var burnedArea = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_accumulated_burned_v1'); // burned cover accumulated area
var qualityPasture = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_pasture_quality_v1');


print("qualityPasture",qualityPasture);

var palette_quality = [
    '#A61C00', // 1. Low
    '#FDAE61', // 2. Medium
    '#2D7BB6', // 3. High
];

//   'pasture': [
//       '#ffffff',
//       '#ffd738',
//   ],

Map.addLayer(qualityPasture.slice(-1),{min:0,max:3,palette:palette_quality},'quality pasture 2023');
Map.addLayer(burnedArea.select("fire_accumulated_1985_2023"),{palette:"red"},'fire_accumulated_1985_2023');

// Define a list of years to export
var years = [2023];
print("qualityPasture.slice(-1)",qualityPasture.slice(-1));

// print(mapbiomas)
// Image area in km2
var pixelArea = ee.Image.pixelArea().divide(1000000);

var regions = ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2');
var regions_img = ee.Image().paint(regions,'region');

var municipios = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/city')
  //.limit(10)
  .map(function(feat){
    return feat.set('feature_id',ee.Number.parse(feat.get('feature_id')));
  });
  
var biomas = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/biomas_IBGE_250mil');
var biomas_img = ee.Image().paint(biomas,'CD_Bioma');
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

var territory = biomas_img;
  // .multiply(10000000).add(municipios_img);
Map.addLayer(territory.randomVisualizer(),{},'territory');
//Map.addLayer(regions, {}, 'Regiões');
Map.addLayer(municipios, {}, 'municipios');

var geometry = biomas.geometry().bounds();

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

var leg_pasture_quality = ee.Dictionary({
  1:"1. Baixo", // '#A61C00'
  2:"2. Médio", // '#FDAE61'
  3:"3. Alto", // '#2D7BB6'
});

var leg_situation = ee.Dictionary({
  0:"Não fogo", // 
  1:"Fogo", //
});




// var legends_municipios = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends_territorios.js');

// var municipiosNM = ee.Dictionary(legends_municipios.get('municipios_name'));

/**
 * Calculate area crossing a cover map (deforestation, mapbiomas)
 * and a region map (states, biomes, municipalites)
 * @param image 
 * @param territory 
 * @param geometry
 */
 print('burnedArea',burnedArea);
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
        var classesAndAreas = ee.List(obj.get('groups'));
    
        var tableRows = classesAndAreas.map(
            function (classAndArea) {
                classAndArea = ee.Dictionary(classAndArea);
    
                var classId = classAndArea.getNumber('class');


                var lulcId = classId.divide(10).int();
                var pastureId = classId.mod(10).int();
                
    
    var leg_situation = ee.Dictionary({
  0:"Não fogo", // 
  1:"Fogo", //
});

            var area = classAndArea.get('sum');
    
                var tableColumns = ee.Feature(null)
                    .set('Área km²', area)
                    .set('Bioma', biomes.get(territory))
                    .set('Fogo', leg_situation.get(lulcId))
                    .set('Vigor da pastagem', pastureId)
                    .set('Vigor da pastagem', leg_pasture_quality.get(pastureId))
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
        var image = burnedArea.select('fire_accumulated_1985_' + year).gte(1).unmask()
          .multiply(10)
          .add(qualityPasture.select("pasture_quality_" + year));
          
        Map.addLayer(image.randomVisualizer(),{},'cruzamento ' + year);

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
