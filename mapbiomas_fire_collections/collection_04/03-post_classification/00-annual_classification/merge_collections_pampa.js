
// Importar as coleções de imagens  
var col4_v1 = ee.ImageCollection("projects/ee-geomapeamentoipam/assets/MAPBIOMAS_FOGO/COLECAO_4/Colecao4_fogo_v1_mask")
  .filterMetadata('biome', 'equals', 'PAMPA'); // Filtrar por bioma 
  //.filter(ee.Filter.calendarRange(2024, 2022, 'year')); // Filtrar por anos 
var col4_v2 = ee.ImageCollection("projects/ee-geomapeamentoipam/assets/MAPBIOMAS_FOGO/COLECAO_4/Colecao4_fogo_v2_mask")
  .filterMetadata('biome', 'equals', 'PAMPA'); // Filtrar por bioma Cerrado
  //.filter(ee.Filter.calendarRange(2013, 2022, 'year')); // Filtrar por anos 
  
// Imprimir informações sobre as coleções
print('col4_v2',col4_v2);
print('col4_v1',col4_v1);

// Mesclar as duas coleções de dados de área queimada
var preSave = col4_v2
  .aggregate_array('system:index')
  .sort()
  .iterate(function(current, previous) {
    var index = ee.String(current);
    var img_1 = col4_v1.filter(ee.Filter.stringContains('system:index', index.slice(0,-3))).first();
    var img_2 = col4_v2.filter(ee.Filter.stringContains('system:index', index.slice(0,-3))).first();

    // Verificar se as imagens existem antes de mesclar
    // Soma das imagens binárias e definindo 1 como o valor final
    var img = img_2.blend(img_1)
      .selfMask()
      .copyProperties(img_2)
      .set({
        'system:time_start': img_2.get('system:time_start'),
        'system:time_end': img_2.get('system:time_end'),
        'system:footprint': img_2.get('system:footprint'),
        // 'system:index': index,
      });
    return ee.List(previous).add(img);

  }, []);

var preSave = ee.ImageCollection(ee.List(preSave));

// Exportar a imagem resultante da mesclagem para o Earth Engine Asset
var assetId = "projects/ee-geomapeamentoipam/assets/MAPBIOMAS_FOGO/COLECAO_4/Colecao4_fogo_v31";
preSave
  .aggregate_array('system:index')
  .sort()
  .evaluate(function(indexs){
    if (indexs) {
      indexs.forEach(function(index){
        var img = preSave.filter(ee.Filter.eq('system:index',index)).first();
        
        Export.image.toAsset({
          image:img,
          description:index,
          assetId:assetId + '/' + index,
          pyramidingPolicy:'mode',
          scale:30,
          maxPixels:1e13,
        });
      });
    }
  });
  
// Visualização das coleções
var saved = ee.ImageCollection(assetId);
print('preSave',preSave);

// Filtrar imagens por ano específico
var specific_year = 2024;
var filter_year = ee.Filter.calendarRange(specific_year, specific_year, 'year');

var col4_v2_year = col4_v2.filter(filter_year);
var col4_v1_year = col4_v1.filter(filter_year);

var preSave_year = preSave.filter(filter_year);
print('preSave_year',preSave_year);
var saved_year = saved.filter(filter_year);

// Plotagem no mapa
// Adicionar camada para col4_v2
Map.addLayer(col4_v2_year, {min: 0, max: 1, palette: ['blue']}, "col4 v2 - "+specific_year);

// Adicionar camada para col4_v1
Map.addLayer(col4_v1_year, {min: 0, max: 1, palette: ['red']}, "col4 v1 - "+specific_year);

// Adicionar camada para coleção ainda não salva
Map.addLayer(preSave_year, {min: 0, max: 1, palette: ['green']}, "pré save - "+specific_year);

// Adicionar camada para coleção já salva
Map.addLayer(saved_year, {min: 0, max: 1, palette: ['yellow']}, "já salvo - "+specific_year,false);
