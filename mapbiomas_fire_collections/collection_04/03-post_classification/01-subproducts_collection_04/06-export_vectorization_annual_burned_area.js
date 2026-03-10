/**** Start of imports. If edited, may not auto-convert in the playground. ****/
var region = 
    /* color: #d63000 */
    /* shown: false */
    ee.Geometry.Polygon(
        [[[-74.54343726505891, 6.276358376106893],
          [-74.54343726505891, -34.4120391807966],
          [-30.68601539005891, -34.4120391807966],
          [-30.68601539005891, 6.276358376106893]]]);
/***** End of imports. If edited, may not auto-convert in the playground. *****/
/**
 * @Description 
 * MapBiomas Fogo Coleção 4
 * Export do subproduto do fogo de área queimada anual por Uso e Cobertura 
 * Exporta uma imagem binária da cobertura de fogo anual (fogo: 1; não fogo: 0)
 * Uma banda para cada ano do período.
 * 
 */

/*
  acesse este notebook python para executar esta etapa
  colab.research.google.com/drive/1sgkU3j8s4UuYpY63Mfggnx6PSnM5dqsU?usp=sharing
  https://colab.research.google.com/drive/1wef9Jo5LLA2vN6xQbn8IdFCw8j6_NrWZ
*/

print(
  ui.Label(
    'A vetorização em 30 metros de resolução é realizada via gdal com python, '
    + 'após exportar os dados anuais de area queimada, utilize o seguinte '
    + 'Google Colab Notebook para vetorizar as cicatrizes de fogo e retornar-las '
    + 'como features collections anuais.'
  ),
  ui.Label(
    'link para o collab',
    {},
    'https://colab.research.google.com/drive/1CtGZsOWJY-Va1h_GknIf8MJL9Eguo_YY'
  )
);

// Coleção de imagens mensais de áreas queimadas
var assetFire = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-monthly-burned-coverage-v1';
var fire = ee.Image(assetFire);

print(fire);

fire.bandNames().evaluate(function(bandNames){
  
  bandNames.forEach(function(bandName){
    var fire_year = fire.select(bandName).gte(1);
    
    var description = 'mapbiomas_fire_collection4_'+bandName;
    
    Export.image.toDrive({
      image:fire_year,
      description:'GT_Fogo-'+description,
      folder:'mapbiomas_fire_collection4',
      fileNamePrefix:description,
      // dimensions:,
      region:fire.geometry(),
      scale:30,
      // crs:,
      // crsTransform:,
      maxPixels:1e11, 
      // shardSize:,
      // fileDimensions:,
      // skipEmptyTiles:,
      // fileFormat:,
      // formatOptions:
    });
  });
});
