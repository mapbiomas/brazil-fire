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
 * Export do subproduto do fogo de frequência de área queimada por Uso e Cobertura   
 * Exporta uma imagem com 85 bandas dos diferentes períodos de frequência 
 * Sendo o valor do pixel a classe de frequência multiplicado por 100 e adicionado o valor do Uso e Cobertura do ultimo ano
 */

// Nome dos arquivos de saída
var outFileNameFrequency = 'mapbiomas-fire-collection4-fire-frequency-v1';
var outFileNameFrequencyCoverage = 'mapbiomas-fire-collection4-fire-frequency-coverage-v1';

// Diretório de saída dos assets
var assetOutput = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos';

// Coleção de imagens anuais de áreas queimadas
var annualBurned_col = ee.ImageCollection('projects/ee-geomapeamentoipam/assets/MAPBIOMAS_FOGO/COLECAO_4/Colecao4_fogo_mask_v1');

// MapBiomas Uso e Cobertura 
var coverageAsset = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1');

// Adiciona a banda de classificação de 2023
coverageAsset = coverageAsset.addBands(coverageAsset.select("classification_2023").rename("classification_2024"));

// Períodos especiais de 5, 10 e 15 anos
var specialPeriods = [
    // A cada 5 anos
    [1990, 1995],
    [1995, 2000],
    [2000, 2005],
    [2005, 2010],
    [2010, 2015],
    [2015, 2020],

    // A cada 10 anos
    [1995, 2005],
    [2005, 2015],

    // A cada 15 anos
    [2000, 2015],

];

// Início de 1985
var start = 1985;

// Lista de períodos de 1 a 40 anos a partir de 1985
var periods1 = Array.apply(0, Array(40))
    .map(
        function (_, i) {
            return [start, start + i];
        }
    );
    
// Início de 2024
var start = 2024;

// Lista de períodos de 1 a 40 anos a partir de 2024
var periods2 = Array.apply(0, Array(40))
    .map(
        function (_, i) {
            return [start - i, start];
        }
    );
    
// Concatena todos os períodos
var periods = periods1.concat(periods2).concat(specialPeriods);

// Imprime os períodos
print(periods);

// Definição das paletas de cores
var palettes = ['F8D71F','DAA118','BD6C12','9F360B','810004','4D0709','190D0D'];

// Paletas de cores a serem utilizadas na visualização
var visParams = {
    bands: 'fire_frequency_1985_2024',
    min: 0,
    max: 40,
    palette: palettes,
    format: 'png'
}

// Escala para exportação
var scale = 30;

// Imagem com o número de queimadas anuais
var annualBurned = ee.Image(
    ee.List.sequence (1985,2024,1)
    .iterate(function(year,prev){
      year = ee.Number(year).int()
      var image = annualBurned_col.filter(ee.Filter.eq("year",year)).mosaic().rename(ee.String('burned_coverage_').cat(year))
      return ee.Image(prev).addBands(image)
    },
      ee.Image().select()
  )
);  

// Lista de imagens para cada período definido
var imageListWithCoverage = periods.map(
    function (period) {
        // Cria uma lista com os nomes das bandas para cada ano no período
        var bandNames = ee.List.sequence(period[0], period[1]).map(
            function (year) {
                return ee.String('burned_coverage_').cat(ee.Number(year).int16());
            }
        );

        // Nome da banda de saída
        var outBandName = 'fire_frequency_' + period[0].toString() + '_' + period[1].toString();
        
        // Máscara do ano corrente
        var year = String(period[1]);
        
        // Recupera a cobertura para o ano corrente
        var lastCoverage = coverageAsset.select('classification_' + year);
        
        // Calcula a frequência de queimadas para o período
        var frequency = annualBurned.select(bandNames).gt(0).reduce(ee.Reducer.sum()).rename(outBandName);
        
        // Adiciona a cobertura atual na imagem com a frequência de queimadas
        var image = frequency.multiply(100).add(lastCoverage);

        return image;
    }
);

// Lista de imagens para cada período definido sem cruzamento com uso e cobertura
var imageListWithoutCoverage = periods.map(
    function (period) {
        // Cria uma lista com os nomes das bandas para cada ano no período
        var bandNames = ee.List.sequence(period[0], period[1]).map(
            function (year) {
                return ee.String('burned_coverage_').cat(ee.Number(year).int16());
            }
        );

        // Nome da banda de saída
        var outBandName = 'fire_frequency_' + period[0].toString() + '_' + period[1].toString();
        
        // Calcula a frequência de queimadas para o período
        var frequency = annualBurned.select(bandNames).gt(0).reduce(ee.Reducer.sum()).rename(outBandName);
        
        return frequency;
    }
);

// Imagem final com uso e cobertura
var fireFrequencyCoverage = ee.Image(imageListWithCoverage);
fireFrequencyCoverage = fireFrequencyCoverage.select(fireFrequencyCoverage.bandNames().remove('fire_frequency_1985_2024_1')).uint16();

// Imagem final sem uso e cobertura
var fireFrequency = ee.Image(imageListWithoutCoverage);
fireFrequency = fireFrequency.select(fireFrequency.bandNames().remove('fire_frequency_1985_2024_1')).uint8();

// Print das imagens
print('annualBurned_col', annualBurned_col); 
print('annualBurned', annualBurned);
print('coverageAsset', coverageAsset);
print('fireFrequency', fireFrequency);
print('fireFrequencyCoverage', fireFrequencyCoverage);

// Adiciona as camadas no mapa
Map.addLayer(annualBurned, {}, 'annualBurned', false);
Map.addLayer(fireFrequency, {bands: ['fire_frequency_1985_2024'], min: 1, max: 40, palette: require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Palettes.js').get('frequencia')}, 'fireFrequency');
Map.addLayer(fireFrequencyCoverage, visParams, 'fireFrequencyCoverage');

// Exporta a imagem de frequência sem cruzamento com uso e cobertura
Export.image.toAsset({
    'image': fireFrequency,
    'description': outFileNameFrequency,
    'assetId': assetOutput + '/' + outFileNameFrequency,
    'pyramidingPolicy': { '.default': 'mode' },
    'region': region,
    'scale': scale,
    'maxPixels': 1e13,
});

// Exporta a imagem de frequência com cruzamento com uso e cobertura
Export.image.toAsset({
    'image': fireFrequencyCoverage,
    'description': outFileNameFrequencyCoverage,
    'assetId': assetOutput + '/' + outFileNameFrequencyCoverage,
    'pyramidingPolicy': { '.default': 'mode' },
    'region': region,
    'scale': scale,
    'maxPixels': 1e13,
});
