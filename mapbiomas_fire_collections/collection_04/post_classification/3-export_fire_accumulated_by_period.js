/**
 * @Description 
 * MapBiomas Fogo Coleção 4
 * Este script exporta dois subprodutos de dados acumulados de área queimada:
 * 1. Dados acumulados de área queimada sem cruzamento com uso e cobertura (0 e 1):
 *    - Representa o total de vezes que um mesmo pixel teve evento de fogo.
 *    - Cada banda indica o número de vezes que um pixel queimou em um período específico.
 *    - Pixels têm valores binários: 0 (não queimado) e 1 (queimado).
 * 
 * 2. Dados acumulados de área queimada com cruzamento com uso e cobertura:
 *    - Representa o total de vezes que um mesmo pixel teve evento de fogo, considerando as classes de uso e cobertura.
 *    - Cada banda indica o número de vezes que um pixel queimou em um período específico, mascarado pela última classe de uso e cobertura.
 */

// Nome dos arquivos de saída
var outFileNameAccumulated = 'mapbiomas-fire-collection4-accumulated-burned-v1'; 
var outFileNameAccumulatedCoverage = 'mapbiomas-fire-collection4-accumulated-burned-coverage-v1'; 

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
var palettes = ['c22121'];

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
        var outBandName = 'fire_accumulated_' + period[0].toString() + '_' + period[1].toString();
        
        // Mascara do ano corrente
        var year = String(period[1]);
        
        // Recupera a cobertura para o ano 
        var lastCoverage = coverageAsset
            .select('classification_' + year);
        
        // Calcula a frequência de queimadas para o período
        var frequency = annualBurned
            .select(bandNames)
            .gt(0)
            .reduce(ee.Reducer.sum());
        
        // Adiciona o uso e cobertura atual mascarada pela área acumulada de queimadas
        var image = lastCoverage
          .updateMask(frequency.gte(1))
          .rename(outBandName);

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
        var outBandName = 'fire_accumulated_' + period[0].toString() + '_' + period[1].toString();
        
        // Calcula a frequência de queimadas para o período e transforma em binário
        var binaryFrequency = annualBurned
            .select(bandNames)
            .gt(0)
            .reduce(ee.Reducer.sum())
            .gt(0)
            .rename(outBandName);

        return binaryFrequency;
    }
);

// Imagem final com uso e cobertura
var fireAccumulatedCoverage = ee.Image(imageListWithCoverage);
fireAccumulatedCoverage = fireAccumulatedCoverage.select(fireAccumulatedCoverage.bandNames().remove('fire_accumulated_1985_2024_1')).uint8();

// Imagem final sem uso e cobertura
var fireAccumulated = ee.Image(imageListWithoutCoverage).uint8();
fireAccumulated = fireAccumulated.select(fireAccumulated.bandNames().remove('fire_accumulated_1985_2024_1')).uint8();

// Print das imagens
print('annualBurned_col', annualBurned_col); 
print('annualBurned', annualBurned);
print('coverageAsset', coverageAsset);
print('fireAccumulated', fireAccumulated);
print('fireAccumulatedCoverage', fireAccumulatedCoverage);

// Adiciona as camadas no mapa
Map.addLayer(annualBurned, {}, 'annualBurned', false);
Map.addLayer(fireAccumulated, {bands: 'fire_accumulated_1985_2024',min: 0, max: 1, palette: ['white', 'black']}, 'fireAccumulated');
Map.addLayer(fireAccumulatedCoverage, {bands: 'fire_accumulated_1985_2024',min: 0, max: 62, palette: require('users/mapbiomas/modules:Palettes.js').get('classification8')}, 'fireAccumulatedCoverage');

// Exporta a imagem acumulada sem cruzamento com uso e cobertura
Export.image.toAsset({
    'image': fireAccumulated,
    'description': outFileNameAccumulated,
    'assetId': assetOutput + '/' + outFileNameAccumulated,
    'pyramidingPolicy': { '.default': 'mode' },
    'region': region,
    'scale': scale,
    'maxPixels': 1e13,
});

// Exporta a imagem acumulada com cruzamento com uso e cobertura
Export.image.toAsset({
    'image': fireAccumulatedCoverage,
    'description': outFileNameAccumulatedCoverage,
    'assetId': assetOutput + '/' + outFileNameAccumulatedCoverage,
    'pyramidingPolicy': { '.default': 'mode' },
    'region': region,
    'scale': scale,
    'maxPixels': 1e13,
});
