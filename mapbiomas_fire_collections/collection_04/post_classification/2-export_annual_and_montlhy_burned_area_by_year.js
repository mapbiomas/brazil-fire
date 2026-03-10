//v1:https://code.earthengine.google.com/5c7ca504a7edc64e17bac0ed44938361
/**
 * @Description 
 * Este script realiza a exportação de duas imagens:
 * 1. Uma imagem com 40 bandas de queimada anual binária, onde fogo é representado pelo valor 1 e ausência de fogo pelo valor 0.
 * 2. Uma imagem com 40 bandas da área queimada mensal para cada ano, com valores de 1 a 12, representando a presença de fogo em cada mês do ano.
 */

// Coleção de imagens mensais de áreas queimadas
var assetFire = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection41-monthly-burned-coverage-v1';

// Nome do arquivo de saída para anual
var outFileNameAnnual = 'mapbiomas-fire-collection4-annual-burned-v1';
// Nome do arquivo de saída para mensal
var outFileNameMonthly = 'mapbiomas-fire-collection4-monthly-burned-v1';

// Diretório de saída dos assets
var assetOutput = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos';

// Anos
var years = [
    '1985', '1986', '1987', '1988', '1989', '1990',
    '1991', '1992', '1993', '1994', '1995', '1996',
    '1997', '1998', '1999', '2000', '2001', '2002',
    '2003', '2004', '2005', '2006', '2007', '2008',
    '2009', '2010', '2011', '2012', '2013', '2014',
    '2015', '2016', '2017', '2018', '2019', '2020',
    '2021', '2022', '2023', '2024'
];

// Export.image.toAsset() params
var scale = 30;

// Convertendo cobertura queimada mensal para área queimada anual
var monthlyBurnedArea = ee.Image(assetFire);
print('monthlyCoverageBurnedArea', monthlyBurnedArea);

// Renomear bandas mensais
var monthlyBandNames = monthlyBurnedArea.bandNames().map(
    function (bandName) {
        return ee.String(bandName).replace('burned_coverage_', 'burned_monthly_')
    }
);

var renamedMonthlyBurnedArea = monthlyBurnedArea.rename(monthlyBandNames).select(monthlyBandNames).divide(100).uint8();
print('monthlyBurnedArea', renamedMonthlyBurnedArea);

var annualBandNames = monthlyBandNames.map(
    function (bandName) {
        return ee.String(bandName).replace('burned_monthly_', 'burned_area_')
    }
);

var annualBurnedArea = renamedMonthlyBurnedArea
    .gt(0)
    .rename(annualBandNames)
    .uint8();

print('annualBurnedArea', annualBurnedArea);

Map.addLayer(annualBurnedArea, {palette: ['000000'], bands: ['burned_area_2024']}, 'burned_area_2024');

// Visualização dos dados mensais
Map.addLayer(renamedMonthlyBurnedArea.select('burned_monthly_2024'), {
    min: 0,
    max: 12,
    palette: require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Palettes.js').get('mensal')
}, 'monthly_burned_area_2024');

// Export da imagem anual
Export.image.toAsset({
    'image': annualBurnedArea,
    'description': outFileNameAnnual,
    'assetId': assetOutput + '/' + outFileNameAnnual,
    'pyramidingPolicy': { '.default': 'mode' },
    'region': region,
    'scale': 30,
    'maxPixels': 1e13,
});

// Export da imagem mensal
Export.image.toAsset({
    'image': renamedMonthlyBurnedArea,
    'description': outFileNameMonthly,
    'assetId': assetOutput + '/' + outFileNameMonthly,
    'pyramidingPolicy': { '.default': 'mode' },
    'region': region,
    'scale': 30,
    'maxPixels': 1e13,
});

