
/** 
 * @Description 
 * Este script exporta dois conjuntos de produtos derivados de dados de área queimada do MapBiomas Fogo Coleção 4,
 * organizados por uso e cobertura do solo. Os dados são processados para cada mês e cada ano, a partir de imagens
 * que representam as áreas queimadas e as classes de uso e cobertura do MapBiomas Coleção 9.
 * 
 * O valor de cada pixel nas imagens mensais é determinado pelo mês (multiplicado por 100) somado à classificação
 * de uso e cobertura do último ano disponível. As imagens são exportadas como 'monthlyBurnedCoverage' para cobertura
 * queimada mensal e 'annualBurnedCoverage' para cobertura queimada anual.
 * 
 */
 

// Nome do arquivo de saída
var assetOutput = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos';
var scale = 30;

//MapBiomas Uso e Cobertura 
var mapbiomas = ee.Image('projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1');
  
//Mascara com o ano corrente
var outFileName_monthly = 'mapbiomas-fire-collection4-monthly-burned-coverage-v1';
var outFileName_annual = 'mapbiomas-fire-collection4-annual-burned-coverage-v1';

var monthlyBurnedArea_col = ee.ImageCollection('projects/ee-geomapeamentoipam/assets/MAPBIOMAS_FOGO/COLECAO_4/Colecao4_fogo_mask_v1');
mapbiomas = mapbiomas.addBands(mapbiomas.select("classification_2023"));

var monthlyBurnedArea = ee.Image(
    ee.List.sequence (1985,2024,1)
    .iterate(function(year,prev){
      year = ee.Number(year).int()
      var image = monthlyBurnedArea_col.filter(ee.Filter.eq("year",year)).mosaic().rename(ee.String('burned_coverage_').cat(year))
      //return image
      return ee.Image(prev).addBands(image)
    },
      ee.Image().select()
  )
);  

print('monthlyBurnedArea_col',monthlyBurnedArea_col);  
print('monthlyBurnedArea',monthlyBurnedArea);
print('mapbiomas',mapbiomas);

var monthlyBurnedCoverage = monthlyBurnedArea
    .multiply(100)
    .add(mapbiomas)
    .uint16();

var annualBurnedCoverage = monthlyBurnedArea
    .gte(1)
    .multiply(mapbiomas)
    .uint8();

print('monthlyBurnedCoverage',monthlyBurnedCoverage);
// Map.addLayer(monthlyBurnedCoverage,{},'monthlyBurnedCoverage');
// Map.addLayer(annualBurnedCoverage,{},'annualBurnedCoverage');

//Export da imagem final
Export.image.toAsset({
    'image': monthlyBurnedCoverage,
    'description': outFileName_monthly,
    'assetId': assetOutput + '/' + outFileName_monthly,
    'pyramidingPolicy': { '.default': 'mode' },
    'region': region,
    'scale': scale,
    'maxPixels': 1e13,
});

Export.image.toAsset({
    'image': annualBurnedCoverage,
    'description': outFileName_annual,
    'assetId': assetOutput + '/' + outFileName_annual,
    'pyramidingPolicy': { '.default': 'mode' },
    'region': region,
    'scale': scale,
    'maxPixels': 1e13,
});

/** - mapbiomas-toolkit/APP/mapbiomas-fogo-col4
 * 
 * 
**/

var b64 = require('users/workspaceipam/packages:mapbiomas-toolkit/utils/b64');

var options = {

  title_toolkit:'MapBiomas Fogo',
  
  logo: b64.get('logo_mapbiomas_fogo'),
  
  filter:{
    table:'País',
    feature:'Brazil'
  },

  filters:{
    'País':{
      id:'FAO/GAUL_SIMPLIFIED_500m/2015/level0',
      features:ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0").filter(ee.Filter.inList('ADM0_NAME',['Brazil'])), 
      propertie:'ADM0_NAME',
    },

    'Unidade Federativa':{
      id:'projects/mapbiomas-workspace/AUXILIAR/estados-2017',
      features:ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/estados-2017'),
      propertie:'NM_ESTADO',
    },
    'Bioma':{
      id:'projects/mapbiomas-workspace/AUXILIAR/biomas_IBGE_250mil',
      features:ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/biomas_IBGE_250mil'), 
      propertie:'Bioma',
    },
    'Terras Indígenas':{
      id:'users/rayalves/pa_br_tis_funai_2023_',
      features:ee.FeatureCollection('users/rayalves/pa_br_tis_funai_2023_'),
      propertie:'terrai_nom',
    },
    'Assentamentos':{
      id:'projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO8/VERSAO-1/settlements',
      features:ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO8/VERSAO-1/settlements'),
      propertie:'NAME',
    },   
    'Territórios Quilombolas':{
      id:'projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO8/VERSAO-1/quilombos',
      features:ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO8/VERSAO-1/quilombos'),
      propertie:'NAME',
    },   
  },
  datasets: {
      'MapBiomas Coleção 4': [
          {
            type: 'image',
            name: 'Área queimada anual',
            id: 'projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_v1',
            eeObject: ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_v1').gte(1),
            visParams: {
                'color': {
                    'min': 0,
                    'max': 1,
                    'palette': ['800000'],
                    'bands':'burned_area_2024',
                },
            },
          },
          {
            type: 'image',
            name: 'Cobertura queimada anual',
            id: 'projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_coverage_v1',
            eeObject: ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_coverage_v1'),
            visParams: {
              'nível4': {
                  'min': 0,
                  'max': 62,
                  'palette': require('users/mapbiomas/modules:Palettes.js').get('classification9'),
                  'bands': 'burned_coverage_2024',
                  'legend': [
                      // { value: 0, label: "Não observado" },
                      // { value: 1, label: "Floresta" },
                      { value: 3, label: "Formação Florestal" },
                      { value: 4, label: "Formação Savânica" },
                      { value: 5, label: "Mangue" },
                      { value: 49, label: "Restinga Arborizada" },
                      // { value: 10, label: "Formação Natural não Florestal" },
                      { value: 11, label: "Campo Alagado e Área Pantanosa" },
                      { value: 12, label: "Formação Campestre" },
                      { value: 32, label: "Apicum" },
                      { value: 29, label: "Afloramento Rochoso" },
                      { value: 50, label: "Restinga Herbácea" },
                      { value: 13, label: "Outras Formações não Florestais" },
                      // { value: 14, label: "Agropecuária" },
                      { value: 15, label: "Pastagem" },
                      { value: 18, label: "Agricultura" },
                      { value: 19, label: "Lavoura Temporária" },
                      // { value: 39, label: "Soja" },
                      // { value: 20, label: "Cana" },
                      // { value: 40, label: "Arroz (beta)" },
                      // { value: 62, label: "Algodão (beta)" },
                      // { value: 41, label: "Outras Lavouras Temporárias" },
                      { value: 36, label: "Lavoura Perene" },
                  //   { value: 46, label: "Café" },
                  //    { value: 47, label: "Citrus" },
                      // { value: 48, label: "Outras Lavouras Perenes" },
                      { value: 9, label: "Silvicultura" },
                      { value: 21, label: "Mosaico de Usos" },
                      // { value: 22, label: "Área não Vegetada" },
                      { value: 23, label: "Praia, Duna e Areal" },
                      { value: 24, label: "Área Urbanizada" },
                      { value: 30, label: "Mineração" },
                      { value: 25, label: "Outras Áreas não Vegetadas" },
                      // { value: 26, label: "Corpo D'água" },
                      { value: 33, label: "Rio, Lago e Oceano" },
                      { value: 31, label: "Aquicultura" }
                      // { value: 27, label: "Não observado" }
                ],
              },
            },
          },
          {
            type: 'image',
            name: 'Área queimada mensal',
            id: 'projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_monthly_burned_v1',
            eeObject: ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_monthly_burned_v1'),
            visParams: {
              'mensal': {
                'min':0,
                'max':12,
                'bands':'burned_monthly_2024',
                'palette': require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Palettes.js').get('mensal'),
                'legend': [
                    { value: 1, label: "Janeiro" },
                    { value: 2, label: "Fevereiro" },
                    { value: 3, label: "Março" },
                    { value: 4, label: "Abril" },
                    { value: 5, label: "Maio" },
                    { value: 6, label: "Junho" },
                    { value: 7, label: "Julho" },
                    { value: 8, label: "Agosto" },
                    { value: 9, label: "Setembro" },
                    { value: 10, label: "Outubro" },
                    { value: 11, label: "Novembro" },
                    { value: 12, label: "Dezembro" }
                ],
              },
            },
          },
      ],
      'MapBiomas Coleção 31': [
          {
            type: 'image',
            name: 'Área queimada anual',
            id: 'projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection31_annual_burned_v1',
            eeObject: annualBurnedCoverage.gte(1),
            visParams: {
                'color': {
                    'min': 0,
                    'max': 1,
                    'palette': ['ff8080'],
                    'bands':'burned_coverage_2023',
                },
            },
          },
          {
            type: 'image',
            name: 'Cobertura queimada anual',
            id: 'projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection31_annual_burned_coverage_v1',
            eeObject: annualBurnedCoverage,
            visParams: {
              'nível4': {
                  'min': 0,
                  'max': 62,
                  'palette': require('users/mapbiomas/modules:Palettes.js').get('classification8'),
                  'bands': 'burned_coverage_2023',
                  'legend': [
                      // { value: 0, label: "Não observado" },
                      // { value: 1, label: "Floresta" },
                      { value: 3, label: "Formação Florestal" },
                      { value: 4, label: "Formação Savânica" },
                      { value: 5, label: "Mangue" },
                      { value: 49, label: "Restinga Arborizada" },
                      // { value: 10, label: "Formação Natural não Florestal" },
                      { value: 11, label: "Campo Alagado e Área Pantanosa" },
                      { value: 12, label: "Formação Campestre" },
                      { value: 32, label: "Apicum" },
                      { value: 29, label: "Afloramento Rochoso" },
                      { value: 50, label: "Restinga Herbácea" },
                      { value: 13, label: "Outras Formações não Florestais" },
                      // { value: 14, label: "Agropecuária" },
                      { value: 15, label: "Pastagem" },
                      { value: 18, label: "Agricultura" },
                      { value: 19, label: "Lavoura Temporária" },
                      // { value: 39, label: "Soja" },
                      // { value: 20, label: "Cana" },
                      // { value: 40, label: "Arroz (beta)" },
                      // { value: 62, label: "Algodão (beta)" },
                      // { value: 41, label: "Outras Lavouras Temporárias" },
                      { value: 36, label: "Lavoura Perene" },
                  //   { value: 46, label: "Café" },
                  //    { value: 47, label: "Citrus" },
                      // { value: 48, label: "Outras Lavouras Perenes" },
                      { value: 9, label: "Silvicultura" },
                      { value: 21, label: "Mosaico de Usos" },
                      // { value: 22, label: "Área não Vegetada" },
                      { value: 23, label: "Praia, Duna e Areal" },
                      { value: 24, label: "Área Urbanizada" },
                      { value: 30, label: "Mineração" },
                      { value: 25, label: "Outras Áreas não Vegetadas" },
                      // { value: 26, label: "Corpo D'água" },
                      { value: 33, label: "Rio, Lago e Oceano" },
                      { value: 31, label: "Aquicultura" }
                      // { value: 27, label: "Não observado" }
                ],
              },
            },
          },
          {
            type: 'image',
            name: 'Área queimada mensal',
            id: 'projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection31_monthly_burned_v1',
            eeObject: monthlyBurnedCoverage.divide(100).int8(),
            visParams: {
              'mensal': {
                'min':0,
                'max':12,
                'bands':'burned_coverage_2023',
                'palette': require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Palettes.js').get('mensal'),
                'legend': [
                    { value: 1, label: "Janeiro" },
                    { value: 2, label: "Fevereiro" },
                    { value: 3, label: "Março" },
                    { value: 4, label: "Abril" },
                    { value: 5, label: "Maio" },
                    { value: 6, label: "Junho" },
                    { value: 7, label: "Julho" },
                    { value: 8, label: "Agosto" },
                    { value: 9, label: "Setembro" },
                    { value: 10, label: "Outubro" },
                    { value: 11, label: "Novembro" },
                    { value: 12, label: "Dezembro" }
                ],
              },
            },
          },
      ],
  },
};
// start toolkit
require('users/workspaceipam/packages:mapbiomas-toolkit/production').start(options);
// require('users/workspaceipam/packages:mapbiomas-toolkit/staging').start(options);


