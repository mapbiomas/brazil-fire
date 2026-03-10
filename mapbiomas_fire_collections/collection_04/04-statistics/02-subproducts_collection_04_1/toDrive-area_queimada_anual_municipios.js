/** @description  - calculate scar fire area for municipalities
 *  
 *    by: Instituto de Pesquisa Ambiental da Amazônia
 *    desenvolvimento: Wallace Silva e Vera Laisa;
 * 
 */

var driverFolder = 'colecao41';
var collection = 'MapBiomas Fogo Coleção 4';
var subproduct = 'Anual';
var scale = 30;
var data = 'col41_municipios';
var description = 'csv-col4_anual-area-municipios';
var mapbiomas = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/2_Subprodutos_col41/mapbiomas-fire-collection41-annual-burned-v1'); // anual burned area
//print(mapbiomas)

// Define a list of years to export
var years = [ 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994,
  1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004,
  2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014,
  2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 
  2024
];


// Image area in km2
var pixelArea = ee.Image.pixelArea().divide(1000000);

// ---------- BASES GEOGRÁFICAS ---------- //
var regions = ee.FeatureCollection('users/geomapeamentoipam/AUXILIAR/regioes_biomas_col2');
var regions_img = ee.Image().paint(regions,'region');
Map.addLayer(regions_img.randomVisualizer(),{},'regioes_e_biomas');

// Municípios
var municipios = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/ESTATISTICAS/COLECAO7/VERSAO-2/city')
  //.limit(10)
  .map(function(feat){return feat.set('feature_id',ee.Number.parse(feat.get('feature_id')))});
var municipios_img = ee.Image().paint(municipios, 'feature_id').unmask(0);
Map.addLayer(municipios_img.randomVisualizer(),{},'municipios_img');  

// Estados (UF)
var states = ee.FeatureCollection('projects/mapbiomas-workspace/AUXILIAR/estados-2017')
  .map(function(feat){return feat.set('CD_GEOCUF',ee.Number.parse(feat.get('CD_GEOCUF')))});
var states_img = ee.Image().paint(states,'CD_GEOCUF');
Map.addLayer(states_img.randomVisualizer(), {}, 'uf_img');

// Converter todas as bases para Double
var uf_img_d         = states_img.toDouble();
var municipios_img_d = municipios_img.toDouble();
var regions_img_d    = regions_img.toDouble();

// ---------- NOVA CODIFICAÇÃO COM BIOMA ---------- //
// territory = municipio * 1e5 + uf * 1e3 + bioma * 1e1 
var territory = municipios_img.multiply(1e5)
  .add(uf_img_d.multiply(1e3))
  .add(regions_img_d.multiply(1e1));

Map.addLayer(territory.randomVisualizer(), {}, 'territory');
var geometry = regions.geometry().bounds();

//legendas
var legends = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends.js');

var regions          = ee.Dictionary(legends.get('fogo_regions_col2')),
    biomes           = ee.Dictionary(legends.get('biomas_id')),
    estados          = ee.Dictionary(legends.get('estados')),
    ufs              = ee.Dictionary(legends.get('ufs')),
    niveis           = ee.Dictionary(legends.get('lulc_mbc10_niveis')),
    nivel_0          = ee.Dictionary(legends.get('lulc_mbc10_nivel_0')),
    nivel_1          = ee.Dictionary(legends.get('lulc_mbc10_nivel_1')),
    nivel_1_1        = ee.Dictionary(legends.get('lulc_mbc10_nivel_1_1')),
    nivel_2          = ee.Dictionary(legends.get('lulc_mbc10_nivel_2')),
    nivel_3          = ee.Dictionary(legends.get('lulc_mbc10_nivel_3')),
    nivel_4          = ee.Dictionary(legends.get('lulc_mbc10_nivel_4'));

var legends_municipios = require('users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/Legends_territorios.js');
var municipiosNM = ee.Dictionary(legends_municipios.get('municipios_name'));

// ---------- FUNÇÕES DE LOOKUP SEGURO ---------- //

// Municipios: chaves string, pular 0
var getMunicipioNome = function(muniCode) {
  muniCode = ee.Number(muniCode).int();
  return ee.String(
    ee.Algorithms.If(
      muniCode.eq(0),
      'SEM_MUNICIPIO',
      municipiosNM.get(muniCode.format())  // chave string
    )
  );
};
// Biomas: chaves numéricas, pular 0
var getBiomaNome = function(biomaCode) {
  biomaCode = ee.Number(biomaCode).int();
  return ee.Algorithms.If(
    biomaCode.eq(0),
    null,
    biomes.get(biomaCode)
  );
};

// UF (sigla) e Estado (nome): chaves numéricas, pular 0
var getUF = function(ufCode) {
  ufCode = ee.Number(ufCode).int();
  return ee.Algorithms.If(
    ufCode.eq(0),
    null,
    ufs.get(ufCode)
  );
};

var getEstado = function(ufCode) {
  ufCode = ee.Number(ufCode).int();
  return ee.Algorithms.If(
    ufCode.eq(0),
    null,
    estados.get(ufCode)
  );
};

/**
 * Calculate area crossing a cover map (deforestation, mapbiomas)
 * and a region map (states, biomes, municipalites)
 * @param image 
 * @param territory 
 * @param geometry
 */
var convert2table = function (obj) {

  obj = ee.Dictionary(obj);

  var territoryVal = ee.Number(obj.get('territory'));
  // territory = municipio * 1e5 + uf * 1e3 + bioma * 1e1 

  var municipioInt  = territoryVal.divide(1e5).floor();
  var resto1    = territoryVal.mod(1e5);
  var uf_int    = resto1.divide(1e3).floor();
  var resto2    = resto1.mod(1e3);
  var bioma_int = resto2.divide(1e1).floor();

  var municipioNome = getMunicipioNome(municipioInt);
  var biomaNome     = getBiomaNome(bioma_int);
  var ufSigla       = getUF(uf_int);
  var ufNome        = getEstado(uf_int);

  var classesAndAreas = ee.List(obj.get('groups'));

  var tableRows = classesAndAreas.map(
    function (classAndArea) {
      classAndArea = ee.Dictionary(classAndArea);

      var classId  = classAndArea.getNumber('class').int();
      var area_km2 = classAndArea.get('sum');
      var area_ha  = ee.Number(area_km2).multiply(100); // km² -> ha

      var tableColumns = ee.Feature(null)
        .set('Cod_municipio', municipioInt)
        .set('Municipio', municipioNome)
        .set('Bioma', biomaNome)
        .set('Área ha', area_ha)
        .set('Área km²', area_km2)
        .set('UF', ufSigla)
        .set('Estados', ufNome)
        
        // .set('Nível 0',  nivel_0.get(classId))
        // .set('Nível 1_1',nivel_1_1.get(classId))
        // .set('Nível 3',  nivel_3.get(classId))
        // .set('Fundiaria n1', fund_n1)
        // .set('Fundiaria n2', fund_n2)
        // .set('Fundiaria n3', fund_n3)
        // .set('Fundiaria Jurisdição', fund_n4)
        // .set('Mês', months_legend.get(month))
        // .set('mes', months_int_legend.get(month))
        ;

      return tableColumns;
    }
  );

  return ee.FeatureCollection(ee.List(tableRows));
};


// ---------- CÁLCULO DE ÁREA ---------- //

/**
 * 
 * Calculate area crossing a cover map (mapbiomas)
 * and territory map (municipio + uf + bioma + base fundiária)
 * @param image 
 * @param territory 
 * @param geometry
 */
var calculateArea = function (image, territory, geometry) {

// Garantir nomes das bandas (opcional, mas ajuda a depurar)
  image     = image.rename('class');
  territory = territory.rename('territory');

  // Ordem das bandas: [0] pixelArea, [1] class, [2] territory
  var stack = pixelArea
    .addBands(image)
    .addBands(territory);

  // Reducer: soma área, agrupando por classe e depois por território
  var reducer = ee.Reducer.sum()
    .group({
      groupField: 1,    // banda 1 = 'class'
      groupName: 'class'
    })
    .group({
      groupField: 2,    // banda 2 = 'territory'
      groupName: 'territory'
    });

  var territotiesData = stack.reduceRegion({
    reducer: reducer,
    geometry: geometry,
    scale: scale,
    maxPixels: 1e12
  });

  territotiesData = ee.List(territotiesData.get('groups'));
  
  var areas = territotiesData.map(convert2table);

  areas = ee.FeatureCollection(areas).flatten();

  return areas;
};

var areas = years.map(
    function (year) {
        var image = mapbiomas.select('burned_area_' + year);

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

areas = ee.FeatureCollection(areas).flatten()
    .filter(ee.Filter.gt('Cod_municipio', 0)); // Remove todos os registros com municipio 0


print(areas.limit(10));

Export.table.toDrive({
    collection: areas,
    description: description,
    folder: driverFolder,
    fileNamePrefix: description,
    fileFormat: 'CSV',
    selectors: ["Ano", "Coleção", "Bioma",  "UF",  "Estados", 
      "Cod_municipio",  "Municipio","Área ha",  "Área km²"]
});
