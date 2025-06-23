// SUBPRODUCTS OF ANNUAL BURNED AREA - MAPBIOMAS FIRE COLLECTION 4
// contact: contato@mapbiomas.org

var palettes = require('users/mapbiomas/modules:Palettes.js');
var fire_palettes = require('users/workspaceipam/packages:mapbiomas-toolkit/utils/palettes');

var visParams = {
  'annual_burned':{
    'min':1,
    'max':1,
    'palette':['#ff0000'],
    'bands':'burned_area_2024'
  },
  'annual_burned_coverage':{
    'min':0,
    'max':69,
    'palette':palettes.get('classification9'),
    'bands':['burned_coverage_2024']
  },
  'burned_monthly':{
    'min':1,
    'max':12,
    'palette':fire_palettes.get('mensal'),
    'bands':['burned_monthly_2024']
  },
  'scar_size_range':{
    'min':1,
    'max':10,
    'palette':fire_palettes.get('tamanho_n2'),
    'bands':'scar_area_ha_2024'
  },
  'vector_annual_burned':{
    'color':'#808080'
  },
};

var annual_burned = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_v1');

Map.addLayer(annual_burned,visParams.annual_burned,'annual_burned 2024');

var annual_burned_coverage = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_coverage_v1');

Map.addLayer(annual_burned_coverage,visParams.annual_burned_coverage,'annual_burned 2024');

var annual_burned_coverage = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_coverage_v1');

Map.addLayer(annual_burned_coverage,visParams.annual_burned_coverage,'annual burned coverage 2024');

var burned_monthly = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_monthly_burned_v1');

Map.addLayer(burned_monthly,visParams.burned_monthly,'burned monthly 2024');

var scar_size_range = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_scar_size_range_v1');

Map.addLayer(scar_size_range,visParams.scar_size_range,'scar size range 2024');

var vector_annual_burned = ee.FeatureCollection('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_vectors_v1/mbfogo_col4_2024_v1');

Map.addLayer(vector_annual_burned,visParams.vector_annual_burned,'vector annual burned 2024');
