// SUBPRODUCTS OF ACCUMULATED BURNED AREA PER PERIOD - MAPBIOMAS FIRE COLLECTION 4
// contact: contato@mapbiomas.org
 
var palettes = require('users/mapbiomas/modules:Palettes.js');
var fire_palettes = require('users/workspaceipam/packages:mapbiomas-toolkit/utils/palettes');

var visParams = {
  'accumulated_burned':{
    'min':1,
    'max':1,
    'palette':['#ff0000'],
    'bands':'fire_accumulated_1985_2024'
  },
  'accumulated_burned_coverage':{
    'min':0,
    'max':69,
    'palette':palettes.get('classification9'),
    'bands':['fire_accumulated_1985_2024']
  },
  'fire_frequency':{
    'min':1,
    'max':40,
    'palette':fire_palettes.get('frequencia'),
    'bands':['fire_frequency_1985_2024']
  },
  'year_last_fire':{
    'min':1985,
    'max':2022,
    'palette':fire_palettes.get('ano_do_ultimo_fogo'),
    'bands':['classification_2023']
  },
  'time_after_fire':{
    'min':0,
    'max':39,
    'palette':fire_palettes.get('ultimo_fogo'),
    'bands':['classification_2023']

  },
};

var accumulated_burned = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_accumulated_burned_v1');

Map.addLayer(accumulated_burned,visParams.accumulated_burned,'accumulated burned 2024');

var accumulated_burned_coverage = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_accumulated_burned_coverage_v1');

Map.addLayer(accumulated_burned_coverage,visParams.accumulated_burned_coverage,'accumulated burned coverage 2024');

var fire_frequency = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_fire_frequency_v1');

Map.addLayer(fire_frequency,visParams.fire_frequency,'fire frequency 2024');

var year_last_fire = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_year_last_fire_v1');

Map.addLayer(year_last_fire,visParams.year_last_fire,'year last fire 2023');

var time_after_fire = ee.Image('projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_time_after_fire_v1');

Map.addLayer(time_after_fire,visParams.time_after_fire,'time after fire 2023');
