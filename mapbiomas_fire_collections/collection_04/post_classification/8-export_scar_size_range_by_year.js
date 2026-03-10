/**
 * @Description
 * This script analyzes the annually burned area from MapBiomas Fire Collection 2 data.
 * It categorizes burned areas into size ranges and exports the results to Assets.
 * v1: https://code.earthengine.google.com/0f292403e2f7d7043603c970fe37d6ea
 */

// Header
print('Burned Area Analysis by Size Range');

// Load burned area image
var scar_size = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-annual-burned-area_ha-v1');

// Print and add burned area layer to the map
print('Burned area size:', scar_size);
Map.addLayer(scar_size.slice(-1), { min: 0, max: 1000 }, 'Burned Area Size (km²)', false);

// Classify burned area into size ranges
var scar_size_ranges = scar_size
  .multiply(0)
  .where(scar_size.lt(10), 1)
  .where(scar_size.gte(10), 2)
  .where(scar_size.gte(250), 3)
  .where(scar_size.gte(500), 4)
  .where(scar_size.gte(5000), 5)
  .where(scar_size.gte(10000), 6)
  .where(scar_size.gte(50000), 7)
  .where(scar_size.gte(100000), 8);

// Color palette for size ranges
var palette = ["#d0ae35", "#c68e2f", "#b96d2b", "#ac4b28", "#933328", "#72251d", "#4e180f", "#240d00"];

// Add size ranges layer to the map
Map.addLayer(scar_size_ranges.slice(-1), { min: 1, max: 10, palette: palette }, 'Burned Area Size Ranges');

// Export image of size ranges to Asset
var description = 'mapbiomas-fire-collection4-annual-burned-scar-size-range-v1';
var assetId = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/' + description;

Export.image.toAsset({
  image: scar_size_ranges,
  description: description,
  assetId: assetId,
  pyramidingPolicy: 'mode',
  region: scar_size.geometry(),
  scale: 30,
  maxPixels: 1e13
});

// Labels for size ranges in hectares
var labels_ha = [
  [1, '< 10 ha'],
  [2, '10 - 250 ha'],
  [3, '250 - 500 ha'],
  [4, '500 - 5,000 ha'],
  [5, '5,000 - 10,000 ha'],
  [6, '10,000 - 50,000 ha'],
  [7, '50,000 - 100,000 ha'],
  [8, '>= 100,000 ha']
];

// Create legend panel
var legendPanel = labels_ha.map(function(list, i) {
  return ui.Panel([
    ui.Label('O', { backgroundColor: palette[i], margin: '1px' }),
    ui.Label(list[0] + ': ' + list[1], { fontSize: 10, margin: '1px' })
  ], ui.Panel.Layout.Flow('horizontal'), { margin: '1px' });
});

// Subtitle for legend panel
var subtitle = ui.Panel([
  ui.Panel(legendPanel, ui.Panel.Layout.Flow('vertical'), { margin: '1px', stretch: 'both', border: '0.5px solid red' }).insert(0, ui.Label('LEGEND')),
], ui.Panel.Layout.Flow('vertical'), { margin: '1px', position: 'bottom-left' });

// Add legend panel to the map
Map.add(subtitle);
