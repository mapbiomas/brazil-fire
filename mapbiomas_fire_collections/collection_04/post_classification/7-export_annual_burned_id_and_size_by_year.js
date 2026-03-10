/**
 * @Description
 * MapBiomas Fire Collection 4
 * Export of subproducts with estimates of the chronosequences of burned areas annually in Brazil.
 * Exports two images with integer values in decimal:
 *  - The first one contains the ID of burned areas annually.
 *  - The second one contains the area of burned areas annually in hectares.
 * Both data have
 */

// Monthly burned area image collection
var assetFire = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-monthly-burned-coverage-v1';

// Output file names
var outFileName_id = 'mapbiomas-fire-collection4-annual-burned-id-v1';
var outFileName_area_ha = 'mapbiomas-fire-collection4-annual-burned-area_ha-v1';

// Output directory of assets
var assetOutput = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/';

// Export.image.toAsset() params
var scale = 30;

// Region for export
var region = ee.Image('projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-accumulated-burned-coverage-v1')
    .geometry();


ee.List.sequence(1985, 2024, 1).evaluate(function(years) {

  var address_to_replace = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-annual-burned-vectors/mbfogo-col4-YEAR-v1';

  var scar_id = ee.Image().select();
  var scar_area = ee.Image().select();

  years.forEach(function(year) {
    var scar_vector_year = ee.FeatureCollection(address_to_replace.replace('YEAR', '' + year))
      .map(function(feat){
        return feat.set({
          area_ha:feat.geometry().area().int().divide(10000),
        });
      });

    // Print limited scar_vector_year
    // print(year, scar_vector_year.limit(10));
    // Add scar_vector_year as a layer to the map
    // Map.addLayer(scar_vector_year, {}, '' + year, false);

    // Add scar ID bands
    scar_id = scar_id.addBands(ee.Image().paint(scar_vector_year, 'id').rename('scar_id_' + year).int());
    // Add scar area bands
    scar_area = scar_area.addBands(ee.Image().paint(scar_vector_year, 'area_ha').rename('scar_area_ha_' + year).float());
  });

  // Add all scar ID bands as a layer to the map
  Map.addLayer(scar_id.aside(print, 'scar_id'), {}, 'All Scar ID', false);
  // Add all scar area bands as a layer to the map
  Map.addLayer(scar_area.aside(print, 'scar_area_ha'), {}, 'All Scar Area', false);

  // Add a random visualizer for one scar ID band as a layer to the map
  Map.addLayer(scar_id.slice(-1).randomVisualizer().aside(print, 'scar_id'), {}, 'Scar ID');
  // Add one scar area band as a layer to the map with custom visualization parameters
  Map.addLayer(scar_area.slice(-1).aside(print, 'scar_area_ha'), { min: 100, max: 100000 }, 'Scar Area');
  
  // Exporting images to Asset
  [
    [outFileName_id, scar_id, 'mode'],
    [outFileName_area_ha, scar_area, 'median'],
  ].forEach(function(list) {

    Export.image.toAsset({
      image: list[1],
      description: 'GT_Fogo-' + list[0],
      assetId: assetOutput + list[0],
      pyramidingPolicy: list[2],
      region: region,
      scale: 30,
      maxPixels: 1e11,
    });
  });
});
