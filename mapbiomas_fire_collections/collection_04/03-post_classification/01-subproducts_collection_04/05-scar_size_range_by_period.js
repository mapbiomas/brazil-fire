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
 * MapBiomas Fire Collection 4
 * Export of subproducts with estimates of the chronosequences of burned areas annually in Brazil.
 * Exports two images with integer values in decimal:
 *  - The first one contains the information of years after fire for the entire mapped area as burned area, at least once,
 *    from 1985 to 2024;
 *  - The second one contains the information of years before fire for the entire mapped area as burned area, at least once,
 *    from 2024 to 1985;
 * Both data have
 */

// Monthly burned area image collection
var assetFire = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos/mapbiomas-fire-collection4-monthly-burned-coverage-v1';

// Output file names
var outFileName_taf = 'mapbiomas-fire-collection4-time-after-fire-v1';
var outFileName_tbf = 'mapbiomas-fire-collection4-time-before-fire-v1';
var outFileName_ylf = 'mapbiomas-fire-collection4-year-last-fire-v1';
var outFileName_ynf = 'mapbiomas-fire-collection4-year-next-fire-v1';

// Output directory of assets
var assetOutput = 'projects/mapbiomas-workspace/FOGO_COL4/1_Subprodutos';

// Export.image.toAsset() params
var scale = 30;

// Converting monthly burned coverage to time after fire
var scars = ee.Image(assetFire).gte(1);

// Iteration over years to calculate time after fire
var years = scars.bandNames().map(function(bandname) {
  return ee.Number.parse(ee.String(bandname).slice(-4));
});

// Function to calculate time after fire
function timeAfterFire(current, previous) {
  var year = ee.Number(current).int();
  var yearPost = year.add(1);

  var sliceIndex = years.indexOf(current);
  sliceIndex = ee.Number(sliceIndex).add(1);
  var sliceList = years.slice(0, sliceIndex, 1);

  var alreadyBurned = sliceList.map(function(y) {
    return scars.select([ee.String('burned_coverage_').cat(ee.Number(y).int())])
      .rename('classification');
  });
  alreadyBurned = ee.ImageCollection(alreadyBurned).mosaic().byte();
  alreadyBurned = ee.Image(1).updateMask(alreadyBurned).byte();

  var burnedThisYear = ee.Image(1).updateMask(scars.select([ee.String('burned_coverage_').cat(year)]))
    .byte();

  var newImage = ee.Image(previous)
    .select(ee.String('classification_').cat(year))
    .add(alreadyBurned).byte();

  newImage = newImage.blend(burnedThisYear);

  return ee.Image(previous)
    .addBands(
      newImage.rename(ee.String('classification_').cat(yearPost))
    );
}

var first = ee.Image(0).mask(0).rename('classification_1985');
var timeAfterFireImage = years.iterate(timeAfterFire, first);
timeAfterFireImage = ee.Image(timeAfterFireImage).slice(1,-1);
// print('timeAfterFireImage',timeAfterFireImage);
// Iteration over years to calculate time before fire
years = years.reverse();

function timeBeforeFire(current, previous) {
  var year = ee.Number(current).int();
  var yearPost = year.subtract(1);

  var sliceIndex = years.indexOf(current);
  sliceIndex = ee.Number(sliceIndex).add(1);
  var sliceList = years.slice(0, sliceIndex, 1);

  var alreadyBurned = sliceList.map(function(y) {
    return scars.select([ee.String('burned_coverage_').cat(ee.Number(y).int())])
      .rename('classification');
  });
  alreadyBurned = ee.ImageCollection(alreadyBurned).mosaic().byte();
  alreadyBurned = ee.Image(1).updateMask(alreadyBurned).byte();

  var burnedThisYear = ee.Image(1).updateMask(scars.select([ee.String('burned_coverage_').cat(year)]))
    .byte();

  var newImage = ee.Image(previous)
    .select(ee.String('classification_').cat(year))
    .add(alreadyBurned).byte();

  newImage = newImage.blend(burnedThisYear);

  return ee.Image(previous)
    .addBands(
      newImage.rename(ee.String('classification_').cat(yearPost))
    );
}

var first = ee.Image(0).mask(0).rename('classification_2024');
var timeBeforeFireImage = years.iterate(timeBeforeFire, first);
timeBeforeFireImage = ee.Image(timeBeforeFireImage);
timeBeforeFireImage = timeBeforeFireImage
  .select(timeBeforeFireImage.bandNames().reverse()).slice(1,-1);
// print('timeBeforeFireImage',timeBeforeFireImage);


// adicionando o valor 0 nas areas de "fogo inedito" (primeira vez que surge o fogo)

timeAfterFireImage = scars.multiply(0).slice(1).blend(timeAfterFireImage).rename(timeAfterFireImage.bandNames());
timeBeforeFireImage = scars.multiply(0).slice(0,-1).blend(timeBeforeFireImage).rename(timeBeforeFireImage.bandNames());

print('timeAfterFireImage',timeAfterFireImage);
print('timeBeforeFireImage',timeBeforeFireImage);

// Visualization
var palette = [
  "FFFFFF", "800000", "850708", "8B0E0F", "901417", "961B1E",
  "9B2226", "9F2222", "A3211E", "A6211A", "AA2016", "AE2012",
  "B42E0F", "B93C0C", "BF4B08", "C45905", "CA6702", "BF7C27",
  "B4924D", "AAA772", "9FBD98", "94D2BD", "78C5B5", "5DB9AD",
  "41ACA6", "26A09E", "0A9460", "0876AB", "0658C0", "043BD5",
  "021DEA", "0000FF", "0000E6", "0000CC", "0000B3", "000080"
];


var year_vis = 2010;
var visParams = {
  bands: ['classification_' + year_vis],
  min: 0,
  max: 40,
  palette: palette
};

var visParams_2 = {
  bands: ['classification_' + year_vis],
  min: 1985,
  max: 2024,
  palette: palette
};

Map.addLayer(timeAfterFireImage, visParams, 'Time After Fire');
Map.addLayer(timeBeforeFireImage, visParams, 'Time Before Fire', false);
Map.addLayer(scars, {}, 'Burned Areas', false);

// Exporting images to Asset
Export.image.toAsset({
  image: timeAfterFireImage.uint8(),
  description: outFileName_taf,
  assetId: assetOutput + '/' + outFileName_taf,
  pyramidingPolicy: 'mode',
  region: scars.geometry(),
  scale: 30,
  maxPixels: 1e13,
});

Export.image.toAsset({
  image: timeBeforeFireImage.uint8(),
  description: outFileName_tbf,
  assetId: assetOutput + '/' + outFileName_tbf,
  pyramidingPolicy: 'mode',
  region: scars.geometry(),
  scale: 30,
  maxPixels: 1e13,
});

// year-last-fire e year-next-fire

var year_last_fire = ee.Image(timeAfterFireImage.bandNames().sort().iterate(function(curr,prev){
  var bandname = ee.String(curr);
  var year = ee.Number.parse(bandname.slice(-4));
  var img = ee.Image(year).rename(bandname).subtract(timeAfterFireImage.select(bandname).selfMask());
  return ee.Image(prev).addBands(img);
},ee.Image().select())).reproject('EPSG:4326', null, 30);
print('year_last_fire',year_last_fire);
Map.addLayer(year_last_fire, visParams_2, 'Year Last Fire');

Export.image.toAsset({
  image: year_last_fire.uint16(),
  description: outFileName_ylf,
  assetId: assetOutput + '/' + outFileName_ylf,
  pyramidingPolicy: 'mode',
  region: scars.geometry(),
  scale: 30,
  maxPixels: 1e13,
});


var year_next_fire = ee.Image(timeBeforeFireImage.bandNames().sort().iterate(function(curr,prev){
  var bandname = ee.String(curr);
  var year = ee.Number.parse(bandname.slice(-4));
  var img = ee.Image(year).rename(bandname).add(timeBeforeFireImage.select(bandname).selfMask());
  return ee.Image(prev).addBands(img);
},ee.Image().select())).reproject('EPSG:4326', null, 30);
print('year_next_fire',year_next_fire);
Map.addLayer(year_next_fire, visParams_2, 'Year Next Fire', false);

Export.image.toAsset({
  image: year_next_fire.uint16(),
  description: outFileName_ynf,
  assetId: assetOutput + '/' + outFileName_ynf,
  pyramidingPolicy: 'mode',
  region: scars.geometry(),
  scale: 30,
  maxPixels: 1e13,
});
