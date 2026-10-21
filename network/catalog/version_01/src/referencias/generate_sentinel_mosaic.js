/**
 * @description Generate Sentinel-2 quality mosaics for MapBiomas Fire Monitor
 *   Monthly and yearly mosaics per country, exported to:
 *
 *   LULC masks applied uniformly to the full country:
 *     26 = Water bodies
 *     22 = Non-vegetated area / bare soil
 *     33 = River, lake and ocean
 *     24 = Urban infrastructure
 *   - GEE Asset: projects/mapbiomas-mosaics/assets/SENTINEL/FIRE/mosaics-countries/
 *   - GCS Bucket: mapbiomas-fire/sudamerica/{country}/monitor/library_images/
 *
 *   stable version: 2026 - MapBiomas Fire Sentinel Monitor
 *   development: @IPAM - Brasília, DF - BR
 *     Wallace Silva, Vera Laísa
 *   contact: wallace.silva@ipam.org.br
 */

// ─── CONFIGURATION ───────────────────────────────────────────────────────────

var config = {
  country: 'peru',
  ee_project: 'mapbiomas-peru',

  // GEE Asset destination
  asset_folder_monthly: 'projects/mapbiomas-mosaics/assets/SENTINEL/FIRE/quaity_mosaics_nbr_countries_monthly-01',
  asset_folder_yearly:  'projects/mapbiomas-mosaics/assets/SENTINEL/FIRE/quaity_mosaics_nbr_countries_yearly-01',

  // GCS destination
  bucket: 'mapbiomas-fire',
  bucket_base: 'sudamerica/peru/monitor/library_images',

  // Grid tile size in degrees (~92km, 1/4 of Landsat WRS scene ~185km)
  // At 10m resolution: ~9250 x 9250 pixels per tile
  tile_size_deg: 0.83,

  // Years to process
  years: [2024],

  // Months to process (null = annual only)
  months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],

  // Process flags
  export_monthly: true,
  export_yearly: true,
  export_to_asset: true,
  export_to_gcs: true,
};

// ─── BAND DEFINITIONS ─────────────────────────────────────────────────────────

// Sentinel-2 Band mapping
var S2_BANDS_IN  = ['B2',   'B3',    'B4',  'B8',  'B11',   'B12'  ];
var S2_BANDS_OUT = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];

// All bands stored in mosaic (spectral + temporal flag)
// Spectral: divide(100).byte() → 0–100 range (reflectance × 100, base-10 divisor)
// dayOfYear: int16, 1–366, no conversion
var BANDS_SPECTRAL = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];
var BANDS_ALL      = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'dayOfYear'];

// Default model bands (always-on defaults + optional extras)
// User selects at training/classification time
var BANDS_MODEL_DEFAULT = ['red', 'nir', 'swir1', 'swir2'];

// NOTE: LULC mask (classes 26, 22, 33, 24) is applied to the CLASSIFICATION OUTPUT,
// not to the mosaic. See M5 post-classification script / masks reference script.


// ─── DATA SOURCES ─────────────────────────────────────────────────────────────

var S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED';
var CS_PLUS       = 'GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED';
var CS_THRESHOLD  = 0.40;

// Fire focus buffer — covers all of South America
// Used to restrict monthly mosaics to potential fire areas
var FOCUS_BUFFER_ASSET = 'projects/workspace-ipam/assets/BUFFER-DOUBLE-MONTHLY-FOCUS-OF-INPE-SULAMERICA';

// Region boundary
var regions_fc = ee.FeatureCollection(
  'projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1'
);
var country_geometry = regions_fc.geometry();
var country_bounds   = country_geometry.bounds();

// ─── FUNCTIONS ────────────────────────────────────────────────────────────────

/**
 * Apply Cloud Score+ mask and rename S2 bands to standard names.
 * Raw S2_SR_HARMONIZED values: 0–10000 (reflectance × 10000).
 * Returns image with bands: [blue, green, red, nir, swir1, swir2] — int16 (0–10000).
 */
function mask_and_rename(image) {
  var mask = image.select('cs').gte(CS_THRESHOLD);
  return image
    .updateMask(mask)
    .select(S2_BANDS_IN, S2_BANDS_OUT);
}

/**
 * Add inverted NBR band for quality mosaic composition.
 * Inverted so burned areas (low NBR) get HIGH values → selected by qualityMosaic().
 * Formula: (-1 × NBR + 1) × 1000 → burned areas near 2000, vegetation near 0.
 */
function add_nbr(image) {
  var nbr = image
    .expression('( b("nir") - b("swir2") ) / ( b("nir") + b("swir2") )')
    .multiply(-1)
    .add(1)
    .multiply(1000)
    .int16()
    .rename('nbr');
  return image.addBands(nbr);
}

/**
 * Add dayOfYear band from image acquisition date.
 * Used as temporal flag: value = Julian day (1–366).
 */
function add_day_of_year(image) {
  var doy = ee.Image(
    ee.Number.parse(
      ee.Date(image.get('system:time_start')).format('D')
    )
  ).int16().rename('dayOfYear');
  return image.addBands(doy);
}

/**
 * Full pre-processing pipeline for a single S2 image.
 */
function preprocess(image) {
  image = mask_and_rename(image);
  image = add_nbr(image);
  image = add_day_of_year(image);
  return image;
}

/**
 * Build the monthly fire focus mask for a given year and month.
 * The buffer asset covers all of South America.
 */
function get_focus_mask(year, month) {
  var buffer = ee.ImageCollection(FOCUS_BUFFER_ASSET)
    .filter(ee.Filter.eq('year', year))
    .filter(ee.Filter.eq('month', month))
    .mean();
  return buffer;
}

/**
 * Generate a grid of tiles covering the country geometry.
 * tile_size: side length in degrees (default ~0.83° ≈ 92km ≈ 1/4 Landsat scene).
 * Returns a FeatureCollection with properties: tile_id, col, row.
 */
function generate_grid(geometry, tile_size) {
  tile_size = tile_size || config.tile_size_deg;
  var bounds        = geometry.bounds();
  var coords        = ee.List(bounds.coordinates().get(0));
  var xmin = ee.Number(ee.List(coords.get(0)).get(0));
  var ymin = ee.Number(ee.List(coords.get(0)).get(1));
  var xmax = ee.Number(ee.List(coords.get(2)).get(0));
  var ymax = ee.Number(ee.List(coords.get(2)).get(1));

  var ncols = xmax.subtract(xmin).divide(tile_size).ceil();
  var nrows = ymax.subtract(ymin).divide(tile_size).ceil();

  var cols = ee.List.sequence(0, ncols.subtract(1));
  var rows = ee.List.sequence(0, nrows.subtract(1));

  var tiles = cols.map(function(col) {
    return rows.map(function(row) {
      col = ee.Number(col);
      row = ee.Number(row);
      var x0 = xmin.add(col.multiply(tile_size));
      var y0 = ymin.add(row.multiply(tile_size));
      var x1 = x0.add(tile_size);
      var y1 = y0.add(tile_size);
      var tile_geom = ee.Geometry.Rectangle([x0, y0, x1, y1]);
      var col_str = col.int().format('%02d');
      var row_str = row.int().format('%02d');
      var tile_id = ee.String('c').cat(col_str).cat('r').cat(row_str);
      return ee.Feature(tile_geom, {
        'tile_id': tile_id,
        'col': col,
        'row': row,
      });
    });
  }).flatten();

  var grid_fc = ee.FeatureCollection(tiles)
    .filterBounds(geometry);  // only tiles intersecting country

  return grid_fc;
}

/**
 * Build mosaic image from a Sentinel-2 collection for a given time window.
 * Returns image with bands: [blue, green, red, nir, swir1, swir2, dayOfYear]
 *   - spectral bands: byte (0–100, divide(100) of raw 0–10000)
 *   - dayOfYear: int16 (1–366)
 * Mosaic is NOT masked by LULC — LULC mask is applied post-classification.
 */
function build_mosaic(start_date, end_date, mask_geometry, focus_mask) {
  var csPlus_bands = ee.ImageCollection(CS_PLUS).first().bandNames();

  var col = ee.ImageCollection(S2_COLLECTION)
    .filterDate(start_date, end_date)
    .filterBounds(mask_geometry)
    .linkCollection(ee.ImageCollection(CS_PLUS), csPlus_bands)
    .map(preprocess);

  // For monthly: apply fire focus mask (restrict to potential fire areas)
  if (focus_mask !== null) {
    col = col.map(function(image) {
      return image.updateMask(focus_mask.unmask(0).gt(0));
    });
  }

  var mosaic = col.qualityMosaic('nbr');

  // ── Spectral: S2 raw (0–10000) ÷ 100 → 0–100 → byte
  var spectral = mosaic.select(BANDS_SPECTRAL)
    .divide(100)
    .byte();

  // ── dayOfYear: keep as int16 (no conversion)
  var doy = mosaic.select('dayOfYear').int16();

  return spectral.addBands(doy);
}

// ─── EXPORT LOGIC ─────────────────────────────────────────────────────────────

/**
 * Export mosaic to GEE Asset and/or GCS bucket (as COG chunks per tile).
 */
function export_mosaic(mosaic, name, region, year, month, period) {
  var time_start, time_end;

  if (period === 'monthly') {
    time_start = ee.Date('' + year + '-' + month + '-01').millis();
    time_end   = ee.Date('' + year + '-' + month + '-01').advance(1, 'month').millis();
  } else {
    time_start = ee.Date('' + year + '-01-01').millis();
    time_end   = ee.Date('' + (year + 1) + '-01-01').millis();
  }

  var mosaic_with_meta = mosaic
    .clip(country_geometry)
    .set({
      'system:time_start': time_start,
      'system:time_end':   time_end,
      'country':  config.country,
      'year':     year,
      'month':    month || 0,
      'period':   period,
      'sensor':   'sentinel2',
      'version':  '1',
      'bands':    BANDS_ALL,
      'name':     name,
    });

  // ── 1. Export to GEE Asset (full country mosaic) ──────────────────────────
  if (config.export_to_asset) {
    var asset_folder = period === 'monthly' ? config.asset_folder_monthly : config.asset_folder_yearly;
    Export.image.toAsset({
      image:       mosaic_with_meta,
      description: 'ASSET_' + name,
      assetId:     asset_folder + '/' + name,
      region:      country_bounds,
      scale:       10,
      maxPixels:   1e13,
      pyramidingPolicy: { '.default': 'median' },
    });
  }

  // ── 2. Export to GCS (per band) ─────────────────────────────
  if (config.export_to_gcs) {
    var period_path = period === 'monthly'
      ? 'monthly/chunks/' + year + '/' + (month < 10 ? '0' + month : '' + month)
      : 'yearly/chunks/' + year;

    var gcs_folder = config.bucket_base + '/' + period_path;

    BANDS_ALL.forEach(function(band) {
      var band_name = name + '_' + band;
      Export.image.toCloudStorage({
        image:       mosaic_with_meta.select(band),
        description: 'GCS_' + band_name,
        bucket:      config.bucket,
        fileNamePrefix: gcs_folder + '/' + band_name,
        region:      country_bounds,
        scale:       10,
        maxPixels:   1e13,
        fileFormat:  'GeoTIFF',
        formatOptions: { cloudOptimized: true },
      });
    });
  }
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────

config.years.forEach(function(year) {

  // ── YEARLY MOSAIC ──────────────────────────────────────────────────────────
  if (config.export_yearly) {
    var yearly_start = ee.Date('' + year + '-01-01');
    var yearly_end   = yearly_start.advance(1, 'year');

    var yearly_mosaic = build_mosaic(
      yearly_start,
      yearly_end,
      country_geometry,
      null  // no focus mask for annual
    );

    var yearly_name = 's2_fire_' + config.country + '_' + year;
    print('yearly:', yearly_name, yearly_mosaic);
    Map.addLayer(
      yearly_mosaic,
      { bands: ['swir1', 'nir', 'red'], min: 0, max: 50 },
      yearly_name,
      false
    );
    export_mosaic(yearly_mosaic, yearly_name, country_geometry, year, null, 'yearly');
  }

  // ── MONTHLY MOSAICS ────────────────────────────────────────────────────────
  if (config.export_monthly) {
    config.months.forEach(function(month) {
      var monthly_start = ee.Date('' + year + '-' + month + '-01');
      var monthly_end   = monthly_start.advance(1, 'month');

      // Fire focus mask (only for monthly)
      var focus_mask = get_focus_mask(year, month);

      var monthly_mosaic = build_mosaic(
        monthly_start,
        monthly_end,
        country_geometry,
        focus_mask
      );

      var month_str    = month < 10 ? '0' + month : '' + month;
      var monthly_name = 's2_fire_' + config.country + '_' + year + '_' + month_str;

      print('monthly:', monthly_name, monthly_mosaic);
      Map.addLayer(
        monthly_mosaic,
        { bands: ['swir1', 'nir', 'red'], min: 0, max: 50 },
        monthly_name,
        false
      );
      export_mosaic(monthly_mosaic, monthly_name, country_geometry, year, month, 'monthly');
    });
  }
});

// ─── VISUALIZATION ────────────────────────────────────────────────────────────
Map.addLayer(regions_fc, {color: 'ff6600'}, 'Peru Regions', true);
Map.centerObject(country_geometry, 5);
