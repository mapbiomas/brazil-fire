"""Small spatial helpers for region-scoped M5 inference."""

import numpy as np


NODATA = -9999


def clip_geometry_to_cell(region_geojson, cell_geojson, source_crs):
    """Intersect a campaign region with one CIM cell and project to raster CRS.

    Earth Engine returns both geometries in EPSG:4326. The output GeoJSON and
    bounds are in ``source_crs`` so they can be used directly with the COG
    transform and window calculations.
    """
    from pyproj import Transformer
    import shapely
    from shapely.geometry import mapping, shape

    region = shape(region_geojson)
    cell = shape(cell_geojson)
    clipped = region.intersection(cell)
    if clipped.is_empty or clipped.area == 0:
        return [], None

    crs_text = str(source_crs).upper()
    if crs_text not in ('EPSG:4326', 'OGC:CRS84', 'CRS84'):
        transformer = Transformer.from_crs(
            'EPSG:4326', source_crs, always_xy=True,
        )
        def project_coordinates(coordinates):
            x, y = transformer.transform(coordinates[:, 0], coordinates[:, 1])
            return np.column_stack((x, y))

        clipped = shapely.transform(clipped, project_coordinates)

    return [mapping(clipped)], tuple(clipped.bounds)


def rasterize_region_mask(geometries, out_shape, transform):
    """Return True for pixel centers inside the requested region geometry."""
    from rasterio.features import geometry_mask

    return geometry_mask(
        geometries,
        out_shape=out_shape,
        transform=transform,
        invert=True,
        all_touched=False,
    )


def classify_region_block(stack, data_valid_mask, region_mask, day_of_year,
                          norm_stats, normalize_fn, predict_fn):
    """Predict only valid pixels inside a region and fill all others as nodata."""
    stack = np.asarray(stack)
    data_valid_mask = np.asarray(data_valid_mask, dtype=bool)
    region_mask = np.asarray(region_mask, dtype=bool)
    if stack.ndim != 3:
        raise ValueError('stack must have shape (height, width, bands)')
    expected_shape = stack.shape[:2]
    if data_valid_mask.shape != expected_shape or region_mask.shape != expected_shape:
        raise ValueError('data and region masks must have the same height and width as stack')

    doy = None if day_of_year is None else np.asarray(day_of_year)
    if doy is not None and doy.shape != expected_shape:
        raise ValueError('day_of_year must have the same height and width as stack')

    valid_mask = data_valid_mask & region_mask
    probability = np.full(expected_shape, NODATA, dtype=np.int16)
    output_doy = np.full(expected_shape, NODATA, dtype=np.int16)
    valid_count = int(valid_mask.sum())
    if not valid_count:
        return {
            'probability': probability,
            'day_of_year': output_doy,
            'valid_pixels': 0,
            'burned_pixels': 0,
            'confidence_sum': 0.0,
            'confidence_count': 0,
        }

    features = np.ascontiguousarray(stack[valid_mask])
    predictions = np.asarray(predict_fn(normalize_fn(features, norm_stats)))
    if predictions.ndim > 1 and predictions.shape[1] == 1:
        predictions = predictions.flatten()
    if predictions.ndim != 1 or len(predictions) != valid_count:
        raise ValueError(
            f'predict_fn returned {predictions.shape} for {valid_count} valid pixels'
        )

    quantized = np.round(predictions * 1000).astype(np.int16)
    probability[valid_mask] = quantized
    burned = quantized > 0
    burned_count = int(burned.sum())
    if doy is not None and burned_count:
        burned_mask = np.zeros(expected_shape, dtype=bool)
        burned_mask[valid_mask] = burned
        output_doy[burned_mask] = doy[burned_mask].astype(np.int16)

    return {
        'probability': probability,
        'day_of_year': output_doy,
        'valid_pixels': valid_count,
        'burned_pixels': burned_count,
        'confidence_sum': float(predictions[burned].sum()) if burned_count else 0.0,
        'confidence_count': burned_count,
    }
