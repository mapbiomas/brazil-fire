"""Experimental, isolated High Volume computePixels transport.

This transport is not wired into M1. It never changes ee.Initialize's global
endpoint and does not access GCS. Credentials and expressions are supplied by
the caller so the GEE user identity cannot be replaced by the GCS identity.
"""
import random
import time

MAX_UNCOMPRESSED_BYTES = 48 * 1024 * 1024
MAX_DIMENSION = 32768
RETRYABLE_STATUS = frozenset((429, 500, 502, 503, 504))


def make_tile_request(expression, tile, pixel_size, tile_pixels, crs, band_bytes):
    """Build a north-up request for a tile from M_export_grid.

    ``expression`` must already be an Earth Engine cloud-serialized image;
    image masks and the eventual GeoTIFF conversion remain the caller's duty.
    """
    import math
    validate_request_size(tile_pixels, tile_pixels, band_bytes)
    if (not isinstance(crs, str) or not crs.strip() or pixel_size <= 0
            or not math.isfinite(pixel_size)):
        raise ValueError('Invalid projected CRS or pixel size')
    width = tile_pixels * pixel_size
    if not (math.isclose(tile.xmax - tile.xmin, width, rel_tol=1e-9, abs_tol=1e-6)
            and math.isclose(tile.ymax - tile.ymin, width, rel_tol=1e-9, abs_tol=1e-6)):
        raise ValueError('Tile bounds do not match the requested pixel grid')
    return {
        'expression': expression,
        'fileFormat': 'GEO_TIFF',
        'grid': {
            'dimensions': {'width': tile_pixels, 'height': tile_pixels},
            'affineTransform': {'scaleX': pixel_size, 'scaleY': -pixel_size,
                                'translateX': tile.xmin, 'translateY': tile.ymax},
            'crsCode': crs,
        },
    }


def validate_request_size(width, height, band_bytes):
    """Reject a request exceeding the documented uncompressed EE limits."""
    if (isinstance(width, bool) or isinstance(height, bool)
            or not isinstance(width, int) or not isinstance(height, int)
            or not 0 < width <= MAX_DIMENSION or not 0 < height <= MAX_DIMENSION):
        raise ValueError('Invalid computePixels dimensions')
    if (not band_bytes or len(band_bytes) > 1024
            or any(isinstance(x, bool) or not isinstance(x, int) or x <= 0 for x in band_bytes)):
        raise ValueError('Invalid computePixels bands')
    if width * height * sum(band_bytes) > MAX_UNCOMPRESSED_BYTES:
        raise ValueError('computePixels request exceeds the 48 MiB limit')


def compute_pixels(project, credentials, request, max_attempts=5, timeout=120,
                   session_factory=None, sleep=time.sleep, jitter=random.random):
    """Request a tile using the user's credentials, with bounded retry.

    The caller must validate grid dimensions and decoded content. HTTP errors
    are raised, not converted into successful empty tiles.
    """
    if not project or '/' in project or credentials is None or max_attempts < 1:
        raise ValueError('A project, user credentials and positive attempts are required')
    if session_factory is None:
        from google.auth.transport.requests import AuthorizedSession
        session_factory = AuthorizedSession
    from requests.exceptions import ConnectionError, Timeout
    url = f'https://earthengine-highvolume.googleapis.com/v1/projects/{project}/image:computePixels'
    session = session_factory(credentials)
    try:
        for attempt in range(max_attempts):
            try:
                response = session.post(url, json=request, timeout=timeout,
                                        headers={'X-Goog-User-Project': project})
                if response.status_code not in RETRYABLE_STATUS or attempt == max_attempts - 1:
                    response.raise_for_status()
                    return response.content
            except (ConnectionError, Timeout):
                if attempt == max_attempts - 1:
                    raise
            sleep(min(30, 2 ** attempt + jitter()))
    finally:
        session.close()
