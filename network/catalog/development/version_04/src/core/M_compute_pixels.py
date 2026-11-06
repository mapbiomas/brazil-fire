"""Experimental, isolated High Volume computePixels transport.

This transport never changes ee.Initialize's global endpoint and does not
access GCS. Credentials and expressions are supplied by
the caller so the GEE user identity cannot be replaced by the GCS identity.
"""
import random
import time
import copy

MAX_UNCOMPRESSED_BYTES = 48 * 1024 * 1024
MAX_DIMENSION = 32768
RETRYABLE_STATUS = frozenset((429, 500, 502, 503, 504))


class RestrictedModeError(RuntimeError):
    """Earth Engine explicitly reported a restricted-mode concurrency limit."""


def _requires_subdivision(error):
    message = str(error).lower()
    return ('user memory limit exceeded' in message
            or ('http 400' in message and 'computation timed out' in message))


def download_adaptive_tile(project, credentials, request, destination, fetch=None,
                           min_pixels=256, on_split=None, dtype='uint8',
                           request_max_pixels=None, on_piece=None):
    """Fetch one manifest tile, subdividing only EE memory-limit failures.

    The saved parent GeoTIFF retains its original grid/name, so existing
    manifests and previously uploaded tiles remain usable. Subtiles are local
    and are uploaded only after the entire parent tile has succeeded.
    """
    import rasterio
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine
    from rasterio.windows import Window
    fetch = fetch or compute_pixels
    grid = request['grid']
    width = grid['dimensions']['width']
    height = grid['dimensions']['height']
    if width != height or width <= 0 or min_pixels <= 0:
        raise ValueError('Adaptive export requires a positive square grid')

    limit = request_max_pixels or width
    if limit <= 0 or width % limit or (limit < width and limit < min_pixels):
        raise ValueError('Request limit must divide the parent tile')
    if width <= limit:
        try:
            payload = fetch(project, credentials, request)
        except RuntimeError as exc:
            if not _requires_subdivision(exc) or width <= min_pixels:
                raise
            if width % 2 or width // 2 < min_pixels:
                raise ValueError('Tile cannot be evenly subdivided to the minimum size') from exc
            limit = width // 2
            if on_split:
                on_split(width, limit)
        else:
            with open(destination, 'wb') as handle:
                handle.write(payload)
            return

    transform = grid['affineTransform']
    affine = Affine(transform['scaleX'], transform.get('shearX', 0), transform['translateX'],
                    transform.get('shearY', 0), transform['scaleY'], transform['translateY'])
    if affine.b != 0 or affine.d != 0:
        raise ValueError('Adaptive export requires a north-up grid')
    if dtype not in ('uint8', 'int16'):
        raise ValueError('Unsupported output dtype')
    piece_counts = {'done': 0, 'total': (width // limit) ** 2}
    def write_child(output, x, y, size):
        child = copy.deepcopy(request)
        child_grid = child['grid']
        child_grid['dimensions'] = {'width': size, 'height': size}
        child_grid['affineTransform']['translateX'] = affine.c + x * affine.a
        child_grid['affineTransform']['translateY'] = affine.f + y * affine.e
        try:
            data = fetch(project, credentials, child)
        except RuntimeError as exc:
            if not _requires_subdivision(exc) or size <= min_pixels:
                raise
            half = size // 2
            if size % 2 or half < min_pixels:
                raise
            piece_counts['total'] += 3
            if on_split:
                on_split(size, half)
            for dy in (0, half):
                for dx in (0, half):
                    write_child(output, x + dx, y + dy, half)
            return
        with MemoryFile(data) as memory:
            with memory.open() as source:
                if source.count != 1 or source.width != size or source.height != size:
                    raise ValueError('Unexpected computePixels tile dimensions')
                output.write(source.read(1, masked=False), 1, window=Window(x, y, size, size))
        piece_counts['done'] += 1
        if on_piece:
            on_piece(piece_counts['done'], piece_counts['total'])

    with rasterio.open(destination, 'w', driver='GTiff', width=width, height=height,
                       count=1, dtype=dtype, crs=grid['crsCode'], transform=affine,
                       nodata=0, tiled=True, blockxsize=256, blockysize=256,
                       compress='LZW') as output:
        for y in range(0, height, limit):
            for x in range(0, width, limit):
                write_child(output, x, y, limit)


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
    from requests.exceptions import ConnectionError, Timeout, HTTPError
    url = f'https://earthengine-highvolume.googleapis.com/v1/projects/{project}/image:computePixels'
    session = session_factory(credentials)
    try:
        for attempt in range(max_attempts):
            try:
                response = session.post(url, json=request, timeout=timeout,
                                        headers={'X-Goog-User-Project': project})
                if response.status_code == 429:
                    try:
                        message = response.json().get('error', {}).get('message', '')
                    except (ValueError, AttributeError):
                        message = ''
                    if isinstance(message, str) and 'restricted mode' in message.lower():
                        raise RestrictedModeError(f'HTTP 429: {message}')
                if response.status_code not in RETRYABLE_STATUS or attempt == max_attempts - 1:
                    try:
                        response.raise_for_status()
                    except HTTPError as exc:
                        try:
                            detail = response.json().get('error', {}).get('message', '')
                        except (ValueError, AttributeError):
                            detail = ''
                        raise RuntimeError(f'HTTP {response.status_code}: {detail or str(exc)}') from exc
                    return response.content
            except (ConnectionError, Timeout):
                if attempt == max_attempts - 1:
                    raise
            sleep(min(30, 2 ** attempt + jitter()))
    finally:
        session.close()
