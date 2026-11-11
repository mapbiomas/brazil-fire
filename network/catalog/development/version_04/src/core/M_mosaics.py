"""
M_mosaics — Mosaic method registry and configuration.

Each method defines its display label and which sensors/periodicities it supports.
"""

MOSAIC_METHODS = {
    'minnbr':         {'label': 'MINNBR',         'sensors': ['sentinel2', 'landsat', 'hls', 'modis'], 'periods': ['monthly', 'yearly']},
    'minnbr_buffer':  {'label': 'MINNBR Buffer',  'sensors': ['sentinel2', 'landsat', 'hls', 'modis'], 'periods': ['monthly', 'yearly']},
    'median':         {'label': 'Median',         'sensors': ['sentinel2', 'landsat', 'hls', 'modis'], 'periods': ['monthly', 'yearly']},
    'minndvi':        {'label': 'MINNDVI',        'sensors': ['sentinel2', 'landsat', 'hls', 'modis'], 'periods': ['monthly', 'yearly']},
}


def all_methods():
    """Return list of all known mosaic method keys."""
    return sorted(MOSAIC_METHODS.keys())


def available_methods(sensor=None, period=None):
    """Return method names filtered by sensor and/or periodicity."""
    result = []
    for name, cfg in MOSAIC_METHODS.items():
        if sensor is not None and sensor not in cfg['sensors']:
            continue
        if period is not None and period not in cfg['periods']:
            continue
        result.append(name)
    return sorted(result)


def methods_for(configured, sensor=None, period=None):
    """Return the configured methods that a sensor/period pair supports.

    The notebook/config decides which methods are active; the registry only
    removes combinations the pipeline does not implement.
    """
    if isinstance(configured, str):
        configured = [configured]
    configured = [str(method).lower() for method in configured]
    supported = set(available_methods(sensor=sensor, period=period))
    if not supported:
        return configured
    return [method for method in configured if method in supported]


def method_label(method):
    """Return human-readable label for a method key."""
    info = MOSAIC_METHODS.get(method)
    return info['label'] if info else method.upper()
