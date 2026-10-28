"""
M0 - Global Configuration and Authentication
MapBiomas Fire Monitor
"""
import ee
import os
import warnings
import logging

from M_periods import ENABLED_PERIODICITIES, PERIODS, parse_period, period_state_key

# Silenciar avisos de autenticacao do Google e loggers verbosos do GCSFS
warnings.filterwarnings("ignore", message="Your application has authenticated using end user credentials")
warnings.filterwarnings("ignore", message=".*gdal.UseExceptions.*", category=FutureWarning)
logging.getLogger("google.auth").setLevel(logging.ERROR)
logging.getLogger("gcsfs").setLevel(logging.CRITICAL)
logging.getLogger("fsspec").setLevel(logging.CRITICAL)

# ---------------------------------------------------------------------------
# Parameter descriptions (for validation)
# ---------------------------------------------------------------------------

_PARAM_DESCRIPTIONS = {
    'country':             'codigo do pais/regiao (ex: "peru", "bolivia")',
    'gcs_bucket':          'nome do bucket Google Cloud Storage (ex: "mapbiomas-fire")',
    'gcs_library_images_prefix': 'pasta GCS onde ficam as imagens base/LIBRARY_IMAGES (ex: "sudamerica/peru/CATALOG_01")',
    'gcs_campaigns_prefix':'pasta GCS onde ficam as campanhas (ex: "sudamerica/peru/CATALOG_01")',
    'gee_project':         'GEE Cloud Project ID (ex: "mapbiomas-peru")',
    'gee_library_images_prefix': 'GEE asset path para imagens (ex: "projects/mapbiomas-peru/assets/FIRE/CATALOG_01")',
    'gee_campaigns_prefix':'GEE asset path para campanhas (ex: "projects/mapbiomas-peru/assets/FIRE/CATALOG_01")',
    'campaign':            'campanha ativa (ex: "MONITOR_01", "WATER_01")',
    'asset_regions':       'GEE FeatureCollection com geometria das regioes (ex: "projects/.../CATALOG_01/MONITOR_01/AUXILIARY/regiones_fuego_peru_v1")',
}

# ---------------------------------------------------------------------------
# CONFIG builder
# ---------------------------------------------------------------------------

def _make_config(
    country,
    bucket,
    gcs_library_images_prefix,
    gcs_campaigns_prefix,
    gee_project,
    gee_library_images_prefix,
    gee_campaigns_prefix,
    campaign,
    asset_regions,
    mosaic_methods,
):
    """Build the CONFIG dict from project-level parameters. All required."""
    return {
        'country': country,
        'bucket': bucket,
        'gee_project': gee_project,
        'campaign': campaign,
        'asset_regions': asset_regions,
        'mosaic_methods': list(mosaic_methods),

        # --- GCS prefixos ---
        'gcs_library_images_prefix': gcs_library_images_prefix,
        'gcs_campaigns_prefix':     gcs_campaigns_prefix,

        # --- GCS paths derivados ---
        'gcs_library_images': f"{gcs_library_images_prefix}/LIBRARY_IMAGES",
        'gcs_library_cache': f"{gcs_library_images_prefix}/LIBRARY_IMAGES/.CACHE",
        'gcs_campaign_cache': f"{gcs_campaigns_prefix}/{campaign}/.CACHE",
        'gcs_cache':          f"{gcs_campaigns_prefix}/.CACHE",  # legacy single cache

        # --- GEE prefixos ---
        'gee_library_images_prefix': gee_library_images_prefix,
        'gee_campaigns_prefix':     gee_campaigns_prefix,

        # Processing params
        'bands_model_default': ['red', 'nir', 'swir1', 'swir2'],
        'bands_all': ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'dayOfYear'],
    }


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _single(val):
    """If val is a list, return first element; otherwise return val as-is."""
    return val[0] if isinstance(val, list) else val

# Deve ser inicializado via set_global_opts()
CONFIG = None

GLOBAL_OPTS = {
    'SENSOR': ['sentinel2'],
    'PERIODICITY': ['monthly'],
    'MOSAIC_METHOD': 'minnbr',
    'PERSONAL_TASK_FLAG': 'CATALOG',
    'FIRE_POTENTIAL_FILTER': False,
}


# ---------------------------------------------------------------------------
# Dynamic path helpers (campaign-aware)
# ---------------------------------------------------------------------------

def gcs_samples_path(campaign=None):
    c = campaign or CONFIG['campaign']
    return f"{CONFIG['gcs_campaigns_prefix']}/{c}/LIBRARY_SAMPLES"

def gcs_models_path(campaign=None):
    c = campaign or CONFIG['campaign']
    return f"{CONFIG['gcs_campaigns_prefix']}/{c}/LIBRARY_MODELS"

def gcs_classifications_path(campaign=None):
    c = campaign or CONFIG['campaign']
    return f"{CONFIG['gcs_campaigns_prefix']}/{c}/LIBRARY_CLASSIFICATIONS"

def gee_samples_path(campaign=None):
    c = campaign or CONFIG['campaign']
    return f"{CONFIG['gee_campaigns_prefix']}/{c}/LIBRARY_SAMPLES"

def gee_classifications_path(campaign=None):
    c = campaign or CONFIG['campaign']
    return f"{CONFIG['gee_campaigns_prefix']}/{c}/LIBRARY_CLASSIFICATIONS"


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

_AUTHENTICATED = False
_GCS_CREDENTIALS = None
_MODEL_SCALE_CACHE = {}


def authenticate(project=None):
    """Autenticar com Google Earth Engine e GCS. Se project for None, usa CONFIG['gee_project']."""
    global _AUTHENTICATED, _GCS_CREDENTIALS
    
    if _AUTHENTICATED:
        return
    
    if project is None:
        project = (CONFIG or {}).get('gee_project', 'mapbiomas-peru')
    
    import ee

    if not getattr(ee.data, '_credentials', None):
        # Deteccao de ambiente Colab para autenticacao GCS
        try:
            import google.colab
            print("[COLAB] Detected Google Colab environment. Authenticating user...")
            from google.colab import auth
            auth.authenticate_user()
            import google.auth
            _GCS_CREDENTIALS, _ = google.auth.default()
        except ImportError:
            pass  # Ambiente local, assume Application Default Credentials (ADC)

        try:
            ee.Initialize(project=project)
            print("[GEE] Inicializado.")
        except Exception:
            print("GEE session expired or null. Starting authentication...")
            ee.Authenticate()
            ee.Initialize(project=project)
            print("[GEE] Authenticated successfully.")
    else:
        ee.Initialize(project=project)
        print("[GEE] Reusing credentials for configured project.")
    
    from M_gcs import authenticate as _mgcs_auth
    _mgcs_auth()
    print("[GCS] GCS/ADC authentication configured.")

    # Deferred cache sync — gsutil auth já está pronto aqui
    try:
        from M_cache import CacheManager
        state = CacheManager.get_state()
        has_cogs = any(
            state.get(period_state_key('cogs', periodicity))
            for periodicity in ENABLED_PERIODICITIES
        )
        if not state.get('updated_at') or not has_cogs:
            CacheManager.build_full_cache()
    except Exception as e:
        print(f"Cache sync deferred: {e}")

    _AUTHENTICATED = True


_fs_instance = None

def _get_fs():
    """Retorna uma instancia GCSFileSystem (lazy, read-only)."""
    from M_gcs import _get_fs as _mgcs_get_fs
    return _mgcs_get_fs()


# ---------------------------------------------------------------------------
# GCS (gsutil) helpers
# ---------------------------------------------------------------------------

def _gcs_download(gcs_rel_path, local_path):
    from M_gcs import download
    download(gcs_rel_path, local_path)


def _gcs_upload(local_path, gcs_rel_path):
    from M_gcs import upload
    upload(local_path, gcs_rel_path)


# ---------------------------------------------------------------------------
# set_global_opts
# ---------------------------------------------------------------------------

def set_global_opts(
    # === Obrigatorios ===
    country,
    gcs_bucket,
    gcs_library_images_prefix,
    gcs_campaigns_prefix,
    gee_project,
    gee_library_images_prefix,
    gee_campaigns_prefix,
    campaign,
    asset_regions,

    # === Opcionais (pipeline) ===
    sensor='sentinel2',
    periodicity='monthly',
    mosaic_methods=None,
    personal_task_flag='MONITOR',
    language='en',
    clean_cache=False,
):
    """Configure global processing options and project paths.

    All required parameters must be provided explicitly.
    No defaults for project-level configuration.

    ``mosaic_methods`` is required: the active methods come from the notebook
    configuration, never from an internal default.

    Passing convention: use a plain string for a single option and a list
    only when selecting more than one (e.g. sensor='landsat' but
    mosaic_methods=['minnbr', 'median']).
    """
    global GLOBAL_OPTS, CONFIG

    requested_sensors = sensor if isinstance(sensor, (list, tuple)) else [sensor]
    requested_periods = periodicity if isinstance(periodicity, (list, tuple)) else [periodicity]
    requested_periods = [str(p).lower() for p in requested_periods]
    if len(requested_periods) != 1 or requested_periods[0] not in ENABLED_PERIODICITIES:
        raise ValueError(
            "periodicity must be a single value from: "
            + ', '.join(ENABLED_PERIODICITIES)
            + '.'
        )

    from M_mosaics import all_methods as _all_mosaic_methods, available_methods as _available_methods
    valid_mosaics = set(_all_mosaic_methods())
    if mosaic_methods is None:
        raise ValueError(
            "mosaic_methods is required; pass one or more of: "
            + ', '.join(sorted(valid_mosaics))
            + "."
        )
    requested_mosaics = mosaic_methods if isinstance(mosaic_methods, (list, tuple)) else [mosaic_methods]
    selected_mosaics = [m for m in requested_mosaics if m in valid_mosaics]
    if not selected_mosaics:
        raise ValueError(f"mosaic_methods must contain one or more of: {', '.join(sorted(valid_mosaics))}.")

    for sensor_name in requested_sensors:
        for period_name in requested_periods:
            supported = set(_available_methods(
                sensor=str(sensor_name).lower(),
                period=str(period_name).lower(),
            ))
            if not supported:
                continue
            unsupported = [m for m in selected_mosaics if m not in supported]
            if unsupported:
                raise ValueError(
                    f"mosaic_methods {unsupported} not supported for "
                    f"{sensor_name}/{period_name}. "
                    f"Available: {', '.join(sorted(supported))}."
                )

    # Build CONFIG
    new_config = _make_config(
        country=country,
        bucket=gcs_bucket,
        gcs_library_images_prefix=gcs_library_images_prefix,
        gcs_campaigns_prefix=gcs_campaigns_prefix,
        gee_project=gee_project,
        gee_library_images_prefix=gee_library_images_prefix,
        gee_campaigns_prefix=gee_campaigns_prefix,
        campaign=campaign,
        asset_regions=asset_regions,
        mosaic_methods=selected_mosaics,
    )

    # Preserve the CONFIG object so modules that imported it keep the active values.
    if CONFIG is None:
        CONFIG = new_config
    else:
        CONFIG.clear()
        CONFIG.update(new_config)

    GLOBAL_OPTS['SENSOR'] = requested_sensors
    GLOBAL_OPTS['PERIODICITY'] = requested_periods
    GLOBAL_OPTS['PERSONAL_TASK_FLAG'] = personal_task_flag
    GLOBAL_OPTS['LANGUAGE'] = language

    GLOBAL_OPTS['MOSAIC_METHOD'] = CONFIG['mosaic_methods'][0]

    from M_lang import L
    L.load_locale(language)

    mosaics_str = ', '.join(CONFIG['mosaic_methods'])
    sensor_str = ', '.join(str(s).upper() for s in requested_sensors)
    period_str = ', '.join(str(p).upper() for p in requested_periods)
    print(f"Global options: Sensor(s): {sensor_str} | Period(s): {period_str} | Mosaics: {mosaics_str} | Campaign: {CONFIG['campaign']} | Task Flag: {personal_task_flag}")

    if clean_cache:
        try:
            from M_cache import CacheManager
            CacheManager.clear()
            print("Cache cleared (library + campaign).")
        except Exception as e:
            print(f"Warning while clearing cache: {e}")
    
    return GLOBAL_OPTS


# ---------------------------------------------------------------------------
# GCS path generators
# ---------------------------------------------------------------------------

def _gcs_library_base():
    """Retorna o base path da library de imagens (sem sensor/period)."""
    sensor = _single(GLOBAL_OPTS['SENSOR']).upper()
    return f"{CONFIG['gcs_library_images']}/{sensor}"

def _gcs_models_base():
    """Base folder for trained models in GCS."""
    return gcs_models_path()

def _period_folder(periodicity=None):
    key = (periodicity or _single(GLOBAL_OPTS['PERIODICITY'])).lower()
    definition = PERIODS.get(key)
    return definition['folder'] if definition else key.upper()

def _gee_folder(periodicity=None):
    key = (periodicity or _single(GLOBAL_OPTS['PERIODICITY'])).lower()
    definition = PERIODS.get(key)
    if not definition:
        return key.upper()
    return definition.get('gee_folder', definition['folder'])

def _gcs_mosaic_path(periodicity, temporal_id, mosaic=None):
    base = _gcs_library_base()
    period = _period_folder(periodicity)
    m = (mosaic or GLOBAL_OPTS.get('MOSAIC_METHOD', 'minnbr')).upper()
    return f"{base}/{period}/{m}/{temporal_id}"

def period_chunk_path(period, mosaic='minnbr', sensor=None):
    info = parse_period(period)
    s = (sensor or _single(GLOBAL_OPTS['SENSOR'])).upper()
    return f"{CONFIG['gcs_library_images']}/{s}/{info.folder}/{mosaic.upper()}/{info.key}/CHUNKS"

def period_cog_path(period, mosaic='minnbr', sensor=None):
    info = parse_period(period)
    s = (sensor or _single(GLOBAL_OPTS['SENSOR'])).upper()
    return f"{CONFIG['gcs_library_images']}/{s}/{info.folder}/{mosaic.upper()}/{info.key}/COG"

def period_mosaic_path(period, mosaic=None):
    info = parse_period(period)
    return _gcs_mosaic_path(info.periodicity, info.key, mosaic=mosaic)

def monthly_mosaic_path(year, month, mosaic=None):
    return period_mosaic_path(f"{year}_{month:02d}", mosaic=mosaic)

def yearly_mosaic_path(year, mosaic=None):
    return period_mosaic_path(str(year), mosaic=mosaic)

def monthly_chunk_path(year, month, mosaic='minnbr', sensor=None):
    return period_chunk_path(f"{year}_{month:02d}", mosaic=mosaic, sensor=sensor)

def yearly_chunk_path(year, mosaic='minnbr', sensor=None):
    return period_chunk_path(str(year), mosaic=mosaic, sensor=sensor)

def monthly_cog_path(year, month, mosaic='minnbr', sensor=None):
    return period_cog_path(f"{year}_{month:02d}", mosaic=mosaic, sensor=sensor)

def yearly_cog_path(year, mosaic='minnbr', sensor=None):
    return period_cog_path(str(year), mosaic=mosaic, sensor=sensor)

def model_path(training_id, shortname, region=None):
    sensor = _single(GLOBAL_OPTS['SENSOR']).lower()
    t_id = str(training_id)
    if t_id.startswith('training_'):
        asset_name = t_id
    else:
        asset_name = f"training_{t_id}_{shortname}_{sensor}"
    
    base = f"{_gcs_models_base()}/{asset_name}"
    if region:
        return f"{base}/{region}"
    return base

def gcs_path(relative):
    """Retorna URL gs:// completa."""
    return f"gs://{CONFIG['bucket']}/{relative}"


# ---------------------------------------------------------------------------
# GEE path generators
# ---------------------------------------------------------------------------

def get_asset_mosaic_collection(sensor=None, periodicity=None, band=None, period=None, mosaic=None, period_key=None):
    if period_key:
        folder_period = _gee_folder(parse_period(period_key).periodicity)
    else:
        folder_period = _gee_folder(period or periodicity)
    m_name = (mosaic or GLOBAL_OPTS.get('MOSAIC_METHOD', 'minnbr')).upper()
    
    sensor_name = (sensor or _single(GLOBAL_OPTS['SENSOR'])).upper()
    
    path = f"{CONFIG['gee_library_images_prefix']}/LIBRARY_IMAGES/{sensor_name}/{folder_period}/{m_name}"
    
    if band:
        path = f"{path}/{band.lower()}"
    return path


# ---------------------------------------------------------------------------
# Nomenclature
# ---------------------------------------------------------------------------

def mosaic_name_for_period(period, band=None, mosaic=None, sensor=None):
    info = parse_period(period)
    country = CONFIG['country']
    s = (sensor or _single(GLOBAL_OPTS['SENSOR'])).lower()
    m = (mosaic or GLOBAL_OPTS.get('MOSAIC_METHOD', 'minnbr')).lower()

    if band:
        return f"image_{country}_fire_{s}_{m}_{band}_{info.key}"
    return f"image_{country}_fire_{s}_{m}_{info.key}"

def mosaic_name(year, month=None, period=None, band=None, mosaic=None, sensor=None):
    key = f"{year}_{month:02d}" if month else f"{year}"
    return mosaic_name_for_period(key, band=band, mosaic=mosaic, sensor=sensor)

def get_temp_dir(subdir=None):
    is_colab = 'COLAB_RELEASE_TAG' in os.environ
    base = '/content/TEMPORARY' if is_colab else os.path.abspath('TEMPORARY')
    os.makedirs(base, exist_ok=True)
    if subdir:
        path = os.path.join(base, subdir)
        os.makedirs(path, exist_ok=True)
        return path
    return base

def check_command_exists(cmd):
    import shutil
    return shutil.which(cmd) is not None or shutil.which(cmd + ".py") is not None

def ensure_gdal_path():
    import os, shutil, platform, sys
    
    gdal_cmds = ['gdalbuildvrt', 'gdal_translate']
    missing = [cmd for cmd in gdal_cmds if shutil.which(cmd) is None]
    
    if not missing:
        return
    
    candidate_dirs = []
    
    if platform.system() == 'Windows':
        py_prefix = sys.prefix
        candidate_dirs += [
            os.path.join(py_prefix, 'Library', 'bin'),
            os.path.join(py_prefix, 'Scripts'),
        ]
        home = os.path.expanduser('~')
        for conda_root in [
            os.path.join(home, 'miniconda3'),
            os.path.join(home, 'anaconda3'),
            os.path.join(home, 'AppData', 'Local', 'miniconda3'),
            r'C:\OSGeo4W\bin',
            r'C:\Program Files\GDAL',
        ]:
            candidate_dirs.append(os.path.join(conda_root, 'Library', 'bin'))
            candidate_dirs.append(os.path.join(conda_root, 'Scripts'))
            candidate_dirs.append(conda_root)
    else:
        py_prefix = sys.prefix
        candidate_dirs += [
            os.path.join(py_prefix, 'bin'),
            '/usr/bin', '/usr/local/bin',
        ]
    
    for d in candidate_dirs:
        if os.path.isdir(d):
            test_bin = 'gdalbuildvrt.exe' if platform.system() == 'Windows' else 'gdalbuildvrt'
            if os.path.exists(os.path.join(d, test_bin)):
                os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
                print(f"GDAL found and added to PATH: {d}")
                return
    
    print(f"Warning: GDAL utilities not found: {missing}")
    if 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ:
        print("Tip: On Google Colab, run: !apt-get install -y gdal-bin")
    elif platform.system() == 'Windows':
        print("Tip: On Windows, ensure GDAL is in your PATH or use Conda environment (gdal).")

def sample_asset_name(temporal_id, version_id):
    clean_version = version_id.replace('sample_', '')
    return f"sample_{clean_version}_{temporal_id}"

def get_asset_samples(temporal_id, version_id):
    asset = sample_asset_name(temporal_id, version_id)
    return f"{gee_samples_path()}/{asset}"

def get_gcs_samples(temporal_id, version_id):
    asset = sample_asset_name(temporal_id, version_id)
    return f"{gcs_samples_path()}/{asset}"

def get_asset_regional(region_id, training_id, temporal_id):
    sensor = _single(GLOBAL_OPTS['SENSOR']).lower()
    suffix = "_BUFFER" if GLOBAL_OPTS['FIRE_POTENTIAL_FILTER'] else ""
    sensor_upper = sensor.upper() + suffix
    period = _single(GLOBAL_OPTS['PERIODICITY']).upper()
    asset_name = f"region_{region_id}_training_{training_id}_{sensor}_{temporal_id}"
    return f"{CONFIG['gee_library_images_prefix']}/LIBRARY_IMAGES/{sensor_upper}/{period}/burned_day_of_year_regional/{asset_name}"

def get_gcs_regional(region_id, training_id, temporal_id):
    base = _gcs_library_base()
    period = _single(GLOBAL_OPTS['PERIODICITY']).lower()
    sensor = _single(GLOBAL_OPTS['SENSOR']).lower()
    asset_name = f"region_{region_id}_training_{training_id}_{sensor}_{temporal_id}"
    return f"{base}/{sensor}/{period}/burned_day_of_year_regional/{asset_name}"

def get_asset_candidate(candidate_id, temporal_id):
    sensor = _single(GLOBAL_OPTS['SENSOR']).lower()
    suffix = "_BUFFER" if GLOBAL_OPTS['FIRE_POTENTIAL_FILTER'] else ""
    sensor_upper = sensor.upper() + suffix
    period = _single(GLOBAL_OPTS['PERIODICITY']).upper()
    asset_name = f"candidate_{candidate_id}_{sensor}_{temporal_id}"
    return f"{CONFIG['gee_library_images_prefix']}/LIBRARY_IMAGES/{sensor_upper}/{period}/burned_day_of_year_candidates/{asset_name}"

def get_gcs_candidate(candidate_id, temporal_id):
    base = _gcs_library_base()
    period = _single(GLOBAL_OPTS['PERIODICITY']).lower()
    sensor = _single(GLOBAL_OPTS['SENSOR']).lower()
    asset_name = f"candidate_{candidate_id}_{sensor}_{temporal_id}"
    return f"{base}/{sensor}/{period}/burned_day_of_year_candidates/{asset_name}"

def get_asset_official(temporal_id):
    sensor = _single(GLOBAL_OPTS['SENSOR']).lower()
    suffix = "_BUFFER" if GLOBAL_OPTS['FIRE_POTENTIAL_FILTER'] else ""
    sensor_upper = sensor.upper() + suffix
    period = _single(GLOBAL_OPTS['PERIODICITY']).upper()
    asset_name = f"burned_day_of_year_{sensor}_{temporal_id}"
    return f"{CONFIG['gee_library_images_prefix']}/LIBRARY_IMAGES/{sensor_upper}/{period}/burned_day_of_year_official/{asset_name}"

def get_gcs_official(temporal_id):
    base = _gcs_library_base()
    period = _single(GLOBAL_OPTS['PERIODICITY']).lower()
    sensor = _single(GLOBAL_OPTS['SENSOR']).lower()
    asset_name = f"burned_day_of_year_{sensor}_{temporal_id}"
    return f"{base}/{sensor}/{period}/burned_day_of_year_official/{asset_name}.tif"

def classification_name(periodicity, year, month=None, version=None):
    """(DEPRECATED)"""
    country = CONFIG['country']
    sensor = _single(GLOBAL_OPTS['SENSOR']).lower()
    suffix = "_buffer" if GLOBAL_OPTS['FIRE_POTENTIAL_FILTER'] else ""
    ver = version or 'version_01'
    region = "r01"
    return f"class_{sensor}{suffix}_{country}_{region}_{year}_{month:02d}_{ver}" if month else f"class_{sensor}{suffix}_{country}_{region}_{year}_{ver}"


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def get_country_geometry():
    return ee.FeatureCollection(CONFIG['asset_regions']).geometry()

def get_fire_reference(year):
    """(DEPRECATED) Requer asset_fire_ref no CONFIG."""
    ref = CONFIG.get('asset_fire_ref')
    if not ref:
        raise RuntimeError("asset_fire_ref not configured.")
    return ee.Image(f"{ref}{year}")

def get_hotspots_collection(year, month=None):
    """(DEPRECATED) Requer asset_hotspots no CONFIG."""
    col = CONFIG.get('asset_hotspots')
    if not col:
        raise RuntimeError("asset_hotspots not configured.")
    col = ee.FeatureCollection(col)
    col = col.filter(ee.Filter.eq('year', year))
    if month:
        col = col.filter(ee.Filter.eq('month', month))
    return col

def get_sensor_scale(sensor=None):
    sensor = (sensor or _single(GLOBAL_OPTS['SENSOR'])).lower()
    scales = {
        'sentinel2': 10,
        'landsat': 30,
        'hls': 30,
        'modis': 500
    }
    if sensor not in scales:
        raise ValueError(f"Unsupported sensor for scale lookup: {sensor}")
    return scales[sensor]

def get_sensor_start_year(sensor=None):
    """First full year available to the M1/M2 sensor collections."""
    sensor = (sensor or _single(GLOBAL_OPTS['SENSOR'])).lower()
    start_years = {
        'sentinel2': 2017,
        'landsat': 1985,
        'hls': 2013,
        'modis': 2000,
    }
    if sensor not in start_years:
        raise ValueError(f"Unsupported sensor for start-year lookup: {sensor}")
    return start_years[sensor]


def resolve_years(years=None, sensor=None):
    """Normalize a year selection to a descending list.

    Accepted forms:
    - None or []: every year available for the sensor (start year..current).
    - int: a single year, e.g. 2020.
    - list/tuple of ints: explicit years, e.g. [1985, 2000, 2020].
    - string or one-item list with an interval: '1985:2000' (also ';' or '-').

    Years outside the sensor range are ignored with a warning; if nothing
    valid remains, a ValueError explains the available range.
    """
    import datetime
    import re

    if sensor is None:
        sensors = GLOBAL_OPTS['SENSOR']
    elif isinstance(sensor, (list, tuple)):
        sensors = list(sensor)
    else:
        sensors = [sensor]
    sensors = [str(item).lower() for item in sensors]
    first_year = min(get_sensor_start_year(item) for item in sensors)
    last_year = datetime.datetime.now().year

    if years is None:
        years = []
    if isinstance(years, int):
        years = [years]
    elif isinstance(years, str):
        years = [years]
    else:
        years = list(years)

    if not years:
        return list(range(last_year, first_year - 1, -1))

    if len(years) == 1 and isinstance(years[0], str):
        text = years[0].strip()
        match = re.fullmatch(r'(\d{4})\s*[:;\-]\s*(\d{4})', text)
        if match:
            start, end = sorted(int(value) for value in match.groups())
            years = list(range(start, end + 1))
        else:
            years = [text]

    normalized = []
    for value in years:
        try:
            normalized.append(int(value))
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid year value {value!r}. Use an int, a list of ints, "
                "or a range like '1985:2000'."
            ) from error

    valid = sorted(
        {year for year in normalized if first_year <= year <= last_year},
        reverse=True,
    )
    ignored = sorted(
        {year for year in normalized if year < first_year or year > last_year}
    )
    if ignored:
        print(
            f"[WARN] Years outside {first_year}-{last_year} for {sensors} "
            f"ignored: {ignored}"
        )
    if not valid:
        raise ValueError(
            f"No valid years for {sensors}: requested {normalized}; "
            f"available range is {first_year}-{last_year}."
        )
    return valid

def get_model_scale(model_id):
    """Resolve pixel size from the sensor encoded in a trained model ID."""
    cache_key = (CONFIG.get('country'), CONFIG.get('campaign'), str(model_id))
    if cache_key in _MODEL_SCALE_CACHE:
        return _MODEL_SCALE_CACHE[cache_key]

    known_sensors = ('sentinel2', 'landsat', 'hls', 'modis')
    tokens = str(model_id).lower().split('_')
    model_sensors = [sensor for sensor in known_sensors if sensor in tokens]
    if model_sensors:
        scales = {get_sensor_scale(sensor) for sensor in model_sensors}
        if len(scales) == 1:
            scale = scales.pop()
            _MODEL_SCALE_CACHE[cache_key] = scale
            return scale

    # Multi-sensor model IDs do not carry one sensor; inference uses the first
    # band in band_order as the output raster grid, so use that band's sensor.
    try:
        import json
        meta_path = f"{CONFIG['bucket']}/{gcs_models_path()}/{model_id}/metadata.json"
        with _get_fs().open(meta_path, 'r') as f:
            metadata = json.load(f)
        band_order = metadata.get('band_order', [])
        bands_config = metadata.get('bands_config', {})
        if band_order:
            sensor = bands_config.get(band_order[0], {}).get('sensor')
            if sensor:
                scale = get_sensor_scale(sensor)
                _MODEL_SCALE_CACHE[cache_key] = scale
                return scale
    except Exception:
        pass

    configured = [s.lower() for s in GLOBAL_OPTS.get('SENSOR', []) if s.lower() in known_sensors]
    configured_scales = {get_sensor_scale(sensor) for sensor in configured}
    if len(configured_scales) == 1:
        scale = configured_scales.pop()
        _MODEL_SCALE_CACHE[cache_key] = scale
        return scale
    raise ValueError(f"Cannot determine one pixel scale for model '{model_id}'.")

def set_country(country):
    """Update country while preserving project-level params."""
    global CONFIG
    if CONFIG is None:
        raise RuntimeError("CONFIG not initialized. Call set_global_opts() first.")
    CONFIG['country'] = country
    return CONFIG

def set_edit_mode(mode):
    global GLOBAL_OPTS
    GLOBAL_OPTS['EDIT_MODE'] = mode

def is_edit_mode():
    return GLOBAL_OPTS.get('EDIT_MODE', False)

def print_config():
    if CONFIG is None:
        print("CONFIG not initialized. Call set_global_opts() first.")
        return
    sensor_str = ', '.join(GLOBAL_OPTS['SENSOR']) if isinstance(GLOBAL_OPTS['SENSOR'], list) else str(GLOBAL_OPTS['SENSOR'])
    period_str = ', '.join(GLOBAL_OPTS['PERIODICITY']) if isinstance(GLOBAL_OPTS['PERIODICITY'], list) else str(GLOBAL_OPTS['PERIODICITY'])
    print(f"Country: {CONFIG['country'].upper()}")
    print(f"GEE Project: {CONFIG['gee_project']}")
    print(f"GCS Bucket: {CONFIG['bucket']}")
    print(f"GCS Images: {CONFIG['gcs_library_images_prefix']}")
    print(f"GCS Campaigns: {CONFIG['gcs_campaigns_prefix']}")
    print(f"GEE Images: {CONFIG['gee_library_images_prefix']}")
    print(f"GEE Campaigns: {CONFIG['gee_campaigns_prefix']}")
    print(f"Campaign: {CONFIG['campaign']}")
    print(f"Asset Regions: {CONFIG['asset_regions']}")
    print(f"Sensor(s): {sensor_str} | Period(s): {period_str} | Task Flag: {GLOBAL_OPTS['PERSONAL_TASK_FLAG']}")
