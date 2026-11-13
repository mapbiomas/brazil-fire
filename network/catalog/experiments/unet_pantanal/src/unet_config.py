"""
U-Net experiment - configuration and GCS paths.

Independent experiment (no import from the catalog versions). It reads the
same mosaics (catalog M1/M2 COGs in LIBRARY_IMAGES) and the same training
polygons (catalog M3 CSVs in LIBRARY_SAMPLES) of a campaign. Everything it
writes goes to ``{GCS_PREFIX}/{CAMPAIGN}/{EXPERIMENT}/``; it never changes
or deletes other files.

Usage in the notebook::

    from unet_config import make_config, paths
    UCFG = make_config(REGION="pantanal", YEARS=[2024])
    P = paths(UCFG)
"""
import copy

# Spectral bands read from the mosaic COGs, in this order (channels 0..4).
DEFAULT_BANDS = ["blue", "red", "nir", "swir1", "swir2"]

# Spectral indices appended after the bands (see unet_features.add_indices).
DEFAULT_INDICES = ["nbr", "nbr2", "mirbi", "ndvi", "bai", "evi"]

DEFAULTS = {
    # --- Project (where the catalog data is) -------------------------------
    "COUNTRY": "brazil",
    "BUCKET": "mapbiomas-fire",
    "GCS_PREFIX": "sudamerica/brazil/CATALOG_01",   # holds LIBRARY_IMAGES and the campaigns
    "CAMPAIGN": "COLLECTION_06",
    "GEE_PROJECT": "mapbiomas-brazil",
    "REGIONS_ASSET": ("projects/mapbiomas-brazil/assets/FIRE/CATALOG_01/COLLECTION_06/"
                      "AUXILIARY/regiones_fuego_brazil_v1"),
    "REGION_PROPERTY": "region_nam",
    "SERVICE_ACCOUNT": "mapbiomas-fire@mapbiomas.iam.gserviceaccount.com",

    # --- Experiment -------------------------------------------------------
    "MODEL": "unet",                 # "unet" | "mlp" (mlp = catalog M4/M5 notebook)
    "EXPERIMENT": "UNET_01",         # output folder inside the campaign
    "MODEL_NAME": "unet_pantanal_v1",
    "REGION": "pantanal",            # value of `region_nam` in the regions asset
    "YEARS": [2024],

    # --- Mosaics (M2 COGs) ------------------------------------------------
    "SENSOR": "landsat",
    "MOSAIC": "minnbr",
    "BANDS": list(DEFAULT_BANDS),
    "VALID_BAND": "dayOfYear",       # pixel is valid where this band > 0 (same rule as M5)
    # M1 exports Landsat reflectance x 100 as uint8 (Sentinel-2/MODIS use other
    # multipliers). Dividing by SCALE gives reflectance 0-1.
    "SCALE": 100.0,
    # Optional override: one multiband GeoTIFF per year instead of one COG per
    # band. Relative to the bucket, `{year}` is replaced. Band order = BANDS,
    # VALID_BAND last.
    "MULTIBAND_FILE": None,

    # --- Samples (M3 CSVs) ------------------------------------------------
    # File names in LIBRARY_SAMPLES (fnmatch pattern; {year} and {region} are
    # replaced). Rows are also filtered by `period == year`.
    "SAMPLE_GLOB": "*{year}*.csv",
    "SAMPLE_FILES": None,            # explicit list of CSV names (overrides SAMPLE_GLOB)
    "FILTER_SAMPLES_BY_REGION": True,  # keep rows whose `region` column == REGION
    "CLASS_FIELD": "fire",

    # --- Features ---------------------------------------------------------
    "INDICES": list(DEFAULT_INDICES),
    "CLIP_PERCENTILES": (1.0, 99.0),
    "STATS_MAX_PIXELS": 2_000_000,   # random train pixels used for clip/mean/std

    # --- Chips ------------------------------------------------------------
    "CHIP": 128,
    "POINTS_PER_POLYGON": 4,
    "MIN_LABELED_FRACTION": 0.01,
    "MAX_NODATA_FRACTION": 0.50,
    "SHARD_MB": 100,

    # --- Spatial split ----------------------------------------------------
    "BLOCK_KM": 25,
    "SPLIT": (0.70, 0.15, 0.15),     # train, val, test (per block)
    "SEED": 42,

    # --- Training ---------------------------------------------------------
    "FILTERS": [32, 64, 128, 256],
    "BOTTOM_FILTERS": 512,
    "BATCH": 16,
    "EPOCHS": 60,
    "LR": 1e-4,
    "PATIENCE": 8,
    "MIXED_PRECISION": True,

    # --- Inference --------------------------------------------------------
    "OVERLAP": 16,                   # border discarded on each side
    "INFER_TILE": 512,               # core size written per window (multiple of 16)
    "INFER_BATCH": 8,

    # --- MLP baseline (5.13) ----------------------------------------------
    "MLP_LAYERS": [7, 14, 7, 14, 7], # version_04 M4 default
    "MLP_ITERS": 7000,
    "MLP_BATCH": 1000,
    "MLP_LR": 0.001,
    "MLP_MAX_PIXELS": 500_000,
}

SPLITS = ("train", "val", "test")
IGNORE = 255  # label value ignored by the loss and the metrics


def make_config(**overrides):
    """Return a copy of DEFAULTS with the given overrides (unknown keys fail)."""
    cfg = copy.deepcopy(DEFAULTS)
    unknown = sorted(set(overrides) - set(cfg))
    if unknown:
        raise KeyError(f"Unknown U-Net config keys: {unknown}")
    cfg.update(copy.deepcopy(overrides))
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg["MODEL"] not in ("unet", "mlp"):
        raise ValueError("MODEL must be 'unet' or 'mlp'.")
    split = tuple(cfg["SPLIT"])
    if len(split) != 3 or abs(sum(split) - 1.0) > 1e-6:
        raise ValueError("SPLIT must have 3 fractions that sum to 1.")
    depth = 2 ** len(cfg["FILTERS"])
    if cfg["CHIP"] % depth:
        raise ValueError(f"CHIP must be a multiple of {depth}.")
    if (cfg["INFER_TILE"] + 2 * cfg["OVERLAP"]) % depth:
        raise ValueError(f"INFER_TILE + 2*OVERLAP must be a multiple of {depth}.")
    if not cfg["YEARS"]:
        raise ValueError("YEARS must list at least one year.")
    return cfg


def channel_names(cfg):
    return list(cfg["BANDS"]) + list(cfg["INDICES"])


def paths(cfg):
    """GCS paths (relative to the bucket) for this experiment.

    The sample folder follows the catalog layout
    (``{GCS_PREFIX}/{CAMPAIGN}/LIBRARY_SAMPLES``), so the U-Net reads exactly
    what the catalog M3 wrote.
    """
    campaign = f"{cfg['GCS_PREFIX']}/{cfg['CAMPAIGN']}"
    root = f"{campaign}/{cfg['EXPERIMENT']}"
    return {
        "bucket": cfg["BUCKET"],
        "samples": f"{campaign}/LIBRARY_SAMPLES",
        "root": root,
        "chips": f"{root}/chips",
        "stats": f"{root}/stats/norm_stats.json",
        "splits": f"{root}/splits",
        "models": f"{root}/models",
        "predict": f"{root}/predict",
        "reports": f"{root}/reports",
    }


def mosaic_band_paths(cfg, year):
    """{band: 'bucket/.../image_..._cog.tif'} for BANDS + VALID_BAND.

    Annual mosaics, same folder and file names the catalog (version_04 M1/M2)
    writes and its M4/M5 read:
    {GCS_PREFIX}/LIBRARY_IMAGES/{SENSOR}/ANNUALLY/{MOSAIC}/{year}/COG/
        image_{country}_fire_{sensor}_{mosaic}_{band}_{year}_cog.tif
    """
    bands = list(cfg["BANDS"]) + [cfg["VALID_BAND"]]
    if cfg.get("MULTIBAND_FILE"):
        path = f"{cfg['BUCKET']}/{cfg['MULTIBAND_FILE'].format(year=year)}"
        return {b: path for b in bands}

    sensor, mosaic = cfg["SENSOR"].lower(), cfg["MOSAIC"].lower()
    folder = (f"{cfg['GCS_PREFIX']}/LIBRARY_IMAGES/{sensor.upper()}/ANNUALLY/"
              f"{mosaic.upper()}/{year}/COG")
    return {
        b: f"{cfg['BUCKET']}/{folder}/image_{cfg['COUNTRY']}_fire_{sensor}_{mosaic}_{b}_{year}_cog.tif"
        for b in bands
    }
