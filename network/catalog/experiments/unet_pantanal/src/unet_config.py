"""
U-Net experiment - configuration and GCS paths.

Independent experiment (no import from the catalog versions). It reads the
same mosaics (catalog M1/M2 COGs in LIBRARY_IMAGES) and the same training
polygons (catalog M3 CSVs in LIBRARY_SAMPLES) of a campaign. Everything it
writes goes to ``{GCS_PREFIX}/{CAMPAIGN}/{EXPERIMENT}/``; it never changes
or deletes other files.

Default: Fire Monitor (MONITOR_01), Sentinel-2, monthly mosaics and monthly
samples. A period is a catalog period key: '2026_10' (monthly) or '2024'
(annual).

Usage in the notebook::

    from unet_config import make_config, paths
    UCFG = make_config(REGION="pantanal", PERIODS=["2026_10"])
    P = paths(UCFG)
"""
import copy

# Spectral bands read from the mosaic COGs, in this order.
# Full sets, for when every band is exported.
FULL_BANDS = ["blue", "red", "nir", "swir1", "swir2"]
FULL_INDICES = ["nbr", "nbr2", "mirbi", "ndvi", "bai", "evi"]

# First test: only the bands exported so far for MONITOR_01 2026_10.
DEFAULT_BANDS = ["green", "swir2"]
# DEFAULT_BANDS = list(FULL_BANDS)

# Spectral indices appended after the bands (see unet_features.add_indices).
# None of them can be computed from green + swir2 only.
DEFAULT_INDICES = []
# DEFAULT_INDICES = list(FULL_INDICES)   # needs blue, red, nir, swir1, swir2

# Bands each index needs (checked by validate_config).
INDEX_BANDS = {
    "nbr": {"nir", "swir2"}, "nbr2": {"swir1", "swir2"}, "mirbi": {"swir1", "swir2"},
    "ndvi": {"nir", "red"}, "bai": {"red", "nir"}, "evi": {"nir", "red", "blue"},
}

DEFAULTS = {
    # --- Project (where the catalog data is) -------------------------------
    "COUNTRY": "brazil",
    "BUCKET": "mapbiomas-fire",
    "GCS_PREFIX": "sudamerica/brazil/CATALOG_01",   # holds LIBRARY_IMAGES and the campaigns
    "CAMPAIGN": "MONITOR_01",
    "GEE_PROJECT": "mapbiomas-brazil",
    "REGIONS_ASSET": ("projects/mapbiomas-brazil/assets/FIRE/CATALOG_01/MONITOR_01/"
                      "AUXILIARY/regiones_fuego_brazil_v1"),
    "REGION_PROPERTY": "region_nam",
    "SERVICE_ACCOUNT": "mapbiomas-fire@mapbiomas.iam.gserviceaccount.com",

    # --- Experiment -------------------------------------------------------
    "MODEL": "unet",                 # "unet" | "mlp" (mlp = catalog M4/M5 notebook)
    "EXPERIMENT": "UNET_01",         # output folder inside the campaign
    "MODEL_NAME": "unet_monitor_pantanal_v1",
    "REGION": "pantanal",            # value of `region_nam` in the regions asset
    # Periods used for training and classification: '2026_10' (monthly) or
    # '2024' (annual). Chips of all periods train one model; each period gets
    # its own map.
    "PERIODS": ["2026_10"],
    # Periods to classify with the trained model (need a mosaic, not samples).
    # None = the same as PERIODS.
    "PREDICT_PERIODS": None,

    # --- Mosaics (M2 COGs) ------------------------------------------------
    "SENSOR": "sentinel2",           # sentinel2 (10 m) | landsat (30 m)
    "MOSAIC": "minnbr",
    "BANDS": list(DEFAULT_BANDS),
    # Pixel is valid where this band > 0 (same rule as M5), and the output
    # band 2 gets its value. None (not exported yet): a pixel is valid where
    # the bands are not nodata and not all 0; output band 2 stays -9999.
    "VALID_BAND": None,              # "dayOfYear" when it is exported
    # M1 exports Sentinel-2 and Landsat reflectance x 100 as uint8.
    # Dividing by SCALE gives reflectance 0-1.
    "SCALE": 100.0,
    # Optional override: one multiband GeoTIFF per period instead of one COG
    # per band. Relative to the bucket, `{period}` is replaced. Band order =
    # BANDS, VALID_BAND last.
    "MULTIBAND_FILE": None,

    # --- Samples (M3 CSVs) ------------------------------------------------
    # File names in LIBRARY_SAMPLES (fnmatch pattern; {period} and {region}
    # are replaced). Rows are also filtered by the `period` column (or by the
    # date at the end of the file name when `period` is empty), as M4 does.
    "SAMPLE_GLOB": "*{period}*.csv",
    "SAMPLE_FILES": None,            # explicit list of CSV names (overrides SAMPLE_GLOB)
    "FILTER_SAMPLES_BY_REGION": True,  # keep rows whose `region` column == REGION
    "CLASS_FIELD": "fire",

    # --- Features ---------------------------------------------------------
    "INDICES": list(DEFAULT_INDICES),
    "CLIP_PERCENTILES": (1.0, 99.0),
    "STATS_MAX_PIXELS": 2_000_000,   # random train pixels used for clip/mean/std

    # --- Chips ------------------------------------------------------------
    # 256 px x 10 m (Sentinel-2) = 2.56 km. For Landsat (30 m), 128 px = 3.8 km.
    "CHIP": 256,
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
    "BATCH": 8,                      # 8 chips of 256 px fit a Colab T4 with mixed precision
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
    if not cfg["BANDS"]:
        raise ValueError("BANDS must list at least one band.")
    for idx in cfg["INDICES"]:
        missing = INDEX_BANDS.get(idx, set()) - set(cfg["BANDS"])
        if missing:
            raise ValueError(f"Index '{idx}' needs bands {sorted(missing)}; add them to BANDS "
                             "or remove the index from INDICES.")
    if not cfg["PERIODS"]:
        raise ValueError("PERIODS must list at least one period (e.g. ['2026_10']).")
    cfg["PERIODS"] = [period_info(p)[0] for p in cfg["PERIODS"]]
    cfg["PREDICT_PERIODS"] = [period_info(p)[0] for p in (cfg["PREDICT_PERIODS"] or cfg["PERIODS"])]
    return cfg


# ---------------------------------------------------------------------------
# Periods (same keys and folders as the catalog M_periods)
# ---------------------------------------------------------------------------

PERIOD_FOLDERS = {"yearly": "ANNUALLY", "monthly": "MONTHLY"}


def normalize_period(value):
    """'2026_10', '202610', 2024, '2024.0' -> '2026_10' / '2024'; empty -> ''."""
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "nat", "none", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit() and len(text) == 6:
        return f"{text[:4]}_{text[4:]}"
    return text


def period_info(value):
    """(key, periodicity) for an annual ('2024') or monthly ('2026_10') period."""
    key = normalize_period(value)
    parts = key.split("_")
    if not parts[0].isdigit() or len(parts[0]) != 4:
        raise ValueError(f"Invalid period '{value}'; use 'YYYY' or 'YYYY_MM'.")
    if len(parts) == 1:
        return key, "yearly"
    if len(parts) == 2 and parts[1].isdigit() and 1 <= int(parts[1]) <= 12:
        return f"{parts[0]}_{int(parts[1]):02d}", "monthly"
    raise ValueError(f"Invalid period '{value}'; use 'YYYY' or 'YYYY_MM'.")


def period_from_file_name(name):
    """Date at the end of a sample file name ('..._2026_10.csv' -> '2026_10'), as M4."""
    parts = name.rsplit(".", 1)[0].split("_")
    if len(parts) >= 2 and len(parts[-2]) == 4 and parts[-2].isdigit() and parts[-1].isdigit():
        return f"{parts[-2]}_{parts[-1]}"
    if len(parts[-1]) == 4 and parts[-1].isdigit():
        return parts[-1]
    return ""


def split_chip_name(path):
    """'.../2026_10_train_003.tfrecord.gz' -> ('2026_10', 'train', '003')."""
    import os
    name = os.path.basename(path).split(".")[0]
    period, split, shard = name.rsplit("_", 2)
    return period, split, shard


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


def mosaic_band_paths(cfg, period):
    """{band: 'bucket/.../image_..._cog.tif'} for BANDS + VALID_BAND (if set).

    Same folder and file names the catalog (version_04 M1/M2) writes and its
    M4/M5 read:
    {GCS_PREFIX}/LIBRARY_IMAGES/{SENSOR}/{MONTHLY|ANNUALLY}/{MOSAIC}/{period}/COG/
        image_{country}_fire_{sensor}_{mosaic}_{band}_{period}_cog.tif
    """
    key, periodicity = period_info(period)
    bands = list(cfg["BANDS"]) + ([cfg["VALID_BAND"]] if cfg.get("VALID_BAND") else [])
    if cfg.get("MULTIBAND_FILE"):
        path = f"{cfg['BUCKET']}/{cfg['MULTIBAND_FILE'].format(period=key)}"
        return {b: path for b in bands}

    sensor, mosaic = cfg["SENSOR"].lower(), cfg["MOSAIC"].lower()
    folder = (f"{cfg['GCS_PREFIX']}/LIBRARY_IMAGES/{sensor.upper()}/{PERIOD_FOLDERS[periodicity]}/"
              f"{mosaic.upper()}/{key}/COG")
    return {
        b: f"{cfg['BUCKET']}/{folder}/image_{cfg['COUNTRY']}_fire_{sensor}_{mosaic}_{b}_{key}_cog.tif"
        for b in bands
    }
