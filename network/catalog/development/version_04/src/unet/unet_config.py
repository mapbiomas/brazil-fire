"""
U-Net experiment - configuration and GCS paths.

The U-Net is an alternative M4/M5 path of the version_04 catalog. It reuses
the same mosaics (M1/M2 COGs in LIBRARY_IMAGES) and the same training
polygons (M3 CSVs in LIBRARY_SAMPLES). Everything it writes goes to a
separate folder, ``{gcs_campaigns_prefix}/{campaign}/{EXPERIMENT}/``; it
never changes or deletes existing files.

Usage in the notebook (after ``set_global_opts()`` and ``authenticate()``)::

    from unet_config import make_config, paths
    UCFG = make_config(REGION="pantanal", YEARS=[2024])
    P = paths(UCFG)
"""
import copy

# Spectral bands read from the M2 COGs, in this order (channels 0..4).
DEFAULT_BANDS = ["blue", "red", "nir", "swir1", "swir2"]

# Spectral indices appended after the bands (see unet_features.add_indices).
DEFAULT_INDICES = ["nbr", "nbr2", "mirbi", "ndvi", "bai", "evi"]

DEFAULTS = {
    # --- Experiment -------------------------------------------------------
    "MODEL": "unet",                 # "unet" | "mlp" (mlp = version_04 M4/M5, untouched)
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

    Mosaic and sample folders come from the version_04 M0 helpers, so the
    U-Net reads exactly what M2 and M3 wrote.
    """
    from M0_auth_config import CONFIG, gcs_samples_path

    root = f"{CONFIG['gcs_campaigns_prefix']}/{CONFIG['campaign']}/{cfg['EXPERIMENT']}"
    return {
        "bucket": CONFIG["bucket"],
        "samples": gcs_samples_path(),
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

    Same names M4_data_extractor and M5_inference use.
    """
    from M0_auth_config import CONFIG, mosaic_name_for_period, period_cog_path

    bands = list(cfg["BANDS"]) + [cfg["VALID_BAND"]]
    if cfg.get("MULTIBAND_FILE"):
        path = f"{CONFIG['bucket']}/{cfg['MULTIBAND_FILE'].format(year=year)}"
        return {b: path for b in bands}

    key = str(year)
    folder = period_cog_path(key, mosaic=cfg["MOSAIC"], sensor=cfg["SENSOR"])
    out = {}
    for b in bands:
        name = mosaic_name_for_period(key, band=b, mosaic=cfg["MOSAIC"], sensor=cfg["SENSOR"])
        out[b] = f"{CONFIG['bucket']}/{folder}/{name}_cog.tif"
    return out
