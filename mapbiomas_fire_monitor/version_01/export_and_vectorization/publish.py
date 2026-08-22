"""Publicacao incremental em 3 etapas (por produto/unidade ativo).

1. publish_mosaic_all  — copia COGs do bucket de processamento para o publico.
2. publish_vector_all  — copia vetores ZIP para o publico.
3. cleanup_temp_all    — apaga tiles de temp/ das unidades consolidadas
                         (COG + ZIP validados no publico).
"""

from . import config
from .state import _get_fs

_OK = {"exists", "copied"}


def _unit_from_name(name, suffix, prefix_strip):
    body = name.replace(prefix_strip, "").replace(suffix, "")
    return body or None


def _art_prefix():
    return f"{config.PRODUCT}-{config.COUNTRY}_"


def _copy_file(fs, src, dst, logger=None, force=False):
    """Copia src -> dst se faltar ou divergir em tamanho. force=True sobrescreve."""
    try:
        src_info = fs.info(src)
        src_size = src_info.get("size")
    except Exception as e:
        if logger:
            logger(f"[ERROR] No info for {src}: {e}")
        return "error"

    if not force:
        try:
            dst_info = fs.info(dst)
            if dst_info.get("size") == src_size:
                return "exists"
        except FileNotFoundError:
            pass
        except Exception as e:
            if logger:
                logger(f"[ERROR] Checking {dst}: {e}")
            return "error"

    try:
        fs.copy(src, dst)
        dst_info = fs.info(dst)
        if dst_info.get("size") != src_size:
            if logger:
                logger(f"[ERROR] Copy size mismatch: {dst}")
            return "error"
        return "copied"
    except Exception as e:
        if logger:
            logger(f"[ERROR] Copy failed ({src} -> {dst}): {e}")
        return "error"


def publish_mosaic_all(logger=None, force=False):
    fs = _get_fs()
    prefix = _art_prefix()
    src = f"{config.BUCKET}/{config.mosaic_prefix()}"
    dst = f"{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}"
    _log = logger or (lambda *_: None)

    _log(f"[PUBLISH MOSAIC] Syncing COGs to gs://{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}/ ...")
    try:
        cogs = sorted(fs.glob(f"{src}/{prefix}*.tif"))
    except Exception as e:
        cogs = []
        _log(f"[ERROR] Listing COGs: {e}")

    status = {}
    for c in cogs:
        name = c.split("/")[-1]
        unit = _unit_from_name(name, ".tif", prefix)
        if unit is None:
            continue
        s = _copy_file(fs, c, f"{dst}/{name}", logger, force=force)
        status[unit] = s
        if s == "copied":
            _log(f"[OK] Published COG: {name}")
        elif s == "exists":
            _log(f"[SKIP] COG already in public: {name}")

    _log(f"[PUBLISH MOSAIC] Summary: {len(status)} COGs verified.", "success")
    return status


def publish_vector_all(logger=None, force=False):
    fs = _get_fs()
    prefix = _art_prefix()
    src = f"{config.BUCKET}/{config.vector_prefix()}"
    dst = f"{config.PUBLIC_BUCKET}/{config.vector_prefix()}"
    _log = logger or (lambda *_: None)

    if not config.is_vectorizable():
        _log("[PUBLISH VECTOR] SKIP: product not vectorizable.", "warning")
        return {}

    _log(f"[PUBLISH VECTOR] Syncing vector ZIPs to gs://{config.PUBLIC_BUCKET}/{config.vector_prefix()}/ ...")
    try:
        zips = sorted(fs.glob(f"{src}/{prefix}*.zip"))
    except Exception as e:
        zips = []
        _log(f"[ERROR] Listing ZIPs: {e}")

    status = {}
    for z in zips:
        name = z.split("/")[-1]
        unit = _unit_from_name(name, ".zip", prefix)
        if unit is None:
            continue
        s = _copy_file(fs, z, f"{dst}/{name}", logger, force=force)
        status[unit] = s
        if s == "copied":
            _log(f"[OK] Published ZIP: {name}")
        elif s == "exists":
            _log(f"[SKIP] ZIP already in public: {name}")

    _log(f"[PUBLISH VECTOR] Summary: {len(status)} ZIPs verified.", "success")
    return status


def cleanup_temp_all(logger=None):
    fs = _get_fs()
    prefix = _art_prefix()
    _log = logger or (lambda *_: None)

    try:
        pub_cogs = set(
            u for u in (
                _unit_from_name(f.split('/')[-1], ".tif", prefix)
                for f in fs.glob(f"{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}/{prefix}*.tif")
            ) if u
        )
    except Exception as e:
        pub_cogs = set()
        _log(f"[ERROR] Listing public COGs: {e}")

    try:
        pub_zips = set(
            u for u in (
                _unit_from_name(f.split('/')[-1], ".zip", prefix)
                for f in fs.glob(f"{config.PUBLIC_BUCKET}/{config.vector_prefix()}/{prefix}*.zip")
            ) if u
        )
    except Exception as e:
        pub_zips = set()
        _log(f"[ERROR] Listing public ZIPs: {e}")

    consolidated = sorted(pub_cogs & pub_zips)
    _log(f"[CLEAN TEMP] {len(consolidated)} units consolidated in public (COG+ZIP).")

    deleted = 0
    for unit in consolidated:
        pattern = f"{config.BUCKET}/{config.tiles_prefix()}/{config.tile_pattern_unit(unit)}*.tif"
        try:
            tiles = fs.glob(pattern)
        except Exception:
            tiles = []
        for t in tiles:
            try:
                fs.rm(t)
                deleted += 1
                _log(f"[DEL] temp: {t.split('/')[-1]}")
            except Exception as e:
                _log(f"[ERROR] Could not delete {t}: {e}")

    _log(f"[CLEAN TEMP] Summary: {deleted} tiles removed.", "success")
    return {"consolidated": consolidated, "deleted_tiles": deleted}


def publish_all(logger=None, force=False):
    """Executa as 3 etapas de publicacao em sequencia."""
    publish_mosaic_all(logger=logger, force=force)
    publish_vector_all(logger=logger, force=force)
    cleanup_temp_all(logger=logger)
