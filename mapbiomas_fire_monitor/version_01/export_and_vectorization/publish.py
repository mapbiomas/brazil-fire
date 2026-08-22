"""Publicacao incremental em 3 etapas.

1. publish_mosaic_all  — copia COGs do bucket de processamento para o publico.
2. publish_vector_all  — copia vetores ZIP do bucket de processamento para o publico.
3. cleanup_temp_all    — apaga tiles de temp/ dos meses com COG + ZIP validados
                         no publico (consolidados).

publish_all() encadeia as tres etapas. Tudo idempotente: pode rodar quando quiser
para pegar os meses que ficaram faltando.
"""

from . import config
from .state import _get_fs

_OK = {"exists", "copied"}


def _month_from_name(name, suffix, prefix_strip):
    body = name.replace(prefix_strip, "").replace(suffix, "")
    parts = body.split("_")
    if len(parts) >= 2:
        try:
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            pass
    return None


def _copy_file(fs, src, dst, logger=None, force=False):
    """Copia src -> dst se faltar ou divergir em tamanho. Retorna status.

    force=True: sempre copia (sobrescreve o destino) e valida o tamanho.
    """
    try:
        src_info = fs.info(src)
        src_size = src_info.get("size")
    except Exception as e:
        if logger:
            logger(f"[ERROR] Sem info de {src}: {e}")
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
                logger(f"[ERROR] Checando {dst}: {e}")
            return "error"

    try:
        fs.copy(src, dst)
        dst_info = fs.info(dst)
        if dst_info.get("size") != src_size:
            if logger:
                logger(f"[ERROR] Copia divergiu em tamanho: {dst}")
            return "error"
        return "copied"
    except Exception as e:
        if logger:
            logger(f"[ERROR] Copia falhou ({src} -> {dst}): {e}")
        return "error"


def publish_mosaic_all(logger=None, force=False):
    """Etapa 5 — espelha os COGs mensais no bucket publico."""
    fs = _get_fs()
    prefix_strip = f"monthly_burned-{config.COUNTRY}_"
    src = f"{config.BUCKET}/{config.mosaic_prefix()}"
    dst = f"{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}"
    _log = logger or (lambda *_: None)

    _log(f"[PUBLISH MOSAIC] Sincronizando COGs para gs://{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}/ ...")
    try:
        cogs = sorted(fs.glob(f"{src}/monthly_burned-{config.COUNTRY}_*.tif"))
    except Exception as e:
        cogs = []
        _log(f"[ERROR] Listando COGs: {e}")

    status = {}
    for c in cogs:
        name = c.split("/")[-1]
        ym = _month_from_name(name, ".tif", prefix_strip)
        if ym is None:
            continue
        s = _copy_file(fs, c, f"{dst}/{name}", logger, force=force)
        status[ym] = s
        if s == "copied":
            _log(f"[OK] Publicado COG: {name}")
        elif s == "exists":
            _log(f"[SKIP] COG ja no publico: {name}")

    _log(f"[PUBLISH MOSAIC] Resumo: {len(status)} COGs verificados.", "success")
    return status


def publish_vector_all(logger=None, force=False):
    """Etapa 6 — espelha os vetores ZIP no bucket publico."""
    fs = _get_fs()
    prefix_strip = f"monthly_burned-{config.COUNTRY}_"
    src = f"{config.BUCKET}/{config.vector_prefix()}"
    dst = f"{config.PUBLIC_BUCKET}/{config.vector_prefix()}"
    _log = logger or (lambda *_: None)

    _log(f"[PUBLISH VECTOR] Sincronizando vetores ZIP para gs://{config.PUBLIC_BUCKET}/{config.vector_prefix()}/ ...")
    try:
        zips = sorted(fs.glob(f"{src}/monthly_burned-{config.COUNTRY}_*.zip"))
    except Exception as e:
        zips = []
        _log(f"[ERROR] Listando ZIPs: {e}")

    status = {}
    for z in zips:
        name = z.split("/")[-1]
        ym = _month_from_name(name, ".zip", prefix_strip)
        if ym is None:
            continue
        s = _copy_file(fs, z, f"{dst}/{name}", logger, force=force)
        status[ym] = s
        if s == "copied":
            _log(f"[OK] Publicado ZIP: {name}")
        elif s == "exists":
            _log(f"[SKIP] ZIP ja no publico: {name}")

    _log(f"[PUBLISH VECTOR] Resumo: {len(status)} ZIPs verificados.", "success")
    return status


def cleanup_temp_all(logger=None):
    """Etapa 7 — apaga tiles de temp/ apenas dos meses consolidados no publico
    (COG + ZIP validados)."""
    fs = _get_fs()
    prefix_strip = f"monthly_burned-{config.COUNTRY}_"
    _log = logger or (lambda *_: None)

    try:
        pub_cogs = set(
            ym for ym in (
                _month_from_name(f.split('/')[-1], ".tif", prefix_strip)
                for f in fs.glob(f"{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}/monthly_burned-{config.COUNTRY}_*.tif")
            ) if ym
        )
    except Exception as e:
        pub_cogs = set()
        _log(f"[ERROR] Listando COGs publicos: {e}")

    try:
        pub_zips = set(
            ym for ym in (
                _month_from_name(f.split('/')[-1], ".zip", prefix_strip)
                for f in fs.glob(f"{config.PUBLIC_BUCKET}/{config.vector_prefix()}/monthly_burned-{config.COUNTRY}_*.zip")
            ) if ym
        )
    except Exception as e:
        pub_zips = set()
        _log(f"[ERROR] Listando ZIPs publicos: {e}")

    consolidated = sorted(pub_cogs & pub_zips)
    _log(f"[CLEAN TEMP] {len(consolidated)} meses consolidados no publico (COG+ZIP).")

    deleted = 0
    for y, m in consolidated:
        pattern = f"{config.BUCKET}/{config.tiles_prefix()}/{config.tile_pattern(y, m)}*.tif"
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
                _log(f"[ERROR] Nao consegui apagar {t}: {e}")

    _log(f"[CLEAN TEMP] Resumo: {deleted} tiles removidos.", "success")
    return {"consolidated": consolidated, "deleted_tiles": deleted}


def publish_all(logger=None, force=False):
    """Executa as 3 etapas de publicacao em sequencia."""
    publish_mosaic_all(logger=logger, force=force)
    publish_vector_all(logger=logger, force=force)
    cleanup_temp_all(logger=logger)
