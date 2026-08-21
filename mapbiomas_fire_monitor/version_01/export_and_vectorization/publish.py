"""Publicacao incremental: copia COGs e vetores ZIP para o bucket publico
(espelho do mapbiomas-public) e limpa os tiles de temp dos meses ja
consolidados e validados la.

Fluxo por mes:
  1. COG existe no bucket de processamento -> copia se faltar no publico.
  2. ZIP vetorial existe no processamento  -> copia se faltar no publico.
  3. Se COG + ZIP validados no publico     -> apaga tiles de temp/ (libera espaco).

Idempotente: pode rodar mensalmente para pegar os meses que ficaram faltando.
"""

import time
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


def _copy_file(fs, src, dst, logger=None):
    """Copia src -> dst se faltar ou divergir em tamanho. Retorna status."""
    try:
        src_info = fs.info(src)
        src_size = src_info.get("size")
    except Exception as e:
        if logger:
            logger(f"[ERROR] Sem info de {src}: {e}")
        return "error"

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


def publish_all(logger=None):
    fs = _get_fs()

    cog_prefix_strip = f"monthly_burned-{config.COUNTRY}_"
    zip_prefix_strip = f"monthly_burned-{config.COUNTRY}_"

    src_mosaic = f"{config.BUCKET}/{config.mosaic_prefix()}"
    dst_mosaic = f"{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}"
    src_vec = f"{config.BUCKET}/{config.vector_prefix()}"
    dst_vec = f"{config.PUBLIC_BUCKET}/{config.vector_prefix()}"
    src_tiles = f"{config.BUCKET}/{config.tiles_prefix()}"

    cog_status = {}
    zip_status = {}

    # 1) COGs
    _log = logger or (lambda *_: None)
    _log(f"[PUBLISH] Sincronizando COGs para gs://{config.PUBLIC_BUCKET}/{config.mosaic_prefix()}/ ...")
    try:
        cogs = sorted(fs.glob(f"{src_mosaic}/monthly_burned-{config.COUNTRY}_*.tif"))
    except Exception as e:
        cogs = []
        _log(f"[ERROR] Listando COGs: {e}")

    for c in cogs:
        name = c.split("/")[-1]
        ym = _month_from_name(name, ".tif", cog_prefix_strip)
        if ym is None:
            continue
        dst = f"{dst_mosaic}/{name}"
        status = _copy_file(fs, c, dst, logger)
        cog_status[ym] = status
        if status == "copied":
            _log(f"[OK] Publicado COG: {name}")
        elif status == "exists":
            _log(f"[SKIP] COG ja no publico: {name}")

    # 2) Vetores ZIP
    _log(f"[PUBLISH] Sincronizando vetores ZIP para gs://{config.PUBLIC_BUCKET}/{config.vector_prefix()}/ ...")
    try:
        zips = sorted(fs.glob(f"{src_vec}/monthly_burned-{config.COUNTRY}_*.zip"))
    except Exception as e:
        zips = []
        _log(f"[ERROR] Listando ZIPs: {e}")

    for z in zips:
        name = z.split("/")[-1]
        ym = _month_from_name(name, ".zip", zip_prefix_strip)
        if ym is None:
            continue
        dst = f"{dst_vec}/{name}"
        status = _copy_file(fs, z, dst, logger)
        zip_status[ym] = status
        if status == "copied":
            _log(f"[OK] Publicado ZIP: {name}")
        elif status == "exists":
            _log(f"[SKIP] ZIP ja no publico: {name}")

    # 3) Limpeza de temp dos meses consolidados (COG + ZIP validados no publico)
    consolidated = []
    for ym in sorted(set(list(cog_status.keys()) + list(zip_status.keys()))):
        if cog_status.get(ym) in _OK and zip_status.get(ym) in _OK:
            consolidated.append(ym)

    deleted_tiles = 0
    for y, m in consolidated:
        pattern = f"{src_tiles}/{config.tile_pattern(y, m)}*.tif"
        try:
            tiles = fs.glob(pattern)
        except Exception:
            tiles = []
        for t in tiles:
            try:
                fs.rm(t)
                deleted_tiles += 1
                _log(f"[DEL] temp: {t.split('/')[-1]}")
            except Exception as e:
                _log(f"[ERROR] Nao consegui apagar {t}: {e}")

    _log(f"[PUBLISH] Resumo: COGs publicados/verificados {len(cog_status)}, "
         f"ZIPs {len(zip_status)}, meses consolidados {len(consolidated)}, "
         f"tiles removidos {deleted_tiles}.", "success")
    return {"cog_status": cog_status, "zip_status": zip_status, "consolidated": consolidated}
