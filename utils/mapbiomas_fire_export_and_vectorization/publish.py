"""Publicacao por produto/unidade selecionado.

1. publish_mosaic_all       — copia COGs do bucket de processamento para o publico.
2. publish_vector_all       — copia vetores ZIP para o publico.
   Com ui informado e selecao nao vazia, ambas iteram sobre TODOS os contextos
   afetados pela selecao multi-painel.
3. cleanup_temp_selected    — apaga os tiles temp/ das unidades selecionadas,
   sem condicionais com as demais etapas (a ordem e apenas sugestao de fluxo).
   A delecao e restrita ao padrao de tiles dentro de temp/: nunca alcanca
   COG, ZIP, bucket publico ou assets GEE.
"""

from . import config
from . import selection
from .state import _get_fs

_OK = {"exists", "copied"}


def _unit_from_name(name, suffix, prefix_strip):
    body = name.replace(prefix_strip, "").replace(suffix, "")
    return body or None


def _art_prefix(context=None):
    return config.art_prefix(context)


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


def _iter_contexts(ui, context):
    """Contextos a processar: os afetados pela selecao multi-painel quando
    houver itens selecionados; caso contrario, o contexto unico recebido
    (ou o global). Devolve dict ordenado {chave: ctx}, onde a chave e a
    tupla do contexto em modo multi-painel ou None em modo simples.
    """
    if ui is not None:
        items = selection.collect_items(ui)
        if items:
            _, seen_ctx = selection.build_jobs(items)
            return {key: seen_ctx[key] for key in sorted(seen_ctx)}
    ctx = context or config.processing_context()
    return {None: ctx}


def publish_mosaic_all(logger=None, force=False, context=None, ui=None):
    fs = _get_fs()
    _log = logger or (lambda *_: None)
    status = {}

    for key, ctx in _iter_contexts(ui, context).items():
        if key is not None:
            _log(f"[PUBLISH MOSAIC] {key[0]}/{key[1]}/{key[2]}/{key[3]}: "
                 f"syncing COGs to gs://{config.PUBLIC_BUCKET}/ ...")
        prefix = _art_prefix(ctx)
        src = f"{config.BUCKET}/{config.mosaic_prefix(ctx)}"
        dst = f"{config.PUBLIC_BUCKET}/{config.mosaic_prefix(ctx)}"

        try:
            cogs = sorted(fs.glob(f"{src}/{prefix}*.tif"))
        except Exception as e:
            cogs = []
            _log(f"[ERROR] Listing COGs: {e}")

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


def publish_vector_all(logger=None, force=False, context=None, ui=None):
    fs = _get_fs()
    _log = logger or (lambda *_: None)
    status = {}

    for key, ctx in _iter_contexts(ui, context).items():
        if not config.is_vectorizable(ctx):
            target = (f"{key[0]}/{key[1]}/{key[2]}/{key[3]}"
                      if key is not None else "product")
            _log(f"[PUBLISH VECTOR] SKIP {target}: product not vectorizable.", "warning")
            continue
        if key is not None:
            _log(f"[PUBLISH VECTOR] {key[0]}/{key[1]}/{key[2]}/{key[3]}: "
                 f"syncing vector ZIPs to gs://{config.PUBLIC_BUCKET}/ ...")
        prefix = _art_prefix(ctx)
        src = f"{config.BUCKET}/{config.vector_prefix(ctx)}"
        dst = f"{config.PUBLIC_BUCKET}/{config.vector_prefix(ctx)}"

        try:
            zips = sorted(fs.glob(f"{src}/{prefix}*.zip"))
        except Exception as e:
            zips = []
            _log(f"[ERROR] Listing ZIPs: {e}")

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


def cleanup_temp_selected(ui, logger=None):
    """Apaga os tiles temp/ das unidades selecionadas em todos os paineis.

    Sem condicionais com as demais etapas: deleta o que casar no padrao de
    tiles da unidade, exista COG/ZIP/publicacao ou nao. A operacao e
    idempotente e restrita ao cenario temp/ — nunca alcanca COGs, ZIPs,
    bucket publico ou assets GEE.
    """
    fs = _get_fs()
    _log = logger or (lambda *_: None)

    items = selection.collect_items(ui)
    if not items:
        _log("[CLEAN TEMP] No unit selected.", "warning")
        return {"deleted_tiles": 0}

    jobs, _ = selection.build_jobs(items)

    deleted = 0
    for unit, ctx in jobs:
        pattern = (f"{config.BUCKET}/{config.tiles_prefix(ctx)}/"
                   f"{config.tile_pattern_unit(unit, ctx)}*.tif")
        try:
            tiles = fs.glob(pattern)
        except Exception as e:
            _log(f"[ERROR] Listing tiles of {unit}: {e}")
            continue
        removed = 0
        for t in tiles:
            try:
                fs.rm(t)
                removed += 1
                deleted += 1
            except Exception as e:
                _log(f"[ERROR] Could not delete {t}: {e}")
        if removed:
            _log(f"[OK] {unit}: {removed} tile(s) removed.")
        else:
            _log(f"[SKIP] {unit}: no temp tiles.")

    _log(f"[CLEAN TEMP] Summary: {deleted} tiles removed.", "success")
    return {"deleted_tiles": deleted}


def publish_all(logger=None, force=False, ui=None):
    """Executa as duas etapas de publicacao em sequencia."""
    publish_mosaic_all(logger=logger, force=force, ui=ui)
    publish_vector_all(logger=logger, force=force, ui=ui)
