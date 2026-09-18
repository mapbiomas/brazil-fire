"""Publicacao no bucket publico (M5).

Espelha o COG da unidade de ``mapbiomas-fire`` para ``mapbiomas-public`` com
checagem de tamanho (idempotente). Sem GDAL; apenas copia de objeto no GCS.
"""

from __future__ import annotations

from ..domain import paths


def _default_fs():
    import gcsfs

    return gcsfs.GCSFileSystem(token="google_default")


def publish_cog(root_path: str, unit: str, *, bucket: str | None = None,
                public_bucket: str | None = None, force: bool = False,
                fs=None, log=None) -> str:
    """Copia ``{bucket}/{root}/{unit}.tif`` -> ``{public}/{root}/{unit}.tif``.

    Retorna ``"copied"``, ``"exists"`` ou ``"error"``.
    """
    _log = log or (lambda _m: None)
    fs = fs or _default_fs()
    src = f"{bucket or paths.BUCKET}/{paths.cog_object(root_path, unit)}"
    dst = f"{public_bucket or paths.PUBLIC_BUCKET}/{paths.cog_object(root_path, unit)}"

    try:
        src_size = fs.info(src).get("size")
    except Exception as exc:  # noqa: BLE001
        _log(f"[ERROR] COG ausente na origem {src}: {exc}")
        return "error"

    if not force:
        try:
            if fs.info(dst).get("size") == src_size:
                _log(f"[SKIP] Ja publicado: {dst}")
                return "exists"
        except FileNotFoundError:
            pass
        except Exception:  # noqa: BLE001
            pass

    try:
        fs.copy(src, dst)
        if fs.info(dst).get("size") != src_size:
            _log(f"[ERROR] Tamanho divergente apos copia: {dst}")
            return "error"
    except Exception as exc:  # noqa: BLE001
        _log(f"[ERROR] Falha ao publicar {src} -> {dst}: {exc}")
        return "error"

    _log(f"[OK] Publicado: {dst}")
    return "copied"
