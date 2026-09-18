"""Layout de paths do mapbiomas_cog (clean break em relacao ao version_01).

Estrutura por unidade::

    {BUCKET}/{BUCKET_PATH}/{country}/{theme}/{collection}/{product}/
        temp/                     tiles intermediarios
        _manifest.json            estado/progresso da unidade
        _failures.json            journal de falhas (retry)
        {unit}.tif                COG final (produto)
"""

from __future__ import annotations

import os

BUCKET = "mapbiomas-fire"
PUBLIC_BUCKET = "mapbiomas-public"
BUCKET_PATH = "initiatives"

MANIFEST_NAME = "_manifest.json"
FAILURES_NAME = "_failures.json"
TILES_DIRNAME = "temp"


def root(country: str, theme: str, collection: str, product: str) -> str:
    return f"{BUCKET_PATH}/{country}/{theme}/{collection}/{product}"


def tiles_dir(root_path: str) -> str:
    return f"{root_path}/{TILES_DIRNAME}"


def cog_object(root_path: str, unit: str) -> str:
    return f"{root_path}/{unit}.tif"


def manifest_object(root_path: str) -> str:
    return f"{root_path}/{MANIFEST_NAME}"


def failures_object(root_path: str) -> str:
    return f"{root_path}/{FAILURES_NAME}"


def gcs_uri(bucket: str, path: str) -> str:
    return f"gs://{bucket}/{path}"


def public_uri(path: str) -> str:
    return gcs_uri(PUBLIC_BUCKET, path)


STATE_DIRNAME = ".mapbiomas_cog"


def local_state_path(root_path: str, state_dir: str | None = None) -> str:
    """Caminho local do manifest de um contexto (estado da sessao/VM)."""
    safe = root_path.replace("/", "__")
    return os.path.join(state_dir or STATE_DIRNAME, f"{safe}.json")


def local_failures_path(root_path: str, state_dir: str | None = None) -> str:
    """Caminho local do journal de falhas de um contexto."""
    safe = root_path.replace("/", "__")
    return os.path.join(state_dir or STATE_DIRNAME, f"{safe}_failures.json")
