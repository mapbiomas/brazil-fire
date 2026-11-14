"""
U-Net experiment - authentication, Google Cloud Storage access and temp dirs.

Self-contained replacement for the catalog M0/M_gcs modules, so this folder
does not depend on any catalog version. It uses the same access scheme as the
catalog (version_04 M0):
  * Earth Engine with the person's own account (Colab pop-up / local ADC);
  * GCS with the shared service account (impersonation), falling back to the
    person's account if impersonation is not allowed.

Paths given to the GCS functions may be 'gs://bucket/a/b', 'bucket/a/b' or
'a/b' (relative to the bucket set by `connect`).
"""
import logging
import os
import warnings
from functools import lru_cache

warnings.filterwarnings("ignore", message="Your application has authenticated using end user credentials")
logging.getLogger("google.auth").setLevel(logging.ERROR)
logging.getLogger("gcsfs").setLevel(logging.CRITICAL)
logging.getLogger("fsspec").setLevel(logging.CRITICAL)

SERVICE_ACCOUNT_GROUP = "mapbiomas-net-fire@googlegroups.com"
SCOPES = [
    "https://www.googleapis.com/auth/earthengine",
    "https://www.googleapis.com/auth/devstorage.full_control",
    "https://www.googleapis.com/auth/cloud-platform",
]

# Filled by connect(); BUCKET can also be set directly (tests).
BUCKET = None
GEE_PROJECT = None
_CREDENTIALS = None


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def _user_credentials():
    try:
        from google.colab import auth  # noqa: F401
        print("[LOGIN] Login do Colab (janela pop-up)...")
        auth.authenticate_user()
    except ImportError:
        pass  # local machine: Application Default Credentials
    import google.auth
    credentials, _ = google.auth.default()
    return credentials


def connect(cfg, earth_engine=True):
    """Set the bucket from the config and authenticate GEE and GCS."""
    global BUCKET, GEE_PROJECT, _CREDENTIALS
    BUCKET = cfg["BUCKET"]
    GEE_PROJECT = cfg["GEE_PROJECT"]
    _storage_client.cache_clear()
    get_fs.cache_clear()

    source = _user_credentials()
    if earth_engine:
        import ee
        try:
            ee.Initialize(project=GEE_PROJECT)
            print(f"[GEE] Earth Engine iniciado com a sua conta | projeto: {GEE_PROJECT}")
        except Exception as exc:
            raise RuntimeError(
                f"Falha no Earth Engine (projeto {GEE_PROJECT}). Confira se a sua conta tem "
                f"acesso a esse projeto. Erro: {exc}") from exc

    sa = cfg.get("SERVICE_ACCOUNT")
    _CREDENTIALS = source
    if sa:
        try:
            from google.auth import impersonated_credentials
            from google.auth.transport.requests import Request
            creds = impersonated_credentials.Credentials(
                source_credentials=source, target_principal=sa, target_scopes=SCOPES)
            creds.refresh(Request())
            _CREDENTIALS = creds
            print(f"[GCS] Conta de serviço pronta | {sa}")
        except Exception as exc:
            if "iam.serviceAccounts.getAccessToken" in str(exc):
                print(f"[GCS][AVISO] Sua conta não pode usar {sa}. Entre no grupo {SERVICE_ACCOUNT_GROUP} "
                      f"ou peca o papel 'Storage Object User' em gs://{BUCKET}.")
            else:
                print(f"[GCS][AVISO] Falha na conta de serviço: {exc}")
            print("[GCS] Usando a sua própria conta para acessar o bucket.")

    ok = _storage_client().bucket(BUCKET).exists()
    print(f"[CHECK] GCS gs://{BUCKET}: {'OK' if ok else 'SEM ACESSO'}")
    if not ok:
        raise PermissionError(f"Sem acesso a gs://{BUCKET}.")


# ---------------------------------------------------------------------------
# GCS
# ---------------------------------------------------------------------------

def _require_bucket():
    if not BUCKET:
        raise RuntimeError("Rode antes unet_env.connect(UCFG) (célula 0.2).")
    return BUCKET


def gs_path(path):
    """'gs://bucket/...' for any accepted path form."""
    p = str(path)
    if p.startswith("gs://"):
        return p
    p = p.lstrip("/")
    bucket = _require_bucket()
    if p.startswith(bucket + "/"):
        return f"gs://{p}"
    return f"gs://{bucket}/{p}"


def _split(path):
    bucket, _, name = gs_path(path)[5:].partition("/")
    return bucket, name


@lru_cache(maxsize=1)
def _storage_client():
    from google.cloud import storage
    kwargs = {}
    if _CREDENTIALS is not None:
        kwargs["credentials"] = _CREDENTIALS
    if GEE_PROJECT:
        kwargs["project"] = GEE_PROJECT
    return storage.Client(**kwargs)


@lru_cache(maxsize=1)
def get_fs():
    """gcsfs file system (reads)."""
    import gcsfs
    if _CREDENTIALS is not None:
        return gcsfs.GCSFileSystem(token=_CREDENTIALS, requests_timeout=60)
    return gcsfs.GCSFileSystem(requests_timeout=60)


def blob_exists(path):
    bucket, name = _split(path)
    return _storage_client().bucket(bucket).blob(name).exists()


def ls(path):
    """Names (not paths) of the entries directly under a folder."""
    try:
        entries = get_fs().ls(gs_path(path)[5:].rstrip("/") + "/")
    except FileNotFoundError:
        return []
    return [e.rstrip("/").split("/")[-1] for e in entries]


def list_files(prefix, suffix=None):
    """gs:// paths of every file under a prefix."""
    try:
        files = get_fs().find(gs_path(prefix)[5:].rstrip("/") + "/")
    except FileNotFoundError:
        return []
    out = [f"gs://{f}" for f in files]
    return [f for f in out if not suffix or f.endswith(suffix)]


def open_file(path, mode="rb"):
    return get_fs().open(gs_path(path)[5:], mode)


def read_text(path):
    with open_file(path, "r") as f:
        return f.read()


def write_text(path, text):
    bucket, name = _split(path)
    _storage_client().bucket(bucket).blob(name).upload_from_string(text)


def upload(local_path, path):
    bucket, name = _split(path)
    _storage_client().bucket(bucket).blob(name).upload_from_filename(local_path)


def download(path, local_path):
    bucket, name = _split(path)
    _storage_client().bucket(bucket).blob(name).download_to_filename(local_path)


def rm_files(paths, batch_size=100):
    """Delete the given objects (missing ones are ignored)."""
    from collections import defaultdict
    by_bucket = defaultdict(list)
    for p in paths:
        if p:
            bucket, name = _split(p)
            by_bucket[bucket].append(name)
    client = _storage_client()
    for bucket, names in by_bucket.items():
        b = client.bucket(bucket)
        for i in range(0, len(names), batch_size):
            b.delete_blobs([b.blob(n) for n in names[i:i + batch_size]], on_error=lambda _b: None)


# ---------------------------------------------------------------------------
# Local temp folder
# ---------------------------------------------------------------------------

def temp_dir(*parts):
    """Local work folder (/content/TEMPORARY/unet in Colab)."""
    base = "/content/TEMPORARY" if "COLAB_RELEASE_TAG" in os.environ else os.path.abspath("TEMPORARY")
    path = os.path.join(base, "unet", *parts)
    os.makedirs(path, exist_ok=True)
    return path
