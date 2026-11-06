"""
M_gcs - Unified GCS Gateway
MapBiomas Fire Monitor

Single module for all Google Cloud Storage operations.
Reads use a gcsfs lazy singleton; writes use the google-cloud-storage client.
Both are bound to the impersonated service-account credentials from M0.
"""

import json
import logging
from functools import lru_cache

_log = logging.getLogger(__name__)


# ─── helpers ────────────────────────────────────────────────────────

def _gcs_path(path):
    """Normalize a path to 'gs://bucket/...' form.

    Accepts: 'gs://bucket/foo/bar', 'bucket/foo/bar', '/foo/bar'.
    Always returns 'gs://bucket/...' using CONFIG['bucket'].
    """
    from M0_auth_config import CONFIG
    p = str(path)
    if p.startswith('gs://'):
        return p
    p = p.lstrip('/')
    bucket = CONFIG['bucket']
    if p.startswith(bucket + '/'):
        return f'gs://{p}'
    return f'gs://{bucket}/{p}'


def _gcs_rel(path):
    """Return the relative path (bucket/foo/bar) from any gs:// form."""
    p = _gcs_path(path)
    return p[5:]  # strip 'gs://'


def _split(path):
    """Split a GCS path into (bucket, object_name)."""
    rel = _gcs_rel(path)
    bucket, _, name = rel.partition('/')
    return bucket, name


# ─── storage client (writes, lazy) ─────────────────────────────────

@lru_cache(maxsize=1)
def _storage_client():
    """Return a google-cloud-storage client bound to the SA credentials."""
    from google.cloud import storage
    from M0_auth_config import _GCS_CREDENTIALS, CONFIG
    kwargs = {}
    if _GCS_CREDENTIALS is not None:
        kwargs['credentials'] = _GCS_CREDENTIALS
    project = (CONFIG or {}).get('gee_project')
    if project:
        kwargs['project'] = project
    return storage.Client(**kwargs)


# ─── fs (read-only, lazy) ───────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_fs():
    """Return lazy gcsfs instance for read-only operations."""
    import gcsfs
    try:
        from M0_auth_config import _GCS_CREDENTIALS, CONFIG
    except ImportError:
        from M0_auth_config import CONFIG
        _GCS_CREDENTIALS = None
    if _GCS_CREDENTIALS is not None:
        return gcsfs.GCSFileSystem(
            token=_GCS_CREDENTIALS, requests_timeout=60
        )
    return gcsfs.GCSFileSystem(requests_timeout=60)


# ─── public API ─────────────────────────────────────────────────────


def authenticate():
    """Ensure the GCS credentials are valid (surfaces actAs errors early)."""
    from M0_auth_config import _GCS_CREDENTIALS
    creds = _GCS_CREDENTIALS
    if creds is not None and not getattr(creds, 'valid', False):
        from google.auth.transport.requests import Request
        creds.refresh(Request())
    _log.debug('GCS authentication ready.')


def blob_exists(path):
    """Check GCS object existence (google-cloud-storage, no listing cache)."""
    bucket, name = _split(path)
    return _storage_client().bucket(bucket).blob(name).exists()


def exists(path):
    """Check if a GCS path exists."""
    return _get_fs().exists(_gcs_path(path))


def list_files(prefix, suffix=None):
    """List files under prefix. Returns list of gs:// paths.

    If suffix is provided (e.g. '.tif'), only files ending with it
    are returned.
    """
    fs = _get_fs()
    clean = _gcs_rel(prefix).rstrip('/') + '/'
    try:
        files = fs.find(clean)
    except FileNotFoundError:
        return []
    out = [f'gs://{f}' for f in files]
    if suffix:
        out = [f for f in out if f.endswith(suffix)]
    return out


def ls(path):
    """List immediate children of a GCS directory. Returns list of gs:// paths."""
    fs = _get_fs()
    clean = _gcs_rel(path).rstrip('/') + '/'
    try:
        entries = fs.ls(clean)
    except FileNotFoundError:
        return []
    return [f'gs://{e}' for e in entries]


def glob(pattern):
    """Match files against a glob pattern on GCS. Returns list of gs:// paths.

    Similar to fs.glob().  Pattern is a relative path with optional
    wildcards (e.g. 'bucket/foo/bar_*.tif').
    """
    fs = _get_fs()
    rel = _gcs_rel(pattern)
    try:
        files = fs.glob(rel)
    except (FileNotFoundError, NotADirectoryError):
        return []
    return [f'gs://{f}' for f in files]


def read_bytes(path):
    """Read raw bytes from GCS."""
    fs = _get_fs()
    with fs.open(_gcs_path(path), 'rb') as f:
        return f.read()


def read_text(path):
    """Read text from GCS."""
    fs = _get_fs()
    with fs.open(_gcs_path(path), 'r') as f:
        return f.read()


def read_json(path):
    """Read and parse a JSON file from GCS."""
    return json.loads(read_text(path))


def download(remote, local):
    """Download a file from GCS to local filesystem.

    Replaces fs.get().
    """
    bucket, name = _split(remote)
    _storage_client().bucket(bucket).blob(name).download_to_filename(local)


def upload(local_path, remote_path):
    """Upload a local file to GCS.

    Replaces fs.put().
    """
    bucket, name = _split(remote_path)
    _storage_client().bucket(bucket).blob(name).upload_from_filename(local_path)


def write_text(path, text):
    """Write text content to a GCS file."""
    bucket, name = _split(path)
    _storage_client().bucket(bucket).blob(name).upload_from_string(text)


def write_json(path, data):
    """Write a JSON-serializable object to a GCS file."""
    write_text(path, json.dumps(data, indent=2, default=str))


def rm(path, recursive=False):
    """Remove a single file or directory from GCS.

    Replaces fs.rm().
    """
    client = _storage_client()
    bucket, name = _split(path)
    b = client.bucket(bucket)
    if recursive:
        prefix = name.rstrip('/') + '/'
        blobs = list(b.list_blobs(prefix=prefix))
        if blobs:
            b.delete_blobs(blobs, on_error=lambda _blob: None)
    else:
        b.blob(name).delete()


def rm_files(paths, batch_size=100):
    """Remove several GCS objects using batched delete_blobs calls.

    Callers pass the exact object list; missing objects are ignored.
    Batching keeps each request under a manageable size.
    """
    from collections import defaultdict
    client = _storage_client()
    by_bucket = defaultdict(list)
    for path in paths:
        if not path:
            continue
        bucket, name = _split(path)
        by_bucket[bucket].append(name)

    for bucket, names in by_bucket.items():
        b = client.bucket(bucket)
        for start in range(0, len(names), batch_size):
            chunk = names[start:start + batch_size]
            b.delete_blobs([b.blob(n) for n in chunk], on_error=lambda _blob: None)


def mkdir(path):
    """Create a directory placeholder in GCS.

    GCS doesn't have real directories; a zero-byte marker object is created.
    Equivalent to fs.mkdir().
    """
    bucket, name = _split(path)
    name = name.rstrip('/') + '/'
    _storage_client().bucket(bucket).blob(name).upload_from_string('')


def copy(src, dest):
    """Copy a file within GCS. Replaces fs.copy()."""
    client = _storage_client()
    src_bucket, src_name = _split(src)
    dst_bucket, dst_name = _split(dest)
    client.bucket(dst_bucket).copy_blob(
        client.bucket(src_bucket).blob(src_name),
        client.bucket(dst_bucket),
        dst_name,
    )


# ─── module-level convenience for step-wise migration ───────────────

class GCS:
    """Wrapper that allows M_gcs.method() calls during transition."""
    authenticate = staticmethod(authenticate)
    blob_exists = staticmethod(blob_exists)
    exists = staticmethod(exists)
    list_files = staticmethod(list_files)
    ls = staticmethod(ls)
    glob = staticmethod(glob)
    read_bytes = staticmethod(read_bytes)
    read_text = staticmethod(read_text)
    read_json = staticmethod(read_json)
    download = staticmethod(download)
    upload = staticmethod(upload)
    write_text = staticmethod(write_text)
    write_json = staticmethod(write_json)
    rm = staticmethod(rm)
    rm_files = staticmethod(rm_files)
    mkdir = staticmethod(mkdir)
    copy = staticmethod(copy)
