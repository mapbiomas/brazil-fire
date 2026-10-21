"""
M_gcs - Unified GCS Gateway
MapBiomas Fire Monitor

Single module for all Google Cloud Storage operations.
Writes use gsutil subprocess (stable, avoids gcsfs 2026.2.0 bugs).
Reads use gcsfs lazy singleton for convenience.
"""

import os
import json
import subprocess
import logging
from functools import lru_cache

_log = logging.getLogger(__name__)

_GSUTIL = 'gsutil.cmd' if os.name == 'nt' else 'gsutil'


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


def _call(args, check=True, capture_output=True):
    """Run a gsutil command. Returns subprocess.CompletedProcess."""
    cmd = [_GSUTIL] + args
    _log.debug('Running: %s', ' '.join(cmd))
    return subprocess.run(
        cmd, check=check, capture_output=capture_output, text=True
    )


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
    """Ensure gsutil is authenticated via ADC."""
    _call(['version'], check=False)  # quick smoke test


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
    """Download a file from GCS to local filesystem via gsutil.

    Replaces fs.get().
    """
    _call(['cp', _gcs_path(remote), local])


def upload(local_path, remote_path):
    """Upload a local file to GCS via gsutil.

    Replaces fs.put().
    """
    _call(['cp', local_path, _gcs_path(remote_path)])


def write_text(path, text):
    """Write text content to a GCS file via gsutil (temp local file)."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.tmp', delete=False) as f:
        f.write(text)
        tmp = f.name
    try:
        upload(tmp, path)
    finally:
        os.unlink(tmp)


def write_json(path, data):
    """Write a JSON-serializable object to a GCS file via gsutil."""
    write_text(path, json.dumps(data, indent=2, default=str))


def rm(path, recursive=False):
    """Remove a single file or directory from GCS.

    Replaces fs.rm().
    """
    args = ['rm']
    if recursive:
        args.append('-r')
    args.append(_gcs_path(path))
    _call(args)


def mkdir(path):
    """Create a directory placeholder in GCS.

    GCS doesn't have real directories, but gsutil cp of an empty
    string to the path creates a zero-byte object as a marker.
    Equivalent to fs.mkdir().
    """
    rel = _gcs_rel(path).rstrip('/') + '/'
    if not rel.endswith('/'):
        rel += '/'
    try:
        _ = _call(['ls', f'gs://{rel}'], check=False, capture_output=True)
        if _.returncode == 0:
            return  # already exists
    except Exception:
        pass
    _call(['cp', '/dev/null' if os.name != 'nt' else 'NUL',
           _gcs_path(path).rstrip('/') + '/'])


def copy(src, dest):
    """Copy a file within GCS. Replaces fs.copy()."""
    _call(['cp', _gcs_path(src), _gcs_path(dest)])


# ─── module-level convenience for step-wise migration ───────────────

class GCS:
    """Wrapper that allows M_gcs.method() calls during transition."""
    authenticate = staticmethod(authenticate)
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
    mkdir = staticmethod(mkdir)
    copy = staticmethod(copy)
