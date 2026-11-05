"""Fetch and parse the live GEE blocklist modules used by the mosaic exports."""

import os
import re
import shutil
import subprocess
import tempfile


ROOT_REQUIRE = 'users/mapbiomasworkspace1/mapbiomas-fire:00_Tools/require-landsat-collection'

# Live sources follow the Monitor Fuego toolkit pattern. Every source is
# optional: an unavailable list must never stop an export.
BLOCKLIST_SOURCES = {
    'landsat': (
        'users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList',
        'users/mapbiomasworkspace1/mapbiomas-fire:00_Tools/require-blockList',
    ),
    'sentinel2': (
        'users/geomapeamentoipam/MapBiomas__Fogo:00_Tools/module-blockList',
        'users/mapbiomasworkspace1/mapbiomas-fire:00_Tools/module-blockList',
    ),
    'hls': (),
    'hls_l30': (),
    'hls_s30': (),
    'modis': (),
    'modis_terra': (),
    'modis_aqua': (),
    'viirs': (),
}
SUPPORTED_SENSORS = tuple(BLOCKLIST_SOURCES)

# Some GEE modules use a different export name than the pipeline sensor key.
BLOCKLIST_EXPORT_NAMES = {
    'sentinel2': 'sentinel',
}

BLOCKLIST_ALIASES = {
    'sentinel': ('sentinel2',),
    'hls': ('hls', 'hls_l30', 'hls_s30'),
    'hls_l30': ('hls_l30', 'hls'),
    'hls_s30': ('hls_s30', 'hls'),
    'modis': ('modis', 'modis_terra', 'modis_aqua'),
    'modis_terra': ('modis_terra', 'modis'),
    'modis_aqua': ('modis_aqua', 'modis'),
}

_BLOCKLISTS = None
_BLOCKLIST_META = None


class BlocklistSourceError(RuntimeError):
    """Raised when current blocklists cannot be fetched or parsed safely."""


def _dedupe(values):
    """Preserve source order while removing empty and duplicate entries."""
    return list(dict.fromkeys(value for value in values if value))


def _alias_chain(sensor):
    sensor = str(sensor).lower()
    if sensor in BLOCKLIST_ALIASES:
        return BLOCKLIST_ALIASES[sensor]
    if sensor in BLOCKLIST_SOURCES:
        return (sensor,)
    return ()


def _source_assets(sensor):
    assets = []
    for key in _alias_chain(sensor):
        for asset in BLOCKLIST_SOURCES.get(key, ()):
            if asset not in assets:
                assets.append(asset)
    return assets


def _normalize_sensors(sensors):
    if sensors is None:
        return list(BLOCKLIST_SOURCES)
    if isinstance(sensors, str):
        sensors = [sensors]
    return [str(sensor).lower() for sensor in sensors]


GEE_GIT_URL = 'https://earthengine.googlesource.com'


def _access_token():
    """Return a fresh OAuth token for GEE git, or None when unavailable."""
    try:
        from M0_auth_config import _GCS_CREDENTIALS
        credentials = _GCS_CREDENTIALS
        if credentials is None:
            import google.auth
            credentials, _ = google.auth.default(
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        if not getattr(credentials, 'valid', False):
            from google.auth.transport.requests import Request
            credentials.refresh(Request())
        return getattr(credentials, 'token', None)
    except Exception:
        return None


def _git_environment(token=None):
    """Build a non-interactive git environment for GEE sources.

    A stale ``extraheader`` inherited from the notebook is reset first. When a
    fresh OAuth token is available it is added back; otherwise git keeps using
    gitcookies/credential helpers. This is what makes the fetch work in Colab,
    where no gitcookies exist, and locally, where they usually do.
    """
    if token is None:
        token = _access_token()

    env = os.environ.copy()
    env['GIT_TERMINAL_PROMPT'] = '0'
    configs = [(f'http.{GEE_GIT_URL}/.extraheader', '')]
    if token:
        configs.append((
            f'http.{GEE_GIT_URL}/.extraheader',
            f'Authorization: Bearer {token}',
        ))

    config_index = int(env.get('GIT_CONFIG_COUNT', '0'))
    for key, value in configs:
        env[f'GIT_CONFIG_KEY_{config_index}'] = key
        env[f'GIT_CONFIG_VALUE_{config_index}'] = value
        config_index += 1
    env['GIT_CONFIG_COUNT'] = str(config_index)
    return env


def _strip_js_comments(source):
    """Remove JavaScript comments without changing quoted strings."""
    result = []
    i = 0
    quote = None
    while i < len(source):
        char = source[i]
        if quote:
            result.append(char)
            if char == '\\' and i + 1 < len(source):
                i += 1
                result.append(source[i])
            elif char == quote:
                quote = None
            i += 1
            continue

        if char in ('\'', '"', '`'):
            quote = char
            result.append(char)
            i += 1
        elif source.startswith('//', i):
            while i < len(source) and source[i] not in '\r\n':
                result.append(' ')
                i += 1
        elif source.startswith('/*', i):
            result.extend((' ', ' '))
            i += 2
            while i < len(source) and not source.startswith('*/', i):
                result.append('\n' if source[i] == '\n' else ' ')
                i += 1
            if i < len(source):
                result.extend((' ', ' '))
                i += 2
        else:
            result.append(char)
            i += 1
    return ''.join(result)


def _require_calls(source):
    clean = _strip_js_comments(source)
    pattern = re.compile(
        r"\brequire\s*\(\s*(['\"])(.*?)\1\s*\)\s*\.\s*([A-Za-z_$][\w$]*)\s*\("
    )
    return [(match.group(2), match.group(3)) for match in pattern.finditer(clean)]


def _matching_bracket(source, start, opening='[', closing=']'):
    depth = 0
    quote = None
    i = start
    while i < len(source):
        char = source[i]
        if quote:
            if char == '\\':
                i += 2
                continue
            if char == quote:
                quote = None
        elif char in ('\'', '"', '`'):
            quote = char
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise BlocklistSourceError(f"Unclosed JavaScript {opening}{closing} list.")


def _string_values(source):
    values = []
    i = 0
    while i < len(source):
        if source[i] not in ('\'', '"'):
            i += 1
            continue
        quote = source[i]
        i += 1
        chars = []
        while i < len(source) and source[i] != quote:
            if source[i] == '\\' and i + 1 < len(source):
                i += 1
            chars.append(source[i])
            i += 1
        if i >= len(source):
            raise BlocklistSourceError("Unclosed JavaScript string in blocklist.")
        values.append(''.join(chars))
        i += 1
    return values


def _exported_list(source, sensor):
    clean = _strip_js_comments(source)
    export = re.search(
        rf"\bexports\s*\.\s*{re.escape(sensor)}\s*=\s*function\s*\([^)]*\)\s*\{{",
        clean,
    )
    if not export:
        return None

    returned = re.search(r"\breturn\b", clean[export.end():])
    if not returned:
        return None
    return_start = export.end() + returned.end()
    list_start = clean.find('[', return_start)
    if list_start < 0:
        return None
    list_end = _matching_bracket(clean, list_start)
    return _string_values(clean[list_start + 1:list_end])


def _parse_asset_id(asset_id):
    match = re.fullmatch(r'users/([^/]+)/([^:]+):(.+)', asset_id)
    if not match:
        raise BlocklistSourceError(f"Unsupported GEE require path: {asset_id}")
    owner, repository, file_path = match.groups()
    remote = f"https://earthengine.googlesource.com/users/{owner}/{repository}"
    return remote, file_path


class _SparseGeeSources:
    """Read only the JavaScript files reachable from a GEE require path."""

    def __init__(self, temp_root):
        self.temp_root = temp_root
        self.repositories = {}
        self.commits = {}
        token = _access_token()
        self.git_env = _git_environment(token=token)
        self.fallback_env = _git_environment(token='') if token else None

    def _alternate_env(self, current):
        if self.fallback_env is None:
            return None
        if current is self.git_env:
            return self.fallback_env
        return self.git_env

    def _clone(self, remote, local_path):
        last_error = None
        for env in (self.git_env, self.fallback_env):
            if env is None:
                continue
            shutil.rmtree(local_path, ignore_errors=True)
            try:
                subprocess.run(
                    ['git', 'clone', '--depth', '1', '--filter=blob:none', '--sparse', remote, local_path],
                    check=True, capture_output=True, text=True, env=env,
                )
                return env
            except (OSError, subprocess.CalledProcessError) as error:
                last_error = error

        detail = (getattr(last_error, 'stderr', None) or str(last_error)).strip()
        raise BlocklistSourceError(
            f"Could not fetch GEE source {remote}: {detail}. "
            "Run the M0 authentication cell or configure GEE git credentials at "
            "https://earthengine.googlesource.com/new-password."
        ) from last_error

    def read(self, asset_id):
        remote, file_path = _parse_asset_id(asset_id)
        repository = self.repositories.get(remote)
        if repository is None:
            local_path = os.path.join(self.temp_root, f"source_{len(self.repositories)}")
            env = self._clone(remote, local_path)
            repository = {'path': local_path, 'files': set(), 'env': env}
            self.repositories[remote] = repository
            try:
                resolved = subprocess.run(
                    ['git', '-C', local_path, 'rev-parse', 'HEAD'],
                    check=True, capture_output=True, text=True, env=env,
                )
                self.commits[remote] = resolved.stdout.strip()
            except (OSError, subprocess.CalledProcessError):
                self.commits[remote] = None

        normalized_path = file_path.replace('\\', '/')
        if normalized_path not in repository['files']:
            # Only persist the path after a successful checkout. A missing
            # optional source must not poison the next sparse-checkout call.
            attempted = set(repository['files']) | {normalized_path}
            sparse_paths = [f"/{path}" for path in sorted(attempted)]
            env = repository.get('env', self.git_env)
            try:
                subprocess.run(
                    ['git', '-C', repository['path'], 'sparse-checkout', 'set', '--no-cone', *sparse_paths],
                    check=True, capture_output=True, text=True, env=env,
                )
            except subprocess.CalledProcessError as error:
                fallback = self._alternate_env(env)
                if fallback is None:
                    detail = (error.stderr or str(error)).strip()
                    raise BlocklistSourceError(f"Could not select GEE source file {file_path}: {detail}") from error
                try:
                    subprocess.run(
                        ['git', '-C', repository['path'], 'sparse-checkout', 'set', '--no-cone', *sparse_paths],
                        check=True, capture_output=True, text=True, env=fallback,
                    )
                except subprocess.CalledProcessError:
                    detail = (error.stderr or str(error)).strip()
                    raise BlocklistSourceError(f"Could not select GEE source file {file_path}: {detail}") from error
                repository['env'] = fallback
            repository['files'].add(normalized_path)

        full_path = os.path.join(repository['path'], *normalized_path.split('/'))
        try:
            with open(full_path, 'r', encoding='utf-8') as source_file:
                return source_file.read()
        except OSError as error:
            raise BlocklistSourceError(f"GEE source file is unavailable: {asset_id}") from error

    def commit_for(self, asset_id):
        try:
            remote, _ = _parse_asset_id(asset_id)
        except BlocklistSourceError:
            return None
        return self.commits.get(remote)


def extract_blocklists(read_source, root_require=ROOT_REQUIRE, root_sensor=None):
    """Resolve require() calls recursively and return distinct exported lists."""
    collected = {}
    visited = set()
    queue = [(root_require, root_sensor)]

    while queue:
        asset_id, sensor = queue.pop(0)
        key = (asset_id, sensor)
        if key in visited:
            continue
        visited.add(key)
        if len(visited) > 128:
            raise BlocklistSourceError("Too many nested require() modules while reading blocklists.")

        source = read_source(asset_id)
        if sensor:
            values = _exported_list(source, sensor)
            if values is not None:
                collected.setdefault(sensor.lower(), []).extend(values)
        queue.extend(_require_calls(source))

    result = {}
    for sensor, values in collected.items():
        # Preserve source order while matching Earth Engine's distinct() behavior.
        result[sensor] = _dedupe(values)
    return result


def _refresh_with(read_source, sensors):
    global _BLOCKLISTS, _BLOCKLIST_META

    reader = getattr(read_source, 'read', read_source)
    found = {}
    meta = {}
    for sensor in sensors:
        values = []
        source_meta = []
        seen = set()

        for key in _alias_chain(sensor):
            export_name = BLOCKLIST_EXPORT_NAMES.get(key, key)
            for asset in BLOCKLIST_SOURCES.get(key, ()):
                if (asset, export_name) in seen:
                    continue
                seen.add((asset, export_name))

                entry = {'asset': asset, 'export': export_name, 'status': 'ok', 'count': 0}
                try:
                    extracted = extract_blocklists(
                        reader, root_require=asset, root_sensor=export_name,
                    )
                    entries = extracted.get(export_name, [])
                    if not entries and export_name != key:
                        entries = extracted.get(key, [])
                    entry['count'] = len(entries)
                    values.extend(entries)
                except Exception as error:
                    entry['status'] = 'error'
                    entry['error'] = str(error)

                commit_for = getattr(read_source, 'commit_for', None)
                if commit_for:
                    try:
                        entry['commit'] = commit_for(asset)
                    except Exception:
                        pass
                source_meta.append(entry)

        values = _dedupe(values)
        found[sensor] = values
        meta[sensor] = {
            'count': len(values),
            'sample': values[:3],
            'sources': source_meta,
        }

    if _BLOCKLISTS is None:
        _BLOCKLISTS = {}
    _BLOCKLISTS.update(found)
    _BLOCKLIST_META = meta if _BLOCKLIST_META is None else {**_BLOCKLIST_META, **meta}
    return {sensor: list(values) for sensor, values in found.items()}


def refresh_blocklists(sensors=None, read_source=None):
    """Fetch available GEE blocklists. Missing sources are reported, not raised."""
    requested = _normalize_sensors(sensors)
    if read_source is not None:
        return _refresh_with(read_source, requested)

    with tempfile.TemporaryDirectory(prefix='mapbiomas_blocklists_') as temp_root:
        return _refresh_with(_SparseGeeSources(temp_root), requested)


def blocklist_report(sensors=None):
    """Return per-sensor source status, counts and samples for verification."""
    requested = _normalize_sensors(sensors)
    missing = [sensor for sensor in requested if not _BLOCKLIST_META or sensor not in _BLOCKLIST_META]
    if missing:
        refresh_blocklists(missing)
    return {
        sensor: dict((_BLOCKLIST_META or {}).get(sensor, {'count': 0, 'sample': [], 'sources': []}))
        for sensor in requested
    }


def get_blocklist(sensor):
    """Return the current blocklist for a sensor, fetching live sources if needed."""
    normalized_sensor = str(sensor).lower()
    if not _alias_chain(normalized_sensor) or not _source_assets(normalized_sensor):
        return []
    if not _BLOCKLISTS or normalized_sensor not in _BLOCKLISTS:
        refresh_blocklists([normalized_sensor])

    values = []
    for key in _alias_chain(normalized_sensor):
        values.extend((_BLOCKLISTS or {}).get(key, []))
    return _dedupe(values)
