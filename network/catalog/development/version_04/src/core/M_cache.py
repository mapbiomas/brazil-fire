"""
M_cache — Cache Manager
MapBiomas Fire Sentinel Monitor

Gerencia cache de estado de exports no GCS para carregamento rapido da interface.
"""

import os
import json
import datetime
import threading
from M0_auth_config import CONFIG, _get_fs
from M_periods import ENABLED_PERIODICITIES, FOLDER_TO_PERIODICITY, parse_period, period_state_key

class CacheManager:
    LIBRARY_CACHE_FILE = "library.json"
    CAMPAIGN_CACHE_FILE = "campaign.json"
    LEGACY_CACHE_FILE = "state.json"
    _state = None
    _lock = threading.RLock() # Lock reentrante para evitar race conditions no mesmo kernel

    @staticmethod
    def _library_scope():
        """Scope of the shared image library cache (no campaign)."""
        from M0_auth_config import CONFIG
        if not CONFIG:
            return None
        return {
            '_cache_format_version': 4,
            'country': CONFIG.get('country'),
            'bucket': CONFIG.get('bucket'),
            'gcs_library_images_prefix': CONFIG.get('gcs_library_images_prefix'),
            'gcs_campaigns_prefix': CONFIG.get('gcs_campaigns_prefix'),
            'gee_library_images_prefix': CONFIG.get('gee_library_images_prefix'),
            'gee_campaigns_prefix': CONFIG.get('gee_campaigns_prefix'),
        }

    @staticmethod
    def _campaign_scope():
        """Scope of the campaign cache (models/samples/classifications)."""
        from M0_auth_config import CONFIG
        if not CONFIG:
            return None
        return {
            '_cache_format_version': 4,
            'country': CONFIG.get('country'),
            'bucket': CONFIG.get('bucket'),
            'campaign': CONFIG.get('campaign'),
            'gcs_campaigns_prefix': CONFIG.get('gcs_campaigns_prefix'),
        }

    @staticmethod
    def _local_candidates(filename):
        return [filename, os.path.join("..", filename), os.path.join("..", "..", filename)]

    @staticmethod
    def _read_local_cache(filename, scope):
        for path in CacheManager._local_candidates(filename):
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
            except Exception:
                continue
            if scope is None or data.get('_cache_scope') == scope:
                return data
        return None

    @staticmethod
    def _read_gcs_cache(gcs_path, scope):
        try:
            fs = _get_fs()
            if not fs.exists(gcs_path):
                return None
            with fs.open(gcs_path, 'r') as f:
                data = json.load(f)
            if scope is None or data.get('_cache_scope') == scope:
                return data
        except Exception:
            pass
        return None

    @staticmethod
    def _write_local_cache(filename, data):
        try:
            with open(filename, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Local cache write failed ({filename}): {e}")

    @staticmethod
    def _apply_library(state, data):
        for key in ('assets_monthly', 'assets_annually', 'gcs_chunks',
                    'cogs_monthly', 'cogs_annually'):
            if key in data:
                state[key] = data[key]
        if 'updated_at' in data:
            state['updated_at'] = data['updated_at']

    @staticmethod
    def _apply_campaign(state, data):
        for key in ('trained_models', 'sample_collections', 'classified_groups'):
            if key in data:
                state[key] = data[key]
        if 'updated_at' in data:
            state['campaign_updated_at'] = data['updated_at']

    @staticmethod
    def _get_library_gcs_path():
        cache = (CONFIG.get('gcs_library_cache')
                 or f"{CONFIG.get('gcs_library_images_prefix')}/LIBRARY_IMAGES/.CACHE")
        return f"gs://{CONFIG['bucket']}/{cache}/{CacheManager.LIBRARY_CACHE_FILE}"

    @staticmethod
    def _get_campaign_gcs_path():
        cache = (CONFIG.get('gcs_campaign_cache')
                 or f"{CONFIG.get('gcs_campaigns_prefix')}/{CONFIG.get('campaign', '')}/.CACHE")
        return f"gs://{CONFIG['bucket']}/{cache}/{CacheManager.CAMPAIGN_CACHE_FILE}"

    @staticmethod
    def _get_legacy_gcs_path():
        cache = (CONFIG.get('gcs_cache')
                 or f"{CONFIG.get('gcs_campaigns_prefix')}/.CACHE")
        return f"gs://{CONFIG['bucket']}/{cache}/{CacheManager.LEGACY_CACHE_FILE}"

    @staticmethod
    def clear():
        """Delete both local and GCS caches and reset memory."""
        with CacheManager._lock:
            CacheManager._state = {}
            for filename in (CacheManager.LIBRARY_CACHE_FILE,
                             CacheManager.CAMPAIGN_CACHE_FILE,
                             CacheManager.LEGACY_CACHE_FILE):
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                    except Exception as e:
                        print(f"Error removing local cache {filename}: {e}")
            try:
                from M_gcs import exists, rm
                for path in (CacheManager._get_library_gcs_path(),
                             CacheManager._get_campaign_gcs_path(),
                             CacheManager._get_legacy_gcs_path()):
                    if exists(path):
                        rm(path)
            except Exception as e:
                print(f"Error clearing GCS cache: {e}")

    @staticmethod
    def load(force=False):
        """Load library + campaign caches (local first, then GCS)."""
        library_scope = CacheManager._library_scope()
        campaign_scope = CacheManager._campaign_scope()
        if CacheManager._state is not None and not force:
            if (CacheManager._state.get('_cache_scope') == library_scope
                    and CacheManager._state.get('_campaign_scope') == campaign_scope):
                return CacheManager._state
            CacheManager._state = None

        with CacheManager._lock:
            CacheManager._state = {
                "updated_at": None,
                "campaign_updated_at": None,
                "country": (CONFIG or {}).get('country'),
                "_cache_scope": library_scope,
                "_campaign_scope": campaign_scope,
                "assets_monthly": [],
                "assets_annually": [],
                "gcs_chunks": {},
                "cogs_monthly": [],
                "cogs_annually": [],
                "trained_models": [],
                "sample_collections": [],
                "classified_groups": [],
            }

            library_local = CacheManager._read_local_cache(
                CacheManager.LIBRARY_CACHE_FILE, library_scope)
            campaign_local = CacheManager._read_local_cache(
                CacheManager.CAMPAIGN_CACHE_FILE, campaign_scope)
            if library_local:
                CacheManager._apply_library(CacheManager._state, library_local)
            if campaign_local:
                CacheManager._apply_campaign(CacheManager._state, campaign_local)

            if library_local is None:
                data = CacheManager._read_gcs_cache(
                    CacheManager._get_library_gcs_path(), library_scope)
                if data:
                    CacheManager._apply_library(CacheManager._state, data)
                    CacheManager._write_local_cache(
                        CacheManager.LIBRARY_CACHE_FILE, data)
            if campaign_local is None:
                data = CacheManager._read_gcs_cache(
                    CacheManager._get_campaign_gcs_path(), campaign_scope)
                if data:
                    CacheManager._apply_campaign(CacheManager._state, data)
                    CacheManager._write_local_cache(
                        CacheManager.CAMPAIGN_CACHE_FILE, data)

            # One-time migration from the old single state.json.
            if library_local is None and campaign_local is None:
                legacy = (CacheManager._read_local_cache(
                              CacheManager.LEGACY_CACHE_FILE, None)
                          or CacheManager._read_gcs_cache(
                              CacheManager._get_legacy_gcs_path(), None))
                if legacy:
                    CacheManager._apply_library(CacheManager._state, legacy)
                    CacheManager._apply_campaign(CacheManager._state, legacy)
                    CacheManager.save(CacheManager._state)
        return CacheManager._state

    @staticmethod
    def build_cache_from_gee(logger=None):
        """Popula cache listando assets do GEE em paralelo para performance."""
        import ee, time
        from concurrent.futures import ThreadPoolExecutor
        from M0_auth_config import CONFIG, get_asset_mosaic_collection
        
        start_time = time.time()
        state = CacheManager.get_state()
        for period_type in ENABLED_PERIODICITIES:
            state[period_state_key('assets', period_type)] = []
        
        bands = CONFIG.get('bands_all', ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'dayOfYear'])
        # Todos os métodos de mosaico disponíveis para este país (SEMPRE escaneia tudo)
        from M_mosaics import all_methods as _all_mosaic_methods
        mosaic_methods = _all_mosaic_methods()
        # CORRIGIDO: Escanear TODOS os sensores relevantes, não apenas o GLOBAL_OPTS atual
        # (o sensor padrão pode ser 'landsat' mas os dados podem ser de 'sentinel2')
        all_sensors = ['sentinel2', 'landsat', 'hls', 'modis']
        tasks = []
        
        for sensor_name in all_sensors:
            for period_type in ENABLED_PERIODICITIES:
                for mosaic_m in mosaic_methods:
                    col_id = get_asset_mosaic_collection(
                        sensor=sensor_name,
                        periodicity=period_type,
                        mosaic=mosaic_m
                    )
                    tasks.append((col_id, period_type))
        
        total_steps = len(tasks)
        completed = 0
        
        if logger: logger("Iniciando sincronización GEE (Paralelo)...", "info")
        
        def fetch_assets(task_info):
            col_id, period_type = task_info
            assets_found = []
            try:
                # 1. Lista sub-coleções de bandas (ex: .../MINNBR/blue)
                band_paths = []
                page_token = None
                while True:
                    params = {'parent': col_id}
                    if page_token: params['pageToken'] = page_token
                    result = ee.data.listAssets(params)
                    for a in result.get('assets', []):
                        band_paths.append(a['name'])
                    page_token = result.get('nextPageToken')
                    if not page_token: break

                # 2. Dentro de cada banda, lista imagens reais
                for band_path in band_paths:
                    inner_token = None
                    while True:
                        params = {'parent': band_path}
                        if inner_token: params['pageToken'] = inner_token
                        result = ee.data.listAssets(params)
                        for a in result.get('assets', []):
                            asset_name = a['name'].split('/')[-1].lower()
                            if asset_name:
                                assets_found.append((period_type, asset_name))
                        inner_token = result.get('nextPageToken')
                        if not inner_token: break
                return assets_found
            except Exception as e:
                return []

        # Executar em paralelo (máximo 8 threads para não sobrecarregar quota GEE)
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(fetch_assets, t) for t in tasks]
            for future in futures:
                res = future.result()
                for period_type, asset_name in res:
                    target_key = period_state_key('assets', period_type)
                    state.setdefault(target_key, [])
                    name_lower = asset_name.lower()
                    if name_lower not in state[target_key]:
                        state[target_key].append(name_lower)
                
                completed += 1
                if logger:
                    elapsed = time.time() - start_time
                    # Estimativa mais realista para paralelo
                    remaining = (elapsed / completed) * (total_steps - completed) if completed > 0 else 0
                    logger(f"Estimated remaining: {int(remaining):02d}s || [{completed}/{total_steps}]")
        
        CacheManager._state = state
        CacheManager.save()
        return CacheManager._state

    @staticmethod
    def build_cache_from_gcs(logger=None, years=None):
        """Escaneia o GCS para detectar chunks e COGs existentes."""
        from M0_auth_config import CONFIG

        bands_all = CONFIG.get('bands_all', ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'dayOfYear'])
        sorted_bands = sorted(bands_all, key=len, reverse=True)

        state = CacheManager.get_state()
        gcs_chunks = {}
        cogs = {periodicity: [] for periodicity in ENABLED_PERIODICITIES}

        try:
            fs = _get_fs()
            
            bucket = CONFIG['bucket']
            
            # CORRIGIDO: Escanear a raiz de TODOS os sensores/métodos
            # (antes usava _gcs_library_base() que dependia do GLOBAL_OPTS['SENSOR'] atual
            # e limitava o scan a apenas um sensor)
            base_search = f"{bucket}/{CONFIG['gcs_library_images']}"
            
            if logger: logger(f"Escaneando GCS: gs://{base_search} ...", "info")
            
            # Listagem recursiva é mais lenta mas garante sincronia total
            all_files = fs.find(base_search)
            
            for fpath in all_files:
                if not fpath.endswith('.tif'): continue
                
                # Identifica se é COG ou Chunk
                basename = fpath.split('/')[-1]
                
                # 1. Processa COGs usando a hierarquia de pastas
                if basename.endswith('_cog.tif'):
                    # O path esperado é: .../LIBRARY_IMAGES/{SENSOR}/{PERIODICITY}/{MOSAIC}/{DATE}/COG/{FILENAME}
                    parts = fpath.split('/')
                    try:
                        # Encontra o índice da pasta LIBRARY_IMAGES
                        lib_idx = -1
                        for i, p in enumerate(parts):
                            if p.upper() == 'LIBRARY_IMAGES':
                                lib_idx = i
                                break
                        
                        if lib_idx != -1 and len(parts) >= lib_idx + 6:
                            sensor = parts[lib_idx + 1].lower()
                            folder_period = parts[lib_idx + 2].upper()
                            periodicity = FOLDER_TO_PERIODICITY.get(folder_period)
                            mosaic = parts[lib_idx + 3].lower()
                            # date   = parts[lib_idx + 4]
                            
                            name_no_ext = basename[:-8].lower() # image_peru_fire_{sensor}_{mosaic}_{band}_{date}
                            
                            if periodicity:
                                bucket_list = cogs.setdefault(periodicity, [])
                                if name_no_ext not in bucket_list:
                                    bucket_list.append(name_no_ext)
                    except Exception:
                        continue
                
                # 2. Processa Chunks
                elif '/CHUNKS/' in fpath.upper():
                    name_no_ext = basename[:-4]
                    
                    # Novo padrão: image_{country}_fire_{sensor}_{mosaic}_{band}_{date}
                    # Antigo padrão: {mosaic}_{sensor}_fire_{country}_{date}_{band}
                    
                    # Tenta identificar a banda e a data de forma robusta
                    found_band = None
                    for band in sorted_bands:
                        if f"_{band}_" in name_no_ext or name_no_ext.endswith(f"_{band}"):
                            found_band = band
                            break
                    
                    if found_band and name_no_ext.lower().startswith('image_'):
                        # New-format chunks encode the date in the folder and
                        # filename. Read the folder first: an annual filename
                        # ends in YYYY_000000..., which a loose regex misreads
                        # as YYYY_00 and prevents M2 from finding its bands.
                        import re as _re
                        parts = fpath.split('/')
                        lib_idx = next((i for i, part in enumerate(parts) if part.upper() == 'LIBRARY_IMAGES'), -1)
                        clean_date = None
                        if lib_idx >= 0 and len(parts) > lib_idx + 4:
                            folder_period = parts[lib_idx + 2].upper()
                            folder_date = parts[lib_idx + 4]
                            periodicity = FOLDER_TO_PERIODICITY.get(folder_period)
                            if periodicity:
                                try:
                                    clean_date = parse_period(folder_date, periodicity=periodicity).key
                                except ValueError:
                                    clean_date = None

                        # Fallback for valid new-format filenames outside the
                        # standard folder layout. A two-digit month is distinct
                        # from the zero-padded shard suffix of annual exports.
                        if clean_date is None:
                            marker = f"_{found_band}_"
                            if marker in name_no_ext:
                                date_parts = name_no_ext.split(marker, 1)[1].split('_')
                                if date_parts and _re.fullmatch(r'\d{4}', date_parts[0]):
                                    if len(date_parts) > 1 and _re.fullmatch(r'0[1-9]|1[0-2]|S[12]|Q[1-4]', date_parts[1]):
                                        clean_date = f"{date_parts[0]}_{date_parts[1]}"
                                    else:
                                        clean_date = date_parts[0]

                        if clean_date is not None:
                            # The UI key is image_{country}_fire_{sensor}_{mosaic}_{date}.
                            prefix = name_no_ext.split(f"_{found_band}", 1)[0]
                            mosaic_key = f"{prefix}_{clean_date}".lower()
                            if mosaic_key not in gcs_chunks:
                                gcs_chunks[mosaic_key] = []
                            if found_band not in gcs_chunks[mosaic_key]:
                                gcs_chunks[mosaic_key].append(found_band)
                            continue

                    # Fallback para o padrão antigo: {mosaic}_{sensor}_fire_{country}_{date}_{band}
                    # Ex: minnbr_sentinel2_fire_peru_2026_01_dayOfYear...
                    found_legacy = False
                    for band in sorted_bands:
                        needle = f"_{band}"
                        if needle in name_no_ext:
                            # Tenta reconstruir a chave da UI (que usa o novo padrão)
                            # Mas para o Cache, precisamos que a chave case com o que o M1 gera.
                            # Se o arquivo é antigo, vamos mapeá-lo para a chave que a UI usa hoje.
                            parts = name_no_ext.split(needle)
                            main_part = parts[0] # minnbr_sentinel2_fire_peru_2026_01
                            
                            # Tenta extrair os componentes do nome antigo
                            p = main_part.split('_')
                            if len(p) >= 5:
                                m_type = p[0]
                                s_type = p[1]
                                c_code = p[3]
                                d_part = p[4]
                                if len(p) > 5: d_part += "_" + p[5] # para YYYY_MM
                                
                                # Chave compatível com mosaic_name atual: image_{country}_fire_{sensor}_{mosaic}_{date}
                                mosaic_key = f"image_{c_code}_fire_{s_type}_{m_type}_{d_part}"
                                
                                if mosaic_key not in gcs_chunks:
                                    gcs_chunks[mosaic_key] = []
                                if band not in gcs_chunks[mosaic_key]:
                                    gcs_chunks[mosaic_key].append(band)
                                found_legacy = True
                                break
                    if found_legacy: continue

            state['gcs_chunks'] = gcs_chunks
            for periodicity, values in cogs.items():
                state[period_state_key('cogs', periodicity)] = values
            
            CacheManager._state = state
            CacheManager.save()

            if logger:
                total_chunks = sum(len(b) for b in gcs_chunks.values())
                total_cogs = sum(len(values) for values in cogs.values())
                logger(f"GCS Sincronizado: {total_cogs} COGs, {total_chunks} fragmentos.", "success")

        except Exception as e:
            if logger: logger(f"Erro ao escanear GCS: {e}", "warning")

        return CacheManager._state

    @staticmethod
    def _scan_gcs_subdir(subdir, logger=None):
        """Escaneia {subdir}/ no GCS e retorna lista de nomes de subdiretórios."""
        from M0_auth_config import CONFIG, _gcs_models_base
        base = f"{CONFIG['bucket']}/{_gcs_models_base()}{'/' + subdir if subdir else ''}"
        names = []
        try:
            fs = _get_fs()
            if logger: logger(f"Escaneando GCS: gs://{base}/ ...", "info")
            items = fs.ls(base)
            for item in items:
                name = item.rstrip('/').split('/')[-1]
                if name and not name.startswith('.'):
                    names.append(name)
        except FileNotFoundError:
            pass
        except Exception as e:
            if logger: logger(f"Erro ao escanear {subdir}: {e}", "warning")
        return sorted(names)

    @staticmethod
    def build_cache_from_models(logger=None):
        """Popula cache com IDs de modelos treinados (training_* em LIBRARY_MODELS/)."""
        state = CacheManager.get_state()
        all_dirs = CacheManager._scan_gcs_subdir('', logger=logger)
        state['trained_models'] = sorted([d for d in all_dirs if d.startswith('training_')])
        CacheManager._state = state
        CacheManager.save()
        if logger:
            logger(f"Modelos sincronizados: {len(state['trained_models'])} encontrados.", "success")
        return CacheManager._state

    @staticmethod
    def build_cache_from_samples(logger=None):
        """Popula cache com coleções de amostras (em LIBRARY_MODELS/samples-ESbeta/)."""
        state = CacheManager.get_state()
        state['sample_collections'] = CacheManager._scan_gcs_subdir('samples-ESbeta', logger=logger)
        CacheManager._state = state
        CacheManager.save()
        if logger:
            logger(f"Coleções de amostras: {len(state['sample_collections'])} encontradas.", "success")
        return CacheManager._state

    @staticmethod
    def build_full_cache(logger=None, years=None):
        """Reconstrói o cache completo."""
        CacheManager.build_cache_from_gee(logger=logger)
        CacheManager.build_cache_from_gcs(logger=logger, years=years)
        CacheManager.build_cache_from_models(logger=logger)
        CacheManager.build_cache_from_samples(logger=logger)
        return CacheManager._state



    @staticmethod
    def save(state=None, touch=True):
        """Persist the merged state into the library and campaign caches."""
        from M_gcs import write_json
        with CacheManager._lock:
            try:
                if state is None:
                    state = CacheManager._state
                now = datetime.datetime.utcnow().isoformat() + 'Z'
                if touch:
                    state['updated_at'] = now
                    state['campaign_updated_at'] = now
                else:
                    state.setdefault('updated_at', None)
                    state.setdefault('campaign_updated_at', now)
                state['country'] = CONFIG['country']
                library_scope = CacheManager._library_scope()
                campaign_scope = CacheManager._campaign_scope()
                state['_cache_scope'] = library_scope
                state['_campaign_scope'] = campaign_scope

                library = {
                    'updated_at': state.get('updated_at'),
                    'country': state['country'],
                    '_cache_scope': library_scope,
                    '_cache_format_version': 4,
                    'assets_monthly': state.get('assets_monthly', []),
                    'assets_annually': state.get('assets_annually', []),
                    'gcs_chunks': state.get('gcs_chunks', {}),
                    'cogs_monthly': state.get('cogs_monthly', []),
                    'cogs_annually': state.get('cogs_annually', []),
                }
                campaign = {
                    'updated_at': state.get('campaign_updated_at'),
                    'country': state['country'],
                    '_cache_scope': campaign_scope,
                    '_cache_format_version': 4,
                    'trained_models': state.get('trained_models', []),
                    'sample_collections': state.get('sample_collections', []),
                    'classified_groups': state.get('classified_groups', []),
                }

                CacheManager._state = state
                CacheManager._write_local_cache(
                    CacheManager.LIBRARY_CACHE_FILE, library)
                CacheManager._write_local_cache(
                    CacheManager.CAMPAIGN_CACHE_FILE, campaign)
                write_json(CacheManager._get_library_gcs_path(), library)
                write_json(CacheManager._get_campaign_gcs_path(), campaign)
            except Exception as e:
                # Aviso limpo sem traceback assustador
                print(f"Warning: Local cache OK (GCS sync pending) - Error: {e}")

    @staticmethod
    def get_state():
        """Retorna estado atual (livraria + campanha mesclados)."""
        with CacheManager._lock:
            if (CacheManager._state is None
                    or CacheManager._state.get('_cache_scope') != CacheManager._library_scope()
                    or CacheManager._state.get('_campaign_scope') != CacheManager._campaign_scope()):
                return CacheManager.load(force=True)
            return CacheManager._state

    @staticmethod
    def cogs_for(periodicity):
        """Return cached COG names for a periodicity."""
        state = CacheManager.get_state()
        return list(state.get(period_state_key('cogs', periodicity), []))

    @staticmethod
    def reset():
        CacheManager._state = None

    @staticmethod
    def mark_stale():
        """Invalidate the library cache stamp and persist it.

        Exports create data asynchronously, so the next module/session must
        rescan instead of trusting the old state. Only the library is affected;
        the campaign cache keeps its own timestamp.
        """
        with CacheManager._lock:
            if CacheManager._state:
                CacheManager._state['updated_at'] = None
                CacheManager.save(CacheManager._state, touch=False)

    @staticmethod
    def build_cache_from_classifications(fs=None, logger=None):
        """Scan GCS para descobrir grupos (modelo, regiao, periodo) classificados."""
        from M5_workplan import gcs_full, classified_tiles_dir
        if fs is None:
            from M0_auth_config import _get_fs
            fs = _get_fs()

        base = gcs_full(classified_tiles_dir('', ''))
        base = base.replace('/CLASSIFIED_TILES', '')
        prefix = base.rsplit('/', 1)[0] if base.endswith('/') else base

        groups = set()
        try:
            all_tiles = fs.glob(f"{prefix}/*/CLASSIFIED_TILES/tile_*.tif")
        except Exception as e:
            if logger:
                logger(f"    [ERROR] scanning classifications: {e}")
            return groups

        for tp in all_tiles:
            basename = os.path.basename(tp)
            parts = basename.replace('tile_', '', 1).split('_')
            if len(parts) < 3:
                continue
            region = parts[0]
            period = parts[-1].replace('.tif', '')
            model_dir = tp.split('/CLASSIFIED_TILES')[0]
            model_id = os.path.basename(model_dir)
            groups.add((model_id, region, period))

        with CacheManager._lock:
            state = CacheManager.get_state()
            state['classified_groups'] = [{'model': m, 'region': r, 'period': p} for m, r, p in sorted(groups)]
            CacheManager.save(state)

        if logger:
            logger(f"    Cache: {len(groups)} classified groups discovered")
        return groups
