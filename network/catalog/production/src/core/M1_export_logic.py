"""
M1 - Exportação GEE Multi-sensor (Lógica)
Converte generate_quality_mosaic_for_multisensors.js para a Python API do Earth Engine.
"""
import ee

from M_periods import parse_period

def ensure_asset_path(asset_id, type_at_end='IMAGE_COLLECTION'):
    """
    Garante que todo o caminho de pastas e a coleção final existam no GEE.
    """
    parts = asset_id.split('/')
    # Índice onde começam os assets após o prefixo do projeto
    # Ex: projects/mapbiomas-mosaics/assets/FIRE/SENTINEL2...
    start_index = 4 if parts[0] == 'projects' else 1
    
    for i in range(start_index, len(parts) + 1):
        partial_path = '/'.join(parts[:i])
        try:
            ee.data.getAsset(partial_path)
        except ee.EEException:
            is_last = (i == len(parts))
            asset_type = type_at_end if is_last else 'FOLDER'
            try:
                print(f"[GEE] Creating {asset_type}: {partial_path}")
                ee.data.createAsset({'type': asset_type}, partial_path)
            except Exception as e:
                print(f"[WARN] Failed to create {asset_type} {partial_path}: {e}")

# ─── FUNÇÕES DE SUPORTE ────────────────────────────────────────────────────────

def clip_board_landsat(image):
    return image.updateMask(ee.Image().paint(image.geometry().buffer(-3000), 0).eq(0))

def bitwise_extract(value, from_bit, to_bit=None):
    if to_bit is None:
        to_bit = from_bit
    maskSize = ee.Number(1).add(to_bit).subtract(from_bit)
    mask = ee.Number(1).leftShift(maskSize).subtract(1)
    return value.rightShift(from_bit).bitwiseAnd(mask)

def add_band_nbr(image):
    exp = '( b("nir") - b("swir2") ) / ( b("nir") + b("swir2") )'
    minimo_nbr = image.expression(exp).multiply(-1).add(1).multiply(1000).toInt16().rename("nbr")
    return image.addBands(minimo_nbr)

def add_doy(image):
    doy = ee.Image(ee.Number.parse(image.date().format('D'))).toInt16().rename('dayOfYear')
    return image.addBands(doy)

# ─── CORREÇÕES RADIOMÉTRICAS ───────────────────────────────────────────────────

def corrections_ls57_col2(image):
    opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    image = image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    qa = image.select('QA_PIXEL')
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0).And(qa.bitwiseAnd(cloudsBitMask).eq(0))

    clear = bitwise_extract(qa, 6)
    water = bitwise_extract(qa, 7)
    radsatQA = image.select('QA_RADSAT')
    anySaturated = bitwise_extract(radsatQA, 0, 6)
    mask_saturation = clear.Or(water).And(anySaturated.Not())

    negative_mask = image.select(['SR_B1']).gt(0).And(image.select(['SR_B2']).gt(0)) \
        .And(image.select(['SR_B3']).gt(0)).And(image.select(['SR_B4']).gt(0)) \
        .And(image.select(['SR_B5']).gt(0)).And(image.select(['SR_B7']).gt(0))

    image = image.updateMask(mask).updateMask(mask_saturation).updateMask(negative_mask)
    oldBands = ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7']
    newBands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    return image.select(oldBands, newBands)

def corrections_ls89_col2(image):
    opticalBands = image.select('SR_B.*').multiply(0.0000275).add(-0.2)
    opticalBands = opticalBands.multiply(10000).subtract(0.0000275 * 0.2 * 1e5 * 100).round().divide(10000)
    thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0)
    image = image.addBands(opticalBands, None, True).addBands(thermalBands, None, True)

    qa = image.select('QA_PIXEL')
    cloud = qa.bitwiseAnd(1 << 3).And(qa.bitwiseAnd(1 << 9)).Or(qa.bitwiseAnd(1 << 4))
    good_pixel = qa.bitwiseAnd(1 << 6).Or(qa.bitwiseAnd(1 << 7))
    radsatQA = image.select('QA_RADSAT')
    saturated = radsatQA.bitwiseAnd(1 << 0).Or(radsatQA.bitwiseAnd(1 << 1)) \
        .Or(radsatQA.bitwiseAnd(1 << 2)).Or(radsatQA.bitwiseAnd(1 << 3)) \
        .Or(radsatQA.bitwiseAnd(1 << 4)).Or(radsatQA.bitwiseAnd(1 << 5)) \
        .Or(radsatQA.bitwiseAnd(1 << 6))

    negative_mask = image.select(['SR_B2']).gt(0).And(image.select(['SR_B3']).gt(0)) \
        .And(image.select(['SR_B4']).gt(0)).And(image.select(['SR_B5']).gt(0)) \
        .And(image.select(['SR_B6']).gt(0)).And(image.select(['SR_B7']).gt(0))

    image = image.updateMask(cloud.Not()).updateMask(good_pixel).updateMask(saturated.Not()).updateMask(negative_mask)
    oldBands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7']
    newBands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    return image.select(oldBands, newBands).toFloat()

def modis_corrections(image):
    oldBands = ['sur_refl_b03', 'sur_refl_b04', 'sur_refl_b01', 'sur_refl_b02', 'sur_refl_b06', 'sur_refl_b07']
    newBands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
    return image.select(oldBands, newBands)

# ─── PIPELINES DE PROCESSAMENTO ────────────────────────────────────────────────

def process_ls57(image):
    return add_doy(add_band_nbr(corrections_ls57_col2(clip_board_landsat(image))))

def process_ls89(image):
    return add_doy(add_band_nbr(corrections_ls89_col2(clip_board_landsat(image))))

def process_s2(img):
    mask = img.select('cs').gte(0.40)
    optical = img.updateMask(mask).select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    return add_doy(add_band_nbr(optical))

def process_modis(image):
    return add_doy(add_band_nbr(modis_corrections(image)))

def process_hls_s30(img):
    mask = img.select('Fmask').bitwiseAnd(1 << 1).eq(0)
    optical = img.updateMask(mask).select(['B2', 'B3', 'B4', 'B8A', 'B11', 'B12'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    out = add_band_nbr(optical).set({'system:time_start': img.get('system:time_start'), 'system:time_end': img.get('system:time_end')})
    return add_doy(out)

def process_hls_l30(img):
    mask = img.select('Fmask').bitwiseAnd(1 << 1).eq(0)
    optical = img.updateMask(mask).select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
    out = add_band_nbr(optical).set({'system:time_start': img.get('system:time_start'), 'system:time_end': img.get('system:time_end')})
    return add_doy(out)

# ─── LÓGICA DE COLEÇÃO E REDUÇÃO ───────────────────────────────────────────────

INPE_BUFFER_COLLECTION = "projects/workspace-ipam/assets/BUFFER-DOUBLE-MONTHLY-FOCUS-OF-INPE-SULAMERICA"

def apply_inpe_buffer_mask(image, year, month):
    """
    Aplica a máscara do buffer de focos de calor do INPE ao mosaico mensal.
    Se não houver dados de buffer para o período, retorna a imagem sem alteração.
    """
    if month is None:
        return image  # Mosaico anual: não aplica máscara
    try:
        buffer_col = ee.ImageCollection(INPE_BUFFER_COLLECTION) \
            .filter(ee.Filter.eq('year', year)) \
            .filter(ee.Filter.eq('month', month))
        buffer_mask = buffer_col.mosaic()
        # Mantém apenas pixels dentro do buffer (onde buffer > 0)
        has_data = buffer_col.size().gt(0)
        return ee.Image(ee.Algorithms.If(
            has_data,
            image.updateMask(buffer_mask.gt(0)),
            image
        ))
    except Exception as e:
        print(f"[WARN] INPE buffer not available for {year}/{month}: {e}")
        return image

def get_landsat_constellation(year):
    if year < 1999: return ['L5']
    elif year <= 2012: return ['L5', 'L7']
    elif year <= 2021: return ['L7', 'L8']
    else: return ['L8', 'L9']

def _blocklist_filter(sensor):
    """Build a scene exclusion filter from the live GEE blocklist module."""
    from M_blocklist import get_blocklist
    try:
        blocked_ids = get_blocklist(sensor)
    except Exception:
        return None
    if not blocked_ids:
        return None
    return ee.Filter.Not(
        ee.Filter.inList('system:index', ee.List(blocked_ids))
    )

def get_quality_mosaic(sensor, year, start_date, end_date, bounds, month=None, method='minnbr'):
    """
    Gera o mosaico de qualidade para a constelação e método escolhidos.
    Suporta: minnbr, minnbr_buffer, median, minndvi
    """
    multiplier = 1
    collection = None
    
    # ... (lógica de coleção permanece igual)
    if sensor == 'landsat':
        constellation = get_landsat_constellation(year)
        lst_collects = []
        blocked_filter = _blocklist_filter(sensor)

        def _filtered_landsat(collection_id, processor):
            collection = ee.ImageCollection(collection_id).filterDate(start_date, end_date).filterBounds(bounds)
            if blocked_filter is not None:
                collection = collection.filter(blocked_filter)
            return collection.map(processor)

        if 'L9' in constellation: lst_collects.append(_filtered_landsat("LANDSAT/LC09/C02/T1_L2", process_ls89))
        if 'L8' in constellation: lst_collects.append(_filtered_landsat("LANDSAT/LC08/C02/T1_L2", process_ls89))
        if 'L7' in constellation: lst_collects.append(_filtered_landsat("LANDSAT/LE07/C02/T1_L2", process_ls57))
        if 'L5' in constellation: lst_collects.append(_filtered_landsat("LANDSAT/LT05/C02/T1_L2", process_ls57))
        collection = lst_collects[0]
        for c in lst_collects[1:]: collection = collection.merge(c)
        multiplier = 100
    elif sensor == 'sentinel2':
        collection = ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterDate(start_date, end_date).filterBounds(bounds)
        blocked_filter = _blocklist_filter('sentinel2')
        if blocked_filter is not None:
            collection = collection.filter(blocked_filter)
        collection = collection.linkCollection(
            ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED'), ['cs']
        ).map(process_s2)
        multiplier = 0.01
    elif sensor == 'hls':
        s30_col = ee.ImageCollection("NASA/HLS/HLSS30/v002").filterDate(start_date, end_date).filterBounds(bounds)
        l30_col = ee.ImageCollection("NASA/HLS/HLSL30/v002").filterDate(start_date, end_date).filterBounds(bounds)
        s30_filter = _blocklist_filter('hls_s30')
        l30_filter = _blocklist_filter('hls_l30')
        if s30_filter is not None:
            s30_col = s30_col.filter(s30_filter)
        if l30_filter is not None:
            l30_col = l30_col.filter(l30_filter)
        collection = s30_col.map(process_hls_s30).merge(l30_col.map(process_hls_l30))
        multiplier = 100
    elif sensor == 'modis':
        mod_col1 = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(start_date, end_date).filterBounds(bounds)
        mod_col2 = ee.ImageCollection("MODIS/061/MYD09A1").filterDate(start_date, end_date).filterBounds(bounds)
        terra_filter = _blocklist_filter('modis_terra')
        aqua_filter = _blocklist_filter('modis_aqua')
        if terra_filter is not None:
            mod_col1 = mod_col1.filter(terra_filter)
        if aqua_filter is not None:
            mod_col2 = mod_col2.filter(aqua_filter)
        collection = mod_col1.merge(mod_col2).map(process_modis)
        multiplier = 0.01

    # Redução Baseada no Método
    method = method.lower()
    if 'minnbr' in method:
        mosaic = collection.qualityMosaic('nbr')
    elif 'minndvi' in method:
        def add_ndvi(img): return img.addBands(img.normalizedDifference(['nir', 'red']).rename('ndvi'))
        mosaic = collection.map(add_ndvi).qualityMosaic('ndvi')
    elif 'median' in method:
        mosaic = collection.median()
    else:
        mosaic = collection.qualityMosaic('nbr')
    
    spectral = mosaic.select(['blue', 'green', 'red', 'nir', 'swir1', 'swir2']).multiply(multiplier).toByte()
    doy = mosaic.select('dayOfYear').toInt16()
    
    full = spectral.addBands(doy).clip(bounds)
    
    # Aplica máscara de buffer se o método solicitar (ex: minnbr_buffer)
    if '_buffer' in method:
        full = apply_inpe_buffer_mask(full, year, month)
    
    return full

# ─── FUNÇÕES DE EXPORTAÇÃO ─────────────────────────────────────────────────────

def export_to_asset(mosaic_obj, name, year, month=None, period='monthly', config=None, band=None, mosaic=None, sensor=None, period_key=None):
    from M0_auth_config import get_country_geometry, get_sensor_scale

    country_geom = config.get_country_geometry() if config else get_country_geometry()
    info = parse_period(period_key, periodicity=period) if period_key else None
    if info:
        year = info.year
        month = info.index if info.periodicity == 'monthly' else None
        period = info.periodicity
        t_start = ee.Date(info.start.strftime('%Y-%m-%d')).millis()
        t_end = ee.Date(info.end.strftime('%Y-%m-%d')).millis()
    elif period == 'monthly':
        t_start = ee.Date(f'{year}-{month:02d}-01').millis()
        t_end = ee.Date(f'{year}-{month:02d}-01').advance(1, 'month').millis()
    else:
        t_start = ee.Date(f'{year}-01-01').millis()
        t_end = ee.Date(f'{year+1}-01-01').millis()

    from M0_auth_config import GLOBAL_OPTS, CONFIG
    
    s = (sensor or _single(GLOBAL_OPTS['SENSOR'])).lower()
    
    properties = {
        'system:time_start': t_start, 'system:time_end': t_end,
        'year': year, 'month': month or 0,
        'period': period, 'name': name,
        'country': CONFIG['country'],
        'sensor': s,
        'mosaic': (mosaic or GLOBAL_OPTS.get('MOSAIC_METHOD', 'minnbr')).lower(),
        'band': band or 'all',
        'produced_for': 'mapbiomas-fire'
    }
    if info:
        properties['period_key'] = info.key
    
    img = mosaic_obj
    if band:
        img = img.select(band)
        
    img = img.clip(country_geom).set(properties)

    from M0_auth_config import get_asset_mosaic_collection, GLOBAL_OPTS
    m_method = mosaic or GLOBAL_OPTS.get('MOSAIC_METHOD', 'minnbr')
    collection_id = get_asset_mosaic_collection(
        period=period, band=band, mosaic=m_method, sensor=s, period_key=period_key,
    )
    
    # Garante que a estrutura de pastas e a ImageCollection existam
    ensure_asset_path(collection_id, 'IMAGE_COLLECTION')

    asset_id = f"{collection_id}/{name}"

    task = ee.batch.Export.image.toAsset(
        image=img, description=f"{GLOBAL_OPTS['PERSONAL_TASK_FLAG']}_ASSET_{name}", assetId=asset_id,
        region=country_geom.bounds(), scale=get_sensor_scale(s), maxPixels=1e13,
        pyramidingPolicy={'.default': 'median'},
        overwrite=True
    )
    task.start()
    return task


def delete_gcs_band(year, month, period, band, sensor=None, mosaic=None, period_key=None):
    """Deleta os chunks de uma banda específica no GCS."""
    from M0_auth_config import monthly_chunk_path, yearly_chunk_path, period_chunk_path, mosaic_name, mosaic_name_for_period, CONFIG, GLOBAL_OPTS, _single
    from M_gcs import glob, rm
    m_method = (mosaic or GLOBAL_OPTS.get('MOSAIC_METHOD', 'minnbr')).lower()
    s = sensor or _single(GLOBAL_OPTS['SENSOR'])
    info = parse_period(period_key, periodicity=period) if period_key else None
    if info:
        folder = period_chunk_path(info.key, mosaic=m_method, sensor=s)
        name = mosaic_name_for_period(info.key, mosaic=m_method, sensor=s)
    else:
        folder = monthly_chunk_path(year, month, mosaic=m_method, sensor=s) if period == 'monthly' else yearly_chunk_path(year, mosaic=m_method, sensor=s)
        name = mosaic_name(year, month, period, mosaic=m_method, sensor=s)
    prefix = f"{CONFIG['bucket']}/{folder}/{name}_{band}"
    print(f"[GCS] Removing band {band}: gs://{prefix}*")
    try:
        files = glob(prefix + "*")
        for f in files:
            rm(f)
    except Exception as e:
        print(f"[WARN] Failed to delete band {band}: {e}")


def export_to_gcs(mosaic_obj, name, year, month=None, period='monthly', bands=None, config_module=None, mosaic=None, sensor=None, period_key=None):
    from M0_auth_config import CONFIG, GLOBAL_OPTS, monthly_chunk_path, yearly_chunk_path, period_chunk_path, mosaic_name, mosaic_name_for_period, get_sensor_scale
    geometry = config_module.get_country_geometry() if config_module else mosaic_obj.geometry()
    
    m_method = mosaic or GLOBAL_OPTS.get('MOSAIC_METHOD', 'minnbr').lower()
    s = sensor or GLOBAL_OPTS.get('SENSOR', 'sentinel2')
    if isinstance(s, list): s = s[0]
    info = parse_period(period_key, periodicity=period) if period_key else None
    if info:
        folder = period_chunk_path(info.key, mosaic=m_method, sensor=s)
    elif period == 'monthly' and month is not None:
        folder = monthly_chunk_path(year, month, mosaic=m_method, sensor=s)
    else:
        folder = yearly_chunk_path(year, mosaic=m_method, sensor=s)

    if not bands:
        bands = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'dayOfYear']
        
    for band in bands:
        # Normalização para evitar problemas com dayOfYear (case-insensitive)
        clean_band = "dayOfYear" if band.lower() == "dayofyear" else band
        
        band_name = (
            mosaic_name_for_period(info.key, band=clean_band, mosaic=m_method, sensor=s)
            if info else mosaic_name(year, month, period, band=clean_band, mosaic=m_method, sensor=s)
        )
        task = ee.batch.Export.image.toCloudStorage(
            image=mosaic_obj.select(band).clip(geometry),
            description=f"{GLOBAL_OPTS['PERSONAL_TASK_FLAG']}_GCS_{band_name}", bucket=CONFIG['bucket'],
            fileNamePrefix=f"{folder}/{band_name}_", region=geometry.bounds(),
            scale=get_sensor_scale(s), maxPixels=1e13, fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True},
        )
        task.start()
