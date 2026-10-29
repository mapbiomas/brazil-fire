import ee
from typing import List, Optional


REFERENCE_LANDCOVER = "projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/reference_native/reference_col9_v1"

LANDCOVER_FOREST_CLASSES = [3, 5, 6]


def build_edge_area() -> ee.Image:
    landcover_base = ee.Image(REFERENCE_LANDCOVER)
    return (
        landcover_base.where(landcover_base.gte(1), 9)
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/edge_area/edge_1000m_col9_v1").gt(1).multiply(8))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/edge_area/edge_600m_col9_v1").gt(1).multiply(7))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/edge_area/edge_300m_col9_v1").gt(1).multiply(6))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/edge_area/edge_150m_col9_v1").gt(1).multiply(5))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/edge_area/edge_120m_col9_v1").gt(1).multiply(4))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/edge_area/edge_90m_col9_v1").gt(1).multiply(3))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/edge_area/edge_60m_col9_v1").gt(1).multiply(2))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/edge_area/edge_30m_col9_v1").gt(1).multiply(1))
    )


def build_fragment_size() -> ee.Image:
    landcover_base = ee.Image(REFERENCE_LANDCOVER)
    return (
        landcover_base.multiply(0)
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/patch_size/size_75ha_col9_v1").gt(1).multiply(6))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/patch_size/size_50ha_col9_v1").gt(1).multiply(5))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/patch_size/size_25ha_col9_v1").gt(1).multiply(4))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/patch_size/size_10ha_col9_v1").gt(1).multiply(3))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/patch_size/size_5ha_col9_v1").gt(1).multiply(2))
        .blend(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/patch_size/size_3ha_col9_v1").gt(1).multiply(1))
    )


def build_distance_100ha() -> ee.Image:
    landcover_base = ee.Image(REFERENCE_LANDCOVER)
    return (
        landcover_base.multiply(0)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/BR_Distance/natural_mask_maior100ha_85_234"), 10)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag50_dist05k_100_v7_85_23"), 2)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag25_dist05k_100_v7_85_23"), 3)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag50_dist10k_100_v7_85_23"), 5)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag25_dist10k_100_v7_85_23"), 6)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag50_dist20k_100_v7_85_23"), 8)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag25_dist20k_100_v7_85_23"), 9)
    )


def build_distance_500ha() -> ee.Image:
    landcover_base = ee.Image(REFERENCE_LANDCOVER)
    return (
        landcover_base.multiply(0)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/BR_Distance/natural_mask_maior500ha_85_234"), 11)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag100_dist05k_500_v7_85_23"), 1)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag50_dist05k_500_v7_85_23"), 2)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag25_dist05k_500_v7_85_23"), 3)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag100_dist10k_500_v7_85_23"), 4)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag50_dist10k_500_v7_85_23"), 5)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag25_dist10k_500_v7_85_23"), 6)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag50_dist20k_500_v7_85_23"), 8)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag25_dist20k_500_v7_85_23"), 9)
    )


def build_distance_1000ha() -> ee.Image:
    landcover_base = ee.Image(REFERENCE_LANDCOVER)
    return (
        landcover_base.multiply(0)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/BR_Distance/natural_mask_maior1000ha_85_234"), 12)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag100_dist05k_1000_v7_85_23"), 1)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag50_dist05k_1000_v7_85_23"), 2)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag25_dist05k_1000_v7_85_23"), 3)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag100_dist10k_1000_v7_85_23"), 4)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag50_dist10k_1000_v7_85_23"), 5)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag25_dist10k_1000_v7_85_23"), 6)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag100_dist20k_1000_v7_85_23"), 7)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag50_dist20k_1000_v7_85_23"), 8)
        .where(ee.Image("projects/mapbiomas-workspace/DEGRADACAO/ISOLATION_col9_v2/nat_uso_frag25_dist20k_1000_v7_85_23"), 9)
    )


def build_secondary_vegetation_coverage() -> ee.Image:
    return (
        ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/secondary_vegetation/secondary_vegetation_age_col9_v1")
        .mod(100)
        .int8()
    )


def build_secondary_vegetation_age() -> ee.Image:
    return (
        ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/secondary_vegetation/secondary_vegetation_age_col9_v1")
        .divide(100)
        .int8()
    )


def build_fire_frequency() -> ee.Image:
    return (
        ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/fire/recurrence_col9_v1")
        .divide(100)
        .int8()
    )


def build_fire_age() -> ee.Image:
    return (
        ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/fire/recurrence_col9_v1")
        .multiply(0)
        .add(
            ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/fire/age_col9_v1")
            .divide(100)
            .int8()
        )
    )


def build_accumulated_burned_coverage() -> ee.Image:
    return (
        ee.Image("projects/mapbiomas-workspace/DEGRADACAO/COLECAO/BETA/PROCESS/fire/recurrence_col9_v1")
        .mod(100)
        .int8()
    )


COL101_FIRE_FREQ = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_fire_frequency_v1"
COL101_FIRE_AGE = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_fire_age_v1"
COL101_VEG_SEC = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_secondary_vegetation_age_v1"
COL101_EDGE_AREA = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_edge_area_v1"
COL11_EDGE_AREA = "projects/mapbiomas-brazil/assets/DEGRADATION/COLLECTION-11/public/degradation_edge_area_col11_v1"
COL101_EDGE_AGE = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_edge_age_v1"
COL101_MORPHOLOGY = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_landscape_morphology_v1"
COL101_PATCH_ID = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_patch_id_v1"
COL101_PATCH_SIZE = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_patch_size_v1"
COL101_CANOPY_DISTURBANCE = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_canopy_disturbance_frequency_v2"
COL101_LOGGING = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_logging_v2"
COL101_COVERAGE = "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_coverage_v1"


def build_natural_coverage_col101() -> ee.Image:
    return ee.Image(COL101_COVERAGE).unmask(0).where(
        ee.Image(COL101_COVERAGE).gte(3), 0
    ).int8()


def decode_fire_frequency_col101() -> ee.Image:
    return ee.Image(COL101_FIRE_FREQ).divide(100).round().unmask(0).int8()


def decode_fire_age_col101() -> ee.Image:
    return ee.Image(COL101_FIRE_AGE).divide(100).round().unmask(0).int8()


def decode_secondary_vegetation_age_col101() -> ee.Image:
    return ee.Image(COL101_VEG_SEC).divide(100).round().unmask(0).int8()


def build_secondary_vegetation_coverage_col101() -> ee.Image:
    img = ee.Image(COL101_VEG_SEC)
    age = img.divide(100).round().int8()
    klass = img.mod(100).unmask(0).int8()
    return klass.where(age.lte(1), 0).int8()


def build_fire_frequency_coverage_col101() -> ee.Image:
    return ee.Image(COL101_FIRE_FREQ).mod(100).unmask(0).int8()


def build_burned_natural_coverage_col101() -> ee.Image:
    img = ee.Image(COL101_FIRE_FREQ)
    freq = img.divide(100).round().int8()
    klass = img.mod(100).unmask(0).int8()
    return klass.where(freq.lt(1), 0)


def build_burned_at_least_once_col101() -> ee.Image:
    img = ee.Image(COL101_FIRE_FREQ)
    freq = img.divide(100).round().unmask(0).int8()
    klass = img.mod(100).unmask(0).int8()
    natural = (klass.eq(3).Or(klass.eq(4)).Or(klass.eq(5)).Or(klass.eq(6))
        .Or(klass.eq(11)).Or(klass.eq(12)).Or(klass.eq(32))
        .Or(klass.eq(49)).Or(klass.eq(50)))
    bands = ee.List(freq.bandNames())

    def iter_cummax(src_list):
        src = ee.List(src_list)
        init = ee.List([ee.Image(0).rename("cm")])
        def fn(cur, prv):
            prv_list = ee.List(prv)
            prev_val = ee.Image(prv_list.get(-1))
            new_val = ee.Image(cur).max(prev_val).rename("cm")
            return prv_list.add(new_val)
        result = ee.List(src.iterate(fn, init))
        result = result.slice(1)
        return ee.ImageCollection.fromImages(result).toBands().rename(bands)

    band_list = bands.map(lambda b: freq.select([b]))
    nat_list = bands.map(lambda b: natural.select([b]))

    burned_so_far = iter_cummax(band_list)

    # ever_natural using the same per-band natural images
    ever_natural = iter_cummax(nat_list)

    # cummax from the end (reverse)
    rev_band_list = ee.List(band_list).reverse()
    cummax_rev = iter_cummax(rev_band_list)
    rev_bands = cummax_rev.bandNames()
    cummax = cummax_rev.select(ee.List(rev_bands).reverse()).rename(bands)

    return (
        freq.multiply(0)
        .where(freq.gte(2), 1)
        .where(freq.eq(1), 2)
        .where(freq.eq(0).And(cummax.gt(freq)).And(natural), 3)
        .where(burned_so_far.gt(0).And(ever_natural.gt(0)).And(natural.Not()), 4)
    )


def build_primary_natural_coverage_col101() -> ee.Image:
    f = ee.Image(COL101_FIRE_FREQ)
    s = ee.Image(COL101_VEG_SEC)
    klass = f.mod(100).unmask(0).int8()
    sec_klass = s.mod(100).unmask(0).int8()
    sec_age = s.divide(100).round().int8()
    natural = (klass.eq(3).Or(klass.eq(4)).Or(klass.eq(5)).Or(klass.eq(6))
        .Or(klass.eq(11)).Or(klass.eq(12)).Or(klass.eq(32))
        .Or(klass.eq(49)).Or(klass.eq(50)))
    is_secondary = sec_klass.gt(0).And(sec_age.gt(1))
    return klass.where(natural.And(is_secondary.Not()).Not(), 0).int8()


def decode_edge_area_col101() -> ee.Image:
    img = ee.Image(COL101_EDGE_AREA)
    classified = (
        img.multiply(0)
        .where(img.gt(0).And(img.lte(30)), 1)
        .where(img.gt(30).And(img.lte(60)), 2)
        .where(img.gt(60).And(img.lte(90)), 3)
        .where(img.gt(90).And(img.lte(120)), 4)
        .where(img.gt(120).And(img.lte(150)), 5)
        .where(img.gt(150).And(img.lte(300)), 6)
        .where(img.gt(300).And(img.lte(600)), 7)
        .where(img.gt(600).And(img.lte(1000)), 8)
        .unmask(0)
        .int8()
    )
    return classified


def decode_edge_area_1000m_col11() -> ee.Image:
    """Versao categorica (0-8) da distancia ate a borda da Colecao 11.

    1 = <=30m, 2 = <=60m, ... 8 = <=1000m. Valores acima de 1000 m e pixels
    sem dado viram 0, que a visualizacao pinta de branco e omite da legenda.
    """
    img = ee.Image(COL11_EDGE_AREA).round()
    return (
        img.multiply(0)
        .where(img.gt(0).And(img.lte(30)), 1)
        .where(img.gt(30).And(img.lte(60)), 2)
        .where(img.gt(60).And(img.lte(90)), 3)
        .where(img.gt(90).And(img.lte(120)), 4)
        .where(img.gt(120).And(img.lte(150)), 5)
        .where(img.gt(150).And(img.lte(300)), 6)
        .where(img.gt(300).And(img.lte(600)), 7)
        .where(img.gt(600).And(img.lte(1000)), 8)
        .unmask(0)
        .int8()
    )


def decode_edge_age_col101() -> ee.Image:
    return ee.Image(COL101_EDGE_AGE).unmask(0).int8()


def decode_morphology_col101() -> ee.Image:
    return ee.Image(COL101_MORPHOLOGY).unmask(0).int8()


def decode_patch_id_col101() -> ee.Image:
    return ee.Image(COL101_PATCH_ID).unmask(0).toInt32()


def decode_patch_size_col101() -> ee.Image:
    """Tamanho do fragmento (contínuo, valores originais em ha)."""
    return ee.Image(COL101_PATCH_SIZE).unmask(0).toFloat()


def decode_patch_size_fragments_col101() -> ee.Image:
    """Tamanho do fragmento discretizado (0-10k ha em 10 categorias)."""
    img = ee.Image(COL101_PATCH_SIZE).unmask(999999)
    return (
        img.multiply(0)
        .where(img.gt(0).And(img.lte(5)), 1)
        .where(img.gt(5).And(img.lte(10)), 2)
        .where(img.gt(10).And(img.lte(25)), 3)
        .where(img.gt(25).And(img.lte(50)), 4)
        .where(img.gt(50).And(img.lte(100)), 5)
        .where(img.gt(100).And(img.lte(250)), 6)
        .where(img.gt(250).And(img.lte(500)), 7)
        .where(img.gt(500).And(img.lte(1000)), 8)
        .where(img.gt(1000).And(img.lte(5000)), 9)
        .where(img.gt(5000).And(img.lte(10000)), 10)
        .unmask(0)
        .int8()
    )


def decode_patch_size_massifs_col101() -> ee.Image:
    """Tamanho do fragmento discretizado (>10k ha em 10 categorias)."""
    img = ee.Image(COL101_PATCH_SIZE).unmask(999999)
    return (
        img.multiply(0)
        .where(img.gt(10000).And(img.lte(20000)), 1)
        .where(img.gt(20000).And(img.lte(50000)), 2)
        .where(img.gt(50000).And(img.lte(100000)), 3)
        .where(img.gt(100000).And(img.lte(250000)), 4)
        .where(img.gt(250000).And(img.lte(500000)), 5)
        .where(img.gt(500000).And(img.lte(1000000)), 6)
        .where(img.gt(1000000).And(img.lte(2500000)), 7)
        .where(img.gt(2500000).And(img.lte(5000000)), 8)
        .where(img.gt(5000000).And(img.lte(7500000)), 9)
        .where(img.gt(7500000), 10)
        .unmask(0)
        .int8()
    )


def decode_canopy_disturbance_col101() -> ee.Image:
    return ee.Image(COL101_CANOPY_DISTURBANCE).unmask(0).int8()


def decode_logging_col101() -> ee.Image:
    return ee.Image(COL101_LOGGING).unmask(0).int8()


def decode_monthly_burned_col5() -> ee.Image:
    asset = "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_monthly_burned_v1"
    return ee.Image(asset).unmask(0).int8()


def decode_unprecedented_fire() -> ee.Image:
    asset = "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_time_after_fire_v1"
    img = ee.Image(asset)
    return img.eq(0).add(1).unmask(0).int8().slice(0, -1)


FIRE_COL5_SEVERITY = "projects/mapbiomas-workspace/FOGO/COLLECTIONS/COL05/1_Subproducts/mapbiomas_fire_collection5_severity_class_v1"
FIRE_COL5_INTERVAL = "projects/mapbiomas-workspace/FOGO/COLLECTIONS/COL05/1_Subproducts/mapbiomas_fire_collection5_interval_since_fire_v1"
FIRE_COL5_NBR_MOSAIC = "projects/mapbiomas-workspace/FOGO/1_mosaics/landsat-view"
ANNUAL_BURNED_COVERAGE = "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_coverage_v1"
ACCUM_BURNED_COVERAGE = "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_accumulated_burned_coverage_v1"



def build_severity_col5() -> ee.Image:
    return ee.Image(FIRE_COL5_SEVERITY).slice(0, 41).unmask(0).int8()


def decode_return_interval_discrete() -> ee.Image:
    img = ee.Image(FIRE_COL5_INTERVAL).slice(0, 41)
    return (
        img.multiply(0)
        .where(img.gte(1).And(img.lt(2)), 1)
        .where(img.gte(2).And(img.lt(3)), 2)
        .where(img.gte(3).And(img.lt(4)), 3)
        .where(img.gte(4).And(img.lt(5)), 4)
        .where(img.gte(5).And(img.lt(6)), 5)
        .where(img.gte(6).And(img.lt(7)), 6)
        .where(img.gte(7).And(img.lt(8)), 7)
        .where(img.gte(8).And(img.lt(9)), 8)
        .where(img.gte(9).And(img.lt(10)), 9)
        .where(img.gte(10).And(img.lt(15)), 10)
        .where(img.gte(15).And(img.lt(20)), 11)
        .where(img.gte(20), 12)
        .unmask(0)
        .int8()
    )


def decode_mean_return_interval_discrete() -> ee.Image:
    img = ee.Image(FIRE_COL5_INTERVAL).slice(0, 41)
    n = int(img.bandNames().length().getInfo())

    def make_cummean(i):
        selected = img.select(list(range(i + 1)))
        mean = selected.selfMask().reduce(ee.Reducer.mean())
        return mean.rename(f"1985_{1985 + i}")

    cummeans = [make_cummean(i) for i in range(n)]
    result = ee.ImageCollection(cummeans).toBands()
    clean_names = result.bandNames().map(
        lambda n: ee.String(n).split('_').slice(1).join('_')
    )
    result = result.rename(clean_names)

    return (
        result.multiply(0)
        .where(result.gte(1).And(result.lt(2)), 1)
        .where(result.gte(2).And(result.lt(3)), 2)
        .where(result.gte(3).And(result.lt(4)), 3)
        .where(result.gte(4).And(result.lt(5)), 4)
        .where(result.gte(5).And(result.lt(6)), 5)
        .where(result.gte(6).And(result.lt(7)), 6)
        .where(result.gte(7).And(result.lt(8)), 7)
        .where(result.gte(8).And(result.lt(9)), 8)
        .where(result.gte(9).And(result.lt(10)), 9)
        .where(result.gte(10).And(result.lt(15)), 10)
        .where(result.gte(15).And(result.lt(20)), 11)
        .where(result.gte(20), 12)
        .unmask(0)
        .int8()
    )


def build_nbr_min_mosaic() -> ee.Image:
    col = ee.ImageCollection(FIRE_COL5_NBR_MOSAIC)
    col = col.filter(ee.Filter.eq('version', 'QMNBR_byte-annual_landsat'))

    images = []
    clean_names = []
    for y in range(1985, 2026):
        img = col.filterDate(f"{y}-01-01", f"{y}-12-31").first()
        names = [f'swir1_{y}', f'nir_{y}', f'red_{y}']
        img = img.select(['swir1', 'nir', 'red'], names).byte()
        images.append(img)
        clean_names.extend(names)

    return ee.ImageCollection(images).toBands().rename(clean_names)


MAPBIOMAS_LANDSAT_MOSAICS2 = "projects/nexgenmap/MapBiomas2/LANDSAT/BRAZIL/mosaics-2"


def _landsat_mosaics2_year(y: int):
    col = ee.ImageCollection(MAPBIOMAS_LANDSAT_MOSAICS2)
    img = col.filter(ee.Filter.eq("year", y)).mosaic()
    return img


def build_landsat_false_color_swir() -> ee.Image:
    images = []
    clean_names = []
    for y in range(1985, 2026):
        img = _landsat_mosaics2_year(y)
        names = [f'swir1_{y}', f'nir_{y}', f'red_{y}']
        img = img.select(["swir1_median", "nir_median", "red_median"], names)
        images.append(img)
        clean_names.extend(names)
    return ee.ImageCollection(images).toBands().rename(clean_names)


def build_landsat_natural_rgb() -> ee.Image:
    images = []
    clean_names = []
    for y in range(1985, 2026):
        img = _landsat_mosaics2_year(y)
        names = [f'red_{y}', f'green_{y}', f'blue_{y}']
        img = img.select(["red_median", "green_median", "blue_median"], names)
        images.append(img)
        clean_names.extend(names)
    return ee.ImageCollection(images).toBands().rename(clean_names)


MAPBIOMAS_LULC11_COVERAGE = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection11/"
    "mapbiomas_brazil_collection11_coverage_v3"
)
MAPBIOMAS_LULC11_DEFSEC = (
    "projects/mapbiomas-public/assets/brazil/lulc/collection11/"
    "mapbiomas_brazil_collection11_deforestation_secondary_vegetation_v5"
)

_LULC11_NATURAL_FOR = [3, 4, 5, 6, 7, 49]
_LULC11_NATURAL_HERB = [11, 12, 32, 50, 77, 84]
_LULC11_NATURAL_ROCK = [29]
_LULC11_WATER = [33]
_LULC11_SILV = [9]
_LULC11_PAST = [15]
_LULC11_AGR_TEMP = [20, 39, 40, 41, 62]
_LULC11_AGR_PER = [35, 46, 47, 48]
_LULC11_MOSAIC = [21]
_LULC11_URBAN = [24]
_LULC11_NONVEG = [23, 25]
_LULC11_MINING = [30]
_LULC11_AQUA = [31]
_LULC11_ENERGY = [75, 91]
_LULC11_ANTROPICO = (
    _LULC11_SILV + _LULC11_PAST + _LULC11_AGR_TEMP + _LULC11_AGR_PER
    + _LULC11_MOSAIC + _LULC11_URBAN + _LULC11_NONVEG + _LULC11_MINING
    + _LULC11_AQUA + _LULC11_ENERGY
)
_LULC11_NATURAL = _LULC11_NATURAL_FOR + _LULC11_NATURAL_HERB + _LULC11_NATURAL_ROCK


def build_lulc_col11() -> ee.Image:
    return ee.Image(MAPBIOMAS_LULC11_COVERAGE).unmask(0).int8()


def _recode_lulc_col11(mapping):
    img = ee.Image(MAPBIOMAS_LULC11_COVERAGE).unmask(0).int8()
    result = img.multiply(0)
    for codes, target in mapping:
        result = result.where(_make_eq_chain(img, codes), target)
    return result.unmask(0).int8()


def build_lulc_col11_nivel0() -> ee.Image:
    return _recode_lulc_col11([
        (_LULC11_NATURAL, 1),
        (_LULC11_ANTROPICO, 2),
        (_LULC11_WATER, 3),
    ])


def build_lulc_col11_nivel1() -> ee.Image:
    return _recode_lulc_col11([
        (_LULC11_NATURAL_FOR, 1),
        (_LULC11_NATURAL_HERB, 2),
        (_LULC11_NATURAL_ROCK, 3),
        (_LULC11_SILV, 4),
        (_LULC11_PAST + _LULC11_AGR_TEMP + _LULC11_AGR_PER + _LULC11_MOSAIC, 5),
        (_LULC11_URBAN + _LULC11_NONVEG + _LULC11_MINING + _LULC11_AQUA + _LULC11_ENERGY, 6),
        (_LULC11_WATER, 7),
    ])


def build_lulc_col11_nivel1_1() -> ee.Image:
    return _recode_lulc_col11([
        ([3, 5, 6, 49], 3),
        ([4, 7], 4),
        ([11], 11),
        (_LULC11_NATURAL_HERB, 12),
        ([29], 29),
        (_LULC11_SILV, 9),
        (_LULC11_PAST, 15),
        (_LULC11_AGR_TEMP + _LULC11_AGR_PER, 18),
        (_LULC11_MOSAIC, 21),
        (_LULC11_URBAN, 24),
        (_LULC11_NONVEG + _LULC11_MINING + _LULC11_AQUA + _LULC11_ENERGY, 22),
        (_LULC11_WATER, 26),
    ])


def build_lulc_col11_nivel2() -> ee.Image:
    return _recode_lulc_col11([
        (_LULC11_NATURAL_FOR, 3),
        ([11], 11),
        ([12, 32, 50, 77, 84], 12),
        ([29], 29),
        (_LULC11_SILV, 9),
        (_LULC11_PAST, 15),
        (_LULC11_AGR_TEMP, 19),
        (_LULC11_AGR_PER, 36),
        (_LULC11_MOSAIC, 21),
        (_LULC11_URBAN, 24),
        ([23], 23),
        ([25, 75, 91], 25),
        (_LULC11_MINING, 30),
        (_LULC11_AQUA, 31),
        (_LULC11_WATER, 26),
    ])


def build_lulc_col11_nivel3() -> ee.Image:
    return _recode_lulc_col11([
        ([3], 3),
        ([4], 4),
        ([5], 5),
        ([6], 6),
        ([7], 7),
        ([9], 9),
        ([11], 11),
        ([12, 50, 77, 84], 12),
        ([15], 15),
        ([20], 20),
        ([21], 21),
        ([23], 23),
        ([24], 24),
        ([25, 75, 91], 25),
        ([29], 29),
        ([30], 30),
        ([31], 31),
        ([32], 32),
        ([33], 33),
        ([41, 62], 41),
        ([46], 46),
        ([47], 47),
        ([35, 48], 48),
        ([49], 49),
    ])


def build_lulc_col11_nivel4() -> ee.Image:
    return build_lulc_col11()


def build_def_seco_col11() -> ee.Image:
    return ee.Image(MAPBIOMAS_LULC11_DEFSEC).unmask(0).int8()


def only_coverage(class_list: List[int], image: ee.Image, landcover: ee.Image) -> ee.Image:
    container = image.multiply(0)
    for cls in class_list:
        container = container.where(landcover.eq(cls).selfMask(), image)
    return ee.Image(container).selfMask()



def _make_eq_chain(img, codes):
    c = img.eq(codes[0])
    for cd in codes[1:]:
        c = c.Or(img.eq(cd))
    return c


def _decode_coverage_nivel0(asset):
    img = ee.Image(asset).unmask(0).int8()
    natural = _make_eq_chain(img, [1,3,4,5,6,10,11,12,13,26,29,32,33,49,50])
    antropico = _make_eq_chain(img, [9,14,15,18,19,20,21,22,23,24,25,30,31,35,36,39,40,41,46,47,48,62,75])
    nao_obs = _make_eq_chain(img, [0,27])
    result = img.multiply(0)
    result = result.where(natural, 1)
    result = result.where(antropico, 14)
    result = result.where(nao_obs, 27)
    return result.unmask(27).int8()


def _decode_coverage_nivel1(asset):
    img = ee.Image(asset).unmask(0).int8()
    floresta = _make_eq_chain(img, [1,3,4,5,6,49])
    veg_herb = _make_eq_chain(img, [10,11,12,29,32,50])
    agropec = _make_eq_chain(img, [9,14,15,18,19,20,21,35,36,39,40,41,46,47,48,62])
    nao_veg = _make_eq_chain(img, [22,23,24,25,30,75])
    agua = _make_eq_chain(img, [26,31,33])
    nao_obs = _make_eq_chain(img, [0,27])
    result = img.multiply(0)
    result = result.where(floresta, 1)
    result = result.where(veg_herb, 10)
    result = result.where(agropec, 14)
    result = result.where(nao_veg, 22)
    result = result.where(agua, 26)
    result = result.where(nao_obs, 27)
    return result.unmask(27).int8()


def _decode_coverage_nivel1_1(asset):
    img = ee.Image(asset).unmask(0).int8()
    form_florestal = _make_eq_chain(img, [1,3,5,6])
    form_savanica = _make_eq_chain(img, [4,49])
    form_campestre = _make_eq_chain(img, [10,12,32,50])
    c_alagado = _make_eq_chain(img, [11])
    pastagem = _make_eq_chain(img, [14,15])
    agricultura = _make_eq_chain(img, [18,19,20,35,36,39,40,41,46,47,48,62])
    silvicultura = _make_eq_chain(img, [9])
    mosaico = _make_eq_chain(img, [21])
    nao_obs = _make_eq_chain(img, [0,27])
    result = img.multiply(0)
    result = result.where(form_florestal, 3)
    result = result.where(form_savanica, 4)
    result = result.where(form_campestre, 12)
    result = result.where(c_alagado, 11)
    result = result.where(pastagem, 15)
    result = result.where(agricultura, 18)
    result = result.where(silvicultura, 9)
    result = result.where(mosaico, 21)
    result = result.where(nao_obs, 27)
    return result.unmask(27).int8()


def _decode_coverage_nivel2(asset):
    img = ee.Image(asset).unmask(0).int8()
    agricultura = _make_eq_chain(img, [19,20,39,40,41,62])
    lavoura_perene = _make_eq_chain(img, [46,47,35,48])
    outras_form = _make_eq_chain(img, [13])
    agua = _make_eq_chain(img, [26,33,31])
    nao_obs = _make_eq_chain(img, [0,27])
    result = img.multiply(0)
    result = result.where(agricultura, 18)
    result = result.where(lavoura_perene, 36)
    result = result.where(outras_form, 10)
    result = result.where(agua, 26)
    result = result.where(nao_obs, 27)
    return result.unmask(27).int8()


def _decode_coverage_nivel3(asset):
    img = ee.Image(asset).unmask(0).int8()
    lavoura_temp = _make_eq_chain(img, [20,39,40,62,41])
    lavoura_perene = _make_eq_chain(img, [46,47,35,48])
    outras_form = _make_eq_chain(img, [13])
    agua = _make_eq_chain(img, [26,33,31])
    nao_obs = _make_eq_chain(img, [0,27])
    result = img.multiply(0)
    result = result.where(lavoura_temp, 19)
    result = result.where(lavoura_perene, 36)
    result = result.where(outras_form, 10)
    result = result.where(agua, 26)
    result = result.where(nao_obs, 27)
    return result.unmask(27).int8()


def decode_annual_burned_coverage_nivel0():
    return _decode_coverage_nivel0(ANNUAL_BURNED_COVERAGE)

def decode_annual_burned_coverage_nivel1():
    return _decode_coverage_nivel1(ANNUAL_BURNED_COVERAGE)

def decode_annual_burned_coverage_nivel1_1():
    return _decode_coverage_nivel1_1(ANNUAL_BURNED_COVERAGE)

def decode_annual_burned_coverage_nivel2():
    return _decode_coverage_nivel2(ANNUAL_BURNED_COVERAGE)

def decode_annual_burned_coverage_nivel3():
    return _decode_coverage_nivel3(ANNUAL_BURNED_COVERAGE)


def decode_accumulated_burned_coverage_nivel0():
    return _decode_coverage_nivel0(ACCUM_BURNED_COVERAGE)

def decode_accumulated_burned_coverage_nivel1():
    return _decode_coverage_nivel1(ACCUM_BURNED_COVERAGE)

def decode_accumulated_burned_coverage_nivel1_1():
    return _decode_coverage_nivel1_1(ACCUM_BURNED_COVERAGE)

def decode_accumulated_burned_coverage_nivel2():
    return _decode_coverage_nivel2(ACCUM_BURNED_COVERAGE)

def decode_accumulated_burned_coverage_nivel3():
    return _decode_coverage_nivel3(ACCUM_BURNED_COVERAGE)


DRC_LULC = "projects/mapbiomas-public/assets/drc/lulc/collection1/mapbiomas_drc_collection1_coverage_v1"


def _decode_drc_lulc_nivel0(asset):
    img = ee.Image(asset).unmask(0).int8()
    naturel = _make_eq_chain(img, [3, 4, 5, 66, 33])
    anthropique = _make_eq_chain(img, [9, 21, 24])
    non_defini = _make_eq_chain(img, [25, 27])
    result = img.multiply(0)
    result = result.where(naturel, 1)
    result = result.where(anthropique, 14)
    result = result.where(non_defini, 27)
    return result.unmask(27).int8()


def _decode_drc_lulc_nivel1(asset):
    img = ee.Image(asset).unmask(0).int8()
    foret = _make_eq_chain(img, [3, 4, 5])
    veg_herb = _make_eq_chain(img, [66])
    agropec = _make_eq_chain(img, [9, 21])
    nao_veg = _make_eq_chain(img, [24, 25])
    eau = _make_eq_chain(img, [33])
    nao_obs = _make_eq_chain(img, [27])
    result = img.multiply(0)
    result = result.where(foret, 1)
    result = result.where(veg_herb, 10)
    result = result.where(agropec, 14)
    result = result.where(nao_veg, 22)
    result = result.where(eau, 26)
    result = result.where(nao_obs, 27)
    return result.unmask(27).int8()


def decode_drc_lulc_nivel0():
    return _decode_drc_lulc_nivel0(DRC_LULC)


def decode_drc_lulc_nivel1():
    return _decode_drc_lulc_nivel1(DRC_LULC)


def decode_drc_lulc_nivel2():
    return ee.Image(DRC_LULC).unmask(0).int8()


# ---------------------------------------------------------------------------
# Peru — Cobertura queimada e cobertura/uso (Collection 2 / Collection 4)
# ---------------------------------------------------------------------------
PERU_BURNED_COVERAGE = "projects/mapbiomas-public/assets/peru/fire/collection2/mapbiomas_peru_fire_collection2_annual_burned_coverage_v1"
PERU_LULC = "projects/mapbiomas-public/assets/peru/collection4/mapbiomas_peru_collection4_coverage_v1"

# nivel 0 — Natural / Antrópico
PERU_N0_NATURAL = [1, 3, 4, 5, 6, 10, 11, 12, 13, 51, 66, 70, 82,
                   23, 52, 61, 68, 92, 26, 33, 34, 29, 32]
PERU_N0_ANTROPICO = [14, 9, 15, 21, 35, 40, 22, 24, 25, 30, 31, 18, 72]

# nivel 1 — 5 grupos
PERU_N1_BOSQUE = [1, 3, 4, 5, 6]
PERU_N1_NAT_NAO_BOSQUE = [10, 11, 12, 13, 51, 66, 70, 82]
PERU_N1_AGRO = [14, 9, 15, 21, 35, 40, 18, 72]
PERU_N1_SEM_VEG = [22, 23, 24, 25, 30, 52, 61, 68, 92, 29, 32]
PERU_N1_AGUA = [26, 31, 33, 34]

PERU_NO_OBSERVADO = [0, 27]


def _decode_peru_nivel0(asset):
    img = ee.Image(asset).unmask(0).int8()
    result = img.multiply(0)
    result = result.where(_make_eq_chain(img, PERU_N0_NATURAL), 1)
    result = result.where(_make_eq_chain(img, PERU_N0_ANTROPICO), 14)
    return result.unmask(0).int8()


def _decode_peru_nivel1(asset):
    img = ee.Image(asset).unmask(0).int8()
    result = img.multiply(0)
    result = result.where(_make_eq_chain(img, PERU_N1_BOSQUE), 3)
    result = result.where(_make_eq_chain(img, PERU_N1_NAT_NAO_BOSQUE), 10)
    result = result.where(_make_eq_chain(img, PERU_N1_AGRO), 14)
    result = result.where(_make_eq_chain(img, PERU_N1_SEM_VEG), 22)
    result = result.where(_make_eq_chain(img, PERU_N1_AGUA), 33)
    result = result.where(_make_eq_chain(img, PERU_NO_OBSERVADO), 0)
    return result.unmask(0).int8()


def _decode_peru_nivel2(asset):
    img = ee.Image(asset).unmask(0).int8()
    result = img
    result = result.where(_make_eq_chain(img, [51, 82]), 11)
    result = result.where(_make_eq_chain(img, [18, 35, 40, 72]), 18)
    result = result.where(_make_eq_chain(img, [29]), 92)
    result = result.where(_make_eq_chain(img, [32]), 52)
    result = result.where(_make_eq_chain(img, PERU_NO_OBSERVADO), 0)
    return result.unmask(0).int8()


def _decode_peru_nivel3(asset):
    img = ee.Image(asset).unmask(0).int8()
    result = img
    result = result.where(_make_eq_chain(img, [29]), 92)
    result = result.where(_make_eq_chain(img, [32]), 52)
    result = result.where(_make_eq_chain(img, PERU_NO_OBSERVADO), 0)
    return result.unmask(0).int8()


def decode_peru_burned_nivel0():
    return _decode_peru_nivel0(PERU_BURNED_COVERAGE)


def decode_peru_burned_nivel1():
    return _decode_peru_nivel1(PERU_BURNED_COVERAGE)


def decode_peru_burned_nivel2():
    return _decode_peru_nivel2(PERU_BURNED_COVERAGE)


def decode_peru_burned_nivel3():
    return _decode_peru_nivel3(PERU_BURNED_COVERAGE)


def decode_peru_lulc_nivel0():
    return _decode_peru_nivel0(PERU_LULC)


def decode_peru_lulc_nivel1():
    return _decode_peru_nivel1(PERU_LULC)


def decode_peru_lulc_nivel2():
    return _decode_peru_nivel2(PERU_LULC)


def decode_peru_lulc_nivel3():
    return _decode_peru_nivel3(PERU_LULC)


PROCESSOR_REGISTRY = {
    "build_edge_area": build_edge_area,
    "build_fragment_size": build_fragment_size,
    "build_distance_100ha": build_distance_100ha,
    "build_distance_500ha": build_distance_500ha,
    "build_distance_1000ha": build_distance_1000ha,
    "build_secondary_vegetation_coverage": build_secondary_vegetation_coverage,
    "build_secondary_vegetation_age": build_secondary_vegetation_age,
    "build_fire_frequency": build_fire_frequency,
    "build_fire_age": build_fire_age,
    "build_accumulated_burned_coverage": build_accumulated_burned_coverage,
    "decode_fire_frequency_col101": decode_fire_frequency_col101,
    "decode_fire_age_col101": decode_fire_age_col101,
    "decode_secondary_vegetation_age_col101": decode_secondary_vegetation_age_col101,
    "build_secondary_vegetation_coverage_col101": build_secondary_vegetation_coverage_col101,
    "build_fire_frequency_coverage_col101": build_fire_frequency_coverage_col101,
    "build_burned_natural_coverage_col101": build_burned_natural_coverage_col101,
    "build_burned_at_least_once_col101": build_burned_at_least_once_col101,
    "build_primary_natural_coverage_col101": build_primary_natural_coverage_col101,
    "build_natural_coverage_col101": build_natural_coverage_col101,
    "decode_edge_area_col101": decode_edge_area_col101,
    "decode_edge_age_col101": decode_edge_age_col101,
    "decode_morphology_col101": decode_morphology_col101,
    "decode_patch_id_col101": decode_patch_id_col101,
    "decode_patch_size_col101": decode_patch_size_col101,
    "decode_patch_size_fragments_col101": decode_patch_size_fragments_col101,
    "decode_patch_size_massifs_col101": decode_patch_size_massifs_col101,
    "decode_canopy_disturbance_col101": decode_canopy_disturbance_col101,
    "decode_logging_col101": decode_logging_col101,
    "decode_annual_burned_coverage_nivel0": decode_annual_burned_coverage_nivel0,
    "decode_annual_burned_coverage_nivel1": decode_annual_burned_coverage_nivel1,
    "decode_annual_burned_coverage_nivel1_1": decode_annual_burned_coverage_nivel1_1,
    "decode_annual_burned_coverage_nivel2": decode_annual_burned_coverage_nivel2,
    "decode_annual_burned_coverage_nivel3": decode_annual_burned_coverage_nivel3,
    "decode_accumulated_burned_coverage_nivel0": decode_accumulated_burned_coverage_nivel0,
    "decode_accumulated_burned_coverage_nivel1": decode_accumulated_burned_coverage_nivel1,
    "decode_accumulated_burned_coverage_nivel1_1": decode_accumulated_burned_coverage_nivel1_1,
    "decode_accumulated_burned_coverage_nivel2": decode_accumulated_burned_coverage_nivel2,
    "decode_accumulated_burned_coverage_nivel3": decode_accumulated_burned_coverage_nivel3,

    "decode_monthly_burned_col5": decode_monthly_burned_col5,
    "decode_unprecedented_fire": decode_unprecedented_fire,
    "build_severity_col5": build_severity_col5,
    "decode_return_interval_discrete": decode_return_interval_discrete,
    "decode_mean_return_interval_discrete": decode_mean_return_interval_discrete,
    "build_nbr_min_mosaic": build_nbr_min_mosaic,
    "build_landsat_false_color_swir": build_landsat_false_color_swir,
    "build_landsat_natural_rgb": build_landsat_natural_rgb,
    "build_lulc_col11": build_lulc_col11,
    "decode_edge_area_1000m_col11": decode_edge_area_1000m_col11,
    "build_lulc_col11_nivel0": build_lulc_col11_nivel0,
    "build_lulc_col11_nivel1": build_lulc_col11_nivel1,
    "build_lulc_col11_nivel1_1": build_lulc_col11_nivel1_1,
    "build_lulc_col11_nivel2": build_lulc_col11_nivel2,
    "build_lulc_col11_nivel3": build_lulc_col11_nivel3,
    "build_lulc_col11_nivel4": build_lulc_col11_nivel4,
    "build_def_seco_col11": build_def_seco_col11,
    "decode_drc_lulc_nivel0": decode_drc_lulc_nivel0,
    "decode_drc_lulc_nivel1": decode_drc_lulc_nivel1,
    "decode_drc_lulc_nivel2": decode_drc_lulc_nivel2,
    "decode_peru_burned_nivel0": decode_peru_burned_nivel0,
    "decode_peru_burned_nivel1": decode_peru_burned_nivel1,
    "decode_peru_burned_nivel2": decode_peru_burned_nivel2,
    "decode_peru_burned_nivel3": decode_peru_burned_nivel3,
    "decode_peru_lulc_nivel0": decode_peru_lulc_nivel0,
    "decode_peru_lulc_nivel1": decode_peru_lulc_nivel1,
    "decode_peru_lulc_nivel2": decode_peru_lulc_nivel2,
    "decode_peru_lulc_nivel3": decode_peru_lulc_nivel3,
}


def run_processor(processor_name: str, **kwargs) -> ee.Image:
    if processor_name not in PROCESSOR_REGISTRY:
        raise KeyError(f"Processador '{processor_name}' não encontrado. Disponíveis: {list(PROCESSOR_REGISTRY.keys())}")
    return PROCESSOR_REGISTRY[processor_name](**kwargs)
