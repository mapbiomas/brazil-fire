"""Configuracao multipais/colecoes/produtos do pipeline Export & Vectorization.

Fonte de verdade: `OBJ` — arvore  pais -> tema -> colecao -> [produtos].

    OBJ = {
        "brasil": {
            "fire": {
                "monitor": [
                    {"product": "monthly_burned",
                     "assetid": "projects/mapbiomas-public/assets/brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1",
                     "type": "byte", "vectorize": True, "visible": True},
                ],
            },
        },
        ...
    }

- `product`: nome curto (vira a pasta GCS).
- `assetid`: asset GEE de origem (projetos podem variar por pais).
- `type`: dtype de salvamento no mosaico (byte | int16 | float32).
- `vectorize`: se o produto gera vetorizacao/upload GEE.
- `visible`: ocultar sem apagar.

Seletores ativos: COUNTRY / THEME / COLLECTION / PRODUCT (lidos em tempo de
chamada -> trocar sempre propaga para os modulos).

Padrao GCS:  {bucket}/initiatives/{country}/{theme}/{collection}/{product}
Vetores:     pasta irma do raster vetorizado, com sufixo _vectors (GCS e GEE);
             ex.: .../monitor/mapbiomas_fire_monthly_burned_v1_vectors/
Assets GEE:  {assetid} (origem)
"""

BUCKET = "mapbiomas-fire"
PUBLIC_BUCKET = "mapbiomas-public"
BUCKET_PATH = "initiatives"

GEE_PROJECT = "mapbiomas-fire-485203"

STATE_FILE = "monitor_state.json"

SCALE = 30

# Timeout (s) do scan de status (Load Data / Load Collection). Se o scan nao
# terminar no prazo, a UI mostra o status parcial e destrava os botoes.
SCAN_TIMEOUT = 180

# Endereco alternativo de saida dos vetores (GEE asset folder / GCS folder).
# None = padrao (pasta irma do raster com sufixo _vectors). Se definido, o
# template e renderizado substituindo placeholders:
#   {country} (codigo OBJ, ex.: brasil) | {COUNTRY} (storage GCS/GEE, ex.: brazil)
#   {theme} | {collection} (chave OBJ, ex.: collection_09)
#   {COLLECTION} (nome GEE normalizado, ex.: collection9)
#   {product} | {raster} (basename do assetid) | {vectors_folder}
# Exemplos (GEE asset folder):
#   VECTOR_OUTPUT_GEE = "projects/mapbiomas-{COUNTRY}/assets/FIRE/COLLECTION2/FINAL_PRODUCTS"
#   VECTOR_OUTPUT_GEE = "projects/mapbiomas-{COUNTRY}/assets/FIRE/CATALOG_01/{COLLECTION}/FINAL_PRODUCTS"
# Exemplo (GCS folder):
#   VECTOR_OUTPUT_GCS = "initiatives/{COUNTRY}/{theme}/{collection}/FINAL_PRODUCTS"
VECTOR_OUTPUT_GEE = None
VECTOR_OUTPUT_GCS = None

# Verbose logging: quando True, o log drawer mostra linhas por-unidade
# ([FOUND], [SKIP], [DEBUG]). Default False: apenas sumarios por lote.
LOG_VERBOSE = False


def _p(country, theme, collection, product, assetid, ptype, vectorize=False, visible=True,
       scale=30, decode=None):
    return {"product": product, "assetid": assetid, "type": ptype,
            "vectorize": vectorize, "visible": visible, "scale": scale, "decode": decode}


def _gee_collection_name(collection):
    """Normaliza nome da coleção para paths GEE (mapbiomas-public).
    collection_02 -> collection2, collection_03_1 -> collection3_1, collection_00 -> collection (beta).
    Mantém a chave original no OBJ/UI."""
    if not collection.startswith("collection_"):
        return collection
    rest = collection[len("collection_"):]
    # Handle collection_00 (beta) -> collection
    if rest == "00":
        return "collection"
    if rest.startswith("00_"):
        return "collection" + rest[2:].replace("_", "_")
    # Handle collection_0X -> collectionX, collection_0X_Y -> collectionX_Y
    if rest[0] == "0" and len(rest) > 1 and rest[1].isdigit():
        return "collection" + rest[1:].replace("_", "_")
    # Already normalized like collection11, collection4_1
    return "collection" + rest


OBJ = {
    "brasil": {
        "fire": {
            "monitor": [
                _p("brasil", "fire", "monitor", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1",
                   "byte",vectorize=True),
            ],
            "collection_05_1": [
                _p("brasil", "fire", "collection5_1", "annual_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_annual_burned_v1", "byte", vectorize=True),
                _p("brasil", "fire", "collection5_1", "annual_burned_area_ha",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_annual_burned_area_ha_v1", "float32"),
                _p("brasil", "fire", "collection5_1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection5_1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_annual_burned_scar_size_range_v1", "float32"),
                _p("brasil", "fire", "collection5_1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_accumulated_burned_v1", "byte"),
                _p("brasil", "fire", "collection5_1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_accumulated_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection5_1", "fire_frequency",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_fire_frequency_v1", "byte"),
                _p("brasil", "fire", "collection5_1", "fire_frequency_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_fire_frequency_coverage_v1", "byte"),
                _p("brasil", "fire", "collection5_1", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_monthly_burned_v1", "byte"),
                _p("brasil", "fire", "collection5_1", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_time_after_fire_v1", "byte"),
                _p("brasil", "fire", "collection5_1", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_year_last_fire_v1", "int16"),
                _p("brasil", "fire", "collection5_1", "severity_class",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_severity_class_v1", "byte"),
                _p("brasil", "fire", "collection5_1", "interval_since_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5_1/mapbiomas_fire_collection51_interval_since_fire_v1", "byte"),
            ],
            "collection_05": [
                _p("brasil", "fire", "collection5", "annual_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_v1", "byte", vectorize=True),
                _p("brasil", "fire", "collection5", "annual_burned_area_ha",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_area_ha_v1", "float32"),
                _p("brasil", "fire", "collection5", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection5", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_scar_size_range_v1", "float32"),
                _p("brasil", "fire", "collection5", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_monthly_burned_v1", "byte"),
                _p("brasil", "fire", "collection5", "accumulated_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_accumulated_burned_v1", "byte"),
                _p("brasil", "fire", "collection5", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_accumulated_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection5", "fire_frequency",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_fire_frequency_v1", "byte"),
                _p("brasil", "fire", "collection5", "fire_frequency_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_fire_frequency_coverage_v1", "byte"),
                _p("brasil", "fire", "collection5", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_time_after_fire_v1", "byte"),
                _p("brasil", "fire", "collection5", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_year_last_fire_v1", "int16"),
                _p("brasil", "fire", "collection5", "severity_class",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_severity_class_v1", "byte"),
                _p("brasil", "fire", "collection5", "interval_since_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_interval_since_fire_v1", "byte"),
            ],
            "collection_04_1": [
                _p("brasil", "fire", "collection4_1", "annual_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_v1", "byte", vectorize=True),
                _p("brasil", "fire", "collection4_1", "annual_burned_area_ha",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_area_ha_v1", "float32"),
                _p("brasil", "fire", "collection4_1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection4_1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_scar_size_range_v1", "float32"),
                _p("brasil", "fire", "collection4_1", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_monthly_burned_v1", "byte"),
                _p("brasil", "fire", "collection4_1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_accumulated_burned_v1", "byte"),
                _p("brasil", "fire", "collection4_1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_accumulated_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection4_1", "fire_frequency",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_fire_frequency_v1", "byte"),
                _p("brasil", "fire", "collection4_1", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_time_after_fire_v1", "byte"),
                _p("brasil", "fire", "collection4_1", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_year_last_fire_v1", "int16"),
            ],
            "collection_04": [
                _p("brasil", "fire", "collection4", "annual_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_v1", "byte", vectorize=True),
                _p("brasil", "fire", "collection4", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection4", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_scar_size_range_v1", "float32"),
                _p("brasil", "fire", "collection4", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_monthly_burned_v1", "byte"),
                _p("brasil", "fire", "collection4", "monthly_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_monthly_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection4", "accumulated_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_accumulated_burned_v1", "byte"),
                _p("brasil", "fire", "collection4", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_accumulated_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection4", "fire_frequency",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_fire_frequency_v1", "byte"),
                _p("brasil", "fire", "collection4", "fire_frequency_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_fire_frequency_coverage_v1", "byte"),
                _p("brasil", "fire", "collection4", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_time_after_fire_v1", "byte"),
                _p("brasil", "fire", "collection4", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_year_last_fire_v1", "int16"),
            ],
            "collection_03_1": [
                _p("brasil", "fire", "collection3_1", "annual_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_annual_burned_v1", "byte", vectorize=True),
                _p("brasil", "fire", "collection3_1", "annual_burned_area_ha",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_annual_burned_area_ha_v1", "float32"),
                _p("brasil", "fire", "collection3_1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3_1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_annual_burned_scar_size_range_v1", "float32"),
                _p("brasil", "fire", "collection3_1", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_monthly_burned_v1", "byte"),
                _p("brasil", "fire", "collection3_1", "monthly_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_monthly_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3_1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_accumulated_burned_v1", "byte"),
                _p("brasil", "fire", "collection3_1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_accumulated_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3_1", "fire_recurrence",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_fire_recurrence_v1", "byte"),
                _p("brasil", "fire", "collection3_1", "fire_recurrence_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_fire_recurrence_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3_1", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_time_after_fire_v1", "byte"),
                _p("brasil", "fire", "collection3_1", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3_1/mapbiomas_fire_collection31_year_last_fire_v1", "int16"),
            ],
            "collection_03": [
                _p("brasil", "fire", "collection3", "annual_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_annual_burned_v1", "byte", vectorize=True),
                _p("brasil", "fire", "collection3", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_annual_burned_scar_size_range_v1", "byte"),
                _p("brasil", "fire", "collection3", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_monthly_burned_v1", "byte"),
                _p("brasil", "fire", "collection3", "monthly_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_monthly_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3", "accumulated_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_accumulated_burned_v1", "byte"),
                _p("brasil", "fire", "collection3", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_accumulated_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3", "fire_frequency",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_fire_frequency_v1", "byte"),
                _p("brasil", "fire", "collection3", "fire_frequency_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_fire_frequency_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_time_after_fire_v1", "byte"),
                _p("brasil", "fire", "collection3", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection3_year_last_fire_v1", "int16"),
            ],
            "collection_02_1": [
                _p("brasil", "fire", "collection2_1", "annual_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection2_1/mapbiomas_fire_collection21_annual_burned_v1", "byte", vectorize=True),
                _p("brasil", "fire", "collection2_1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection2_1/mapbiomas_fire_collection21_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection2_1", "monthly_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection2_1/mapbiomas_fire_collection21_monthly_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection2_1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection2_1/mapbiomas_fire_collection21_accumulated_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection2_1", "fire_frequency_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection2_1/mapbiomas_fire_collection21_fire_frequency_coverage_v1", "byte"),
            ],
            "collection_02": [
                _p("brasil", "fire", "collection2", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection2/mapbiomas_fire_collection2_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection2", "monthly_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection2/mapbiomas_fire_collection2_monthly_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection2", "fire_frequency_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection2/mapbiomas_fire_collection2_fire_frequency_coverage_v1", "byte"),
            ],
            "collection_01_1": [
                _p("brasil", "fire", "collection1_1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection1_1/mapbiomas_fire_collection11_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection1_1", "monthly_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection1_1/mapbiomas_fire_collection11_monthly_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection1_1", "fire_frequency_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection1_1/mapbiomas_fire_collection11_fire_frequency_coverage_v1", "byte"),
            ],
            "collection_01": [
                _p("brasil", "fire", "collection1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection1/mapbiomas_fire_collection1_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection1", "monthly_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection1/mapbiomas_fire_collection1_monthly_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection1", "fire_frequency_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection1/mapbiomas_fire_collection1_fire_frequency_coverage_v1", "byte"),
            ],
            "beta": [
                _p("brasil", "fire", "beta", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/beta/mapbiomas_fire_beta_annual_burned_coverage_v2", "byte"),
                _p("brasil", "fire", "beta", "fire_frequency_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/beta/mapbiomas_fire_beta_fire_frequency_coverage_v2", "byte"),
            ],

        },
        "lulc": {
            "collection_11": [
                _p("brasil", "lulc", "collection11", "agriculture_irrigation_systems", "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_agriculture_irrigation_systems_v1", "byte"),
                _p("brasil", "lulc", "collection11", "agriculture_number_cycles_mean", "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_agriculture_number_cycles_mean_v1", "byte"),
                _p("brasil", "lulc", "collection11", "agriculture_number_cycles", "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_agriculture_number_cycles_v1", "byte"),
                _p("brasil", "lulc", "collection11", "agriculture_second_crop", "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_agriculture_second_crop_v1", "byte"),
                _p("brasil", "lulc", "collection11", "coverage", "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_coverage_v3", "byte"),
                _p("brasil", "lulc", "collection11", "deforestation_secondary_vegetation", "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_deforestation_secondary_vegetation_v5", "byte"),
                _p("brasil", "lulc", "collection11", "mining_substances", "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_mining_substances_v1", "byte"),
                _p("brasil", "lulc", "collection11", "pasture_age", "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_pasture_age_v1", "byte"),
                _p("brasil", "lulc", "collection11", "pasture_vigor", "projects/mapbiomas-public/assets/brazil/lulc/collection11/mapbiomas_brazil_collection11_pasture_vigor_v1", "byte"),
            ],
            "collection_10": [
                _p("brasil", "lulc", "collection10", "agriculture_irrigation_systems",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_agriculture_irrigation_systems_v3", "byte"),
                _p("brasil", "lulc", "collection10", "agriculture_number_cycles",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_agriculture_number_cycles_v2", "byte", scale=10),
                _p("brasil", "lulc", "collection10", "agriculture_number_cycles_mean",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_agriculture_number_cycles_mean_v2", "float32", scale=10),
                _p("brasil", "lulc", "collection10", "agriculture_second_crop",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_agriculture_second_crop_v1", "byte"),
                _p("brasil", "lulc", "collection10", "aspect",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_aspect_v2", "byte"),
                _p("brasil", "lulc", "collection10", "coverage",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_coverage_v2", "byte"),
                _p("brasil", "lulc", "collection10", "deforestation_secondary_vegetation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_deforestation_secondary_vegetation_v2", "byte"),
                _p("brasil", "lulc", "collection10", "degradation_edge_size",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_degradation_edge_size_v1", "byte"),
                _p("brasil", "lulc", "collection10", "degradation_isolation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_degradation_isolation_v1", "byte", scale=100),
                _p("brasil", "lulc", "collection10", "degradation_patch_size",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_degradation_patch_size_v1", "byte"),
                _p("brasil", "lulc", "collection10", "geomorphology_ibge",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_geomorphology_ibge_v2", "byte"),
                _p("brasil", "lulc", "collection10", "hipsometry",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_hipsometry_v2", "byte"),
                _p("brasil", "lulc", "collection10", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_integration_v2", "byte"),
                _p("brasil", "lulc", "collection10", "mining_annual",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_mining_annual_v1", "float32"),
                _p("brasil", "lulc", "collection10", "mining_substances",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_mining_substances_v6", "float32"),
                _p("brasil", "lulc", "collection10", "pasture",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_pasture_v1", "byte"),
                _p("brasil", "lulc", "collection10", "pasture_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_pasture_age_v2", "int16"),
                _p("brasil", "lulc", "collection10", "pasture_biomass",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_pasture_biomass_v2", "float32"),
                _p("brasil", "lulc", "collection10", "pasture_vigor",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_pasture_vigor_v3", "byte"),
                _p("brasil", "lulc", "collection10", "pedology_ibge",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_pedology_ibge_v2", "byte"),
                _p("brasil", "lulc", "collection10", "reefs",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_reefs_v2", "byte"),
                _p("brasil", "lulc", "collection10", "slope",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_slope_v2", "byte"),
                _p("brasil", "lulc", "collection10", "urban_hand",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_urban_hand_v1", "byte"),
                _p("brasil", "lulc", "collection10", "urban_periods",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_urban_periods_v1", "byte"),
                _p("brasil", "lulc", "collection10", "urban_risk",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_urban_risk_v1", "byte"),
                _p("brasil", "lulc", "collection10", "urban_slope",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_urban_slope_v1", "byte"),
                _p("brasil", "lulc", "collection10", "urban_slum",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_urban_slum_v1", "byte"),
                _p("brasil", "lulc", "collection10", "urban_tracts",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_urban_tracts_v1", "byte"),
                _p("brasil", "lulc", "collection10", "vegetation_ibge",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10/mapbiomas_brazil_collection10_vegetation_ibge_v4", "byte"),
            ],
            "collection_10_1": [
                _p("brasil", "lulc", "collection10_1", "coverage",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_coverage_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "deforestation_secondary_vegetation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_deforestation_secondary_vegetation_v3", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_canopy_disturbance_frequency",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_canopy_disturbance_frequency_v2", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_edge_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_edge_age_v1", "int16"),
                _p("brasil", "lulc", "collection10_1", "degradation_edge_area",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_edge_area_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_fire_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_fire_age_v1", "int16"),
                _p("brasil", "lulc", "collection10_1", "degradation_fire_frequency",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_fire_frequency_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_isolation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_isolation_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_isolation_1000ha",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_isolation_1000ha_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_isolation_100ha",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_isolation_100ha_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_isolation_500ha",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_isolation_500ha_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_landscape_morphology",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_landscape_morphology_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_logging",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_logging_v2", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_native_vegetation_mask",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_native_vegetation_mask_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_patch_id",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_patch_id_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_patch_size",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_patch_size_v1", "byte"),
                _p("brasil", "lulc", "collection10_1", "degradation_secondary_vegetation_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_degradation_secondary_vegetation_age_v1", "int16"),
                _p("brasil", "lulc", "collection10_1", "urban_heat_island_classification",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_urban_heat_island_classification_v1", "float32"),
                _p("brasil", "lulc", "collection10_1", "urban_heat_island_intensity",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection10_1/mapbiomas_brazil_collection10_1_urban_heat_island_intensity_v1", "float32"),
            ],
            "collection_02_3": [
                _p("brasil", "lulc", "collection2_3", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection2_3/mapbiomas_collection23_integration_v1", "byte"),
                _p("brasil", "lulc", "collection2_3", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection2_3/mapbiomas_collection23_transitions_v1", "byte"),
            ],
            "collection_03": [
                _p("brasil", "lulc", "collection3", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection3/mapbiomas_collection30_integration_v1", "byte"),
                _p("brasil", "lulc", "collection3", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection3/mapbiomas_collection30_transitions_v1", "byte"),
            ],
            "collection_03_1": [
                _p("brasil", "lulc", "collection3_1", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection3_1/mapbiomas_collection31_integration_v1", "byte"),
                _p("brasil", "lulc", "collection3_1", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection3_1/mapbiomas_collection31_transitions_v1", "byte"),
            ],
            "collection_04": [
                _p("brasil", "lulc", "collection4", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection4/mapbiomas_collection40_integration_v1", "byte"),
                _p("brasil", "lulc", "collection4", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection4/mapbiomas_collection40_transitions_v3", "byte"),
            ],
            "collection_04_1": [
                _p("brasil", "lulc", "collection4_1", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection4_1/mapbiomas_collection41_integration_v1", "byte"),
                _p("brasil", "lulc", "collection4_1", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection4_1/mapbiomas_collection41_transitions_v1", "byte"),
            ],
            "collection_05": [
                _p("brasil", "lulc", "collection5", "burned_cover",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_burned_cover_v1", "byte"),
                _p("brasil", "lulc", "collection5", "burned_cover_cumulated",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_burned_cover_cumulated_v1", "byte"),
                _p("brasil", "lulc", "collection5", "deforestation_primary_vegetation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_deforestation_primary_vegetation_v1", "byte"),
                _p("brasil", "lulc", "collection5", "deforestation_regeneration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_deforestation_regeneration_v1", "byte"),
                _p("brasil", "lulc", "collection5", "deforestation_secondary_vegetation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_deforestation_secondary_vegetation_v1", "byte"),
                _p("brasil", "lulc", "collection5", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_integration_v1", "byte"),
                _p("brasil", "lulc", "collection5", "irrigated_agriculture",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_irrigated_agriculture_v1", "byte"),
                _p("brasil", "lulc", "collection5", "pasture_quality",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_pasture_quality_v1", "byte"),
                _p("brasil", "lulc", "collection5", "quality",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_quality_v1", "byte"),
                _p("brasil", "lulc", "collection5", "secondary_vegetation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_secondary_vegetation_v1", "byte"),
                _p("brasil", "lulc", "collection5", "secondary_vegetation_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection5/mapbiomas_collection50_secondary_vegetation_age_v1", "int16"),
            ],
            "collection_06": [
                _p("brasil", "lulc", "collection6", "deforestation_frequency",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_deforestation_frequency_v2", "byte"),
                _p("brasil", "lulc", "collection6", "deforestation_regeneration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_deforestation_regeneration_v1", "byte"),
                _p("brasil", "lulc", "collection6", "deforestation_secondary_vegetation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_deforestation_secondary_vegetation_v1", "byte"),
                _p("brasil", "lulc", "collection6", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_integration_v1", "byte"),
                _p("brasil", "lulc", "collection6", "irrigated_agriculture",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_irrigated_agriculture_v1", "byte"),
                _p("brasil", "lulc", "collection6", "mined_substance",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_mined_substance_v1", "byte"),
                _p("brasil", "lulc", "collection6", "pasture_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_pasture_age_v1", "int16"),
                _p("brasil", "lulc", "collection6", "pasture_quality",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_pasture_quality_v1", "byte"),
                _p("brasil", "lulc", "collection6", "quality",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection80_quality_v1", "byte"),
                _p("brasil", "lulc", "collection6", "secondary_vegetation_accumulated",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_secondary_vegetation_accumulated_v1", "byte"),
                _p("brasil", "lulc", "collection6", "secondary_vegetation_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_secondary_vegetation_age_v2", "int16"),
                _p("brasil", "lulc", "collection6", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection6/mapbiomas_collection60_transitions_v2", "byte"),
            ],
            "collection_07": [
                _p("brasil", "lulc", "collection7", "deforestation_frequency",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7/mapbiomas_collection70_deforestation_frequency_v2", "byte"),
                _p("brasil", "lulc", "collection7", "deforestation_regeneration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7/mapbiomas_collection70_deforestation_regeneration_v1", "byte"),
                _p("brasil", "lulc", "collection7", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7/mapbiomas_collection70_integration_v2", "byte"),
                _p("brasil", "lulc", "collection7", "irrigated_agriculture",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7/mapbiomas_collection70_irrigated_agriculture_v3", "byte"),
                _p("brasil", "lulc", "collection7", "mined_substance",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7/mapbiomas_collection70_mined_substance_v1", "byte"),
                _p("brasil", "lulc", "collection7", "pasture_quality",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7/mapbiomas_collection70_pasture_quality_v2", "byte"),
                _p("brasil", "lulc", "collection7", "secondary_vegetation_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7/mapbiomas_collection70_secondary_vegetation_age_v2", "int16"),
                _p("brasil", "lulc", "collection7", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7/mapbiomas_collection70_transitions_v3", "byte"),
            ],
            "collection_07_1": [
                _p("brasil", "lulc", "collection7_1", "deforestation_frequency",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7_1/mapbiomas_collection71_deforestation_frequency_v1", "byte"),
                _p("brasil", "lulc", "collection7_1", "deforestation_regeneration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7_1/mapbiomas_collection71_deforestation_regeneration_v1", "byte"),
                _p("brasil", "lulc", "collection7_1", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7_1/mapbiomas_collection71_integration_v1", "byte"),
                _p("brasil", "lulc", "collection7_1", "secondary_vegetation_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7_1/mapbiomas_collection71_secondary_vegetation_age_v1", "int16"),
                _p("brasil", "lulc", "collection7_1", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection7_1/mapbiomas_collection71_transitions_v1", "byte"),
            ],
            "collection_08": [
                _p("brasil", "lulc", "collection8", "deforestation_secondary_vegetation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection8/mapbiomas_collection80_deforestation_secondary_vegetation_v2", "byte"),
                _p("brasil", "lulc", "collection8", "deforestation_secondary_vegetation_accumulated",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection8/mapbiomas_collection80_deforestation_secondary_vegetation_accumulated_v2", "byte"),
                _p("brasil", "lulc", "collection8", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection8/mapbiomas_collection80_integration_v1", "byte"),
                _p("brasil", "lulc", "collection8", "irrigated_agriculture",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection8/mapbiomas_collection80_irrigated_agriculture_v1", "byte"),
                _p("brasil", "lulc", "collection8", "mined_substance",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection8/mapbiomas_collection80_mined_substance_v1", "byte"),
                _p("brasil", "lulc", "collection8", "pasture_quality",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection8/mapbiomas_collection80_pasture_quality_v1", "byte"),
                _p("brasil", "lulc", "collection8", "secondary_vegetation_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection8/mapbiomas_collection80_secondary_vegetation_age_v2", "int16"),
                _p("brasil", "lulc", "collection8", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection8/mapbiomas_collection80_transitions_v1", "byte"),
            ],
            "collection_09": [
                _p("brasil", "lulc", "collection9", "def_sec_veg_accumulated",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_def_sec_veg_accumulated_v1", "byte"),
                _p("brasil", "lulc", "collection9", "deforestation_secondary_vegetation",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_deforestation_secondary_vegetation_v1", "byte"),
                _p("brasil", "lulc", "collection9", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_integration_v1", "byte"),
                _p("brasil", "lulc", "collection9", "irrigated_agriculture",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_irrigated_agriculture_v1", "byte"),
                _p("brasil", "lulc", "collection9", "mined_substance",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_mined_substance_v1", "byte"),
                _p("brasil", "lulc", "collection9", "pasture_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_pasture_age_v1", "int16"),
                _p("brasil", "lulc", "collection9", "pasture_detection_year",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_pasture_detection_year_v1", "int16"),
                _p("brasil", "lulc", "collection9", "pasture_gpp",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_pasture_gpp_v1", "float32"),
                _p("brasil", "lulc", "collection9", "pasture_last_time_mapped",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_pasture_last_time_mapped_v1", "byte"),
                _p("brasil", "lulc", "collection9", "pasture_quality",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_pasture_quality_v1", "byte"),
                _p("brasil", "lulc", "collection9", "pasture_vigor",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_pasture_vigor_v2", "byte"),
                _p("brasil", "lulc", "collection9", "pasture_vigor_transition",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_pasture_vigor_transition_v1", "byte"),
                _p("brasil", "lulc", "collection9", "quality",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_quality_v1", "byte"),
                _p("brasil", "lulc", "collection9", "secondary_vegetation_age",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_secondary_vegetation_age_v1", "int16"),
                _p("brasil", "lulc", "collection9", "transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_transitions_v1", "byte"),
                _p("brasil", "lulc", "collection9", "urban_epochs",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_urban_epochs_v1", "byte"),
                _p("brasil", "lulc", "collection9", "urban_height_above_nearest_drainage",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_urban_height_above_nearest_drainage_v1", "byte"),
                _p("brasil", "lulc", "collection9", "urban_risk",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_urban_risk_v1", "byte"),
                _p("brasil", "lulc", "collection9", "urban_slope",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_urban_slope_v1", "byte"),
                _p("brasil", "lulc", "collection9", "urban_slum",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_urban_slum_v1", "byte"),
                _p("brasil", "lulc", "collection9", "urban_tracts",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_urban_tracts_v1", "byte"),
                _p("brasil", "lulc", "collection9", "urban_transitions",
                   "projects/mapbiomas-public/assets/brazil/lulc/collection9/mapbiomas_collection90_urban_transitions_v1", "byte"),
            ],
        },
        "lulc_10m": {
            "collection_04": [
                _p("brasil", "lulc_10m", "collection4", "coverage",
                   "projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1",
                   "byte", scale=10),
            ],
            "collection_02": [
                _p("brasil", "lulc_10m", "collection2", "agriculture_mean_cycles_2017_2023",
                   "projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_agriculture_mean_cycles_2017_2023_v1", "float32", scale=10),
                _p("brasil", "lulc_10m", "collection2", "agriculture_number_cycles",
                   "projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_agriculture_number_cycles_v1", "byte", scale=10),
                _p("brasil", "lulc_10m", "collection2", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc_10m/collection2/mapbiomas_10m_collection2_integration_v1", "byte", scale=10),
            ],
            "collection_03": [
                _p("brasil", "lulc_10m", "collection3", "integration",
                   "projects/mapbiomas-public/assets/brazil/lulc_10m/collection3/mapbiomas_10m_collection3_integration_v1", "byte", scale=10),
            ],
        },
        "soil": {
            "collection_03": [
                _p("brasil", "soil", "collection3", "soil_carbon", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_carbon_v1", "int16"),
                _p("brasil", "soil", "collection3", "soil_clay_fraction", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_clay_fraction_v1", "byte"),
                _p("brasil", "soil", "collection3", "soil_sand_fraction", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_sand_fraction_v1", "byte"),
                _p("brasil", "soil", "collection3", "soil_silt_fraction", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_silt_fraction_v1", "byte"),
                _p("brasil", "soil", "collection3", "soil_stoniness", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_stoniness_v2", "byte"),
                _p("brasil", "soil", "collection3", "soil_textural_class", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_textural_class_v1", "byte"),
                _p("brasil", "soil", "collection3", "soil_textural_group", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_textural_group_v1", "byte"),
                _p("brasil", "soil", "collection3", "soil_textural_subgroup", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_textural_subgroup_v1", "byte"),
            ],
            "collection_01": [
                _p("brasil", "soil", "collection1", "carbon_coverage",
                   "projects/mapbiomas-public/assets/brazil/soil/collection1/mapbiomas_soil_collection1_carbon_coverage_v1", "int16"),
                _p("brasil", "soil", "collection1", "soil_organic_carbon_0_30cm_kg_m2",
                   "projects/mapbiomas-public/assets/brazil/soil/collection1/mapbiomas_soil_collection1_soil_organic_carbon_0_30cm_kg_m2_v1", "int16"),
                _p("brasil", "soil", "collection1", "soil_organic_carbon_0_30cm_t_ha",
                   "projects/mapbiomas-public/assets/brazil/soil/collection1/mapbiomas_soil_collection1_soil_organic_carbon_0_30cm_t_ha_v1", "int16"),
            ],
            "collection_02": [
                _p("brasil", "soil", "collection2", "granulometry_clay_percentage",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_granulometry_clay_percentage", "byte"),
                _p("brasil", "soil", "collection2", "granulometry_sand_percentage",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_granulometry_sand_percentage", "byte"),
                _p("brasil", "soil", "collection2", "granulometry_silt_percentage",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_granulometry_silt_percentage", "byte"),
                _p("brasil", "soil", "collection2", "soc_kg_m2_000_030cm",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_soc_kg_m2_000_030cm", "int16"),
                _p("brasil", "soil", "collection2", "soc_t_ha_000_030cm",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_soc_t_ha_000_030cm", "int16"),
                _p("brasil", "soil", "collection2", "soc_t_ha_000_030cm_coverage",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_soc_t_ha_000_030cm_coverage", "int16"),
                _p("brasil", "soil", "collection2", "textural_classes",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_textural_classes", "byte"),
                _p("brasil", "soil", "collection2", "textural_groups",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_textural_groups", "byte"),
                _p("brasil", "soil", "collection2", "textural_subgroups",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_textural_subgroups", "byte"),
                _p("brasil", "soil", "collection2", "textural_triangle",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2/mapbiomas_soil_collection2_textural_triangle", "byte"),
            ],
            "collection_02_1": [
                _p("brasil", "soil", "collection2_1", "soil_carbon",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2_1/mapbiomas_brazil_collection21_soil_carbon_v3", "int16"),
                _p("brasil", "soil", "collection2_1", "soil_clay_fraction",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2_1/mapbiomas_brazil_collection21_soil_clay_fraction_v2", "byte"),
                _p("brasil", "soil", "collection2_1", "soil_sand_fraction",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2_1/mapbiomas_brazil_collection21_soil_sand_fraction_v2", "byte"),
                _p("brasil", "soil", "collection2_1", "soil_silt_fraction",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2_1/mapbiomas_brazil_collection21_soil_silt_fraction_v2", "byte"),
                _p("brasil", "soil", "collection2_1", "soil_textural_class",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2_1/mapbiomas_brazil_collection21_soil_textural_class_v2", "byte"),
                _p("brasil", "soil", "collection2_1", "soil_textural_group",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2_1/mapbiomas_brazil_collection21_soil_textural_group_v2", "byte"),
                _p("brasil", "soil", "collection2_1", "soil_textural_subgroup",
                   "projects/mapbiomas-public/assets/brazil/soil/collection2_1/mapbiomas_brazil_collection21_soil_textural_subgroup_v2", "byte"),
            ],
        },
        "water": {
            "collection_05": [
                _p("brasil", "water", "collection5", "water_annual", "projects/mapbiomas-public/assets/brazil/water/collection5/mapbiomas_brazil_collection5_water_annual_v4", "byte"),
                _p("brasil", "water", "collection5", "water_bodies", "projects/mapbiomas-public/assets/brazil/water/collection5/mapbiomas_brazil_collection5_water_bodies_v4", "byte"),
                _p("brasil", "water", "collection5", "water_monthly", "projects/mapbiomas-public/assets/brazil/water/collection5/mapbiomas_brazil_collection5_water_monthly_v4", "byte"),
                _p("brasil", "water", "collection5", "water_transition", "projects/mapbiomas-public/assets/brazil/water/collection5/mapbiomas_brazil_collection5_water_transition_v4", "byte"),
                _p("brasil", "water", "collection5", "water_trend", "projects/mapbiomas-public/assets/brazil/water/collection5/mapbiomas_brazil_collection5_water_trend_v4", "byte"),
            ],
            "collection_02": [
                _p("brasil", "water", "collection2", "annual_water_coverage",
                   "projects/mapbiomas-public/assets/brazil/water/collection2/mapbiomas_water_annual_water_coverage_v1", "byte"),
                _p("brasil", "water", "collection2", "frequency",
                   "projects/mapbiomas-public/assets/brazil/water/collection2/mapbiomas_water_frequency_v1", "byte"),
            ],
            "collection_03": [
                _p("brasil", "water", "collection3", "annual_water_coverage",
                   "projects/mapbiomas-public/assets/brazil/water/collection3/mapbiomas_water_annual_water_coverage_v1", "byte"),
                _p("brasil", "water", "collection3", "frequency",
                   "projects/mapbiomas-public/assets/brazil/water/collection3/mapbiomas_water_frequency_v1", "byte"),
            ],
            "collection_04": [
                _p("brasil", "water", "collection4", "water",
                   "projects/mapbiomas-public/assets/brazil/water/collection4/mapbiomas_brazil_collection4_water_v3", "byte"),
                _p("brasil", "water", "collection4", "water_bodies",
                   "projects/mapbiomas-public/assets/brazil/water/collection4/mapbiomas_brazil_collection4_water_bodies_v1", "byte"),
            ],
        },
        "atmosphere": {
            "collection_01": [
                _p("brasil", "atmosphere", "collection1", "air_temperature_annual",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_air_temperature_annual_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "air_temperature_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_air_temperature_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "air_temperature_monthly_maximum",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_air_temperature_monthly_maximum_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "air_temperature_monthly_mean",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_air_temperature_monthly_mean_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "air_temperature_monthly_minimum",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_air_temperature_monthly_minimum_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "air_temperature_trend_maximum",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_air_temperature_trend_maximum_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "air_temperature_trend_mean",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_air_temperature_trend_mean_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "air_temperature_trend_minimum",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_air_temperature_trend_minimum_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "lulc_l0",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_lulc_l0_v1", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "lulc_l1",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_lulc_l1_v1", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "lulc_l2",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_lulc_l2_v1", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "persistent_rain_annual",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_persistent_rain_annual_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "persistent_rain_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_persistent_rain_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "pm10_annual",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_pm10_annual_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "pm10_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_pm10_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "pm2p5_annual",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_pm2p5_annual_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "pm2p5_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_pm2p5_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "precipitation_annual",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_precipitation_annual_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "precipitation_anomaly_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_precipitation_anomaly_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "precipitation_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_precipitation_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "rain_free_days_annual",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_rain_free_days_annual_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "rain_free_days_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_rain_free_days_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "surface_temperature_annual",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_surface_temperature_annual_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "surface_temperature_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_surface_temperature_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "temperature_anomaly_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_temperature_anomaly_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "vapor_pressure_deficit_annual",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_vapor_pressure_deficit_annual_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "vapor_pressure_deficit_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_vapor_pressure_deficit_monthly_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "water_availability_annual",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_water_availability_annual_v2", "float32", scale=11132),
                _p("brasil", "atmosphere", "collection1", "water_availability_monthly",
                   "projects/mapbiomas-public/assets/brazil/atmosphere/collection1/mapbiomas_brazil_collection1_water_availability_monthly_v2", "float32", scale=11132),
            ],
        },
        "climate_risk": {
            "collection_01": [
                _p("brasil", "climate_risk", "collection1", "urban_flood_risk",
                   "projects/mapbiomas-public/assets/brazil/climate_risk/collection1/mapbiomas_brazil_collection1_urban_flood_risk_v1", "byte"),
                _p("brasil", "climate_risk", "collection1", "urban_landslide_risk",
                   "projects/mapbiomas-public/assets/brazil/climate_risk/collection1/mapbiomas_brazil_collection1_urban_landslide_risk_v1", "byte"),
                _p("brasil", "climate_risk", "collection1", "water_security_index",
                   "projects/mapbiomas-public/assets/brazil/climate_risk/collection1/mapbiomas_brazil_collection1_water_security_index_v1", "byte"),
            ],
            "collection_02": [
                _p("brasil", "climate_risk", "collection2", "urban_flood_risk",
                   "projects/mapbiomas-public/assets/brazil/climate_risk/collection2/mapbiomas_brazil_collection2_urban_flood_risk_v1", "byte"),
                _p("brasil", "climate_risk", "collection2", "urban_landslide_risk",
                   "projects/mapbiomas-public/assets/brazil/climate_risk/collection2/mapbiomas_brazil_collection2_urban_landslide_risk_v1", "byte"),
            ],
        },
        "urban": {
            "collection_02": [
                _p("brasil", "urban", "collection2", "urban_hand",
                   "projects/mapbiomas-public/assets/brazil/urban/collection2/mapbiomas_brazil_collection2_urban_hand_v2", "byte"),
                _p("brasil", "urban", "collection2", "urban_periods",
                   "projects/mapbiomas-public/assets/brazil/urban/collection2/mapbiomas_brazil_collection2_urban_periods_v2", "byte"),
                _p("brasil", "urban", "collection2", "urban_slope",
                   "projects/mapbiomas-public/assets/brazil/urban/collection2/mapbiomas_brazil_collection2_urban_slope_v2", "byte"),
                _p("brasil", "urban", "collection2", "urban_vegetation",
                   "projects/mapbiomas-public/assets/brazil/urban/collection2/mapbiomas_brazil_collection2_urban_vegetation_v2", "byte"),
            ],
            "collection_03": [
                _p("brasil", "urban", "collection3", "urban_hand",
                   "projects/mapbiomas-public/assets/brazil/urban/collection3/mapbiomas_brazil_collection3_urban_hand_v1", "byte"),
                _p("brasil", "urban", "collection3", "urban_nightlight",
                   "projects/mapbiomas-public/assets/brazil/urban/collection3/mapbiomas_brazil_collection3_urban_nightlight_v1", "byte", scale=250),
                _p("brasil", "urban", "collection3", "urban_periods",
                   "projects/mapbiomas-public/assets/brazil/urban/collection3/mapbiomas_brazil_collection3_urban_periods_v1", "byte"),
                _p("brasil", "urban", "collection3", "urban_slope",
                   "projects/mapbiomas-public/assets/brazil/urban/collection3/mapbiomas_brazil_collection3_urban_slope_v1", "byte"),
                _p("brasil", "urban", "collection3", "urban_vegetation",
                   "projects/mapbiomas-public/assets/brazil/urban/collection3/mapbiomas_brazil_collection3_urban_vegetation_v1", "byte"),
            ],
        },
    },
    "indonesia": {
        "fire": {
            "monitor": [
                _p("indonesia", "fire", "monitor", "monthly_burned",
                   "projects/mapbiomas-public/assets/indonesia/fire/monitor/mapbiomas_fire_monthly_burned_v1",
                   "byte", scale=10),
            ],
            "collection_01": [
                _p("indonesia", "fire", "collection1", "annual_burned",
                   "projects/mapbiomas-public/assets/indonesia/fire/collection1/mapbiomas_fire_collection1_annual_burned_v1", "byte", vectorize=True),
                _p("indonesia", "fire", "collection1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/indonesia/fire/collection1/mapbiomas_fire_collection1_annual_burned_coverage_v1", "byte"),
                _p("indonesia", "fire", "collection1", "monthly_burned",
                   "projects/mapbiomas-public/assets/indonesia/fire/collection1/mapbiomas_fire_collection1_monthly_burned_v1", "byte"),
                _p("indonesia", "fire", "collection1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/indonesia/fire/collection1/mapbiomas_fire_collection1_accumulated_burned_v1", "byte"),
                _p("indonesia", "fire", "collection1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/indonesia/fire/collection1/mapbiomas_fire_collection1_accumulated_burned_coverage_v1", "byte"),
                _p("indonesia", "fire", "collection1", "fire_frequency",
                   "projects/mapbiomas-public/assets/indonesia/fire/collection1/mapbiomas_fire_collection1_fire_frequency_v1", "int16"),
            ],
            # Collection 02 is publicly distributed as GeoTIFFs in GCS.  Keep
            # the public prefix as the source so the catalog can discover the
            # already-published products without using private GEE assets.
            "collection_02": [
                _p("indonesia", "fire", "collection2", "annual_burned",
                   "projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_annual_burned_v1", "byte", vectorize=True),
                _p("indonesia", "fire", "collection2", "annual_burned_coverage", "projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_annual_burned_coverage_v1", "byte"),
                _p("indonesia", "fire", "collection2", "monthly_burned", "projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_monthly_burned_v1", "byte"),
                _p("indonesia", "fire", "collection2", "accumulated_burned", "projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_accumulated_burned_v1", "byte"),
                _p("indonesia", "fire", "collection2", "accumulated_burned_coverage", "projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_accumulated_burned_coverage_v1", "byte"),
                _p("indonesia", "fire", "collection2", "fire_frequency", "projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_frequency_burned_v1", "byte"),
            ],
        },
        "lulc": {
            "collection_04_1": [
                _p("indonesia", "lulc", "collection4_1", "coverage",
                   "projects/mapbiomas-public/assets/indonesia/lulc/collection4_1/mapbiomas_indonesia_collection41_coverage_v1",
                   "byte"),
            ],
        },

    },

    "bolivia": {
        "fire": {
            "collection_01": [
                _p("bolivia", "fire", "collection1", "annual_burned",
                   "projects/mapbiomas-public/assets/bolivia/fire/collection1/mapbiomas_bolivia_fire_collection1_annual_burned_v1", "byte", vectorize=True),
                _p("bolivia", "fire", "collection1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/bolivia/fire/collection1/mapbiomas_bolivia_fire_collection1_annual_burned_scar_size_range_v1", "byte"),
                _p("bolivia", "fire", "collection1", "monthly_burned",
                   "projects/mapbiomas-public/assets/bolivia/fire/collection1/mapbiomas_bolivia_fire_collection1_monthly_burned_v1", "byte"),
                _p("bolivia", "fire", "collection1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/bolivia/fire/collection1/mapbiomas_bolivia_fire_collection1_accumulated_burned_v1", "byte"),
                _p("bolivia", "fire", "collection1", "frequency_burned",
                   "projects/mapbiomas-public/assets/bolivia/fire/collection1/mapbiomas_bolivia_fire_collection1_frequency_burned_v1", "byte"),
                _p("bolivia", "fire", "collection1", "year_last_fire",
                   "projects/mapbiomas-public/assets/bolivia/fire/collection1/mapbiomas_bolivia_fire_collection1_year_last_fire_v1", "int16"),
            ],
        },
    },
    "peru": {
        "fire": {
            "collection_01": [
                _p("peru", "fire", "collection1", "annual_burned",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_annual_burned_v1", "byte"),
                _p("peru", "fire", "collection1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_annual_burned_coverage_v1", "byte"),
                _p("peru", "fire", "collection1", "annual_burned_area_ha",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_annual_burned_area_ha_v1", "float32"),
                _p("peru", "fire", "collection1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_annual_burned_scar_size_range_v1", "byte"),
                _p("peru", "fire", "collection1", "monthly_burned",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_monthly_burned_v1", "byte"),
                _p("peru", "fire", "collection1", "monthly_burned_coverage",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_monthly_burned_coverage_v1", "byte"),
                _p("peru", "fire", "collection1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_accumulated_burned_v1", "byte"),
                _p("peru", "fire", "collection1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_accumulated_burned_coverage_v1", "byte"),
                _p("peru", "fire", "collection1", "frequency_burned",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_frequency_burned_v1", "byte"),
                _p("peru", "fire", "collection1", "frequency_burned_coverage",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_frequency_burned_coverage_v1", "int16"),
                _p("peru", "fire", "collection1", "year_last_fire",
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_year_last_fire_v1", "int16"),
            ],
        },
    },
    "paraguay": {
        "fire": {
            "collection_01": [
                _p("paraguay", "fire", "collection1", "annual_burned",
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_annual_burned_v1", "byte"),
                _p("paraguay", "fire", "collection1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_annual_burned_coverage_v1", "byte"),
                _p("paraguay", "fire", "collection1", "monthly_burned",
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_monthly_burned_v1", "byte"),
                _p("paraguay", "fire", "collection1", "monthly_burned_coverage",
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_monthly_burned_coverage-v1", "byte"),
                _p("paraguay", "fire", "collection1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_accumulated_burned_v1", "byte"),
                _p("paraguay", "fire", "collection1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_accumulated_burned_coverage_v1", "byte"),
                _p("paraguay", "fire", "collection1", "frequency_burned",
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_frequency_burned_v1", "byte"),
                _p("paraguay", "fire", "collection1", "frequency_burned_coverage",
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_frequency_burned_coverage_v1", "int16"),
            ],
        },
    },
    "chile": {
        "fire": {
            "collection_01": [
                _p("chile", "fire", "collection1", "annual_burned",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_annual_burned_v1", "byte", vectorize=True),
                _p("chile", "fire", "collection1", "annual_burned_area_ha",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_annual_burned_area_ha_v1", "float32"),
                _p("chile", "fire", "collection1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_annual_burned_coverage_v1", "byte"),
                _p("chile", "fire", "collection1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_annual_burned_scar_size_range_v1", "float32"),
                _p("chile", "fire", "collection1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_accumulated_burned_v1", "byte"),
                _p("chile", "fire", "collection1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_accumulated_burned_coverage_v1", "byte"),
                _p("chile", "fire", "collection1", "frequency_burned",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_frequency_burned_v1", "byte"),
                _p("chile", "fire", "collection1", "frequency_burned_coverage",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_frequency_burned_coverage_v1", "byte"),
                _p("chile", "fire", "collection1", "monthly_burned",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_monthly_burned_v1", "byte"),
                _p("chile", "fire", "collection1", "year_last_fire",
                   "projects/mapbiomas-public/assets/chile/fire/collection1/mapbiomas_chile_fire_collection1_year_last_fire_v1", "int16"),
            ],
        },
        "lulc": {
            "collection_02": [
                _p("chile", "lulc", "collection2", "coverage",
                   "projects/mapbiomas-public/assets/chile/lulc/collection2/mapbiomas_chile_collection2_coverage_v2", "byte"),
            ],
        },
    },
    "argentina": {
        "fire": {
            "collection_01": [
                _p("argentina", "fire", "collection1", "annual_burned",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_annual_burned_v1", "byte", vectorize=True),
                _p("argentina", "fire", "collection1", "annual_burned_area_ha",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_annual_burned_area_ha_v1", "float32"),
                _p("argentina", "fire", "collection1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_annual_burned_coverage_v1", "byte"),
                _p("argentina", "fire", "collection1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_annual_burned_scar_size_range_v1", "byte"),
                _p("argentina", "fire", "collection1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_accumulated_burned_v1", "byte"),
                _p("argentina", "fire", "collection1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_accumulated_burned_coverage_v1", "byte"),
                _p("argentina", "fire", "collection1", "frequency_burned",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_frequency_burned_v1", "byte"),
                _p("argentina", "fire", "collection1", "frequency_burned_coverage",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_frequency_burned_coverage_v1", "byte"),
                _p("argentina", "fire", "collection1", "monthly_burned",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_monthly_burned_v1", "byte"),
                _p("argentina", "fire", "collection1", "year_last_fire",
                   "projects/mapbiomas-public/assets/argentina/fire/collection1/mapbiomas_argentina_fire_collection1_year_last_fire_v1", "int16"),
            ],
        },
        "lulc": {
            "collection_02": [
                _p("argentina", "lulc", "collection2", "deforestation_secondary_vegetation",
                   "projects/mapbiomas-public/assets/argentina/lulc/collection2/mapbiomas_argentina_collection2_deforestation_secondary_vegetation_v1", "byte"),
                _p("argentina", "lulc", "collection2", "integration",
                   "projects/mapbiomas-public/assets/argentina/lulc/collection2/mapbiomas_argentina_collection2_integration_v3", "byte"),
            ],
        },
    },
    "colombia": {
        "fire": {
            "collection_01": [
                _p("colombia", "fire", "collection1", "annual_burned",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_annual_burned_v1", "byte", vectorize=True),
                _p("colombia", "fire", "collection1", "annual_burned_area_ha",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_annual_burned_area_ha_v1", "float32"),
                _p("colombia", "fire", "collection1", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_annual_burned_coverage_v1", "byte"),
                _p("colombia", "fire", "collection1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_annual_burned_scar_size_range_v1", "float32"),
                _p("colombia", "fire", "collection1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_accumulated_burned_v1", "byte"),
                _p("colombia", "fire", "collection1", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_accumulated_burned_coverage_v1", "byte"),
                _p("colombia", "fire", "collection1", "frequency_burned",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_frequency_burned_v1", "byte"),
                _p("colombia", "fire", "collection1", "frequency_burned_coverage",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_frequency_burned_coverage_v1", "byte"),
                _p("colombia", "fire", "collection1", "monthly_burned",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_monthly_burned_v1", "byte"),
                _p("colombia", "fire", "collection1", "year_last_fire",
                   "projects/mapbiomas-public/assets/colombia/fire/collection1/mapbiomas_colombia_fire_collection1_year_last_fire_v1", "int16"),
            ],
        },
        "lulc": {
            "collection_03": [
                _p("colombia", "lulc", "collection3", "coverage",
                   "projects/mapbiomas-public/assets/colombia/lulc/collection3/mapbiomas_colombia_collection3_coverage_v2", "byte"),
                _p("colombia", "lulc", "collection3", "deforestation_secondary_vegetation",
                   "projects/mapbiomas-public/assets/colombia/lulc/collection3/mapbiomas_colombia_collection3_deforestation_secondary_vegetation_v3", "byte"),
            ],
        },
        "water": {
            "collection_03": [
                _p("colombia", "water", "collection3", "water",
                   "projects/mapbiomas-public/assets/colombia/water/collection3/mapbiomas_colombia_collection3_water_v1", "byte"),
                _p("colombia", "water", "collection3", "water_bodies",
                   "projects/mapbiomas-public/assets/colombia/water/collection3/mapbiomas_colombia_collection3_water_bodies_v1", "byte"),
                _p("colombia", "water", "collection3", "water_monthly",
                   "projects/mapbiomas-public/assets/colombia/water/collection3/mapbiomas_colombia_collection3_water_monthly_v1", "byte"),
                _p("colombia", "water", "collection3", "glacier",
                   "projects/mapbiomas-public/assets/colombia/water/collection3/mapbiomas_colombia_collection3_glacier_v1", "byte"),
            ],
        },
    },
    "venezuela": {
        "lulc": {
            "collection_03": [
                _p("venezuela", "lulc", "collection3", "coverage",
                   "projects/mapbiomas-public/assets/venezuela/lulc/collection3/mapbiomas_venezuela_collection3_coverage_v1", "byte"),
                _p("venezuela", "lulc", "collection3", "vegetation_loss_secondary_vegetation",
                   "projects/mapbiomas-public/assets/venezuela/lulc/collection3/mapbiomas_venezuela_collection3_vegetation_loss_secondary_vegetation_v2", "byte"),
            ],
        },
        "water": {
            "collection_03": [
                _p("venezuela", "water", "collection3", "water",
                   "projects/mapbiomas-public/assets/venezuela/water/collection3/mapbiomas_venezuela_collection3_water_v2", "byte"),
                _p("venezuela", "water", "collection3", "water_bodies",
                   "projects/mapbiomas-public/assets/venezuela/water/collection3/mapbiomas_venezuela_collection3_water_bodies_v2", "byte"),
                _p("venezuela", "water", "collection3", "water_monthly",
                   "projects/mapbiomas-public/assets/venezuela/water/collection3/mapbiomas_venezuela_collection3_water_monthly_v2", "byte"),
                _p("venezuela", "water", "collection3", "glacier",
                   "projects/mapbiomas-public/assets/venezuela/water/collection3/mapbiomas_venezuela_collection3_glacier_v1", "byte"),
            ],
        },
    },
}

# --- seletores ativos ---
COUNTRY = "brasil"
THEME = "fire"
COLLECTION = "monitor"
PRODUCT = "monthly_burned"

# Temas exibidos como abas na UI. None ou [] = todos os temas disponiveis.
# Ex.: THEMES = ["fire"] restringe a interface ao tema de fogo.
THEMES = None

COUNTRIES_AVAILABLE = list(OBJ)
COUNTRIES_FLAGS = {
    "brasil": "🇧🇷",
    "indonesia": "🇮🇩",
    "bolivia": "🇧🇴",
    "peru": "🇵🇪",
    "paraguay": "🇵🇾",
    "chile": "🇨🇱",
    "argentina": "🇦🇷",
    "colombia": "🇨🇴",
    "venezuela": "🇻🇪",
}

# Compat com ui (valida chaves por pais)
COUNTRIES = {code: {} for code in OBJ}

COUNTRY_STORAGE = {"brasil": "brazil"}


def storage_country(country=None):
    """Retorna o codigo usado nos caminhos GCS e nos assets publicados."""
    code = country or COUNTRY
    return COUNTRY_STORAGE.get(code, code)


def flag(code):
    return COUNTRIES_FLAGS.get(code, "🌍")


def list_countries():
    return [c for c, themes in OBJ.items() if any(
        any(prods for prods in collections.values()) for collections in themes.values())]


def list_collections(country, theme=None):
    theme = theme or THEME
    return {coll: [p["product"] for p in prods if p.get("visible", True)]
            for coll, prods in OBJ.get(country, {}).get(theme, {}).items()
            if [p for p in prods if p.get("visible", True)]}


def list_products(country, theme=None, collection=None):
    theme = theme or THEME
    collection = collection or COLLECTION
    return [p for p in OBJ.get(country, {}).get(theme, {}).get(collection, [])
            if p.get("visible", True)]


def find_product(country, theme, collection, product):
    for p in OBJ.get(country, {}).get(theme, {}).get(collection, []):
        if p["product"] == product:
            return p
    return None


def set_country(name, verbose=True):
    global COUNTRY, THEME, COLLECTION, PRODUCT
    if name not in OBJ:
        raise ValueError(f"Country '{name}' not configured. Available: {sorted(OBJ)}")
    COUNTRY = name
    themes = OBJ[name]
    THEME = next(iter(themes), "fire")
    colls = themes.get(THEME, {})
    COLLECTION = next(iter(colls), None)
    prods = colls.get(COLLECTION, [])
    PRODUCT = prods[0]["product"] if prods else None
    if verbose:
        print("Country:", COUNTRY, "| Theme:", THEME, "| Collection:", COLLECTION, "| Product:", PRODUCT)


def set_theme(name):
    global THEME, COLLECTION, PRODUCT
    if name not in OBJ.get(COUNTRY, {}):
        raise ValueError(f"Theme '{name}' not configured for {COUNTRY}.")
    THEME = name
    colls = OBJ[COUNTRY][THEME]
    COLLECTION = next(iter(colls), None)
    prods = colls.get(COLLECTION, [])
    visible = [p for p in prods if p.get("visible", True)]
    PRODUCT = visible[0]["product"] if visible else None


def set_collection(name):
    global COLLECTION, PRODUCT
    colls = OBJ.get(COUNTRY, {}).get(THEME, {})
    if name not in colls:
        raise ValueError(f"Collection '{name}' not configured for {COUNTRY}/{THEME}.")
    COLLECTION = name
    prods = colls.get(COLLECTION, [])
    visible = [p for p in prods if p.get("visible", True)]
    PRODUCT = visible[0]["product"] if visible else None


def set_product(name):
    global PRODUCT
    if find_product(COUNTRY, THEME, COLLECTION, name) is None:
        raise ValueError(f"Product '{name}' not configured for {COUNTRY}/{THEME}/{COLLECTION}.")
    PRODUCT = name


def active_product():
    return find_product(COUNTRY, THEME, COLLECTION, PRODUCT)


def product_meta():
    return active_product() or {"product": PRODUCT, "assetid": "", "type": "byte",
                                "vectorize": False, "visible": True}


def product_context():
    """Metadados seguros para apresentar na interface."""
    p = product_meta()
    return {
        "product": p.get("product", PRODUCT),
        "assetid": p.get("assetid", ""),
        "type": p.get("type", "byte"),
        "scale": p.get("scale", SCALE),
        "vectorize": bool(p.get("vectorize", False)),
    }


def image_collection(context=None):
    if context is not None:
        return context["assetid"]
    return active_product()["assetid"]


def scale(context=None):
    return (context or processing_context()).get("scale", SCALE)


def theme():
    return THEME


def collection():
    return COLLECTION


def product():
    return PRODUCT


def product_kind(product=None):
    n = (product or PRODUCT).lower()
    if "monthly" in n:
        return "monthly"
    if "annual" in n:
        return "annual"
    return "period"


def unit_key_for_image(kind, time_start_ms):
    import datetime
    dt = datetime.datetime.utcfromtimestamp(time_start_ms / 1000)
    if kind == "monthly":
        return f"{dt.year}_{dt.month:02d}"
    return f"{dt.year}"


def is_vectorizable(context=None):
    if context is not None:
        return bool(context.get("vectorize"))
    return bool(product_meta().get("vectorize"))


def save_options(context=None):
    """Mapeia o type declarado do produto para o salvamento do mosaico."""
    declared = context.get("type") if context is not None else product_meta().get("type")
    t = (declared or "byte").lower()
    if t in ("float32", "float64", "float"):
        return {"ot": "Float32", "nodata": 0, "predictor": 3, "compression": "DEFLATE"}
    if t in ("int16", "uint16", "int32"):
        return {"ot": "Int16", "nodata": 0, "predictor": 2, "compression": "DEFLATE"}
    return {"ot": "Byte", "nodata": 0, "predictor": 2, "compression": "DEFLATE"}


def _rel(prod):
    physical = prod
    if COLLECTION == "monitor" and prod == "monthly_burned":
        physical = "mapbiomas_fire_monthly_burned_v1"
    return f"{BUCKET_PATH}/{storage_country()}/{THEME}/{COLLECTION}/{physical}"


def tiles_prefix(context=None):
    ctx = context or processing_context()
    return f"{ctx['root']}/temp"


def mosaic_prefix(context=None):
    return (context or processing_context())["root"]


def _raster_base(ctx):
    """Nome fisico do raster vetorizado (basename do assetid de origem)."""
    assetid = (ctx.get("assetid") or "").rstrip("/")
    base = assetid.split("/")[-1] if assetid else ""
    return base or ctx["storage_product"]


def vector_folder_name(context=None):
    """Nome da pasta de vetores: irma do raster, com sufixo _vectors."""
    ctx = context or processing_context()
    return f"{_raster_base(ctx)}_vectors"


def _render_template(template, ctx):
    """Renderiza um template de caminho substituindo os placeholders de vetor."""
    repl = {
        "{country}": ctx.get("country", ""),
        "{COUNTRY}": ctx.get("storage_country", ""),
        "{theme}": ctx.get("theme", ""),
        "{collection}": ctx.get("collection", ""),
        "{COLLECTION}": ctx.get("gee_collection", ""),
        "{product}": ctx.get("product", ""),
        "{raster}": _raster_base(ctx),
        "{vectors_folder}": vector_folder_name(ctx),
    }
    out = template
    for k, v in repl.items():
        out = out.replace(k, v)
    return out.rstrip("/")


def vector_prefix(context=None):
    """Pasta GCS de vetores — por padrao irma do raster; usa VECTOR_OUTPUT_GCS
    quando configurado."""
    ctx = context or processing_context()
    if VECTOR_OUTPUT_GCS:
        return _render_template(VECTOR_OUTPUT_GCS, ctx)
    head = ctx["root"].rsplit("/", 1)[0]
    return f"{head}/{vector_folder_name(ctx)}"


def vector_asset_prefix(context=None):
    """Pasta GEE de vetores — por padrao irma do raster em mapbiomas-public;
    usa VECTOR_OUTPUT_GEE quando configurado."""
    ctx = context or processing_context()
    if VECTOR_OUTPUT_GEE:
        return _render_template(VECTOR_OUTPUT_GEE, ctx)
    assetid = (ctx.get("assetid") or "").rstrip("/")
    folder = vector_folder_name(ctx)
    if assetid.startswith("projects/"):
        return f"{assetid.rsplit('/', 1)[0]}/{folder}"
    return (f"projects/mapbiomas-public/assets/{ctx['storage_country']}/{ctx['theme']}/"
            f"{ctx['gee_collection']}/{folder}")


def art_prefix(context=None):
    """Prefixo de artefatos por unidade: {product}-{storage_country}_."""
    ctx = context or processing_context()
    return f"{ctx['product']}-{ctx['storage_country']}_"


def tile_pattern(year, month):
    return f"fire_monitor_v1_{PRODUCT}_{COUNTRY}_{year}_{month:02d}"


def mosaic_name(year, month):
    return f"{PRODUCT}-{COUNTRY}_{year}_{month:02d}"


def vector_name(year, month):
    return f"{PRODUCT}-{COUNTRY}_{year}_{month:02d}"


# --- nomes por unidade (multibanda/IC) ---
def _sanitize(unit):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in str(unit))


def state_file(country=None, theme=None, collection=None, product=None):
    """Arquivo de estado isolado por contexto de processamento."""
    coll = _gee_collection_name(collection or COLLECTION)
    parts = [country or COUNTRY, theme or THEME, coll, product or PRODUCT]
    return "monitor_state_" + "_".join(_sanitize(p) for p in parts if p) + ".json"


def processing_context(country=None, theme=None, collection=None, product=None):
    """Retorna contexto independente dos seletores globais da UI."""
    country = country or COUNTRY
    theme = theme or THEME
    collection = collection or COLLECTION
    product = product or PRODUCT
    storage_product = product
    if collection == "monitor" and product == "monthly_burned":
        storage_product = "mapbiomas_fire_monthly_burned_v1"
    gee_collection = _gee_collection_name(collection)
    meta = find_product(country, theme, collection, product) or {}
    return {
        "country": country,
        "storage_country": storage_country(country),
        "theme": theme,
        "collection": collection,
        "gee_collection": gee_collection,
        "product": product,
        "storage_product": storage_product,
        "root": f"{BUCKET_PATH}/{storage_country(country)}/{theme}/{collection}/{storage_product}",
        "assetid": meta.get("assetid", ""),
        "type": (meta.get("type") or "byte").lower(),
        "scale": meta.get("scale", SCALE),
        "vectorize": bool(meta.get("vectorize", False)),
        "kind": product_kind(product),
    }


def tile_pattern_unit(unit, context=None):
    ctx = context or processing_context()
    return f"fire_monitor_v1_{ctx['product']}_{ctx['storage_country']}_{_sanitize(unit)}"


def mosaic_name_unit(unit, context=None):
    ctx = context or processing_context()
    return f"{ctx['product']}-{ctx['storage_country']}_{_sanitize(unit)}"


def vector_name_unit(unit, context=None):
    ctx = context or processing_context()
    return f"{ctx['product']}-{ctx['storage_country']}_{_sanitize(unit)}"


def gcs_object_url(bucket, path):
    """Retorna o direct link publico para um objeto GCS."""
    return f"https://storage.googleapis.com/{bucket}/{path}"


def add_collection(country, theme, collection, products):
    """Insere/atualiza uma colecao no OBJ (qualquer tema). products = lista de
    dicts com product/assetid/type/vectorize/visible."""
    OBJ.setdefault(country, {}).setdefault(theme, {})[collection] = [
        _p(country, theme, collection, p.get("product"), p.get("assetid"),
           p.get("type", "byte"), vectorize=p.get("vectorize", False),
           visible=p.get("visible", True), scale=p.get("scale", SCALE),
           decode=p.get("decode"))
        for p in products
    ]
    if country not in COUNTRIES_AVAILABLE:
        COUNTRIES_AVAILABLE.append(country)
        COUNTRIES[country] = {}


def set_product_visible(country, theme, collection, product, visible):
    """Oculta/mostra um produto no OBJ (sem apagar)."""
    for p in OBJ.get(country, {}).get(theme, {}).get(collection, []):
        if p["product"] == product:
            p["visible"] = bool(visible)
            return True
    return False


def remove_collection(country, theme, collection):
    """Remove uma colecao inteira do OBJ."""
    themes = OBJ.get(country, {})
    if theme in themes and collection in themes[theme]:
        del themes[theme][collection]
        return True
    return False


# ============================================================
# FILTER SYSTEM — include/exclude lists + presets + load_data cache
# ============================================================
FILTERS_FILE = "monitor_filters.json"

DEFAULT_FILTERS = {
    "preset": "fire_monitor",
    "include_countries": [],
    "exclude_countries": [],
    "include_themes": [],
    "exclude_themes": [],
    "include_collections": {},
    "exclude_collections": {},
    "include_products": {},
    "exclude_products": {},
    "load_data_cache": {},
    "schema_version": 2,
}


def _load_json_file(path, default):
    import json, os
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default


def _save_json_file(path, data):
    import json
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"[WARN] Could not save {path}: {e}")
        return False


def load_filters(path=None):
    """Carrega filtros do arquivo (default: monitor_filters.json)."""
    path = path or FILTERS_FILE
    return _load_json_file(path, DEFAULT_FILTERS.copy())


def save_filters(filters, path=None):
    """Salva filtros no arquivo."""
    path = path or FILTERS_FILE
    return _save_json_file(path, filters)


def get_filters(path=None):
    """Retorna filtros atuais (carrega do disco se necessario)."""
    return load_filters(path)


def apply_filters(obj, filters=None):
    """Aplica filtros de inclusao/exclusao ao OBJ e retorna OBJ filtrado.
    
    Estrutura dos filtros:
        include_countries: []  # vazio = todos
        exclude_countries: []
        include_themes: []     # vazio = todos
        exclude_themes: []
        include_collections: {"brasil": {"fire": ["collection_04"]}}  # por pais/tema
        exclude_collections: {}
        include_products: {"brasil": {"fire": {"collection_04": ["annual_burned"]}}}
        exclude_products: {}
        preset: "fire_monitor" | "all" | "lulc_only" | "custom"
    """
    if filters is None:
        filters = load_filters()
    
    # Presets
    preset = filters.get("preset", "custom")
    if preset == "fire_monitor":
        filters = dict(filters)
        filters.setdefault("include_themes", []).append("fire")
        # ensure fire collections are included
        filters.setdefault("include_collections", {}).setdefault("brasil", {}).setdefault("fire", []).extend([
            "monitor", "collection_01", "collection_02", "collection_03",
            "collection_04", "collection_04_1", "collection_05"
        ])
        for c in ["indonesia", "bolivia", "peru", "paraguay"]:
            filters.setdefault("include_collections", {}).setdefault(c, {}).setdefault("fire", []).extend([
                "monitor", "collection_01"
            ])
    elif preset == "lulc_only":
        filters = dict(filters)
        filters.setdefault("include_themes", []).extend(["lulc", "lulc_10m"])
    elif preset == "all":
        pass  # no filtering
    
    # Deep copy
    import copy
    result = copy.deepcopy(obj)
    
    inc_c = set(filters.get("include_countries", []))
    exc_c = set(filters.get("exclude_countries", []))
    inc_t = set(filters.get("include_themes", []))
    exc_t = set(filters.get("exclude_themes", []))
    
    # Filter countries
    if inc_c:
        result = {k: v for k, v in result.items() if k in inc_c}
    if exc_c:
        result = {k: v for k, v in result.items() if k not in exc_c}
    
    # Filter themes per country
    for country, themes in result.items():
        if inc_t:
            themes = {k: v for k, v in themes.items() if k in inc_t}
        if exc_t:
            themes = {k: v for k, v in themes.items() if k not in exc_t}
        
        # Filter collections per theme
        for theme, collections in themes.items():
            inc_colls = filters.get("include_collections", {}).get(country, {}).get(theme, [])
            exc_colls = filters.get("exclude_collections", {}).get(country, {}).get(theme, [])
            if inc_colls:
                collections = {k: v for k, v in collections.items() if k in inc_colls}
            if exc_colls:
                collections = {k: v for k, v in collections.items() if k not in exc_colls}
            
            # Filter products per collection
            for coll, prods in collections.items():
                inc_prods = filters.get("include_products", {}).get(country, {}).get(theme, {}).get(coll, [])
                exc_prods = filters.get("exclude_products", {}).get(country, {}).get(theme, {}).get(coll, [])
                if inc_prods:
                    prods = [p for p in prods if p["product"] in inc_prods]
                if exc_prods:
                    prods = [p for p in prods if p["product"] not in exc_prods]
                collections[coll] = prods
            
            themes[theme] = {k: v for k, v in collections.items() if v}
        
        result[country] = {k: v for k, v in themes.items() if v}
    
    result = {k: v for k, v in result.items() if v}
    return result


def get_load_data_cache(filters=None):
    """Retorna o cache de load_data (bandas/unidades por produto)."""
    if filters is None:
        filters = load_filters()
    return filters.get("load_data_cache", {})


def set_load_data_cache(country, theme, collection, product, units, filters=None):
    """Atualiza cache de load_data para um produto."""
    if filters is None:
        filters = load_filters()
    cache = filters.setdefault("load_data_cache", {})
    key = f"{country}/{theme}/{collection}/{product}"
    cache[key] = {
        "units": units,
        "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
    }
    save_filters(filters)
    return True


def clear_load_data_cache(filters=None):
    """Limpa todo o cache de load_data."""
    if filters is None:
        filters = load_filters()
    filters["load_data_cache"] = {}
    save_filters(filters)
    return True


def sync_filters_to_github(repo_path=".", commit_message="Update monitor_filters.json (load data cache)", logger=print):
    """Commit and push monitor_filters.json to GitHub.
    
    Args:
        repo_path: Path to the git repository (default: current directory)
        commit_message: Commit message
        logger: Function to log messages
    
    Returns:
        True if successful, False otherwise
    """
    import subprocess
    filters_file = "monitor_filters.json"
    try:
        # Check if file exists
        import os
        if not os.path.exists(filters_file):
            logger(f"[WARN] {filters_file} not found in {repo_path}")
            return False
        
        # Git add
        result = subprocess.run(
            ["git", "add", filters_file],
            cwd=repo_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            logger(f"[ERROR] git add failed: {result.stderr}")
            return False
        
        # Git commit
        result = subprocess.run(
            ["git", "commit", "-m", commit_message],
            cwd=repo_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            # Check if nothing to commit
            if "nothing to commit" in result.stdout.lower():
                logger("[INFO] No changes to commit")
                return True
            logger(f"[ERROR] git commit failed: {result.stderr}")
            return False
        
        # Git push
        result = subprocess.run(
            ["git", "push"],
            cwd=repo_path, capture_output=True, text=True
        )
        if result.returncode != 0:
            logger(f"[ERROR] git push failed: {result.stderr}")
            return False
        
        logger("[SUCCESS] Filters synced to GitHub")
        return True
        
    except Exception as e:
        logger(f"[ERROR] Git sync failed: {e}")
        return False
