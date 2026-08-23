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
Assets GEE:  {assetid} (origem); vetores em projetos/mapbiomas-public/assets/{country}/{theme}/{collection}/{product}_vectors_v01
"""

BUCKET = "mapbiomas-fire"
PUBLIC_BUCKET = "mapbiomas-public"
BUCKET_PATH = "initiatives"

GEE_PROJECT = "mapbiomas-fire-485203"

STATE_FILE = "monitor_state.json"

SCALE = 30


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
                   "byte", vectorize=True),
            ],
            "collection_05": [
                _p("brasil", "fire", "collection5", "annual_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_v1", "byte", vectorize=True),
                _p("brasil", "fire", "collection5", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection5", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_scar_size_range_v1", "byte"),
                _p("brasil", "fire", "collection5", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_monthly_burned_v1", "byte"),
                _p("brasil", "fire", "collection5", "accumulated_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_accumulated_burned_v1", "byte"),
                _p("brasil", "fire", "collection5", "accumulated_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_accumulated_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection5", "fire_frequency",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_fire_frequency_v1", "byte"),
                _p("brasil", "fire", "collection5", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_time_after_fire_v1", "int16"),
                _p("brasil", "fire", "collection5", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_year_last_fire_v1", "byte"),
                _p("brasil", "fire", "collection5", "severity_class",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_severity_class_v1", "byte"),
                _p("brasil", "fire", "collection5", "interval_since_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_interval_since_fire_v1", "byte"),
            ],
            "collection_04_1": [
                _p("brasil", "fire", "collection4_1", "annual_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_v1", "byte", vectorize=True),
                _p("brasil", "fire", "collection4_1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_annual_burned_scar_size_range_v1", "byte"),
                _p("brasil", "fire", "collection4_1", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_monthly_burned_v1", "byte"),
                _p("brasil", "fire", "collection4_1", "accumulated_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_accumulated_burned_v1", "byte"),
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
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_scar_size_range_v1", "byte"),
                _p("brasil", "fire", "collection4", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_monthly_burned_v1", "byte"),
                _p("brasil", "fire", "collection4", "accumulated_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_accumulated_burned_v1", "byte"),
                _p("brasil", "fire", "collection4", "fire_frequency",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_fire_frequency_v1", "int16"),
                _p("brasil", "fire", "collection4", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_time_after_fire_v1", "int16"),
                _p("brasil", "fire", "collection4", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_year_last_fire_v1", "int16"),
            ],
            "collection_03": [
                _p("brasil", "fire", "collection3", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection31_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection31_monthly_burned_v1", "byte"),
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
        },
        "lulc_10m": {
            "collection_04": [
                _p("brasil", "lulc_10m", "collection4", "coverage",
                   "projects/mapbiomas-public/assets/brazil/lulc_10m/collection4/mapbiomas_10m_collection4_coverage_v1",
                   "byte", scale=10),
            ],
        },
        "soil": {
            "collection_03": [
                _p("brasil", "soil", "collection3", "soil_carbon", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_carbon_v1", "float32"),
                _p("brasil", "soil", "collection3", "soil_clay_fraction", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_clay_fraction_v1", "float32"),
                _p("brasil", "soil", "collection3", "soil_sand_fraction", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_sand_fraction_v1", "float32"),
                _p("brasil", "soil", "collection3", "soil_silt_fraction", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_silt_fraction_v1", "float32"),
                _p("brasil", "soil", "collection3", "soil_stoniness", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_stoniness_v2", "float32"),
                _p("brasil", "soil", "collection3", "soil_textural_class", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_textural_class_v1", "byte"),
                _p("brasil", "soil", "collection3", "soil_textural_group", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_textural_group_v1", "byte"),
                _p("brasil", "soil", "collection3", "soil_textural_subgroup", "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_textural_subgroup_v1", "byte"),
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
        },
    },
    "indonesia": {
        "fire": {
            "monitor": [
                _p("indonesia", "fire", "monitor", "monthly_burned",
                   "projects/mapbiomas-public/assets/indonesia/fire/monitor/mapbiomas_fire_monthly_burned_v1",
                   "byte", vectorize=True),
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
                   "projects/mapbiomas-public/assets/indonesia/fire/collection2/mapbiomas_indonesia_fire_collection2_annual_burned_v1", "byte", vectorize=True)),
                _p("indonesia", "fire", "collection2", "annual_burned_coverage", "projects/mapbiomas-public/assets/indonesia/collection_2/fire-col1/fire-simplifed/FIRE_ANNUAL_COVERAGE", "byte"),
                _p("indonesia", "fire", "collection2", "monthly_burned", "projects/mapbiomas-public/assets/indonesia/collection_2/fire-col1/fire-simplifed/FIRE_MONTHLY_TOTAL", "byte"),
                _p("indonesia", "fire", "collection2", "accumulated_burned", "projects/mapbiomas-public/assets/indonesia/collection_2/fire-col1/fire-simplifed/FIRE_ACCUMULATED_TOTAL", "byte"),
                _p("indonesia", "fire", "collection2", "accumulated_burned_coverage", "projects/mapbiomas-public/assets/indonesia/collection_2/fire-col1/fire-simplifed/FIRE_ACCUMULATED_COVERAGE", "byte"),
                _p("indonesia", "fire", "collection2", "fire_frequency", "projects/mapbiomas-public/assets/indonesia/collection_2/fire-col1/fire-simplifed/FIRE_FREQUENCY_TOTAL", "int16"),
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
                   "projects/mapbiomas-public/assets/bolivia/fire/collection1/mapbiomas_bolivia_fire_collection1_frequency_burned_v1", "int16"),
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
                   "projects/mapbiomas-public/assets/peru/fire/collection1/mapbiomas_peru_fire_collection1_frequency_burned_v1", "int16"),
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
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_frequency_burned_v1", "int16"),
                _p("paraguay", "fire", "collection1", "frequency_burned_coverage",
                   "projects/mapbiomas-public/assets/paraguay/fire/collection1/mapbiomas_paraguay_fire_collection1_frequency_burned_coverage_v1", "int16"),
            ],
        },
    },
    "chile": {
        "fire": {},
    },
}

# --- seletores ativos ---
COUNTRY = "brasil"
THEME = "fire"
COLLECTION = "monitor"
PRODUCT = "monthly_burned"

COUNTRIES_AVAILABLE = list(OBJ)
COUNTRIES_FLAGS = {
    "brasil": "🇧🇷",
    "indonesia": "🇮🇩",
    "bolivia": "🇧🇴",
    "peru": "🇵🇪",
    "paraguay": "🇵🇾",
    "chile": "🇨🇱",
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


def image_collection():
    return active_product()["assetid"]


def scale():
    return product_meta().get("scale", SCALE)


def theme():
    return THEME


def collection():
    return COLLECTION


def product():
    return PRODUCT


def product_kind():
    n = PRODUCT.lower()
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


def is_vectorizable():
    return bool(product_meta().get("vectorize"))


def save_options():
    """Mapeia o type declarado do produto para o salvamento do mosaico."""
    t = (product_meta().get("type") or "byte").lower()
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


def tiles_prefix():
    return f"{_rel(PRODUCT)}/temp"


def mosaic_prefix():
    return _rel(PRODUCT)


def vector_prefix():
    return f"{_rel(PRODUCT)}_vectors"


def vector_asset_prefix():
    gee_coll = _gee_collection_name(COLLECTION)
    return (f"projects/mapbiomas-public/assets/{storage_country()}/{THEME}/{gee_coll}/"
            f"{PRODUCT}_vectors_v01")


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
    return {
        "country": country,
        "storage_country": storage_country(country),
        "theme": theme,
        "collection": collection,
        "gee_collection": gee_collection,
        "product": product,
        "storage_product": storage_product,
        "root": f"{BUCKET_PATH}/{storage_country(country)}/{theme}/{collection}/{storage_product}",
    }


def tile_pattern_unit(unit):
    return f"fire_monitor_v1_{PRODUCT}_{storage_country()}_{_sanitize(unit)}"


def mosaic_name_unit(unit):
    return f"{PRODUCT}-{storage_country()}_{_sanitize(unit)}"


def vector_name_unit(unit):
    return f"{PRODUCT}-{storage_country()}_{_sanitize(unit)}"


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
