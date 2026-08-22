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


OBJ = {
    "brasil": {
        "fire": {
            "monitor": [
                _p("brasil", "fire", "monitor", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1",
                   "byte", vectorize=True),
            ],
            "collection_03": [
                _p("brasil", "fire", "collection3", "annual_burned_coverage",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection31_annual_burned_coverage_v1", "byte"),
                _p("brasil", "fire", "collection3", "monthly_burned",
                   "projects/mapbiomas-public/assets/brazil/fire/collection3/mapbiomas_fire_collection31_monthly_burned_v1", "byte"),
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
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_fire_frequency_v1", "int16"),
                _p("brasil", "fire", "collection4_1", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_time_after_fire_v1", "int16"),
                _p("brasil", "fire", "collection4_1", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection4_1/mapbiomas_fire_collection41_year_last_fire_v1", "int16"),
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
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_fire_frequency_v1", "int16"),
                _p("brasil", "fire", "collection5", "time_after_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_time_after_fire_v1", "int16"),
                _p("brasil", "fire", "collection5", "year_last_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_year_last_fire_v1", "int16"),
                _p("brasil", "fire", "collection5", "severity_class",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_severity_class_v1", "byte"),
                _p("brasil", "fire", "collection5", "interval_since_fire",
                   "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_interval_since_fire_v1", "int16"),
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
                _p("peru", "fire", "collection1", "annual_burned_coverage",
                   "projects/mapbiomas-peru/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_peru_fire_collection1_annual_burned_coverage_v1", "byte", decode={"div": 100, "dtype": "byte"}),
                _p("peru", "fire", "collection1", "monthly_burned_coverage",
                   "projects/mapbiomas-peru/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_peru_fire_collection1_monthly_burned_coverage_v1", "byte", decode={"div": 100, "dtype": "byte"}),
                _p("peru", "fire", "collection1", "frequency_burned_coverage",
                   "projects/mapbiomas-peru/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_peru_fire_collection1_frequency_burned_coverage_v1", "int16", decode={"div": 100, "dtype": "int16"}),
                _p("peru", "fire", "collection1", "accumulated_burned_coverage",
                   "projects/mapbiomas-peru/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_peru_fire_collection1_accumulated_burned_coverage_v1", "byte", decode={"div": 100, "dtype": "byte"}),
                _p("peru", "fire", "collection1", "annual_burned_scar_size_range",
                   "projects/mapbiomas-peru/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_peru_fire_collection1_annual_burned_scar_size_range_v1", "byte", decode={"div": 100, "dtype": "byte"}),
                _p("peru", "fire", "collection1", "year_last_fire",
                   "projects/mapbiomas-peru/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_peru_fire_collection1_year_last_fire_v1", "int16", decode={"div": 100, "dtype": "int16"}),
            ],
        },
    },
    "paraguay": {
        "fire": {
            "collection_01": [
                _p("paraguay", "fire", "collection1", "annual_burned_coverage",
                   "projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_paraguay_fire_collection1_annual_burned_coverage_v1", "byte", decode={"div": 100, "dtype": "byte"}),
                _p("paraguay", "fire", "collection1", "monthly_burned_coverage",
                   "projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_paraguay_fire_collection1_monthly_burned_coverage-v1", "byte", decode={"div": 100, "dtype": "byte"}),
                _p("paraguay", "fire", "collection1", "frequency_burned_coverage",
                   "projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_paraguay_fire_collection1_frequency_burned_coverage_v1", "int16", decode={"div": 100, "dtype": "int16"}),
                _p("paraguay", "fire", "collection1", "accumulated_burned_coverage",
                   "projects/mapbiomas-paraguay/assets/FIRE/COLLECTION1/FINAL_PRODUCTS/mapbiomas_paraguay_fire_collection1_accumulated_burned_coverage_v1", "byte", decode={"div": 100, "dtype": "byte"}),
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
    return f"{BUCKET_PATH}/{storage_country()}/{THEME}/{COLLECTION}/{prod}"


def tiles_prefix():
    return f"{_rel(PRODUCT)}/temp"


def mosaic_prefix():
    return _rel(PRODUCT)


def vector_prefix():
    return f"{_rel(PRODUCT)}_vectors"


def vector_asset_prefix():
    return (f"projects/mapbiomas-public/assets/{storage_country()}/{THEME}/{COLLECTION}/"
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
    parts = [country or COUNTRY, theme or THEME, collection or COLLECTION, product or PRODUCT]
    return "monitor_state_" + "_".join(_sanitize(p) for p in parts if p) + ".json"


def processing_context(country=None, theme=None, collection=None, product=None):
    """Retorna contexto independente dos seletores globais da UI."""
    country = country or COUNTRY
    theme = theme or THEME
    collection = collection or COLLECTION
    product = product or PRODUCT
    return {
        "country": country,
        "storage_country": storage_country(country),
        "theme": theme,
        "collection": collection,
        "product": product,
        "root": f"{BUCKET_PATH}/{storage_country(country)}/{theme}/{collection}/{product}",
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
