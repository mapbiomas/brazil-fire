"""Configuracao multi-pais do pipeline Export & Vectorization.

Todos os paths sao derivados por funcao (lidos em tempo de chamada), entao
redefinir COUNTRY/BUCKET/etc. no notebook sempre propaga para os modulos
(state, export, mosaic, vectorize, publish) sem precisar re-importar nada.

Novo padrao de bucket (GCS):  {bucket}/initiatives/{country}/{theme}/{collection}/{product}
Novo padrao de assets (GEE):  projects/mapbiomas-{country}/assets/FIRE/MONITOR/{product}
"""

BUCKET = "mapbiomas-fire"
PUBLIC_BUCKET = "mapbiomas-public"
BUCKET_PATH = "initiatives"

GEE_PROJECT = "mapbiomas-fire-485203"

STATE_FILE = "monitor_state.json"

COUNTRIES = {
    "brazil": {
        "image_collection": "projects/mapbiomas-public/assets/brazil/fire/monitor/mapbiomas_fire_monthly_burned_v1",
        "theme": "fire",
        "collection": "monitor",
        "product": "mapbiomas_fire_monthly_burned_v1",
        "product_vectors": "mapbiomas_fire_monthly_burned_vectors_v01",
        "scale": 30,
    },
    "indonesia": {
        "image_collection": "projects/mapbiomas-public/assets/indonesia/fire/monitor/mapbiomas_fire_monthly_burned_v1",
        "theme": "fire",
        "collection": "monitor",
        "product": "mapbiomas_fire_monthly_burned_v1",
        "product_vectors": "mapbiomas_fire_monthly_burned_vectors_v01",
        "scale": 30,
    },
}

COUNTRY = "brazil"

# Paises expostos como abas na UI (ordem de exibicao).
COUNTRIES_AVAILABLE = ["brazil", "indonesia"]

COUNTRIES_FLAGS = {
    "brazil": "🇧🇷",
    "indonesia": "🇮🇩",
}


def flag(code):
    return COUNTRIES_FLAGS.get(code, "🌍")


def _country():
    return COUNTRIES.get(COUNTRY, COUNTRIES["brazil"])


def set_country(name, verbose=True):
    """Seleciona o pais ativo. Valida e (se verbose) imprime os caminhos."""
    global COUNTRY
    if name not in COUNTRIES:
        raise ValueError(
            f"Pais '{name}' nao configurado. Disponiveis: {sorted(COUNTRIES)}"
        )
    COUNTRY = name
    if not verbose:
        return
    print("Pais:", COUNTRY)
    print("Colecao:", image_collection())
    print("GCS tiles:", f"gs://{BUCKET}/{tiles_prefix()}")
    print("GCS COG:", f"gs://{BUCKET}/{mosaic_prefix()}")
    print("GCS vetores:", f"gs://{BUCKET}/{vector_prefix()}")
    print("GEE assets:", vector_asset_prefix())


def image_collection():
    return _country()["image_collection"]


def scale():
    return _country()["scale"]


def theme():
    return _country()["theme"]


def collection():
    return _country()["collection"]


def product():
    return _country()["product"]


def product_vectors():
    return _country()["product_vectors"]


def _rel(product_name):
    return f"{BUCKET_PATH}/{COUNTRY}/{theme()}/{collection()}/{product_name}"


def tiles_prefix():
    return f"{_rel(product())}/temp"


def mosaic_prefix():
    return _rel(product())


def vector_prefix():
    return _rel(product_vectors())


def vector_asset_prefix():
    # Pasta irma da ImageCollection de entrada (Etapa 1): mapbiomas-public/{country}/fire/monitor
    return f"projects/mapbiomas-public/assets/{COUNTRY}/fire/monitor/{product_vectors()}"


def tile_pattern(year, month):
    return f"fire_monitor_v1_monthly_burned_{COUNTRY}_{year}_{month:02d}"


def mosaic_name(year, month):
    return f"monthly_burned-{COUNTRY}_{year}_{month:02d}"


def vector_name(year, month):
    return f"monthly_burned-{COUNTRY}_{year}_{month:02d}"
