"""Exemplo inicial; validar resultado visual antes de uso em produção."""

LAYERS = [{
    "id": "annual_burned", "name": "Área queimada anual", "theme": "Fogo", "collection": "Coleção 5",
    "asset": "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_v1",
    "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [{"label": "Não observado", "color": "fdfdfd"}, {"label": "Fogo", "color": "800000"}],
}, {
    "id": "annual_burned_col4", "name": "Área queimada anual", "theme": "Fogo", "collection": "Coleção 4",
    "asset": "projects/mapbiomas-public/assets/brazil/fire/collection4/mapbiomas_fire_collection4_annual_burned_v1",
    "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [{"label": "Não observado", "color": "fdfdfd"}, {"label": "Fogo", "color": "800000"}],
    "status": "experimental",
}, {
    "id": "accumulated_burned_col5", "name": "Área queimada acumulada", "theme": "Fogo", "collection": "Coleção 5",
    "asset": "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_accumulated_burned_v1",
    "bands_slice": [0, 41], "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [{"label": "Não observado", "color": "fdfdfd"}, {"label": "Fogo", "color": "800000"}], "status": "experimental",
}, {
    "id": "peru_fire2_annual", "name": "Área quemada anual", "initiative": "Peru", "theme": "Fuego", "collection": "Colección 2",
    "asset": "projects/mapbiomas-public/assets/peru/fire/collection2/mapbiomas_peru_fire_collection2_annual_burned_v1",
    "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [{"label": "No observado", "color": "fdfdfd"}, {"label": "Fuego", "color": "800000"}], "status": "experimental",
}, {
    "id": "peru_fire2_accumulated", "name": "Área quemada acumulada", "initiative": "Peru", "theme": "Fuego", "collection": "Colección 2",
    "asset": "projects/mapbiomas-public/assets/peru/fire/collection2/mapbiomas_peru_fire_collection2_accumulated_burned_v1",
    "bands_slice": [0, 36], "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [{"label": "No observado", "color": "fdfdfd"}, {"label": "Fuego", "color": "800000"}], "status": "experimental",
}]

_BIOMES_ASSET = "projects/ee-ipam/assets/IBGE/limite_biomas_IBGE_2026"
TERRITORIES = [
    {"id": "bioma_amazonia", "name": "Amazônia", "asset": _BIOMES_ASSET,
     "filter": {"field": "Bioma", "value": "Amazônia"}},
    {"id": "bioma_caatinga", "name": "Caatinga", "asset": _BIOMES_ASSET,
     "filter": {"field": "Bioma", "value": "Caatinga"}},
    {"id": "bioma_cerrado", "name": "Cerrado", "asset": _BIOMES_ASSET,
     "filter": {"field": "Bioma", "value": "Cerrado"}},
    {"id": "bioma_mata_atlantica", "name": "Mata Atlântica",
     "asset": "users/wallacesilva/mata_atlantica_"},
    {"id": "bioma_pampa", "name": "Pampa", "asset": _BIOMES_ASSET,
     "filter": {"field": "Bioma", "value": "Pampa"}},
    {"id": "bioma_pantanal", "name": "Pantanal", "asset": _BIOMES_ASSET,
     "filter": {"field": "Bioma", "value": "Pantanal"}},
    {"id": "peru", "name": "Perú", "asset": "FAO/GAUL/2015/level0", "filter": {"field": "ADM0_NAME", "value": "Peru"}},
    {"id": "peru_bioma_amazonia", "name": "Amazonía (Perú)", "asset": "projects/mapbiomas-peru/assets/Biomas-peru", "filter": {"field": "nombre", "value": "Amazonía"}},
    {"id": "peru_bioma_andes", "name": "Andes (Perú)", "asset": "projects/mapbiomas-peru/assets/Biomas-peru", "filter": {"field": "nombre", "value": "Andes"}},
    {"id": "peru_bioma_desierto", "name": "Desierto costero (Perú)", "asset": "projects/mapbiomas-peru/assets/Biomas-peru", "filter": {"field": "nombre", "value": "Desierto costero"}},
    {"id": "peru_bioma_bosque_seco", "name": "Bosque seco ecuatorial (Perú)", "asset": "projects/mapbiomas-peru/assets/Biomas-peru", "filter": {"field": "nombre", "value": "Bosque seco ecuatorial"}},
    {"id": "pais_brasil", "name": "Brasil", "asset": "FAO/GAUL/2015/level0", "filter": {"field": "ADM0_NAME", "value": "Brazil"}},
]
