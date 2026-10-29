"""Exemplo inicial; validar resultado visual antes de uso em produção."""

LAYERS = [{
    "id": "annual_burned", "name": "Área queimada anual — Fogo coleção 5",
    "asset": "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_v1",
    "bands": ["burned_area_2023", "burned_area_2024"],
    "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [{"label": "Não observado", "color": "fdfdfd"}, {"label": "Fogo", "color": "800000"}],
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
]
