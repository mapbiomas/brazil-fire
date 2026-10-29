"""Exemplo inicial; validar resultado visual antes de uso em produção."""

LAYERS = [{
    "id": "annual_burned", "name": "Área queimada anual — Fogo coleção 5",
    "asset": "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_annual_burned_v1",
    "bands": ["burned_area_2023", "burned_area_2024"],
    "visualization": {"min": 0, "max": 1, "palette": ["fdfdfd", "800000"]},
    "legend": [{"label": "Não observado", "color": "fdfdfd"}, {"label": "Fogo", "color": "800000"}],
}]

TERRITORIES = [{
    "id": "uf_df", "name": "Distrito Federal",
    "asset": "projects/mapbiomas-workspace/AUXILIAR/estados-2017",
    "filter": {"field": "NM_ESTADO", "value": "DISTRITO FEDERAL"},
}]
