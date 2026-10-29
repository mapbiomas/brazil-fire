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

LAYERS.append({
    "id": "soil_carbon_col3", "name": "Carbono orgânico do solo", "theme": "Solo", "collection": "Coleção 3",
    "asset": "projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_soil_carbon_v1",
    "visualization": {"min": -1, "max": 80, "palette": ["ffffff", "ffffe5", "fff7bc", "fee391", "fec44f", "fe9929", "ec7014", "cc4c02", "993404", "662506"], "cmap_type": "sequential", "unit": "t/ha"},
    "legend": [{"label": "-1 t/ha", "color": "ffffff"}, {"label": "80 t/ha", "color": "662506"}],
    "legend_label": "Carbono orgânico do solo (t/ha)", "status": "experimental",
})

_FREQ_BR = ["fdfdfd", "faf3cd", "fdf1b0", "fbe981", "fae258", "f9de44", "f8d71f", "f1ca1e", "e9bc1c", "e2af1a", "daa118", "d29316", "cc8715", "c57b14", "bd6c12", "b76011", "b0540f", "a8450d", "9f360b", "992a0a", "921e08", "8c1307", "810004", "770105", "6a0306", "600407", "5a0508", "4b0709", "450709", "3f0708", "390608", "330607", "2d0506", "270506", "220505", "1c0404", "170404", "120303", "0d0203", "080202", "040101"]
_FREQ_PE = ["ffffff", "faf3cd", "fdf1b0", "fbe981", "fae258", "f9de44", "f6d21e", "eebe1b", "e4b11a", "daa418", "d19416", "c88515", "be7013", "b66011", "ad500f", "a4400c", "9b300a", "922108", "8a1506", "810104", "740205", "6a0306", "610407", "590508", "500608", "470709", "3f0708", "380608", "310607", "2a0506", "240505", "1d0404", "170404", "120303", "0d0203", "080202", "040101"]
LAYERS.extend([
    {"id": "fire_frequency_br5", "name": "Frequência da área queimada", "theme": "Fogo", "collection": "Coleção 5",
     "asset": "projects/mapbiomas-public/assets/brazil/fire/collection5/mapbiomas_fire_collection5_fire_frequency_v1", "bands_slice": [0, 41],
     "visualization": {"min": 0, "max": 40, "palette": _FREQ_BR, "cmap_type": "sequential"},
     "legend": [{"label": "0", "color": _FREQ_BR[0]}, {"label": "40", "color": _FREQ_BR[-1]}],
     "legend_label": "Frequência de queimadas", "status": "experimental"},
    {"id": "fire_frequency_pe2", "name": "Frecuencia del área quemada", "initiative": "Peru", "theme": "Fuego", "collection": "Colección 2",
     "asset": "projects/mapbiomas-public/assets/peru/fire/collection2/mapbiomas_peru_fire_collection2_frequency_burned_v1", "bands_slice": [0, 36],
     "visualization": {"min": 0, "max": 36, "palette": _FREQ_PE, "cmap_type": "sequential"},
     "legend": [{"label": "0", "color": _FREQ_PE[0]}, {"label": "36", "color": _FREQ_PE[-1]}],
     "legend_label": "Frecuencia de incendios", "status": "experimental"},
])

for product, title, band, version in [
    ("soil_clay_fraction", "Fração de argila", "clay_fraction", "v1"),
    ("soil_sand_fraction", "Fração de areia", "sand_fraction", "v1"),
    ("soil_silt_fraction", "Fração de silte", "silt_fraction", "v1"),
    ("soil_stoniness", "Pedregosidade", "stoniness", "v2"),
    ("soil_textural_class", "Classe textural", "textural_class", "v1"),
    ("soil_textural_group", "Grupo textural", "textural_group", "v1"),
    ("soil_textural_subgroup", "Subgrupo textural", "textural_subgroup", "v1"),
]:
    LAYERS.append({"id": product + "_col3", "name": title, "theme": "Solo", "collection": "Coleção 3",
                   "asset": f"projects/mapbiomas-public/assets/brazil/soil/collection3/mapbiomas_brazil_collection3_{product}_{version}",
                   "axis": "depth", "depth_band": band, "enabled": False,
                   "reason": "Validar legenda, banda e visualização para esta coleção antes de habilitar."})

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
