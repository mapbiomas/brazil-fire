"""Registro de legendas do catálogo unificado.

Fontes:
- legends.json: colhido de config/visualization.yaml da Fábrica anterior
  (fogo, cobertura, uso, degradação, solo, desmatamento).
- vera_fire_legends.json: snapshot do registro fire_legends, com identidade
  explícita por iniciativa, coleção, produto e asset.
- Outras sugestões provenientes de pesquisa não são automaticamente oficiais.

Tudo permanece `experimental` até validação visual no GEE: a presença de uma
paleta não comprova acesso ao asset, bandas, máscara ou interpretação.
"""
import json
import re
from pathlib import Path


def _registry():
    path = Path(__file__).with_name("legends.json")
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


_REGISTRY = _registry()

# Snapshot independente do repositório de origem; não implica validação no GEE.
_VERA_PATH = Path(__file__).with_name("vera_fire_legends.json")


def _collection_key(value):
    """Normaliza apenas versões numéricas, preservando todos os componentes."""
    if not isinstance(value, str):
        return value
    match = re.fullmatch(r"(?:collection_)?([0-9]+(?:[._][0-9]+)*)", value)
    if not match:
        return value
    return ".".join(str(int(part)) for part in re.split(r"[._]", match.group(1)))


def _vera_key(key):
    initiative, collection, product, asset = key
    return initiative, _collection_key(collection), product, asset


_VERA = {}
for _entry in json.loads(_VERA_PATH.read_text(encoding="utf-8"))["entries"]:
    _key = _vera_key(_entry["key"])
    if _key in _VERA:
        raise ValueError(f"Chave de legenda duplicada após normalização: {_key}")
    _VERA[_key] = _entry


def _vera_style(entry, lang):
    if not entry["representable"]:
        return None
    classes = entry["classes"]
    return {
        "visualization": {"min": classes[0]["value"], "max": classes[-1]["value"],
                          "palette": [c["color"] for c in classes], "cmap_type": "categorical",
                          "prefix_labels": False, "mask_color": "ffffff"},
        "legend": [{"label": c["labels"][lang], "color": c["color"]} for c in classes],
    }

INITIATIVES = {
    "brasil": "Brasil", "peru": "Peru", "paraguay": "Paraguai",
    "argentina": "Argentina", "bolivia": "Bolívia", "chile": "Chile",
    "colombia": "Colômbia", "ecuador": "Equador", "indonesia": "Indonésia",
    "venezuela": "Venezuela",
}

LANGUAGES = {
    "brasil": "pt",
    "peru": "es", "paraguay": "es", "argentina": "es", "bolivia": "es",
    "chile": "es", "colombia": "es", "ecuador": "es", "venezuela": "es",
    "indonesia": "en",
}

THEME_CANON = {
    "fire": "fire", "fogo": "fire", "fuego": "fire",
    "soil": "soil", "solo": "soil", "suelo": "soil",
    "lulc": "lulc", "lulc_10m": "lulc_10m", "land_cover": "lulc",
    "water": "water", "agua": "water", "água": "water",
    "atmosphere": "atmosphere", "atmosfera": "atmosphere", "atmósfera": "atmosphere",
    "climate_risk": "climate_risk",
    "urban": "urban", "urbano": "urban",
    "degradation": "degradation", "degradacao": "degradation", "degradação": "degradation", "degradacion": "degradation", "degradación": "degradation",
}

THEME_LABELS = {
    "fire": {"pt": "Fogo", "es": "Fuego", "en": "Fire"},
    "soil": {"pt": "Solo", "es": "Suelo", "en": "Soil"},
    "lulc": {"pt": "Uso e cobertura", "es": "Uso y cobertura", "en": "Land use and cover"},
    "lulc_10m": {"pt": "Uso e cobertura 10m", "es": "Uso y cobertura 10m", "en": "Land use and cover 10m"},
    "water": {"pt": "Água", "es": "Agua", "en": "Water"},
    "atmosphere": {"pt": "Atmosfera", "es": "Atmósfera", "en": "Atmosphere"},
    "climate_risk": {"pt": "Risco climático", "es": "Riesgo climático", "en": "Climate risk"},
    "urban": {"pt": "Urbano", "es": "Urbano", "en": "Urban"},
    "degradation": {"pt": "Degradação", "es": "Degradación", "en": "Degradation"},
}


def canonical_theme(theme):
    if theme is None:
        return "outros"
    key = str(theme).strip().lower()
    return THEME_CANON.get(key, key)


def theme_label(canonical, lang="pt"):
    return THEME_LABELS.get(canonical, {}).get(lang, canonical.replace("_", " ").title())


def normalize_collection(value):
    """Normaliza nomes de coleção para um formato numérico consistente.

    "collection_05_1" -> "5.1", "collection_05" -> "5", "Coleção 5.1" -> "5.1",
    "Colección 2" -> "2", "Collection 11" -> "11", "5.1" -> "5.1".
    Outros valores ("monitor", "beta", "Sem coleção") permanecem.
    """
    text = str(value or "").strip()
    m = re.match(r"collection_(\d+)(?:_(\d+))?$", text)
    if m:
        return f"{int(m.group(1))}.{int(m.group(2))}" if m.group(2) else str(int(m.group(1)))
    m = re.match(r"(?:cole[cç][aã]o|coleccion|colección|collection|colecao)\s+(\d+(?:\.\d+)?)$", text, re.IGNORECASE)
    if m:
        return m.group(1)
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        return text
    return text


def lang_for(code):
    return LANGUAGES.get(code, "pt")


# --- Paletas do relatório (não presentes no offline) -----------------------
# Cada classe: (value, pt, en, es, color)
_WATER_BINARY = [
    (0, "Sem água", "Non-water", "Sin agua", "ffffff"),
    (1, "Água", "Water", "Agua", "0000ff"),
]
_WATER_FREQUENCY = [
    (1, "1 ano", "1 year", "1 año", "c6dbef"),
    (5, "5 anos", "5 years", "5 años", "9ecae1"),
    (10, "10 anos", "10 years", "10 años", "6baed6"),
    (20, "20 anos", "20 years", "20 años", "3182bd"),
    (30, "30 anos", "30 years", "30 años", "1565c0"),
    (35, "35 anos", "35 years", "35 años", "08519c"),
    (41, "41 anos", "41 years", "41 años", "022364"),
]
_WATER_BODIES = [
    (1, "Natural", "Natural", "Natural", "2532e4"),
    (2, "Represa", "Reservoir", "Embalse", "00ffff"),
    (3, "Aquicultura", "Aquaculture", "Acuicultura", "091077"),
    (4, "Outros artificiais", "Other artificial", "Otros artificiales", "519799"),
]
_WATER_TREND = [
    (1, "Forte decréscimo", "Strong decrease", "Fuerte disminución", "b2182b"),
    (2, "Médio decréscimo", "Medium decrease", "Disminución moderada", "ef8a62"),
    (3, "Leve decréscimo", "Slight decrease", "Ligera disminución", "fddbc7"),
    (4, "Leve acréscimo", "Slight increase", "Ligero aumento", "d1e5f0"),
    (5, "Médio acréscimo", "Medium increase", "Aumento moderado", "67a9cf"),
    (6, "Forte acréscimo", "Strong increase", "Fuerte aumento", "2166ac"),
]
_SOIL_CARBON = [
    (1, "0 a 10 t/ha", "0 to 10 t/ha", "0 a 10 t/ha", "ffffe5"),
    (2, "10 a 20 t/ha", "10 to 20 t/ha", "10 a 20 t/ha", "fff7bc"),
    (3, "20 a 30 t/ha", "20 to 30 t/ha", "20 a 30 t/ha", "fee391"),
    (4, "30 a 40 t/ha", "30 to 40 t/ha", "30 a 40 t/ha", "fec44f"),
    (5, "40 a 50 t/ha", "40 to 50 t/ha", "40 a 50 t/ha", "fe9929"),
    (6, "50 a 60 t/ha", "50 to 60 t/ha", "50 a 60 t/ha", "ec7014"),
    (7, "60 a 70 t/ha", "60 to 70 t/ha", "60 a 70 t/ha", "cc4c02"),
    (8, "70 a 80 t/ha", "70 to 80 t/ha", "70 a 80 t/ha", "993404"),
    (9, "> 80 t/ha", "> 80 t/ha", "> 80 t/ha", "662506"),
]
_SOIL_CLAY = [(v, f"{v*10-10} – {v*10}%", f"{v*10-10} – {v*10}%", f"{v*10-10} – {v*10}%", c) for v, c in enumerate(
    ["debbbd", "aa8686", "956262", "9c5050", "8c3306", "932703", "7a1d03", "6c1902", "5c1402", "4e1101"], 1)]
_SOIL_SILT = [(v, f"{v*10-10} – {v*10}%", f"{v*10-10} – {v*10}%", f"{v*10-10} – {v*10}%", c) for v, c in enumerate(
    ["d1e9c9", "a9d0a1", "88c997", "7db78e", "6a8e74", "517a64", "406a55", "2e5a49", "1f4a3c", "123029"], 1)]
_SOIL_SAND = [(v, f"{v*10-10} – {v*10}%", f"{v*10-10} – {v*10}%", f"{v*10-10} – {v*10}%", c) for v, c in enumerate(
    ["ffe5d8", "ffccb3", "f7a886", "ff9680", "e08370", "d67258", "b05841", "95413b", "6a2a12", "522010"], 1)]
_SOIL_TEXTURE = [
    (1, "Muito argilosa", "Very clayey", "Muy arcillosa", "6c1902"),
    (2, "Argilosa", "Clayey", "Arcillosa", "9c5050"),
    (3, "Argilo arenosa", "Sandy clay", "Arcillo arenosa", "aa8686"),
    (4, "Argilo siltosa", "Silty clay", "Arcillo limosa", "3481a7"),
    (5, "Franco argilosa", "Clay loam", "Franco arcillosa", "e9a9a9"),
    (6, "Franco argilo siltosa", "Silty clay loam", "Franco arcillo limosa", "80b1d3"),
    (7, "Argilo arenosa média", "Sandy clay intermediate", "Arcillo arenosa media", "c994c7"),
    (8, "Franco argilo arenosa", "Sandy clay loam", "Franco arcillo arenosa", "f4a582"),
    (9, "Franca", "Loam", "Franca", "d7c5a5"),
    (10, "Franco arenosa", "Sandy loam", "Franco arenosa", "f8d488"),
    (11, "Areia", "Sand", "Arena", "fffe73"),
    (12, "Areia franca", "Loamy sand", "Arena franca", "e4b074"),
    (13, "Franco siltosa", "Silt loam", "Franco limosa", "abba7c"),
    (14, "Silte", "Silt", "Limo", "298289"),
    (15, "Rocha", "Rock", "Roca", "505050"),
]
_SOIL_STONINESS = [
    (1, "0 – 10 cm", "0 – 10 cm", "0 – 10 cm", "56008f"),
    (2, "10 – 20 cm", "10 – 20 cm", "10 – 20 cm", "7d00a3"),
    (3, "20 – 30 cm", "20 – 30 cm", "20 – 30 cm", "c300a9"),
    (4, "30 – 40 cm", "30 – 40 cm", "30 – 40 cm", "ff2f9a"),
    (5, "40 – 50 cm", "40 – 50 cm", "40 – 50 cm", "ff4f78"),
    (6, "50 – 60 cm", "50 – 60 cm", "50 – 60 cm", "ff6e63"),
    (7, "60 – 70 cm", "60 – 70 cm", "60 – 70 cm", "ff9678"),
    (8, "70 – 80 cm", "70 – 80 cm", "70 – 80 cm", "ffc5a1"),
    (9, "80 – 90 cm", "80 – 90 cm", "80 – 90 cm", "ffdfc6"),
    (10, "90 – 100 cm", "90 – 100 cm", "90 – 100 cm", "fff8eb"),
    (11, "100 cm", "100 cm", "100 cm", "fbf9f6"),
]
_ATMO_TEMP = [
    (0, "0 °C", "0 °C", "0 °C", "0000ff"),
    (15, "15 °C", "15 °C", "15 °C", "00ffff"),
    (25, "25 °C", "25 °C", "25 °C", "ffff00"),
    (35, "35 °C", "35 °C", "35 °C", "ff7700"),
    (45, "45 °C", "45 °C", "45 °C", "d4271e"),
]
_ATMO_PRECIP = [
    (0, "0 mm", "0 mm", "0 mm", "ffffff"),
    (500, "500 mm", "500 mm", "500 mm", "e0f3f8"),
    (1500, "1500 mm", "1500 mm", "1500 mm", "67a9cf"),
    (2500, "2500 mm", "2500 mm", "2500 mm", "02818a"),
    (4000, "4000 mm", "4000 mm", "4000 mm", "014636"),
]
_ATMO_VPD = [
    (0, "0.0 kPa", "0.0 kPa", "0.0 kPa", "2c7bb6"),
    (1, "1.0 kPa", "1.0 kPa", "1.0 kPa", "abd9e9"),
    (2, "2.0 kPa", "2.0 kPa", "2.0 kPa", "ffffbf"),
    (3, "3.0 kPa", "3.0 kPa", "3.0 kPa", "fdae61"),
    (4, "4.0 kPa", "4.0 kPa", "4.0 kPa", "d7191c"),
]
_ATMO_AIRQ = [
    (10, "Boa", "Good", "Buena", "00e400"),
    (25, "Moderada", "Moderate", "Moderada", "ffff00"),
    (50, "Ruim", "Unhealthy", "Mala", "ff7e00"),
    (75, "Muito ruim", "Very unhealthy", "Muy mala", "ff0000"),
    (150, "Crítica", "Hazardous", "Peligrosa", "7e0023"),
]
_ATMO_HUMID = [
    (20, "20%", "20%", "20%", "d7191c"),
    (40, "40%", "40%", "40%", "fdae61"),
    (60, "60%", "60%", "60%", "ffffbf"),
    (80, "80%", "80%", "80%", "abd9e9"),
    (100, "100%", "100%", "100%", "2c7bb6"),
]
_ATMO_WATER = [
    (-1000, "Déficit severo", "Severe deficit", "Déficit severo", "d7191c"),
    (-500, "Déficit moderado", "Moderate deficit", "Déficit moderado", "fdae61"),
    (0, "Balanço neutro", "Neutral balance", "Balance neutro", "ffffbf"),
    (500, "Superávit moderado", "Moderate surplus", "Superávit moderado", "abd9e9"),
    (1500, "Superávit elevado", "High surplus", "Superávit elevado", "2c7bb6"),
]
_URBAN_ANNUAL = [
    (0, "Não urbanizada", "Non-urbanized", "No urbanizada", "ffffff"),
    (1, "Urbanizada", "Urbanized", "Urbanizada", "d4271e"),
]
_URBAN_PERIODS = [
    (1, "Até 1985", "By 1985", "Hasta 1985", "4a148c"),
    (2, "1986 – 1990", "1986 – 1990", "1986 – 1990", "6a1b9a"),
    (3, "1991 – 1995", "1991 – 1995", "1991 – 1995", "8e24aa"),
    (4, "1996 – 2000", "1996 – 2000", "1996 – 2000", "ab47bc"),
    (5, "2001 – 2005", "2001 – 2005", "2001 – 2005", "ba68c8"),
    (6, "2006 – 2010", "2006 – 2010", "2006 – 2010", "ce93d8"),
    (7, "2011 – 2020", "2011 – 2020", "2011 – 2020", "e1bee7"),
    (8, "2021 – 2025", "2021 – 2025", "2021 – 2025", "f3e5f5"),
]
_CLIMATE_LANDSLIDE = [
    (1, "Baixa", "Low", "Baja", "fee5d9"),
    (2, "Média", "Moderate", "Moderada", "fcae91"),
    (3, "Alta", "High", "Alta", "fb6a4a"),
    (4, "Muito alta", "Very high", "Muy alta", "cb181d"),
]
_CLIMATE_FLOOD = [
    (1, "Baixa", "Low", "Baja", "eff3ff"),
    (2, "Média", "Moderate", "Moderada", "bdd7e7"),
    (3, "Alta", "High", "Alta", "6baed6"),
    (4, "Muito alta", "Very high", "Muy alta", "2171b5"),
]


def _categorical(classes, lang="pt"):
    colors = [c for _, _, _, _, c in classes]
    idx = {"pt": 1, "en": 2, "es": 3}[lang]
    labels = [row[idx] for row in classes]
    values = [row[0] for row in classes]
    return {
        "visualization": {"min": min(values), "max": max(values), "palette": colors, "cmap_type": "categorical", "prefix_labels": False},
        "legend": [{"label": label, "color": color} for label, color in zip(labels, colors)],
    }


def _sequential(classes, lang="pt"):
    colors = [c for _, _, _, _, c in classes]
    values = [row[0] for row in classes]
    idx = {"pt": 1, "en": 2, "es": 3}[lang]
    labels = [row[idx] for row in classes]
    return {
        "visualization": {"min": min(values), "max": max(values), "palette": colors, "cmap_type": "sequential"},
        "legend": [{"label": labels[0], "color": colors[0]}, {"label": labels[-1], "color": colors[-1]}],
    }


def _from_registry(key, lang="pt"):
    entry = _REGISTRY.get(key)
    if entry is None:
        return None
    palette = entry["palette"]
    viz = {"min": entry.get("min", 0), "max": entry.get("max", 1), "palette": palette, "cmap_type": entry.get("cmap_type", "sequential")}
    if "nodata_value" in entry:
        viz["nodata_value"] = entry["nodata_value"]
    if entry.get("nodata_color"):
        viz["nodata_color"] = entry["nodata_color"]
    labels = entry.get("discrete_labels")
    if labels:
        legend = [{"label": str(labels[i]) if i < len(labels) else "", "color": palette[i]}
                  for i in range(len(palette))]
    else:
        legend = [{"label": str(entry.get("min", 0)), "color": palette[0]},
                  {"label": str(entry.get("max", 1)), "color": palette[-1]}]
    return {"visualization": viz, "legend": legend}


# Mapeamento produto -> chave de legenda do offline (ou função EXTRA).
def style_for(record, lang="pt"):
    """Return {visualization, legend} for a record, or None when unknown."""
    theme = canonical_theme(record.get("theme"))
    name = str(record.get("product") or "").lower().replace("_", " ").strip()
    code = record.get("initiative", "brasil")

    if theme == "fire":
        key = _vera_key(tuple(record.get(k) for k in ("initiative", "collection", "product", "asset")))
        entry = _VERA.get(key)
        return _vera_style(entry, lang) if entry is not None else None

    if theme == "lulc" or theme == "lulc_10m":
        if name in {"lulc nivel0", "nivel0", "lulc l0"}:
            return _from_registry("lulc_nivel0", lang)
        if name in {"lulc nivel1", "nivel1", "lulc l1"}:
            return _from_registry("lulc_nivel1", lang)
        if name in {"lulc nivel1 1"}:
            return _from_registry("lulc_nivel1_1", lang)
        if name in {"lulc nivel2", "nivel2", "lulc l2"}:
            return _from_registry("lulc_nivel2", lang)
        if code == "peru":
            return _from_registry("lulc_peru", lang)
        if code == "drc":
            return _from_registry("lulc_drc_nivel0", lang)
        return _from_registry("lulc", lang)

    if theme == "water":
        if "frequency" in name:
            return _sequential(_WATER_FREQUENCY, lang)
        if "trend" in name:
            return _categorical(_WATER_TREND, lang)
        if "bodies" in name or "body" in name:
            return _categorical(_WATER_BODIES, lang)
        return _categorical(_WATER_BINARY, lang)

    if theme == "soil":
        if "soc" in name or "carbon" in name or "organic" in name:
            return _categorical(_SOIL_CARBON, lang)
        if "clay" in name or "argila" in name:
            return _categorical(_SOIL_CLAY, lang)
        if "silt" in name or "silte" in name:
            return _categorical(_SOIL_SILT, lang)
        if "sand" in name or "areia" in name:
            return _categorical(_SOIL_SAND, lang)
        if "textural" in name or "texture" in name or "textura" in name:
            return _categorical(_SOIL_TEXTURE, lang)
        if "stoniness" in name or "pedregosidade" in name:
            return _categorical(_SOIL_STONINESS, lang)
        return None

    if theme == "atmosphere":
        if name in {"lulc l0", "lulc l1", "lulc l2"}:
            return _from_registry({"lulc l0": "lulc_nivel0", "lulc l1": "lulc_nivel1", "lulc l2": "lulc_nivel2"}[name], lang)
        if "temperature" in name:
            return _sequential(_ATMO_TEMP, lang)
        if "precipitation" in name or "rain" in name:
            return _sequential(_ATMO_PRECIP, lang)
        if "vapor pressure" in name or "vpd" in name:
            return _sequential(_ATMO_VPD, lang)
        if "pm2" in name or "pm10" in name or "air quality" in name or "poluente" in name:
            return _sequential(_ATMO_AIRQ, lang)
        if "humidity" in name or "umidade" in name:
            return _sequential(_ATMO_HUMID, lang)
        if "water availability" in name or "disponibilidade" in name:
            return _sequential(_ATMO_WATER, lang)
        return None

    if theme == "urban":
        if "period" in name:
            return _categorical(_URBAN_PERIODS, lang)
        return _categorical(_URBAN_ANNUAL, lang)

    if theme == "climate_risk":
        if "flood" in name or "inunda" in name:
            return _categorical(_CLIMATE_FLOOD, lang)
        if "landslide" in name or "desliz" in name:
            return _categorical(_CLIMATE_LANDSLIDE, lang)
        return None

    return None
