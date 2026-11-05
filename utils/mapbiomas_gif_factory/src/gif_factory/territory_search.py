"""Consulta controlada de territórios em FeatureCollections do usuário."""
import hashlib
import json

# Same GAUL administrative source already used for country presets. This is
# the FAO 2025 snapshot, not the current official IBGE municipal boundary.
BRAZIL_MUNICIPALITIES = "FAO/GAUL/2025/level2"
TERRITORY_SEARCH_SOURCES = {
    "Município": {"asset": BRAZIL_MUNICIPALITIES, "name_field": "GAUL2_NAME", "id_field": "GAUL2_CODE"},
    # Add verified sources and field names here; never infer territory geometry.
    "Unidade de conservação": {"asset": None, "name_field": None, "id_field": None},
    "Terra indígena": {"asset": None, "name_field": None, "id_field": None},
}
UF = {"Acre": "AC", "Alagoas": "AL", "Amapa": "AP", "Amazonas": "AM",
      "Bahia": "BA", "Ceara": "CE", "Distrito Federal": "DF", "Espirito Santo": "ES",
      "Goias": "GO", "Maranhao": "MA", "Mato Grosso": "MT", "Mato Grosso do Sul": "MS",
      "Minas Gerais": "MG", "Para": "PA", "Paraiba": "PB", "Parana": "PR",
      "Pernambuco": "PE", "Piaui": "PI", "Rio de Janeiro": "RJ", "Rio Grande do Norte": "RN",
      "Rio Grande do Sul": "RS", "Rondonia": "RO", "Roraima": "RR", "Santa Catarina": "SC",
      "Sao Paulo": "SP", "Sergipe": "SE", "Tocantins": "TO"}


def search_municipalities(ee, text, limit=30):
    if len(text.strip()) < 2:
        raise ValueError("Informe ao menos duas letras do município.")
    fc = (ee.FeatureCollection(BRAZIL_MUNICIPALITIES)
          .filter(ee.Filter.eq("ISO3_CODE", "BRA"))
          .filter(ee.Filter.stringContains("GAUL2_NAME", text.strip()))
          .select(["GAUL2_NAME", "GAUL1_NAME", "GAUL2_CODE"], retainGeometry=False)
          .limit(limit))
    result = []
    for feature in fc.getInfo().get("features", []):
        props = feature.get("properties") or {}
        if props.get("GAUL2_NAME") and props.get("GAUL1_NAME") and props.get("GAUL2_CODE") is not None:
            import unicodedata
            state = props["GAUL1_NAME"]
            plain = "".join(c for c in unicodedata.normalize("NFKD", state) if not unicodedata.combining(c))
            result.append({"name": f"{props['GAUL2_NAME']} ({UF.get(plain, state)})",
                           "value": props["GAUL2_CODE"], "asset": BRAZIL_MUNICIPALITIES,
                           "field": "GAUL2_CODE"})
    return result


def search(ee, asset, name_field, id_field, text, limit=30):
    if not all(x.strip() for x in (asset, name_field, id_field, text)):
        raise ValueError("Informe asset, campos e busca / Enter asset, fields and search / Indique asset, campos y búsqueda.")
    collection = ee.FeatureCollection(asset).filter(ee.Filter.stringContains(name_field, text.strip()))
    data = collection.limit(limit).getInfo()
    results = []
    for feature in data.get("features", []):
        props = feature.get("properties") or {}
        identifier = feature.get("id") if id_field == "system:index" else props.get(id_field)
        if identifier is not None and props.get(name_field) is not None:
            results.append({"name": str(props[name_field]), "value": identifier,
                            "asset": asset, "field": id_field})
    return results


def search_configured(ee, kind, text, limit=30):
    source = TERRITORY_SEARCH_SOURCES.get(kind) or {}
    if not all(source.get(key) for key in ("asset", "name_field", "id_field")):
        raise ValueError("A fonte oficial deste tipo ainda não está conectada.")
    return search(ee, source["asset"], source["name_field"], source["id_field"], text, limit)


def territory(records):
    if not records:
        raise ValueError("Selecione um resultado / Select a result / Seleccione un resultado.")
    assets = {(r["asset"], r["field"]) for r in records}
    if len(assets) != 1:
        raise ValueError("Use o mesmo asset e campo / Use the same asset and field / Use el mismo asset y campo.")
    asset, field = assets.pop()
    values = sorted({json.dumps(r["value"], sort_keys=True, ensure_ascii=False) for r in records})
    identity = json.dumps([asset, field, values], ensure_ascii=False)
    tid = "busca_" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16]
    return {"id": tid, "name": " + ".join(sorted({r["name"] for r in records})),
            "asset": asset, "filter": {"field": field, "values": [json.loads(v) for v in values]}}
