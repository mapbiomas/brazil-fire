"""Consulta controlada de territórios em FeatureCollections do usuário."""
import hashlib
import json


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
