"""Territórios sem programar: busca nas bases oficiais, desenho, arquivo ou asset.

Todo território vira o mesmo dicionário usado pelo motor:
  {"id", "name", "asset", "filter": {"field", "values"}}   (tabela do GEE)
  {"id", "name", "geometry": <GeoJSON>}                    (desenho ou arquivo)
"""
import hashlib
import io
import json
import re
import unicodedata
import zipfile
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG = Path(__file__).with_name("config") / "territory_sources.yaml"
MAX_GEOJSON_BYTES = 900_000  # acima disso a geometria é simplificada antes de ir ao GEE


def plain(text):
    """Minúsculas sem acento, para buscar 'sao paulo' e achar 'São Paulo'."""
    text = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(c for c in text if not unicodedata.combining(c)).casefold().strip()


def display_name(raw):
    """'[Porto Velho][RO][1100205]' -> 'Porto Velho (RO)'."""
    parts = [p.strip() for p in re.findall(r"\[([^\]]*)\]", str(raw))] or [str(raw).strip()]
    parts = [p for p in parts if p and not re.fullmatch(r"[\d.]+", p)]
    if not parts:
        return str(raw)
    head, rest = parts[0], [p for p in parts[1:] if p != parts[0]]
    return f"{head} ({', '.join(rest)})" if rest else head


def _identifier(prefix, payload):
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return f"{prefix}_{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


@lru_cache(maxsize=1)
def _all_sources():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8")) or {}


def sources_for(initiative_code):
    """Fontes da iniciativa; as do Brasil servem de padrão para mapas personalizados."""
    sources = _all_sources()
    result = []
    for item in sources.get(initiative_code) or sources.get("brasil") or []:
        source = {"name_field": "NAME", "id_field": "FEATURE_ID", "where": None, "whole": False}
        source.update(item)
        result.append(source)
    return result


def _collection(ee, source):
    fc = ee.FeatureCollection(source["asset"])
    if source.get("where"):
        fc = fc.filter(ee.Filter.eq(source["where"]["field"], source["where"]["value"]))
    return fc


_INDEX = {}


def index(ee, source):
    """[(nome legível, id, texto de busca)] da fonte, carregado uma vez por sessão."""
    key = (source["asset"], json.dumps(source.get("where"), sort_keys=True))
    if key not in _INDEX:
        fc = _collection(ee, source)
        data = ee.Dictionary({"names": fc.aggregate_array(source["name_field"]),
                              "ids": fc.aggregate_array(source["id_field"])}).getInfo()
        names, ids = data.get("names") or [], data.get("ids") or []
        if len(names) != len(ids):
            raise RuntimeError("A fonte tem itens sem nome ou sem identificador.")
        rows = []
        for raw, identifier in zip(names, ids):
            name = display_name(raw) if source["name_field"] == "NAME" else str(raw)
            rows.append((name, identifier, plain(name)))
        rows.sort(key=lambda row: row[2])
        _INDEX[key] = rows
    return _INDEX[key]


def search(ee, source, text, limit=40):
    """Itens cujo nome contém o texto (sem diferenciar acento ou maiúscula)."""
    rows = index(ee, source)
    term = plain(text)
    if not term:
        return [_record(source, name, identifier) for name, identifier, _ in rows[:limit]]
    starts = [row for row in rows if row[2].startswith(term)]
    contains = [row for row in rows if term in row[2] and not row[2].startswith(term)]
    return [_record(source, name, identifier) for name, identifier, _ in (starts + contains)[:limit]]


def _record(source, name, identifier):
    return {"name": name, "value": identifier, "asset": source["asset"], "field": source["id_field"],
            "kind": source["label"]}


def from_records(records, combine=False):
    """Territórios a partir de itens buscados: um por item, ou um só juntando todos."""
    if not records:
        raise ValueError("Escolha pelo menos um item da lista.")
    groups = [records] if combine else [[record] for record in records]
    result = []
    for group in groups:
        sources = {(r["asset"], r["field"]) for r in group}
        if len(sources) != 1:
            raise ValueError("Só é possível juntar itens do mesmo tipo de território.")
        asset, field = sources.pop()
        values = sorted({r["value"] for r in group}, key=str)
        names = [r["name"] for r in sorted(group, key=lambda r: plain(r["name"]))]
        name = names[0] if len(names) == 1 else (" + ".join(names) if len(names) <= 3
                                                 else f"{names[0]} + {len(names) - 1} outros")
        result.append({"id": _identifier("busca", [asset, field, values]), "name": name,
                       "asset": asset, "filter": {"field": field, "values": values}})
    return result


def whole(source):
    """A fonte inteira como um único território (ex.: Amazônia Legal)."""
    item = {"id": _identifier("fonte", [source["asset"], source.get("where")]), "name": source["label"],
            "asset": source["asset"]}
    if source.get("where"):
        item["filter"] = {"field": source["where"]["field"], "values": [source["where"]["value"]]}
    return item


def from_asset(asset, name=None, field=None, value=None):
    """Território de um asset colado (FeatureCollection), opcionalmente filtrado."""
    asset = str(asset or "").strip()
    if not asset:
        raise ValueError("Cole o caminho do asset (ex.: projects/meu-projeto/assets/minha_area).")
    item = {"name": (name or "").strip() or asset.rstrip("/").split("/")[-1], "asset": asset}
    if field and str(value or "").strip():
        values = [v.strip() for v in str(value).split(",") if v.strip()]
        values = [int(v) if re.fullmatch(r"-?\d+", v) else v for v in values]
        item["filter"] = {"field": field.strip(), "values": values}
    item["id"] = _identifier("asset", [asset, item.get("filter")])
    return item


def from_geometry(geometry, name):
    """Território de uma geometria GeoJSON (desenho no mapa ou arquivo enviado)."""
    if not geometry or "type" not in geometry:
        raise ValueError("Geometria vazia.")
    geometry = _fit_size(geometry)
    name = (name or "").strip() or "Área desenhada"
    return {"id": _identifier("area", geometry), "name": name, "geometry": geometry}


def merge_drawings(features):
    """Junta os desenhos do mapa (lista de Features GeoJSON) em uma geometria."""
    geometries = [f.get("geometry") for f in features if f and f.get("geometry")]
    if not geometries:
        raise ValueError("Desenhe um retângulo ou polígono no mapa primeiro.")
    if len(geometries) == 1:
        return geometries[0]
    polygons = []
    for geometry in geometries:
        if geometry["type"] == "Polygon":
            polygons.append(geometry["coordinates"])
        elif geometry["type"] == "MultiPolygon":
            polygons.extend(geometry["coordinates"])
        else:
            raise ValueError("Use retângulos ou polígonos; pontos e linhas não delimitam uma área.")
    return {"type": "MultiPolygon", "coordinates": polygons}


def read_upload(filename, content):
    """GeoJSON de um arquivo enviado: .geojson/.json, .kml, .gpkg ou shapefile .zip."""
    suffix = Path(filename).suffix.lower()
    if suffix in (".geojson", ".json"):
        data = json.loads(bytes(content).decode("utf-8"))
        return _dissolve_geojson(data)
    try:
        import geopandas as gpd
    except ImportError:
        raise ValueError("Este tipo de arquivo precisa do geopandas; envie um .geojson.")
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / Path(filename).name
        path.write_bytes(bytes(content))
        if suffix == ".zip":
            with zipfile.ZipFile(path) as archive:
                archive.extractall(tmp)
            shapes = list(Path(tmp).rglob("*.shp"))
            if len(shapes) != 1:
                raise ValueError("O .zip deve conter exatamente um shapefile (.shp com .shx, .dbf e .prj).")
            frame = gpd.read_file(shapes[0])
        elif suffix in (".kml", ".gpkg", ".shp"):
            frame = gpd.read_file(path)
        else:
            raise ValueError("Formato não suportado. Use .geojson, .kml, .gpkg ou um shapefile em .zip.")
    if frame.empty:
        raise ValueError("O arquivo não tem feições.")
    if frame.crs is not None and frame.crs.to_epsg() != 4326:
        frame = frame.to_crs(4326)
    geometry = frame.geometry.union_all() if hasattr(frame.geometry, "union_all") else frame.geometry.unary_union
    return json.loads(gpd.GeoSeries([geometry], crs=4326).to_json())["features"][0]["geometry"]


def _dissolve_geojson(data):
    if data.get("type") == "FeatureCollection":
        features = data.get("features") or []
    elif data.get("type") == "Feature":
        features = [data]
    else:
        features = [{"geometry": data}]
    return merge_drawings(features)


def _fit_size(geometry):
    """Simplifica geometrias muito detalhadas para caber no pedido ao Earth Engine."""
    if len(json.dumps(geometry)) <= MAX_GEOJSON_BYTES:
        return geometry
    try:
        from shapely.geometry import mapping, shape
    except ImportError:
        raise ValueError("A geometria é muito detalhada; simplifique o arquivo antes de enviar.")
    shp = shape(geometry)
    tolerance = 0.0002
    while len(json.dumps(mapping(shp))) > MAX_GEOJSON_BYTES and tolerance < 0.1:
        shp = shape(geometry).simplify(tolerance, preserve_topology=True)
        tolerance *= 2
    return json.loads(json.dumps(mapping(shp)))


def feature_collection(ee, territory):
    """FeatureCollection do território, para mapa e prévia."""
    if territory.get("geometry"):
        return ee.FeatureCollection([ee.Feature(ee.Geometry(territory["geometry"]))])
    fc = ee.FeatureCollection(territory["asset"])
    filt = territory.get("filter")
    if filt:
        values = filt["values"] if "values" in filt else [filt["value"]]
        fc = fc.filter(ee.Filter.inList(filt["field"], values))
    return fc
