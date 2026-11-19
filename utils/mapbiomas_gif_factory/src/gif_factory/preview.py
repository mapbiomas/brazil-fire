"""Prévias no Earth Engine: camada no mapa, miniatura e anos disponíveis."""
from .engine import catalog_image, discover, is_catalog, load_image, prepare_layer
from .temporal import temporal_key
from .territories import feature_collection


def visualization(layer):
    """Parâmetros efetivos (min, max, palette) da camada, com tokens de tema resolvidos."""
    if is_catalog(layer):
        from mapbiomas_data.core import VisualizationManager
        from .catalog import config_loader
        return VisualizationManager(config_loader()).get_viz_params(layer["visualization_key"])
    return dict(layer["visualization"])


def years(layer):
    """(primeiro ano, último ano) do produto, ou None se não for série temporal."""
    all_bands, _ = discover(dict(layer, start_year=None, end_year=None, bands=None))
    keys = [temporal_key(band) for band in all_bands]
    if not keys or any(key is None for key in keys):
        return None
    return min(key[0] for key in keys), max(key[0] for key in keys)


def styled_image(ee, layer, territory=None, year=None):
    """Imagem RGB de um ano (o último, se não informado), recortada no território."""
    if is_catalog(layer):
        image, product = catalog_image(layer)
    else:
        prepared = prepare_layer(layer)
        image, product = load_image(prepared), prepared
    bands = image.bandNames().getInfo()
    if year is not None:
        matching = [b for b in bands if (temporal_key(b) or (None,))[0] == year]
        bands = matching or bands
    viz = visualization(layer)
    if product.get("rgb"):
        tail = bands[-1][-4:]
        group = [b for b in bands if b.endswith(tail)][:3]
        styled = image.select(group).visualize(min=viz.get("min", 0), max=viz.get("max", 1))
    elif viz.get("random_viz") or not viz.get("palette"):
        styled = image.select(bands[-1]).randomVisualizer().select(["viz-red", "viz-green", "viz-blue"])
    else:
        styled = image.select(bands[-1]).visualize(min=viz.get("min", 0), max=viz.get("max", 1),
                                                   palette=viz["palette"])
    label = bands[-1]
    if territory:
        styled = styled.clip(feature_collection(ee, territory).geometry())
    return styled, label


def tile_url(ee, image, vis=None):
    return ee.Image(image).getMapId(vis or {})["tile_fetcher"].url_format


def with_lines(ee, image, territory):
    """Desenha as linhas escolhidas (cor e espessura) dentro do território, como no arquivo final."""
    inside = ee.Image().paint(feature_collection(ee, territory), 0).eq(0)
    for line in territory.get("overlays") or []:
        painted = ee.Image().byte().paint(feature_collection(ee, line), 1, int(line.get("width", 1))).updateMask(inside)
        image = image.blend(painted.visualize(palette=[str(line.get("color", "000000")).lstrip("#")]))
    return image


def thumbnail_url(ee, layer, territory, year=None, size=640):
    image, label = styled_image(ee, layer, territory, year)
    image = with_lines(ee, image, territory)
    region = feature_collection(ee, territory).geometry().bounds()
    return image.getThumbURL({"region": region, "dimensions": size, "format": "png"}), label


def layer_thumbnail(ee, layer, territory=None, size=560):
    """Miniatura do último período: no território, se houver; senão na área do próprio asset."""
    if territory:
        return thumbnail_url(ee, layer, territory, size=size)
    image, label = styled_image(ee, layer)
    footprint = image.geometry().bounds()
    # Imagens sem área definida (globais) caem num recorte do Brasil.
    unbounded = ee.Number(footprint.area(1000)).gt(1e14)
    region = ee.Geometry(ee.Algorithms.If(unbounded, ee.Geometry.Rectangle([-74, -34, -34, 5.5]), footprint))
    return image.getThumbURL({"region": region, "dimensions": size, "format": "png"}), label


def outline_url(ee, territory, color="d81b60"):
    fc = feature_collection(ee, territory)
    return tile_url(ee, fc.style(color=color, fillColor="00000000", width=2))


def center_zoom(box):
    """Centro e zoom aproximados que enquadram [[sul, oeste], [norte, leste]] num mapa de ~600 px."""
    import math
    (south, west), (north, east) = box
    span = max(east - west, (north - south) * 1.4, 1e-4)
    zoom = int(max(3, min(14, math.floor(math.log2(360 / span)) + 1)))
    return ((south + north) / 2, (west + east) / 2), zoom


def bounds(ee, territory):
    """[[sul, oeste], [norte, leste]] para enquadrar o mapa."""
    ring = feature_collection(ee, territory).geometry().bounds().coordinates().get(0).getInfo()
    lons, lats = [p[0] for p in ring], [p[1] for p in ring]
    return [[min(lats), min(lons)], [max(lats), max(lons)]]
