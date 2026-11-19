"""Adaptador de configurações do notebook para o pipeline cartográfico offline."""
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import nullcontext
import re
import shutil
from datetime import datetime, timezone
from .layout import safe_name
from .temporal import resolve_bands


def validate(layers, territories):
    for label, items, required in (("camada", layers, ("id", "name", "asset", "visualization", "legend")),
                                   ("território", territories, ("id", "name", "asset"))):
        ids = set()
        for item in items:
            if label == "camada" and is_catalog(item):
                missing = [key for key in ("id", "name", "dataset", "product") if not item.get(key)]
                if missing:
                    raise ValueError(f"camada de catálogo: campos ausentes: {', '.join(missing)}")
                if item["id"] in ids:
                    raise ValueError(f"camada: id repetido: {item['id']}")
                ids.add(item["id"])
                continue
            if item.get("enabled", True) is False:
                if not item.get("id") or not item.get("name") or not item.get("reason"):
                    raise ValueError(f"{label}: item pendente precisa de id, name e reason")
                if not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]) or item["id"] in ids:
                    raise ValueError(f"{label}: id inválido ou repetido: {item['id']}")
                ids.add(item["id"])
                continue
            missing = [key for key in required if not item.get(key)
                       and not (label == "território" and key == "asset" and item.get("geometry"))]
            if missing:
                raise ValueError(f"{label}: campos ausentes: {', '.join(missing)}")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]) or item["id"] in ids:
                raise ValueError(f"{label}: id inválido ou repetido: {item['id']}")
            ids.add(item["id"])
            if label == "camada" and item.get("bands") is not None and (not isinstance(item["bands"], list) or not all(isinstance(b, str) and b for b in item["bands"])):
                raise ValueError("bands deve ser uma lista de nomes de bandas.")
            if label == "camada":
                start, end = item.get("start_year"), item.get("end_year")
                if item.get("axis", "time") not in ("time", "depth"):
                    raise ValueError(f"{item['name']}: eixo não suportado.")
                if item.get("axis") == "depth" and (start is not None or end is not None):
                    raise ValueError(f"{item['name']}: filtro de ano não se aplica à profundidade.")
                if any(value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1) for value in (start, end)):
                    raise ValueError(f"{item['name']}: start_year e end_year devem ser anos inteiros positivos.")
                if start is not None and end is not None and start > end:
                    raise ValueError(f"{item['name']}: ano inicial maior que o ano final.")
            if label == "território" and item.get("filter"):
                filt = item["filter"]
                if not isinstance(filt, dict) or not filt.get("field") or ("value" in filt) == ("values" in filt):
                    raise ValueError(f"{item['name']}: filtro precisa de field e de value ou values, não ambos.")
                values = filt.get("values") if "values" in filt else filt.get("value")
                if "values" in filt and (not isinstance(values, list) or not values):
                    raise ValueError(f"{item['name']}: values deve ser uma lista não vazia.")
                if "value" in filt and isinstance(values, (list, dict)):
                    raise ValueError(f"{item['name']}: use values para uma lista de valores.")
                if isinstance(values, list) and any(value is None or value == "" or isinstance(value, (list, dict)) for value in values):
                    raise ValueError(f"{item['name']}: values deve conter apenas valores simples não vazios.")
                if values is None or values == "":
                    raise ValueError(f"{item['name']}: filtro sem valor.")


def _map_lang(layer):
    from .i18n import code_for, map_language
    return code_for(layer.get("map_lang") or map_language())


def _translated_product(product, lang):
    from .i18n import map_text
    product = dict(product)
    for key in ("name", "collection_title"):
        if isinstance(product.get(key), str):
            product[key] = map_text(product[key], lang)
    return product


class NotebookConfig:
    def __init__(self, layer, territory, project):
        from .i18n import translate_visualization
        lang = _map_lang(layer)
        viz = translate_visualization(layer["visualization"], lang)
        viz.setdefault("cmap_type", "binary" if len(layer["legend"]) == 2 else "discrete")
        viz.setdefault("discrete_labels", [x["label"] for x in layer["legend"]])
        viz.setdefault("label", layer.get("legend_label", ""))
        product = {"name": layer["name"], "asset": layer["asset"], "bands": layer.get("bands", []), "visualization": "notebook", "start_year": layer.get("start_year"), "end_year": layer.get("end_year"), "bands_slice": layer.get("bands_slice"), "year_offset": layer.get("year_offset", 0), "axis": layer.get("axis", "time")}
        product = _translated_product(product, lang)
        self.datasets = {"notebook": {"products": {layer["id"]: product}, "title_collection": layer.get("collection_title", "")}}
        self.territories = _territory_entries(territory)
        self.visualizations = {"notebook": viz}
        self.categories = {}
        self.ee_project_id = project

    def get_processing_config(self, key):
        return {"gif_creation": {"frame_duration": 300, "loop_count": 1000, "quality": 95}}.get(key, {})


def is_catalog(layer):
    return layer.get("source") == "catalog"


def _territory_entry(territory):
    if territory.get("geometry"):
        return {"name": territory["name"], "geometry": territory["geometry"]}
    entry = {"name": territory["name"], "source": territory["asset"]}
    if territory.get("filter"):
        filt = territory["filter"]
        values = filt["values"] if "values" in filt else [filt["value"]]
        entry["filter_in"] = {"field": filt["field"], "values": values}
    return entry


def _territory_entries(territory):
    if not territory:
        return {"biomes": {}}
    entry = _territory_entry(territory)
    if territory.get("overlays"):
        # Territórios usados como linhas sobre o mapa (cor e espessura de cada um).
        entry["overlays"] = [dict(_territory_entry(line), color=line.get("color", "000000"),
                                  width=line.get("width", 1)) for line in territory["overlays"]]
    return {"biomes": {territory["id"]: entry}}


class CatalogConfig:
    """Configuração do projeto local para um produto do catálogo.

    Produto, processador e visualização vêm de `gif_factory/config`; só o
    território (escolhido na interface) e o período são injetados.
    """
    def __init__(self, layer, territory, project):
        from .catalog import config_loader
        base = config_loader()
        dataset = base.datasets[layer["dataset"]]
        product = dict(dataset["products"][layer["product"]])
        for key in ("start_year", "end_year"):
            if layer.get(key) is not None:
                product[key] = layer[key]
        if layer.get("bands"):
            product["bands"] = list(layer["bands"])
        from .i18n import translate_visualization
        lang = _map_lang(layer)
        product = _translated_product(product, lang)
        self.dataset_id, self.product_id = layer["dataset"], layer["product"]
        self.product = product
        self.datasets = {self.dataset_id: dict(dataset, products={self.product_id: product})}
        self.territories = _territory_entries(territory)
        # Só a visualização deste produto é traduzida (as outras ficam como estão).
        self.visualizations = dict(base.visualizations)
        viz_key = product.get("visualization")
        if isinstance(viz_key, str) and viz_key in self.visualizations:
            self.visualizations[viz_key] = translate_visualization(self.visualizations[viz_key], lang)
        self.categories = base.categories
        self.ee_project_id = project

    def get_processing_config(self, key):
        return {"gif_creation": {"frame_duration": 300, "loop_count": 1000, "quality": 95}}.get(key, {})


def special_catalog_image(product):
    """Imagem de produtos do catálogo que o carregador padrão do pipeline não monta.

    - `axis: depth`: coleção com uma imagem por camada (propriedade `depth`, ex.
      `000_010`) vira uma imagem com bandas `depth_000_010`, `depth_010_020`...;
    - `collection_filter: {property, value}`: escolhe uma imagem da coleção (ex. a
      temperatura máxima entre máxima, média e mínima).
    Devolve None para os demais produtos.
    """
    import ee
    if product.get("remap"):
        # Recodificação por tabela (níveis de legenda gerados): cada banda vira os
        # códigos do nível pedido; valores fora da tabela viram 0 e a máscara se mantém.
        rule = product["remap"]
        source = ee.Image(product["asset"])
        names = source.bandNames().getInfo()
        return ee.Image.cat([source.select(name).remap(rule["from"], rule["to"], 0).rename(name)
                             for name in names])
    if product.get("collection_filter"):
        rule = product["collection_filter"]
        collection = ee.ImageCollection(product["asset"]).filter(ee.Filter.eq(rule["property"], rule["value"]))
        return ee.Image(collection.first())
    if product.get("axis") == "depth":
        prepared = prepare_layer({"asset": product["asset"]})
        if prepared.get("axis") != "depth":
            raise ValueError(f"O asset não tem a estrutura por profundidade esperada: {product['asset']}")
        return load_image(prepared)
    return None


def catalog_image(layer, project=None):
    """Imagem do produto do catálogo, com processador e bands_slice aplicados."""
    from mapbiomas_data.core.pipeline import Pipeline
    config = CatalogConfig(dict(layer, bands=None), None, project)
    image = special_catalog_image(config.product)
    if image is None:
        image = Pipeline(config)._load_product_image(config.product)
    bands_slice = config.product.get("bands_slice")
    if bands_slice and len(bands_slice) == 2:
        image = image.slice(bands_slice[0], bands_slice[1])
    return image, config.product


def asset_type(layer):
    """Resolve the real GEE asset type; never infer it from a product name."""
    import ee
    info = ee.data.getAsset(layer["asset"])
    kind = info.get("type", "")
    if kind not in ("IMAGE", "IMAGE_COLLECTION"):
        raise ValueError(f"Tipo de asset não suportado: {kind or 'desconhecido'}")
    return kind


def _collection_frames(metadata, requested_band=None):
    """Choose one band per timestamp, rejecting ambiguous mosaics or periods."""
    if not metadata:
        raise ValueError("A coleção não contém imagens.")
    parsed = []
    for item in metadata:
        index, timestamp, names = item.get("index"), item.get("time"), item.get("bands") or []
        if not index or (timestamp is None and item.get("year") is None):
            raise ValueError("Coleção sem system:index ou data/ano em todas as imagens.")
        if requested_band:
            if requested_band not in names:
                raise ValueError(f"Banda '{requested_band}' ausente em {index}.")
            band = requested_band
        elif len(names) == 1:
            band = names[0]
        else:
            raise ValueError("A coleção tem imagens com múltiplas bandas; é necessário configurar qual banda animar.")
        try:
            date = (datetime.fromtimestamp(float(timestamp) / 1000, tz=timezone.utc)
                    if timestamp is not None else datetime(int(item["year"]), 1, 1, tzinfo=timezone.utc))
        except (ValueError, OverflowError, TypeError, OSError):
            raise ValueError(f"Data inválida na imagem {index}.")
        parsed.append((date, index, band))
    years = [d.year for d, _, _ in parsed]
    monthly = len(years) != len(set(years))
    names = [f"year_{d.year}_{d.month:02d}" if monthly else f"year_{d.year}" for d, _, _ in parsed]
    if len(names) != len(set(names)):
        raise ValueError("Mais de uma imagem no mesmo período; configure uma regra de seleção ou mosaico.")
    return sorted([(name, idx, band) for name, (_, idx, band) in zip(names, parsed)])


def collection_variables(asset):
    """List multiband variables when a collection is not a time sequence."""
    import ee
    if asset_type({"asset": asset}) != "IMAGE_COLLECTION":
        return []
    collection = ee.ImageCollection(asset)
    size = collection.size().getInfo()
    if size > 500:
        raise ValueError("Coleção com mais de 500 imagens; requer filtro temporal.")
    if not size:
        raise ValueError("A coleção não contém imagens.")
    images = collection.toList(size)
    metadata = ee.List.sequence(0, size - 1).map(
        lambda n: ee.Dictionary({"index": ee.Image(images.get(n)).get("system:index"),
                                 "metric": ee.Image(images.get(n)).get("metric"),
                                 "time": ee.Image(images.get(n)).get("system:time_start"),
                                 "year": ee.Image(images.get(n)).get("year"),
                                 "bands": ee.Image(images.get(n)).bandNames()})
    ).getInfo()
    if len(metadata) > 1 and all(m.get("time") is None and m.get("year") is None and len(m.get("bands") or []) > 1 for m in metadata):
        if any(not m.get("index") for m in metadata):
            raise ValueError("Imagens sem identificador; não é possível selecionar uma variável.")
        return [(str(m.get("metric") or m["index"]), m["index"]) for m in metadata]
    return []


def prepare_layer(layer):
    """Detect known depth layouts, leaving ordinary time-series unchanged."""
    import ee
    result = dict(layer)
    if result.get("axis"):
        return result
    kind = asset_type(result)
    if kind == "IMAGE_COLLECTION":
        collection = ee.ImageCollection(result["asset"])
        if collection.size().getInfo() == 0:
            raise ValueError("A coleção não contém imagens.")
        first = ee.Image(collection.first())
        depth = first.get("depth").getInfo()
        if isinstance(depth, str) and re.fullmatch(r"\d{3}_\d{3}", depth):
            bands = first.bandNames().getInfo()
            if len(bands) != 1:
                raise ValueError("Coleção de profundidade com várias bandas; configure depth_band.")
            result.update(axis="depth", depth_band=bands[0])
    else:
        names = ee.Image(result["asset"]).bandNames().getInfo()
        matches = [re.fullmatch(r".+_(\d{3})_(\d{3})cm", name, re.IGNORECASE) for name in names]
        if names and all(matches):
            normalized = [f"depth_{m.group(1)}_{m.group(2)}" for m in matches]
            if len(set(normalized)) != len(normalized):
                raise ValueError("Profundidades duplicadas no asset; selecione a variável antes de animar.")
            result.update(axis="depth", depth_source_bands=names, depth_output_bands=normalized)
    return result


def load_image(layer):
    import ee
    if layer.get("axis") == "depth":
        if layer.get("depth_source_bands"):
            return ee.Image(layer["asset"]).select(layer["depth_source_bands"], layer["depth_output_bands"])
        if not layer.get("depth_band"):
            raise ValueError("Informe depth_band para a coleção de profundidade.")
        collection = ee.ImageCollection(layer["asset"])
        depths = collection.aggregate_array("depth").getInfo()
        indices = collection.aggregate_array("system:index").getInfo()
        if not depths or len(depths) != len(indices) or len(set(depths)) != len(depths):
            raise ValueError("Profundidades ausentes, duplicadas ou inconsistentes na coleção.")
        bands = []
        for depth, index in zip(depths, indices):
            if not isinstance(depth, str) or not re.fullmatch(r"\d{3}_\d{3}", depth):
                raise ValueError(f"Profundidade inválida: {depth}")
            image = ee.Image(collection.filter(ee.Filter.eq("system:index", index)).first())
            bands.append(image.select(layer["depth_band"]).rename(f"depth_{depth}"))
        return ee.Image.cat(bands)
    if asset_type(layer) == "IMAGE_COLLECTION":
        collection = ee.ImageCollection(layer["asset"])
        size = collection.size().getInfo()
        if size > 500:
            raise ValueError("Coleção com mais de 500 imagens; configure filtro temporal antes de processar.")
        images = collection.toList(size)
        metadata = ee.List.sequence(0, size - 1).map(
            lambda n: ee.Dictionary({"index": ee.Image(images.get(n)).get("system:index"),
                                     "time": ee.Image(images.get(n)).get("system:time_start"),
                                     "year": ee.Image(images.get(n)).get("year"),
                                     "metric": ee.Image(images.get(n)).get("metric"),
                                     "bands": ee.Image(images.get(n)).bandNames()})
        ).getInfo() if size else []
        if layer.get("collection_member"):
            metadata = [m for m in metadata if m.get("index") == layer["collection_member"]]
            if not metadata:
                raise ValueError("A imagem configurada não existe nesta coleção.")
        if len(metadata) == 1 and len(metadata[0].get("bands") or []) > 1:
            image = ee.Image(collection.filter(ee.Filter.eq("system:index", metadata[0]["index"])).first())
            if layer.get("bands_slice"):
                image = image.slice(*layer["bands_slice"])
            return image
        if metadata and any(item.get("time") is None and item.get("year") is None for item in metadata):
            choices = ", ".join(f"{m.get('metric') or m.get('index')} ({m.get('index')})" for m in metadata[:20])
            raise ValueError("Coleção contém imagens multibanda ou sem data; selecione a variável/imagem desejada. Opções: " + choices)
        frames = _collection_frames(metadata, layer.get("collection_band"))
        bands = [ee.Image(collection.filter(ee.Filter.eq("system:index", idx)).first()).select(band).rename(name)
                 for name, idx, band in frames]
        return ee.Image.cat(bands)
    image = ee.Image(layer["asset"])
    if layer.get("bands_slice"):
        image = image.slice(*layer["bands_slice"])
    if layer.get("reclass"):
        src = image.round()
        out = src.multiply(0)
        for gt, lte, value in layer["reclass"]:
            condition = src.gt(gt).And(src.lte(lte)) if lte is not None else src.gt(gt)
            out = out.where(condition, value)
        image = out.unmask(0).int8()
    return image


def discover(layer):
    if is_catalog(layer):
        image, product = catalog_image(layer)
        available = image.bandNames().getInfo()
        axis = product.get("axis", "time")
        # Como no projeto local: bandas do YAML que não existem são ignoradas.
        configured = [b for b in product.get("bands") or [] if b in available] or None
        grouped = bool(product.get("rgb"))
        all_bands = resolve_bands(available, configured, axis=axis, grouped=grouped)
        selected = resolve_bands(available, layer.get("bands") or configured,
                                 layer.get("start_year"), layer.get("end_year"), axis=axis, grouped=grouped)
        return all_bands, selected
    layer = prepare_layer(layer)
    image = load_image(layer)
    available = image.bandNames().getInfo()
    axis = layer.get("axis", "time")
    all_bands = resolve_bands(available, axis=axis)
    selected = resolve_bands(available, layer.get("bands"), layer.get("start_year"), layer.get("end_year"), axis=axis)
    return all_bands, selected


def generate(layer, territory, root, kinds, on_artifact=None, dimension=2000, project=None, on_progress=None, work_dir=None, variant="completo"):
    from mapbiomas_data.core.pipeline import Pipeline
    catalog = is_catalog(layer)
    if not catalog:
        layer = prepare_layer(layer)
    validate([layer], [territory])
    if not project:
        raise ValueError("Informe o projeto Earth Engine usado na autenticação.")
    out = Path(root)
    from .i18n import prepare_map_rendering
    prepare_map_rendering(_map_lang(layer))
    if catalog:
        # Produto do catálogo: o pipeline carrega asset/processador e aplica
        # bands_slice e a visualização exatamente como no projeto local.
        config = CatalogConfig(layer, territory, project)
        pipeline = Pipeline(config)
        dataset_id, product_id = config.dataset_id, config.product_id
        special = special_catalog_image(config.product)
        if special is not None:
            pipeline._load_product_image = lambda _info: special
    else:
        config = NotebookConfig(layer, territory, project)
        pipeline = Pipeline(config)
        # load_image applies bands_slice itself (also used by discover); prevent
        # the pipeline from applying that same slice a second time.
        config.datasets["notebook"]["products"][layer["id"]]["bands_slice"] = None
        pipeline._load_product_image = lambda _: load_image(layer)
        dataset_id, product_id = "notebook", layer["id"]
    if on_progress:
        def notify(*args):
            try:
                on_progress(*args)
            except Exception as exc:
                print(f"Aviso: não foi possível atualizar progresso: {exc}")
        pipeline.downloader.progress_callback = notify
        pipeline.progress_callback = notify
    else:
        def notify(*args):
            pass
    produced = []
    # Keep pipeline intermediates separate from the requested deliverables.
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=False)
    context = nullcontext(str(work_dir)) if work_dir is not None else TemporaryDirectory(prefix="gif_factory_")
    with context as tmp:
        result = pipeline.run(dataset_id, product_id, territory["id"], output_dir=tmp,
                              create_collage="colagem" in kinds, add_labels=True,
                              vertical_dimension=dimension)
        if result.get("status") != "success":
            raise RuntimeError(result.get("error", "Falha no pipeline cartográfico."))
        notify("stage", "Organizando arquivos selecionados")
        folders = {"gif": ("gifs", "GIFs"), "colagem": ("collages_combined", "Colagens"),
                   "brutos": ("frames_pure", "Frames/Mapa_basico"), "preparados": ("frames_clean", "Frames/Mapa_com_ano_e_escala"),
                   "finais": ("frames_maps", "Frames/Mapa_completo")}
        for kind, (source_folder, dest_folder) in folders.items():
            if kind not in kinds:
                continue
            if variant != "completo" and kind in {"brutos", "preparados", "finais"}:
                dest_folder = str(Path(dest_folder) / variant)
            for source in sorted((Path(tmp) / source_folder).glob("*")):
                if not source.is_file() or source.suffix.lower() not in {".gif", ".png"}:
                    continue
                # Preserve the source basename (including year, period and special mode)
                # and add the territory, so a downloaded file is identifiable alone.
                suffix = f"_{variant}" if variant != "completo" and kind in {"gif", "colagem"} else ""
                target = out / dest_folder / f"{safe_name(territory['name'])}_{source.stem}{suffix}{source.suffix}"
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                produced.append(target)
                if on_artifact:
                    on_artifact(target)
        if not produced:
            raise RuntimeError("O pipeline não produziu arquivos dos tipos selecionados.")
    return produced
