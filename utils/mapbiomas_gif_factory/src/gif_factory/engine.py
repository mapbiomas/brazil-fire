"""Adaptador de configurações do notebook para o pipeline cartográfico offline."""
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import nullcontext
import re
import shutil
from .layout import safe_name
from .temporal import resolve_bands


def validate(layers, territories):
    for label, items, required in (("camada", layers, ("id", "name", "asset", "visualization", "legend")),
                                   ("território", territories, ("id", "name", "asset"))):
        ids = set()
        for item in items:
            if item.get("enabled", True) is False:
                if not item.get("id") or not item.get("name") or not item.get("reason"):
                    raise ValueError(f"{label}: item pendente precisa de id, name e reason")
                if not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]) or item["id"] in ids:
                    raise ValueError(f"{label}: id inválido ou repetido: {item['id']}")
                ids.add(item["id"])
                continue
            missing = [key for key in required if not item.get(key)]
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


class NotebookConfig:
    def __init__(self, layer, territory, project):
        viz = dict(layer["visualization"])
        viz.setdefault("cmap_type", "binary" if len(layer["legend"]) == 2 else "discrete")
        viz.setdefault("discrete_labels", [x["label"] for x in layer["legend"]])
        viz.setdefault("label", layer.get("legend_label", ""))
        product = {"name": layer["name"], "asset": layer["asset"], "bands": layer.get("bands", []), "visualization": "notebook", "start_year": layer.get("start_year"), "end_year": layer.get("end_year"), "bands_slice": layer.get("bands_slice"), "year_offset": layer.get("year_offset", 0), "axis": layer.get("axis", "time")}
        self.datasets = {"notebook": {"products": {layer["id"]: product}, "title_collection": layer.get("collection_title", "")}}
        entry = {"name": territory["name"], "source": territory["asset"]}
        if territory.get("filter"):
            filt = territory["filter"]
            values = filt["values"] if "values" in filt else [filt["value"]]
            entry["filter_in"] = {"field": filt["field"], "values": values}
        self.territories = {"biomes": {territory["id"]: entry}}
        self.visualizations = {"notebook": viz}
        self.categories = {}
        self.ee_project_id = project

    def get_processing_config(self, key):
        return {"gif_creation": {"frame_duration": 300, "loop_count": 1000, "quality": 95}}.get(key, {})


def load_image(layer):
    import ee
    if layer.get("axis") == "depth":
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
    image = ee.Image(layer["asset"])
    if layer.get("bands_slice"):
        image = image.slice(*layer["bands_slice"])
    return image


def discover(layer):
    image = load_image(layer)
    available = image.bandNames().getInfo()
    axis = layer.get("axis", "time")
    all_bands = resolve_bands(available, axis=axis)
    selected = resolve_bands(available, layer.get("bands"), layer.get("start_year"), layer.get("end_year"), axis=axis)
    return all_bands, selected


def generate(layer, territory, root, kinds, on_artifact=None, dimension=2000, project=None, on_progress=None, work_dir=None, variant="completo"):
    from mapbiomas_data.core.pipeline import Pipeline
    validate([layer], [territory])
    if not project:
        raise ValueError("Informe o projeto Earth Engine usado na autenticação.")
    out = Path(root)
    config = NotebookConfig(layer, territory, project)
    pipeline = Pipeline(config)
    if layer.get("axis") == "depth":
        pipeline._load_product_image = lambda _: load_image(layer)
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
        result = pipeline.run("notebook", layer["id"], territory["id"], output_dir=tmp,
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
