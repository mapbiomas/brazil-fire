"""Adaptador de configurações do notebook para o pipeline cartográfico offline."""
from pathlib import Path
from tempfile import TemporaryDirectory
import re
import shutil
from .layout import safe_name


def validate(layers, territories):
    for label, items, required in (("camada", layers, ("id", "name", "asset", "bands", "visualization", "legend")),
                                   ("território", territories, ("id", "name", "asset"))):
        ids = set()
        for item in items:
            missing = [key for key in required if not item.get(key)]
            if missing:
                raise ValueError(f"{label}: campos ausentes: {', '.join(missing)}")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]) or item["id"] in ids:
                raise ValueError(f"{label}: id inválido ou repetido: {item['id']}")
            ids.add(item["id"])
            if label == "camada" and (not isinstance(item["bands"], list) or not all(isinstance(b, str) and b for b in item["bands"])):
                raise ValueError("bands deve ser uma lista de nomes de bandas.")


class NotebookConfig:
    def __init__(self, layer, territory, project):
        viz = dict(layer["visualization"])
        viz.setdefault("cmap_type", "binary" if len(layer["legend"]) == 2 else "discrete")
        viz.setdefault("discrete_labels", [x["label"] for x in layer["legend"]])
        viz.setdefault("label", layer.get("legend_label", ""))
        product = {"name": layer["name"], "asset": layer["asset"], "bands": layer["bands"], "visualization": "notebook"}
        self.datasets = {"notebook": {"products": {layer["id"]: product}, "title_collection": layer.get("collection_title", "")}}
        entry = {"name": territory["name"], "source": territory["asset"]}
        if territory.get("filter"):
            entry["filter_in"] = {"field": territory["filter"]["field"], "values": [territory["filter"]["value"]]}
        self.territories = {"biomes": {territory["id"]: entry}}
        self.visualizations = {"notebook": viz}
        self.categories = {}
        self.ee_project_id = project

    def get_processing_config(self, key):
        return {"gif_creation": {"frame_duration": 300, "loop_count": 1000, "quality": 95}}.get(key, {})


def generate(layer, territory, root, kinds, on_artifact=None, dimension=2000, project=None, on_progress=None):
    from mapbiomas_data.core.pipeline import Pipeline
    validate([layer], [territory])
    if not project:
        raise ValueError("Informe o projeto Earth Engine usado na autenticação.")
    out = Path(root)
    config = NotebookConfig(layer, territory, project)
    pipeline = Pipeline(config)
    if on_progress:
        pipeline.downloader.progress_callback = on_progress
        pipeline.progress_callback = on_progress
    produced = []
    # Keep pipeline intermediates separate from the requested deliverables.
    with TemporaryDirectory(prefix="gif_factory_") as tmp:
        result = pipeline.run("notebook", layer["id"], territory["id"], output_dir=tmp,
                              create_collage="colagem" in kinds, add_labels=True,
                              vertical_dimension=dimension)
        if result.get("status") != "success":
            raise RuntimeError(result.get("error", "Falha no pipeline cartográfico."))
        folders = {"gif": ("gifs", "GIFs"), "colagem": ("collages_combined", "Colagens"),
                   "brutos": ("frames_pure", "Frames/Mapa_basico"), "preparados": ("frames_clean", "Frames/Mapa_com_ano_e_escala"),
                   "finais": ("frames_maps", "Frames/Mapa_completo")}
        for kind, (source_folder, dest_folder) in folders.items():
            if kind not in kinds:
                continue
            for source in sorted((Path(tmp) / source_folder).glob("*")):
                if not source.is_file() or source.suffix.lower() not in {".gif", ".png"}:
                    continue
                # Preserve the source basename (including year, period and special mode)
                # and add the territory, so a downloaded file is identifiable alone.
                target = out / dest_folder / f"{safe_name(territory['name'])}_{source.name}"
                target.parent.mkdir(parents=True, exist_ok=True)
                with source.open("rb") as src, target.open("xb") as dst:
                    shutil.copyfileobj(src, dst)
                produced.append(target)
                if on_artifact:
                    on_artifact(target)
        if not produced:
            raise RuntimeError("O pipeline não produziu arquivos dos tipos selecionados.")
    return produced
