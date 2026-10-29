import os
import yaml
import warnings
from typing import Any, Dict, List, Optional


def _deep_merge(base: Dict, overlay: Dict) -> Dict:
    result = base.copy()
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _try_load_yaml(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class ConfigLoader:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self._datasets: Dict[str, Any] = {}
        self._territories: Dict[str, Any] = {}
        self._visualizations: Dict[str, Any] = {}
        self._paths: Dict[str, Any] = {}
        self._categories: Dict[str, Any] = {}
        self._initiatives: Dict[str, Any] = {}
        self._loaded_initiatives = False

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        path = os.path.join(self.config_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_yaml_with_includes(self, filename: str) -> Dict[str, Any]:
        data = self._load_yaml(filename)
        includes = data.pop("includes", [])
        for inc in includes:
            inc_path = os.path.join(self.config_dir, inc)
            if not os.path.exists(inc_path):
                warnings.warn(f"Include not found, skipping: {inc}")
                continue
            inc_data = self._load_yaml(inc)
            data = _deep_merge(data, inc_data)
        return data

    def _load_initiatives(self):
        if self._loaded_initiatives:
            return
        self._loaded_initiatives = True

        initiatives_dir = os.path.join(self.config_dir, "initiatives")
        if not os.path.isdir(initiatives_dir):
            return

        initiatives: Dict[str, Any] = {}
        datasets: Dict[str, Any] = {}
        territories: Dict[str, Any] = {}
        seen_types: Dict[str, str] = {}  # normalize type names

        for init_name in sorted(os.listdir(initiatives_dir)):
            init_dir = os.path.join(initiatives_dir, init_name)
            if not os.path.isdir(init_dir):
                continue

            meta = _try_load_yaml(os.path.join(init_dir, "initiative.yaml")) or {}
            init_id = meta.get("id", init_name)
            init_info = {
                "id": init_id,
                "name": meta.get("name", init_name),
                "description": meta.get("description", ""),
                "dir": init_dir,
            }

            # --- Load collections ---
            coll_dir = os.path.join(init_dir, "collections")
            init_collections = []
            if os.path.isdir(coll_dir):
                for fname in sorted(os.listdir(coll_dir)):
                    if not fname.endswith(".yaml"):
                        continue
                    coll = _try_load_yaml(os.path.join(coll_dir, fname))
                    if coll is None:
                        continue
                    ds_id = coll.get("dataset")
                    if not ds_id:
                        continue
                    entry = {
                        "description": coll.get("description", ""),
                        "title_collection": coll.get("title_collection", ""),
                        "project": coll.get("project", ""),
                        "category": coll.get("category", ""),
                        "collection": coll.get("collection"),
                        "products": coll.get("products", {}),
                    }
                    datasets[ds_id] = entry
                    init_collections.append({
                        "id": fname.replace(".yaml", ""),
                        "name": coll.get("name", fname.replace(".yaml", "")),
                        "dataset_id": ds_id,
                        "category": coll.get("category", ""),
                    })

            init_info["collections"] = init_collections

            # --- Load territories ---
            terr_dir = os.path.join(init_dir, "territories")
            init_territory_groups = []
            if os.path.isdir(terr_dir):
                for fname in sorted(os.listdir(terr_dir)):
                    if not fname.endswith(".yaml"):
                        continue
                    grp = _try_load_yaml(os.path.join(terr_dir, fname))
                    if grp is None:
                        continue
                    ttype = grp.get("type", fname.replace(".yaml", ""))
                    ttype_normalized = ttype
                    tgroup = grp.get("territories", {})
                    if ttype_normalized not in territories:
                        territories[ttype_normalized] = {}
                    territories[ttype_normalized].update(tgroup)
                    init_territory_groups.append({
                        "id": fname.replace(".yaml", ""),
                        "name": grp.get("name", fname.replace(".yaml", "")),
                        "description": grp.get("description", ""),
                        "type": ttype_normalized,
                        "territory_ids": sorted(tgroup.keys()),
                    })

            init_info["territory_groups"] = init_territory_groups
            initiatives[init_id] = init_info

        self._initiatives = initiatives

        if datasets:
            self._datasets = datasets
        if territories:
            self._territories = territories

        # Load categories.yaml
        cat_data = _try_load_yaml(os.path.join(self.config_dir, "categories.yaml"))
        if cat_data:
            self._categories = cat_data.get("categories", {})

    def load_all(self):
        self._load_initiatives()

        if not self._paths:
            paths_data = _try_load_yaml(os.path.join(self.config_dir, "paths.yaml"))
            if paths_data:
                self._paths = paths_data

        if not self._visualizations:
            viz_data = _try_load_yaml(os.path.join(self.config_dir, "visualization.yaml"))
            if viz_data:
                self._visualizations = viz_data.get("visualizations", viz_data)

        return self

    @property
    def datasets(self) -> Dict[str, Any]:
        if not self._datasets:
            self._load_initiatives()
        return self._datasets

    @property
    def categories(self) -> Dict[str, Any]:
        if not self._categories:
            self._load_initiatives()
        return self._categories

    @property
    def territories(self) -> Dict[str, Any]:
        if not self._territories:
            self._load_initiatives()
        return self._territories

    @property
    def initiatives(self) -> Dict[str, Any]:
        if not self._loaded_initiatives:
            self._load_initiatives()
        return self._initiatives

    @property
    def visualizations(self) -> Dict[str, Any]:
        if not self._visualizations:
            viz_data = _try_load_yaml(os.path.join(self.config_dir, "visualization.yaml"))
            if viz_data:
                self._visualizations = viz_data.get("visualizations", viz_data)
                self._normalize_palettes()
        return self._visualizations

    def _normalize_palettes(self) -> None:
        """Garante que toda cor de paleta seja string.

        O YAML interpreta cores so com digitos (ex.: 922108) como inteiro, e o
        Earth Engine exige strings em `Image.visualize`. Tambem valida o formato.
        """
        for viz_key, viz in self._visualizations.items():
            if not isinstance(viz, dict):
                continue
            palette = viz.get("palette")
            if not isinstance(palette, list):
                continue
            normalized = []
            for idx, color in enumerate(palette):
                if not isinstance(color, str):
                    color = str(color)
                color = color.strip()
                if color.startswith("#"):
                    color = color[1:]
                if len(color) not in (3, 6) or any(
                    ch not in "0123456789abcdefABCDEF" for ch in color
                ):
                    raise ValueError(
                        f"Visualizacao '{viz_key}': cor invalida no indice {idx}: {color!r}"
                    )
                normalized.append(color)
            viz["palette"] = normalized

    @property
    def paths(self) -> Dict[str, Any]:
        if not self._paths:
            paths_data = _try_load_yaml(os.path.join(self.config_dir, "paths.yaml"))
            if paths_data:
                self._paths = paths_data
        return self._paths

    @property
    def runtime_mode(self) -> str:
        return self.paths.get("runtime", {}).get("mode", "local")

    @property
    def ee_project_id(self) -> str:
        return (
            self.paths.get("paths", self.paths)
            .get("earth_engine", {})
            .get("project_id", "workspace-ipam")
        )

    def get_output_dir(self) -> str:
        paths = self.paths.get("paths", self.paths)
        mode = self.runtime_mode
        if mode == "colab":
            return paths.get("google_drive", {}).get("output_root", "/content/drive/MyDrive/IPAM FRAMES AND GIFS/")
        return paths.get("local", {}).get("output_dir", "./output/")

    def get_cache_dir(self) -> str:
        paths = self.paths.get("paths", self.paths)
        return paths.get("local", {}).get("cache_dir", "./cache/")

    def get_processing_config(self, key: str) -> Dict[str, Any]:
        proc = self.paths.get("processing", {})
        return proc.get(key, {})
