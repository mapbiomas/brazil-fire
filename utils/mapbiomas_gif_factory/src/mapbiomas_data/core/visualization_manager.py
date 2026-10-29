from typing import Any, Dict, List, Optional
from ..config import ConfigLoader


class VisualizationManager:
    """Gerenciar parâmetros de visualização a partir do YAML."""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self._visualizations = config.visualizations

    def list_viz_keys(self) -> List[str]:
        return sorted(self._visualizations.keys())

    def get_viz_params(self, viz_key: str) -> Dict[str, Any]:
        if viz_key not in self._visualizations:
            raise KeyError(f"Visualização '{viz_key}' não encontrada")
        viz = self._visualizations[viz_key]
        base = {
            "name": viz.get("name", viz_key),
            "min": viz.get("min", 0),
            "max": viz.get("max", 1),
            "palette": viz.get("palette", []),
            "label": viz.get("label", ""),
            "unit": viz.get("unit", ""),
            "cmap_type": viz.get("cmap_type", "sequential"),
        }
        for k, v in viz.items():
            if k not in base:
                base[k] = v
        return base

    def get_palette(self, viz_key: str) -> List[str]:
        return self._visualizations.get(viz_key, {}).get("palette", [])

    def get_range(self, viz_key: str) -> tuple:
        viz = self._visualizations.get(viz_key, {})
        return (viz.get("min", 0), viz.get("max", 1))

    def build_ee_vis_params(self, viz_key: str, band: Optional[str] = None) -> Dict[str, Any]:
        params = self.get_viz_params(viz_key)
        vis = {
            "min": params["min"],
            "max": params["max"],
            "palette": params["palette"],
        }
        if band:
            vis["bands"] = [band]
        return vis

    def validate_palette(self, viz_key: str) -> bool:
        try:
            viz = self._visualizations[viz_key]
            if viz.get("random_viz", False):
                return True
            palette = viz.get("palette", [])
            if not palette:
                return False
            for color in palette:
                color = color.lstrip("#")
                if len(color) not in (3, 6):
                    return False
                int(color, 16)
            return True
        except (ValueError, KeyError):
            return False
