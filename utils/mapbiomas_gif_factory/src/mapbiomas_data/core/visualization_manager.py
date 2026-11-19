from typing import Any, Dict, List, Optional
from ..config import ConfigLoader
from . import theme as theme_mod


class VisualizationManager:
    """Gerenciar parâmetros de visualização a partir do YAML."""

    def __init__(self, config: ConfigLoader):
        self.config = config
        self._visualizations = config.visualizations

    def list_viz_keys(self) -> List[str]:
        return sorted(self._visualizations.keys())

    def get_viz_params(self, viz_key: str,
                       theme: Optional[str] = None) -> Dict[str, Any]:
        """Parâmetros efetivos da visualização para um tema.

        - `palette_dark` (se declarada e com o mesmo tamanho) substitui `palette`
        - token `auto_bg` na paleta/nodata resolve para o fundo do tema
        - `nodata_color` é derivado do fundo do tema quando não declarado
          (mantém o comportamento histórico no tema claro)
        """
        if viz_key not in self._visualizations:
            raise KeyError(f"Visualização '{viz_key}' não encontrada")
        viz = self._visualizations[viz_key]
        declared_theme = viz.get("theme")
        resolved_theme = theme_mod.resolve_theme(theme, declared_theme)

        palette = viz.get("palette", [])
        palette_dark = viz.get("palette_dark")
        if resolved_theme == "dark" and isinstance(palette_dark, list) and palette_dark:
            if len(palette_dark) != len(palette):
                raise ValueError(
                    f"Visualização '{viz_key}': palette_dark tem "
                    f"{len(palette_dark)} cores, palette tem {len(palette)}"
                )
            palette = palette_dark

        base = {
            "name": viz.get("name", viz_key),
            "min": viz.get("min", 0),
            "max": viz.get("max", 1),
            "palette": theme_mod.resolve_palette(palette, resolved_theme),
            "label": viz.get("label", ""),
            "unit": viz.get("unit", ""),
            "cmap_type": viz.get("cmap_type", "sequential"),
            "theme": resolved_theme,
        }
        for k, v in viz.items():
            if k not in base:
                base[k] = v

        nodata_color = base.get("nodata_color")
        if nodata_color is None and resolved_theme == "dark":
            nodata_color = theme_mod.nodata_hex(resolved_theme)
        base["nodata_color"] = theme_mod.resolve_color(nodata_color, resolved_theme)

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
            palette = theme_mod.resolve_palette(viz.get("palette", []), "light")
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
