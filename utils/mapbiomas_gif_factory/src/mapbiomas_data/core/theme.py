"""Tema visual (claro/escuro) como eixo ortogonal ao catálogo.

Um tema NÃO é um produto. Ele muda apenas apresentação: fundo, fonte,
separadores, cores de apoio de gráficos e da composição. A paleta do mapa só
muda quando a visualização declara `palette_dark` (ou usa o token `auto_bg`,
que resolve para o fundo do tema).

Precedência: `--theme` / `item.theme` > alias `*_black` > `viz.theme` > light.
"""

import os
import threading
from typing import Any, Dict, List, Optional, Tuple

ThemeColors = Dict[str, Tuple[int, int, int]]

LIGHT_THEME: ThemeColors = {
    "bg": (255, 255, 255),
    "fg": (0, 0, 0),
    "sep": (200, 200, 200),
    "box": (100, 100, 100),
    "text": (51, 51, 51),
    "muted": (85, 85, 85),
    "faint": (153, 153, 153),
    "panel": (245, 245, 245),
    "legend_bg": (248, 248, 248),
    "unknown": (204, 204, 204),
    "unburned": (224, 224, 224),
    "up": (231, 76, 60),
    "down": (52, 152, 219),
}

DARK_THEME: ThemeColors = {
    "bg": (0, 0, 0),
    "fg": (240, 240, 240),
    "sep": (70, 70, 70),
    "box": (150, 150, 150),
    "text": (230, 230, 230),
    "muted": (160, 160, 160),
    "faint": (110, 110, 110),
    "panel": (20, 20, 20),
    "legend_bg": (16, 16, 16),
    "unknown": (70, 70, 70),
    "unburned": (55, 55, 55),
    "up": (231, 76, 60),
    "down": (52, 152, 219),
}

THEMES: Dict[str, ThemeColors] = {"light": LIGHT_THEME, "dark": DARK_THEME}
VALID_THEMES = tuple(THEMES)
DEFAULT_THEME = "light"
ALIAS_SUFFIX = "_black"

# Cores de paleta que significam "fundo do tema" (nodata / classe vazia).
AUTO_BG_TOKENS = {"auto_bg", "auto", "theme_bg", "background"}

_THEME = threading.local()


# --- Resolução -------------------------------------------------------------

def normalize_theme(name: Optional[str]) -> str:
    if not name:
        return DEFAULT_THEME
    value = str(name).strip().lower()
    if value not in THEMES:
        raise ValueError(
            f"Tema '{name}' inválido. Use um de: {', '.join(VALID_THEMES)}"
        )
    return value


def resolve_theme(explicit: Optional[str] = None,
                  default: Optional[str] = None) -> str:
    """`explicit` vence; `default` é o theme declarado na visualização."""
    return normalize_theme(explicit or default or DEFAULT_THEME)


def is_dark(name: Optional[str]) -> bool:
    return normalize_theme(name) == "dark"


def theme_alias(product_id: str) -> Tuple[str, Optional[str]]:
    """Resolve o alias depreciado `produto_black` -> (`produto`, "dark")."""
    if product_id and product_id.endswith(ALIAS_SUFFIX):
        base = product_id[: -len(ALIAS_SUFFIX)]
        if base:
            return base, "dark"
    return product_id, None


# --- Estado do tema (thread-local) -----------------------------------------

def set_theme(name: Optional[str]) -> None:
    theme = normalize_theme(name)
    _THEME.name = theme
    _THEME.value = THEMES[theme]


def current_theme() -> str:
    return getattr(_THEME, "name", DEFAULT_THEME)


def get_theme() -> ThemeColors:
    return getattr(_THEME, "value", None) or LIGHT_THEME


def get_style(name: Optional[str] = None) -> ThemeColors:
    return THEMES[normalize_theme(name or current_theme())]


# --- Cores -----------------------------------------------------------------

def to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
    h = str(hex_color).lstrip("#")
    return tuple(int(h[i: i + 2], 16) for i in (0, 2, 4))


def resolve_color(color: Optional[str], theme: Optional[str] = None) -> Optional[str]:
    """Resolve tokens `auto_bg` para a cor de fundo do tema."""
    if color is None:
        return None
    value = str(color).strip()
    if value.lower() in AUTO_BG_TOKENS:
        value = to_hex(get_style(theme)["bg"])
    return value.lstrip("#")


def resolve_palette(palette: Optional[List[Any]],
                    theme: Optional[str] = None) -> List[str]:
    return [resolve_color(c, theme) or "" for c in (palette or [])]


def background_hex(theme: Optional[str] = None) -> str:
    return to_hex(get_style(theme)["bg"]).lstrip("#")


def nodata_hex(theme: Optional[str] = None) -> str:
    """Cor de nodata derivada do fundo (light=#ffffff, dark=#000000)."""
    return background_hex(theme)


# --- Layout ----------------------------------------------------------------

def combo_dir(base_dir: str, dataset: str, product: str,
              territory: str, theme: Optional[str] = None) -> str:
    """Diretório do combo. Light fica no nível base; dark ganha subpasta.

    <base>/<ds>/<prod>/<terr>        -> light (layout histórico, sem migração)
    <base>/<ds>/<prod>/<terr>/dark   -> dark
    """
    path = os.path.join(base_dir, dataset, product, territory)
    name = normalize_theme(theme)
    if name != DEFAULT_THEME:
        path = os.path.join(path, name)
    return path
