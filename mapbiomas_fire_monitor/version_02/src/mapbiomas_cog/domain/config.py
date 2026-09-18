"""Adapter do catalogo (reusa o ``OBJ`` do version_01).

O ``config.OBJ`` continua sendo a fonte de verdade dos paises/temas/colecoes/
produtos. Aqui carregamos esse arquivo por caminho (sem importar o pacote
inteiro) e expomos helpers de descoberta equivalentes ao
``config.resolve_countries`` do version_01.
"""

from __future__ import annotations

import importlib.util
import os

# .../version_02/src/mapbiomas_cog/domain -> .../mapbiomas_fire_monitor
_MONITOR_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
)

DEFAULT_CONFIG_PATH = os.environ.get(
    "MAPBIOMAS_COG_CONFIG",
    os.path.join(_MONITOR_ROOT, "version_01", "export_and_vectorization", "config.py"),
)


def load_obj(path: str | None = None) -> dict:
    """Carrega e devolve o dict ``OBJ`` do arquivo de catalogo."""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(f"Catalogo nao encontrado: {path}")
    spec = importlib.util.spec_from_file_location("mapbiomas_cog_v1_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Nao foi possivel carregar o catalogo: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "OBJ")


def resolve_countries(obj: dict, countries=None, themes=None) -> list:
    """Paises das abas: lista vazia/None = todos; senao valida contra o OBJ."""
    base = list(countries) if countries else list(obj)
    unknown = [c for c in base if c not in obj]
    if unknown:
        raise ValueError(
            f"Pais(es) nao configurado(s) no catalogo: {unknown}. "
            f"Disponiveis: {sorted(obj)}."
        )
    allowed = list(themes or [])

    def _has_theme(code: str) -> bool:
        if not allowed:
            return True
        return any(
            t in obj[code]
            and any(
                any(p.get("visible", True) for p in prods)
                for prods in obj[code][t].values()
            )
            for t in allowed
        )

    result = [c for c in base if _has_theme(c)]
    return sorted(result, key=lambda c: (c != "brasil", c))


def products(obj: dict, country: str, theme: str, collection: str) -> list:
    """Lista de produtos (dicts do OBJ) de um contexto."""
    return [
        p
        for p in obj.get(country, {}).get(theme, {}).get(collection, [])
        if p.get("visible", True)
    ]


def find_product(obj: dict, country: str, theme: str, collection: str, product: str) -> dict | None:
    """Entrada do OBJ de um produto (ou None)."""
    for p in obj.get(country, {}).get(theme, {}).get(collection, []):
        if p.get("product") == product:
            return p
    return None


def make_product(obj: dict, country: str, theme: str, collection: str, product: str):
    """Constroi um ``domain.models.Product`` a partir do OBJ."""
    from .models import Product

    entry = find_product(obj, country, theme, collection, product)
    if entry is None:
        raise KeyError(
            f"Produto '{product}' nao encontrado em {country}/{theme}/{collection}."
        )
    return Product.from_obj(country, theme, collection, entry)


def collections(obj: dict, country: str, theme: str) -> list:
    """Colecoes disponiveis de um pais/tema."""
    return list(obj.get(country, {}).get(theme, {}))
