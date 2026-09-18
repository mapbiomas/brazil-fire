"""Inventario do catalogo (M5).

Deriva paises/temas/colecoes/produtos do ``OBJ`` (via ``domain.config``). O
enriquecimento com metadados do GEE fica para depois; aqui o inventario e
estatico e barato (offline).
"""

from __future__ import annotations

from ..domain.config import load_obj, resolve_countries
from ..domain.models import Product


def build_inventory(countries=None, themes=None, config_path: str | None = None) -> dict:
    """Inventario {country: {theme: {collection: [produtos]}}}.

    Cada produto traz ``product``, ``assetid``, ``kind``, ``dtype``, ``scale``,
    ``vectorize`` e ``root`` (layout do mapbiomas_cog).
    """
    obj = load_obj(config_path)
    selected = resolve_countries(obj, countries, themes)
    inventory: dict = {}
    for country in selected:
        inventory[country] = {}
        for theme, collections in obj[country].items():
            theme_inv: dict = {}
            for collection, products in collections.items():
                entries = []
                for entry in products:
                    if not entry.get("visible", True):
                        continue
                    product = Product.from_obj(country, theme, collection, entry)
                    entries.append(
                        {
                            "product": product.product,
                            "assetid": product.assetid,
                            "kind": product.kind,
                            "dtype": product.dtype,
                            "scale": product.scale,
                            "vectorize": product.vectorize,
                            "root": product.root,
                        }
                    )
                if entries:
                    theme_inv[collection] = entries
            if theme_inv:
                inventory[country][theme] = theme_inv
    return inventory
