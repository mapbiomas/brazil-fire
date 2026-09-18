"""Modelos de dominio (stdlib dataclasses; sem dependencias pesadas)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .grid import TileGrid


def product_kind(product: str) -> str:
    """Classifica o produto: monthly | annual | period."""
    name = (product or "").lower()
    if "monthly" in name:
        return "monthly"
    if "annual" in name:
        return "annual"
    return "period"


def unit_key_from_time_start(kind: str, time_start_ms: int) -> str:
    """Deriva a chave da unidade (YYYY_MM ou YYYY) a partir de system:time_start."""
    import datetime

    dt = datetime.datetime.utcfromtimestamp(time_start_ms / 1000)
    if kind == "monthly":
        return f"{dt.year}_{dt.month:02d}"
    return f"{dt.year}"


@dataclass(frozen=True)
class Unit:
    """Uma unidade processavel de um produto (mes, ano ou periodo)."""

    key: str
    kind: str = "period"  # monthly | annual | period


@dataclass(frozen=True)
class Product:
    """Produto do catalogo (entrada do OBJ enriquecida com contexto)."""

    country: str
    theme: str
    collection: str
    product: str
    assetid: str
    dtype: str = "byte"
    vectorize: bool = False
    scale: int = 30
    decode: Optional[str] = None
    kind: str = "period"

    @property
    def root(self) -> str:
        from .paths import root

        return root(self.country, self.theme, self.collection, self.product)

    @classmethod
    def from_obj(cls, country: str, theme: str, collection: str, entry: dict) -> "Product":
        """Constroi um Product a partir de uma entrada do OBJ."""
        product = entry["product"]
        return cls(
            country=country,
            theme=theme,
            collection=collection,
            product=product,
            assetid=entry["assetid"],
            dtype=(entry.get("type") or "byte").lower(),
            vectorize=bool(entry.get("vectorize", False)),
            scale=entry.get("scale", 30),
            decode=entry.get("decode"),
            kind=product_kind(product),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Especificacao de uma unidade a exportar/processar por um runner."""

    product: Product
    unit: Unit
    grid: Optional[TileGrid] = None
    runner: str = "ee_batch_tiled"
    output_prefix: str = ""
    force: bool = False
    tags: dict = field(default_factory=dict)

    @property
    def unit_key(self) -> str:
        return self.unit.key
