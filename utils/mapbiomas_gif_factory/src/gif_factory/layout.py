"""Nomes legíveis e seguros para pastas compartilhadas."""
from pathlib import Path
import re
import unicodedata


def safe_name(value, fallback="Sem_nome"):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_-")[:80].rstrip("_-")
    if not text:
        text = fallback
    if text.upper() in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        text = f"_{text}"
    return text


def product_path(origin, layer):
    theme = layer.get("theme") or ("Fogo" if origin == "Brasil" else "Sem_tema")
    collection = layer.get("collection") or ("Colecao_5" if origin == "Brasil" else "Sem_colecao")
    return Path(*(safe_name(x) for x in (origin, theme, collection, layer["name"])))


def combination_path(origin, layer, territory):
    return product_path(origin, layer) / safe_name(territory["name"])


def ensure_unique_paths(combos):
    seen = {}
    for origin, layer, territory in combos:
        path = combination_path(origin, layer, territory)
        key = str(path).casefold()
        if key in seen:
            raise ValueError(f"Duas seleções geram a mesma pasta: {path}. Ajuste os nomes ou desmarque a duplicata.")
        seen[key] = True
