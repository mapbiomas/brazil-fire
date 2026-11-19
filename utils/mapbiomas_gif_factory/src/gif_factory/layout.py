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
    group = "_".join(safe_name(x) for x in (origin, theme, collection))
    return Path(group) / safe_name(layer["name"])


def combination_path(origin, layer, territory):
    folder = safe_name(territory["name"])
    lines = territory.get("overlays") or []
    if lines:
        # Variação com linhas: pasta própria, para não se misturar com o mapa "cru".
        import hashlib
        import json
        spec = [{k: line.get(k) for k in ("asset", "filter", "geometry", "color", "width")} for line in lines]
        digest = hashlib.sha256(json.dumps(spec, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:6]
        folder = f"{folder}_linhas_{digest}"
    return product_path(origin, layer) / folder


def ensure_unique_paths(combos):
    seen = {}
    for origin, layer, territory in combos:
        path = combination_path(origin, layer, territory)
        key = str(path).casefold()
        if key in seen:
            raise ValueError(f"Duas seleções geram a mesma pasta: {path}. Ajuste os nomes ou desmarque a duplicata.")
        seen[key] = True
