"""Gera a parte do catálogo que vem direto das fontes oficiais.

Entradas:
  - legends_source/platform/<projeto>.json  (sync/legend_sync.py)
  - inventory/gee_public.json               (sync/inventory_sync.py)
  - sync/generator.yaml                     (regras escritas por pessoas)

Saídas (ARQUIVOS GERADOS — não editar à mão; rode este script de novo):
  - src/gif_factory/config/initiatives/<código>/collections/generated_lulc.yaml
  - src/gif_factory/config/initiatives/<código>/initiative.yaml   (só se ainda não existir)
  - src/gif_factory/config/visualization_generated.yaml

Para cada iniciativa: escolhe o asset de uso e cobertura da coleção mais recente e
cria um produto por nível da legenda oficial (0 = Natural/Antrópico, 1..N = níveis da
árvore; o último é o dado bruto). Cada nível recodifica o mapa com `remap`, levando
cada classe ao seu ancestral naquele nível, e ganha um estilo com as cores e os nomes
oficiais. Nada escrito à mão é sobrescrito: o que já é curado fica de fora.

Uso:  python sync/catalog_build.py [--check]   (--check: só diz se algo mudaria)
"""
import argparse
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src" / "gif_factory" / "config"
LEGENDS = ROOT / "legends_source" / "platform"
INVENTORY = ROOT / "inventory" / "gee_public.json"
RULES = ROOT / "sync" / "generator.yaml"
STYLES_FILE = CONFIG / "visualization_generated.yaml"
HEADER = ("# ARQUIVO GERADO por sync/catalog_build.py a partir das fontes oficiais. Não edite à mão:\n"
          "# ajuste sync/generator.yaml e rode o script de novo.\n")
ASSET = re.compile(r"/(?:lulc/)?collection(\d+)(?:_(\d+))?/([^/]+?)_(coverage|integration)_v(\d+)$")
BASENAME_COLLECTION = re.compile(r"collection(\d+)_(\d+)_(?:coverage|integration)")


def name_in(names, lang):
    names = names or {}
    for key in (lang, "pt-BR", "es-ES", "en-US", "fr-FR"):
        if names.get(key):
            return " ".join(str(names[key]).split())
    return next(iter(names.values()), "")


def walk(items, ancestors=()):
    """(nó, ancestrais) de toda a árvore."""
    for item in items:
        yield item, ancestors
        yield from walk(item.get("children") or [], ancestors + (item,))


def leaves(items):
    return [(node, anc) for node, anc in walk(items) if not node.get("children")]


def pick_asset(gee_folder, inventory):
    """Asset de cobertura da coleção mais recente: (id, rótulo da coleção)."""
    best = None
    prefix = f"projects/mapbiomas-public/assets/{gee_folder}/"
    for asset in inventory:
        if not asset["id"].startswith(prefix) or asset["type"] != "IMAGE":
            continue
        match = ASSET.search(asset["id"])
        if not match:
            continue
        major, minor, _stem, kind, version = match.groups()
        sub = BASENAME_COLLECTION.search(asset["id"].rsplit("/", 1)[-1])
        minor = minor or (sub.group(2) if sub and sub.group(1) == major else None)
        key = (int(major), int(minor or 0), kind == "coverage", int(version))
        if best is None or key > best[0]:
            label = f"{int(major)}.{int(minor)}" if minor else str(int(major))
            best = (key, asset["id"], label)
    return (best[1], best[2]) if best else (None, None)


def level_style(name, nodes, lang):
    """Estilo indexado pelo código do pixel: só as classes do nível têm cor e rótulo."""
    top = max(int(node["pixelValue"]) for node in nodes)
    palette, labels = ["auto_bg"] * (top + 1), [""] * (top + 1)
    for node in sorted(nodes, key=lambda n: int(n["pixelValue"])):
        code = int(node["pixelValue"])
        palette[code] = node["color"].lstrip("#").lower()
        labels[code] = name_in(node.get("name"), lang)
    return {"name": name, "min": 0, "max": top, "palette": palette, "discrete_labels": labels,
            "label": "Classe", "cmap_type": "categorical"}


def build_initiative(project, rule, legend_data, inventory):
    asset, collection = pick_asset(rule["gee"], inventory)
    if not asset:
        return None, None, f"nenhum asset de cobertura em {rule['gee']}"
    legends = legend_data["legends"]
    coverage = max(legends, key=lambda l: sum(1 for _ in walk(l["items"])))
    # Natural/Antrópico: a outra legenda do subtema, com até 3 classes no topo (Natural,
    # Antrópico, Não definido); os códigos variam por iniciativa (100/200/300 ou 1/14/200).
    natural = next((l for l in legends if l is not coverage and 2 <= len(l["items"]) <= 3
                    and any(item.get("children") for item in l["items"])), None)
    code, lang = rule["code"], rule.get("lang", "pt-BR")
    leaf_nodes = leaves(coverage["items"])
    depth = max(int(node["level"]) for node, _ in walk(coverage["items"]))
    products, styles = {}, {}
    if natural:
        pairs = [(int(leaf["pixelValue"]), int(anc[0]["pixelValue"])) for leaf, anc in leaves(natural["items"]) if anc]
        key = f"gen_{code}_lulc_nivel0"
        styles[key] = level_style("Uso e cobertura - Nível 0 (natural e antrópico)",
                                  [n for n in natural["items"]], lang)
        products["lulc_nivel0"] = {"name": "Uso e cobertura - Nível 0 (natural e antrópico)", "asset": asset,
                                   "remap": {"from": [a for a, _ in pairs], "to": [b for _, b in pairs]},
                                   "visualization": key}
    for level in range(1, depth + 1):
        targets = {}
        for leaf, ancestors in leaf_nodes:
            chain = list(ancestors) + [leaf]
            target = next((n for n in chain if int(n["level"]) == level), leaf)
            targets[int(leaf["pixelValue"])] = target
        shown = {int(n["pixelValue"]): n for n in targets.values()}
        key = f"gen_{code}_lulc_nivel{level}"
        title = f"Uso e cobertura - Nível {level}" + (" (todas as classes)" if level == depth else "")
        styles[key] = level_style(title, list(shown.values()), lang)
        product = {"name": title, "asset": asset, "visualization": key}
        if any(source != int(t["pixelValue"]) for source, t in targets.items()):
            product["remap"] = {"from": sorted(targets), "to": [int(targets[c]["pixelValue"]) for c in sorted(targets)]}
        products[f"lulc_nivel{level}"] = product
    collection_yaml = {
        "name": f"Uso e cobertura — Coleção {collection}",
        "description": f"Gerado de {legend_data['source']} ({legend_data['fetched']}) e {asset}",
        "dataset": f"gen_{code}_lulc",
        "title_collection": f"MapBiomas {rule['name']} · Coleção {collection}",
        "category": "land_cover",
        "collection": collection,
        "products": products,
    }
    return collection_yaml, styles, None


def dump(data):
    return yaml.safe_dump(data, allow_unicode=True, sort_keys=False, width=1000)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true", help="não grava; sai com 1 se algo mudaria")
    args = parser.parse_args()
    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))["assets"]
    outputs, all_styles, report = {}, {}, []
    for project, rule in rules["initiatives"].items():
        legend_path = LEGENDS / f"{project}.json"
        if rule.get("skip_lulc"):
            report.append(f"--  {project}: pulado ({rule['skip_lulc']})")
            continue
        if not legend_path.exists():
            report.append(f"--  {project}: sem legenda em {legend_path.name}")
            continue
        collection, styles, problem = build_initiative(project, rule, json.loads(legend_path.read_text(encoding="utf-8")),
                                                       inventory)
        if problem:
            report.append(f"ERRO {project}: {problem}")
            continue
        folder = CONFIG / "initiatives" / rule["code"]
        outputs[folder / "collections" / "generated_lulc.yaml"] = HEADER + dump(collection)
        initiative_file = folder / "initiative.yaml"
        if not initiative_file.exists():
            outputs[initiative_file] = dump({"name": rule["name"], "description": f"MapBiomas {rule['name']}",
                                             "id": rule["code"]})
        all_styles.update(styles)
        report.append(f"ok  {project}: coleção {collection['collection']}, {len(collection['products'])} produtos, "
                      f"{collection['products']['lulc_nivel1']['asset'].rsplit('/', 1)[-1]}")
    outputs[STYLES_FILE] = HEADER + dump({"visualizations": all_styles})
    changed = [path for path, text in outputs.items()
               if not path.exists() or path.read_text(encoding="utf-8") != text]
    print("\n".join(report))
    if args.check:
        print(f"\n{len(changed)} arquivo(s) mudariam.")
        return 1 if changed else 0
    for path in changed:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(outputs[path], encoding="utf-8", newline="\n")
    print(f"\n{len(changed)} arquivo(s) gravado(s); {len(all_styles)} estilos gerados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
