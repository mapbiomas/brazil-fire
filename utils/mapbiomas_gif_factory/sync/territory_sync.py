"""Fontes de território de cada iniciativa, a partir do catálogo oficial da Plataforma MapBiomas.

A Plataforma publica, por iniciativa, as categorias de território
(`/{projeto}/territories/categories`): nome em cada idioma, ordem, visibilidade e o
asset do Earth Engine (`featureCollectionId`). Para cada categoria este script:
  1. abre o asset no Earth Engine e descobre os campos de nome e de identificador;
  2. conta as feições (uma só = o território inteiro, sem busca);
  3. registra se a leitura é pública.

Grava `src/gif_factory/config/territory_sources_generated.yaml` (ARQUIVO GERADO).
As listas escritas à mão em `territory_sources.yaml` continuam valendo e vêm
primeiro; as camadas geradas entram depois, sem repetir um asset já curado.
Precisa do Earth Engine.

Uso:  python sync/territory_sync.py --project mapbiomas-brazil [--only peru,chile]
"""
import argparse
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "gif_factory" / "config" / "territory_sources_generated.yaml"
RULES = ROOT / "sync" / "generator.yaml"
API = "https://prd.plataforma.mapbiomas.org/api/v1"
HEADERS = {"tenant-id": "mapbiomas", "User-Agent": "mapbiomas-gif-factory territory sync"}
HEADER = ("# ARQUIVO GERADO por sync/territory_sync.py a partir das categorias de território da\n"
          "# Plataforma MapBiomas. Não edite à mão: rode o script de novo.\n")
NAME_FIELDS = ["NAME", "name", "NOMBRE", "nombre", "NOME", "nome", "NAME_1", "label", "LABEL"]
ID_FIELDS = ["FEATURE_ID", "feature_id", "GEOCODE", "geocode", "id", "ID", "fid", "FID", "OBJECTID"]
LANGS = ["pt-BR", "es-ES", "en-US", "fr-FR", "nl-NL", "id-ID"]


def get(path):
    for attempt in range(3):
        try:
            request = urllib.request.Request(f"{API}{path}", headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except OSError:
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))


def label_of(category):
    names = category.get("name") or {}
    for lang in LANGS:
        if names.get(lang):
            return " ".join(str(names[lang]).split())
    return category["key"].replace("_", " ").title()


def probe(ee, asset):
    """Campos de nome/id, número de feições e leitura pública do asset."""
    fc = ee.FeatureCollection(asset)
    info = ee.Dictionary({"size": fc.size(), "first": ee.Feature(fc.first()).toDictionary()}).getInfo()
    props = info["first"] or {}
    name_field = next((f for f in NAME_FIELDS if f in props), None) or \
        next((k for k, v in props.items() if isinstance(v, str) and "name" in k.lower()), None)
    id_field = next((f for f in ID_FIELDS if f in props), "system:index")
    try:
        public = bool(ee.data.getAssetAcl(asset).get("all_users_can_read"))
    except Exception:
        public = None  # sem permissão para ver a ACL: não dá para afirmar
    return {"size": info["size"], "name_field": name_field, "id_field": id_field, "public": public}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="mapbiomas-brazil")
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    import ee
    ee.Initialize(project=args.project)
    rules = yaml.safe_load(RULES.read_text(encoding="utf-8"))["initiatives"]
    only = {p.strip() for p in args.only.split(",") if p.strip()}
    current = yaml.safe_load(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    result = dict(current or {})
    for project, rule in rules.items():
        if only and project not in only:
            continue
        try:
            categories = get(f"/{project}/territories/categories?orgKey=mapbiomas").get("categories", [])
        except Exception as exc:
            print(f"ERRO {project}: {exc}")
            continue
        categories = sorted((c for c in categories if c.get("featureCollectionId") and c.get("visibility") == "public"),
                            key=lambda c: (c.get("order") is None, c.get("order") or 0, label_of(c)))

        def one(category):
            try:
                return category, probe(ee, category["featureCollectionId"]), None
            except Exception as exc:
                return category, None, str(exc)[:160]

        sources, problems = [], []
        with concurrent.futures.ThreadPoolExecutor(6) as pool:
            for category, found, error in pool.map(one, categories):
                if error or not found or not found["size"]:
                    problems.append(f"{label_of(category)}: {error or 'sem feições'}")
                    continue
                source = {"label": label_of(category), "asset": category["featureCollectionId"],
                          "id_field": found["id_field"]}
                if found["size"] == 1:
                    source["whole"] = True
                if found["name_field"]:
                    source["name_field"] = found["name_field"]
                elif found["size"] > 1:
                    problems.append(f"{source['label']}: sem campo de nome")
                    continue
                if found["public"] is False:
                    source["restricted"] = True
                sources.append(source)
        result[rule["code"]] = {"name": rule["name"], "sources": sources}
        print(f"ok   {project}: {len(sources)} camadas" + (f"; ignoradas: {problems}" if problems else ""))
    OUT.write_text(HEADER + yaml.safe_dump(result, allow_unicode=True, sort_keys=False, width=1000),
                   encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
