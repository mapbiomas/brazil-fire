"""Baixa da Plataforma MapBiomas as legendas oficiais de uso e cobertura de cada iniciativa.

Para cada projeto da Plataforma (`/projects?type=dashboard`):
  1. acha o subtema anual de cobertura (o de legenda com mais classes);
  2. lista as legendas desse subtema (`/{projeto}/legends`): a de cobertura e,
     quando existe, a de Natural/Antrópico;
  3. baixa cada legenda inteira (`/{projeto}/legends/{id}`), que vem como árvore:
     cada classe com valor do pixel, nome (pt/es/en/fr), cor, nível e filhos.

Grava `legends_source/platform/<projeto>.json`. A data da leitura fica no arquivo,
e o git guarda o histórico: uma mudança de cor ou de nome aparece no diff do PR.
Não precisa do Earth Engine.

Uso:  python sync/legend_sync.py [--only brazil,peru]
"""
import argparse
import json
import sys
import time
import urllib.request
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "legends_source" / "platform"
API = "https://prd.plataforma.mapbiomas.org/api/v1"
HEADERS = {"tenant-id": "mapbiomas", "User-Agent": "mapbiomas-gif-factory legend sync"}


def get(path, retries=3):
    for attempt in range(retries):
        try:
            request = urllib.request.Request(f"{API}{path}", headers=HEADERS)
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except OSError:
            if attempt == retries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def paged(path, key):
    first = get(f"{path}{'&' if '?' in path else '?'}page=1")
    items = list(first.get(key, []))
    for page in range(2, (first.get("numberOfPages") or 1) + 1):
        items += get(f"{path}{'&' if '?' in path else '?'}page={page}").get(key, [])
    return items


def coverage_subtheme(themes):
    """O subtema anual cuja legenda tem mais classes (a cobertura completa)."""
    annual = [sub for theme in themes for sub in theme.get("subthemes", [])
              if sub.get("dataType") == "annual" and sub.get("legends")]
    if not annual:
        return None
    return max(annual, key=lambda sub: max(len(legend.get("items") or []) for legend in sub["legends"]))


def sync_project(project):
    themes = paged(f"/{project['key']}/themes?expand=true&pageSize=100", "themes")
    subtheme = coverage_subtheme(themes)
    if subtheme is None:
        return None
    legends = [legend for legend in paged(f"/{project['key']}/legends?pageSize=100", "legends")
               if (legend.get("subtheme") or {}).get("key") == subtheme["key"]]
    trees = []
    for legend in legends:
        detail = get(f"/{project['key']}/legends/{legend['id']}").get("legend", {})
        trees.append({"id": legend["id"], "key": legend["key"], "name": legend.get("name"),
                      "items": detail.get("items", [])})
    return {"project": project["key"], "name": project.get("name"), "fetched": date.today().isoformat(),
            "source": f"{API}/{project['key']}/legends", "subtheme": subtheme["key"],
            "subtheme_name": subtheme.get("name"), "legends": trees}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", default="", help="projetos separados por vírgula")
    args = parser.parse_args()
    only = {p.strip() for p in args.only.split(",") if p.strip()}
    OUT.mkdir(parents=True, exist_ok=True)
    projects = get("/projects?type=dashboard").get("projects", [])
    failed = []
    for project in projects:
        if only and project["key"] not in only:
            continue
        try:
            data = sync_project(project)
        except Exception as exc:  # uma iniciativa fora do ar não derruba as outras
            failed.append(project["key"])
            print(f"ERRO {project['key']}: {exc}")
            continue
        if data is None:
            print(f"--   {project['key']}: sem legenda anual de cobertura")
            continue
        path = OUT / f"{project['key']}.json"
        previous = json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
        if previous and {k: v for k, v in previous.items() if k != "fetched"} == \
                {k: v for k, v in data.items() if k != "fetched"}:
            print(f"=    {project['key']}: sem mudança")
            continue  # mantém a data antiga: o PR só mostra o que mudou de verdade
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
        sizes = [len(legend["items"]) for legend in data["legends"]]
        print(f"ok   {project['key']}: {data['subtheme']} · legendas {sizes}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
