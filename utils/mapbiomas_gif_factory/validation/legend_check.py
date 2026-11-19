"""Compara as legendas do catálogo com as legendas oficiais do MapBiomas.

Três fontes oficiais, cada estilo comparado com a sua:
  - uso e cobertura: a Plataforma (https://prd.plataforma.mapbiomas.org/api/v1), que
    publica por iniciativa as classes com valor do pixel, nome, cor e nível (a coleção
    atual de cobertura, hoje a 11 no Brasil);
  - fogo: as legendas oficiais de `fire_legends_source/legends/*.csv`, copiadas de
    `utils/fire_legends` do brazil-fire. A cobertura queimada segue a coleção de
    cobertura usada pelo fogo, que pode ter cores diferentes da coleção atual;
  - solo: `soil_legends_source/collection_03beta_legends.js`, copiado do repositório
    mapbiomas/brazil-soil (paletas e legendas oficiais do MapBiomas Solo, Coleção 3).

Para cada estilo de classes do catálogo (`config/visualization*.yaml`) aponta:
  - cor diferente da oficial para um valor de pixel que o estilo usa;
  - nome da classe diferente do oficial (ignorando acento, caixa e o "N - " inicial).
Para os estilos contínuos do Solo, confere paleta, mínimo e máximo.

Valores que o estilo deixa como fundo (`auto_bg`) ou cinza neutro não são comparados:
em geral são classes "mãe" que o processador nunca produz naquele nível.

Uso:
    python validation/legend_check.py            # relatório no terminal
    python validation/legend_check.py --json out.json
"""
import argparse
import csv
import json
import re
import sys
import unicodedata
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

API = "https://prd.plataforma.mapbiomas.org/api/v1"
HEADERS = {"tenant-id": "mapbiomas", "User-Agent": "mapbiomas-gif-factory legend check"}
NEUTRAL = {"auto_bg", "auto", "theme_bg", "background", "808080"}
FIRE_LEGENDS = HERE.parent / "fire_legends_source" / "legends"
SOIL_LEGENDS = HERE.parent / "soil_legends_source" / "collection_03beta_legends.js"
HEX = re.compile(r"#([0-9a-fA-F]{6})")

# estilo do catálogo -> (fonte, identificador na fonte, idioma ou coluna do nome)
STYLE_SOURCES = {
    "lulc": ("plataforma", "brazil", "pt-BR"),
    "lulc_peru": ("plataforma", "peru", "es-ES"),
    "lulc_peru_nivel3": ("plataforma", "peru", "es-ES"),
    "lulc_drc_nivel2": ("plataforma", "democratic-republic-of-congo", "fr-FR"),
    "coverage_nivel4": ("fogo", "brazil_col5_annual_burned_coverage", "name_pt"),
    "fire_col5_monthly": ("fogo", "brazil_col5_monthly_burned_area", "name_pt"),
    "fire_col5_frequency": ("fogo", "brazil_col5_fire_frequency", "name_pt"),
    "scar_size_range": ("fogo", "brazil_col5_fire_scar_size", "name_pt"),
    "fire_col5_year_last_fire": ("fogo", "brazil_col5_last_fire_year", "name_pt"),
    "fire_col5_severity": ("fogo", "brazil_col5_fire_severity", "name_pt"),
    "fire_col5_return_interval": ("fogo", "brazil_col5_fire_return_interval", "name_pt"),
    "monthly_peru": ("fogo", "peru_col2_monthly_burned_area", "name_es"),
    "fire_col2_frequency": ("fogo", "peru_col2_fire_frequency", "name_es"),
    "fire_col2_year_last_fire": ("fogo", "peru_col2_last_fire_year", "name_es"),
    "fire_col2_scar_size_range": ("fogo", "peru_col2_fire_scar_size", "name_es"),
    "soil_texture_l1": ("solo", "L1", None),
    "soil_texture_l2": ("solo", "L2", None),
    "soil_texture_l3": ("solo", "L3", None),
}

# estilos contínuos do Solo -> nome do bloco no arquivo oficial
SOIL_CONTINUOUS = {"soil_carbon_col3": "carbon", "soil_sand_col3": "areia",
                   "soil_silt_col3": "silte", "soil_clay_col3": "argila"}

# Diferenças conhecidas e aceitas, com o motivo (o relatório mostra, mas não reprova).
ACCEPTED = {
    "fire_col5_return_interval": "o processador agrupa 10–15, 15–20 e 20+ anos nas classes 10, 11 e 12",
    "fire_col2_scar_size_range": "estilo não usado por nenhum produto do catálogo",
    "scar_size_range": "mesmas faixas, escritas com 'a' em vez de '-'",
    "lulc_drc_nivel2": "a Plataforma traz o nome da classe 21 em inglês no campo em francês",
}


def plain(text):
    text = unicodedata.normalize("NFKD", str(text or ""))
    return "".join(c for c in text if not unicodedata.combining(c)).casefold().strip()


def clean_label(label):
    """'15 - Pastagem' / 'Pastagem (15)' -> 'Pastagem'."""
    label = re.sub(r"^\s*\d+\s*[-–:]\s+", "", str(label))
    return re.sub(r"\s*\(\d+\)\s*$", "", label).strip()


def fetch(project):
    request = urllib.request.Request(f"{API}/{project}/themes?expand=true&pageSize=100", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def official_coverage(project, lang):
    """{valor do pixel: (cor, nome, nível)} da legenda anual de cobertura mais completa."""
    themes = fetch(project).get("themes", [])
    legends = [legend for theme in themes for sub in theme.get("subthemes", [])
               if sub.get("dataType") == "annual" for legend in sub.get("legends") or []]
    if not legends:
        raise RuntimeError(f"{project}: nenhuma legenda anual na Plataforma")
    legend = max(legends, key=lambda item: len(item.get("items") or []))
    result = {}
    for item in legend["items"]:
        names = item.get("name") or {}
        name = names.get(lang) or names.get("en-US") or next(iter(names.values()), "")
        result[int(item["pixelValue"])] = (item["color"].lstrip("#").lower(), name, item.get("level"))
    return result


def official_fire(name, column):
    """{valor do pixel: (cor, nome, None)} de uma legenda oficial de fogo."""
    with (FIRE_LEGENDS / f"{name}.csv").open(encoding="utf-8") as handle:
        return {int(row["pixel_value"]): (row["hex"].lstrip("#").lower(), row[column], None)
                for row in csv.DictReader(handle)}


def _soil_block(pattern):
    match = re.search(pattern, SOIL_LEGENDS.read_text(encoding="utf-8"), re.S)
    if not match:
        raise RuntimeError(f"Trecho não encontrado em {SOIL_LEGENDS.name}: {pattern}")
    return match.group(1)


def official_soil(level, _column=None):
    """{valor do pixel: (cor, nome em português, None)} de TEXTURA_PALETTE/LEGEND do nível."""
    colors = [c.lower() for c in HEX.findall(_soil_block(r"TEXTURA_PALETTE\." + level + r"\s*=\s*\[(.*?)\]"))]
    legend = _soil_block(r"TEXTURA_LEGEND\." + level + r"\s*=\s*\{(.*?)\}")
    names = {int(k): v.split("|")[0].strip() for k, v in re.findall(r"(\d+)\s*:\s*'([^']*)'", legend)}
    return {pixel: (colors[pixel], names.get(pixel, ""), None) for pixel in range(len(colors))}


def check_soil_continuous(visualizations):
    """Paleta, mínimo e máximo dos estilos contínuos do Solo iguais aos oficiais."""
    problems = []
    for style_key, name in SOIL_CONTINUOUS.items():
        block = _soil_block(r"\b" + name + r"\s*:\s*\{(.*?)\}")
        official = {"min": float(re.search(r"min:\s*(-?[\d.]+)", block).group(1)),
                    "max": float(re.search(r"max:\s*(-?[\d.]+)", block).group(1)),
                    "palette": [c.lower() for c in HEX.findall(block)]}
        viz = visualizations.get(style_key) or {}
        ours = {"min": float(viz.get("min", "nan")), "max": float(viz.get("max", "nan")),
                "palette": [str(c).lstrip("#").lower() for c in viz.get("palette") or []]}
        if ours != official:
            problems.append({"estilo": style_key, "oficial": official, "catalogo": ours})
    return problems


def compare(style_key, viz, official):
    palette = [str(c).lstrip("#").lower() for c in viz.get("palette") or []]
    labels = viz.get("discrete_labels") or []
    start = int(viz.get("min", 0))
    colors, names = [], []
    for pixel, (color, name, _level) in sorted(official.items()):
        index = pixel - start
        if not 0 <= index < len(palette) or palette[index] in NEUTRAL:
            continue
        if palette[index] != color:
            colors.append({"pixel": pixel, "classe": name, "oficial": color, "catalogo": palette[index]})
        if isinstance(labels, dict):  # rótulos por valor do pixel
            label = labels.get(pixel, labels.get(str(pixel)))
        else:
            label = labels[index] if index < len(labels) else None
        if label and plain(clean_label(label)) != plain(name):
            names.append({"pixel": pixel, "oficial": name, "catalogo": clean_label(label)})
    return {"estilo": style_key, "cores": colors, "nomes": names}


LOADERS = {"plataforma": official_coverage, "fogo": official_fire, "solo": official_soil}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--json", help="grava o resultado neste arquivo")
    args = parser.parse_args()
    from gif_factory.catalog import config_loader
    visualizations = config_loader().visualizations
    cache, report = {}, []
    for style_key, (kind, source, lang) in STYLE_SOURCES.items():
        if style_key not in visualizations:
            continue
        if (kind, source, lang) not in cache:
            cache[(kind, source, lang)] = LOADERS[kind](source, lang)
        item = compare(style_key, visualizations[style_key], cache[(kind, source, lang)])
        item["fonte"] = f"{kind}:{source}"
        report.append(item)
    for item in report:
        if not item["cores"] and not item["nomes"]:
            status = "ok"
        elif item["estilo"] in ACCEPTED:
            status = f"aceito — {ACCEPTED[item['estilo']]}"
        else:
            status = "DIVERGE"
        print(f"\n{item['estilo']} [{item['fonte']}]: {status} "
              f"({len(item['cores'])} cor(es), {len(item['nomes'])} nome(s))")
        for c in item["cores"]:
            print(f"  cor  {c['pixel']:>4} {c['classe']}: oficial #{c['oficial']} · catálogo #{c['catalogo']}")
        for n in item["nomes"]:
            print(f"  nome {n['pixel']:>4}: oficial “{n['oficial']}” · catálogo “{n['catalogo']}”")
    failing = [item["estilo"] for item in report
               if (item["cores"] or item["nomes"]) and item["estilo"] not in ACCEPTED]
    continuous = check_soil_continuous(visualizations)
    for problem in continuous:
        print(f"\n{problem['estilo']} [solo]: DIVERGE")
        print(f"  oficial  {problem['oficial']}")
        print(f"  catálogo {problem['catalogo']}")
        failing.append(problem["estilo"])
    total = len(report) + len(SOIL_CONTINUOUS)
    summary = f"{total - len(failing)} de {total} estilos conferem com a legenda oficial"
    print("\n" + summary + (f"; divergem: {', '.join(failing)}" if failing else "."))
    if args.json:
        Path(args.json).write_text(json.dumps({"classes": report, "solo_continuo": continuous},
                                              ensure_ascii=False, indent=2), encoding="utf-8")
    return 1 if failing else 0


if __name__ == "__main__":
    sys.exit(main())
