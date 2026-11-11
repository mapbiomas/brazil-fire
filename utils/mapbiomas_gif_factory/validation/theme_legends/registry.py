"""Snapshot de estilos locais, sem inferir associações a produtos."""

import csv
import hashlib
from pathlib import Path

import yaml


SOURCE = Path(__file__).with_name("visualization.source.yaml")
OUTPUT = Path(__file__).with_name("styles.csv")

# Seleção editorial de estilos não-fogo presentes diretamente no arquivo raiz.
STYLE_IDS = (
    "random", "lulc", "lulc_nivel0", "lulc_nivel1", "lulc_nivel1_1",
    "lulc_nivel2", "landsat_swir", "landsat_rgb", "def_seco", "soc",
    "edge_area_col101", "edge_area_col11", "edge_area_1000m",
    "edge_age_col101", "edge_area", "distances", "patch_size",
    "secondary_vegetation_coverage", "morphology", "secondary_vegetation_age",
    "lulc_drc_nivel0", "lulc_drc_nivel1", "lulc_drc_nivel2",
    "lulc_peru_nivel0", "lulc_peru_nivel1", "lulc_peru_nivel2",
    "lulc_peru_nivel3", "lulc_peru",
)
FIELDS = ("style_id", "source", "source_sha256", "source_yaml", "status",
          "country_link", "collection_link", "asset_link", "processor_link", "validation")


class StrictLoader(yaml.SafeLoader):
    pass


def _mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"Chave YAML duplicada: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _mapping)


def build_rows(source=SOURCE):
    data = source.read_bytes()
    text = data.decode("utf-8")
    parsed = yaml.load(text, Loader=StrictLoader)
    root = yaml.compose(text, Loader=StrictLoader)
    visualizations = next(v for k, v in root.value if k.value == "visualizations")
    nodes = {key.value: (key, value) for key, value in visualizations.value}
    digest = hashlib.sha256(data).hexdigest()
    rows = []
    for style_id in STYLE_IDS:
        key, value = nodes[style_id]
        # Preserva grafia, aspas, ordem, listas e demais metadados originais.
        snippet = text[key.start_mark.index:value.end_mark.index]
        standalone = yaml.load(snippet, Loader=StrictLoader)
        if standalone != {style_id: parsed["visualizations"][style_id]}:
            raise ValueError(f"Trecho não reproduz metadados de {style_id}")
        rows.append(dict(zip(FIELDS, (style_id, "config/visualization.yaml", digest,
                                     snippet, "provisorio", *(["nao_comprovado"] * 5)))))
    return rows


def write_csv(rows, output=OUTPUT):
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    write_csv(build_rows())
