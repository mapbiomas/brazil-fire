"""Regenerate the packaged Vera registry from the pinned, vendored CSVs."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "fire_legends_source"
OUTPUT = ROOT / "src/gif_factory/vera_fire_legends.json"
REVISION = "ad70d566e5676402c597339d10907e1fdf2163f9"


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def build():
    entries = []
    seen = set()
    for item in rows(SOURCE / "registry.csv"):
        if item["status"] != "active" or not all(item[k] for k in
                ("gee_asset", "app_country", "app_collection", "app_product", "file")):
            continue
        key = tuple(item[k] for k in ("app_country", "app_collection", "app_product", "gee_asset"))
        if key in seen:
            raise ValueError(f"duplicate registry key: {key}")
        seen.add(key)
        classes = rows(SOURCE / "legends" / item["file"])
        parsed = []
        for row in classes:
            try:
                value = int(row["pixel_value"])
            except ValueError:
                parsed = []
                break
            color = row["hex"].removeprefix("#").lower()
            if not re.fullmatch(r"[0-9a-f]{6}", color):
                parsed = []
                break
            parsed.append({"value": value, "color": color, "labels": {
                lang: row[f"name_{lang}"] for lang in ("pt", "en", "es")}})
        parsed.sort(key=lambda row: row["value"])
        values = [row["value"] for row in parsed]
        representable = len(values) >= 2 and values == list(range(values[0], values[-1] + 1))
        entries.append({"key": list(key), "legend_id": item["legend_id"],
                        "file": item["file"], "representable": representable,
                        "classes": parsed})
    return {"source": "https://github.com/mapbiomas/brazil-fire/tree/" + REVISION + "/utils/fire_legends",
            "revision": REVISION, "entries": entries}


if __name__ == "__main__":
    OUTPUT.write_text(json.dumps(build(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
