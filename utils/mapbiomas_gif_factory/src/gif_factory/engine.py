"""Geração experimental de GIFs a partir de imagens EE multibanda."""
from pathlib import Path
from tempfile import TemporaryDirectory
import re


def validate(layers, territories):
    for label, items, required in (("camada", layers, ("id", "name", "asset", "bands", "visualization", "legend")),
                                   ("território", territories, ("id", "name", "asset"))):
        ids = set()
        for item in items:
            missing = [key for key in required if not item.get(key)]
            if missing:
                raise ValueError(f"{label}: campos ausentes: {', '.join(missing)}")
            if not re.fullmatch(r"[A-Za-z0-9_-]+", item["id"]) or item["id"] in ids:
                raise ValueError(f"{label}: id inválido ou repetido: {item['id']}")
            ids.add(item["id"])
            if label == "camada" and (not isinstance(item["bands"], list) or not all(isinstance(b, str) and b for b in item["bands"])):
                raise ValueError("bands deve ser uma lista de nomes de bandas.")


def generate(layer, territory, root, kinds, on_artifact=None, dimension=600):
    import ee
    import requests
    from PIL import Image, ImageDraw, ImageFont
    validate([layer], [territory])
    root = Path(root) / layer["id"] / territory["id"]
    root.mkdir(parents=True, exist_ok=True)
    region = ee.FeatureCollection(territory["asset"])
    if territory.get("filter"):
        f = territory["filter"]
        region = region.filter(ee.Filter.eq(f["field"], f["value"]))
    if region.size().getInfo() == 0:
        raise ValueError("Território sem feições após aplicação do filtro.")
    geometry = region.geometry()
    image = ee.Image(layer["asset"])
    existing = image.bandNames().getInfo()
    missing = [b for b in layer["bands"] if b not in existing]
    if missing:
        raise ValueError(f"Bandas não encontradas: {missing}; disponíveis: {existing}")
    produced = []

    def save(img, relative, **kwargs):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Arquivo já existe: {path}. Use outra pasta de execução.")
        img.save(path, **kwargs)
        produced.append(path)
        if on_artifact:
            on_artifact(path)

    finals = []
    with TemporaryDirectory() as tmp:
        for index, band in enumerate(layer["bands"]):
            params = dict(layer["visualization"], dimensions=dimension, region=geometry.bounds(), format="png")
            url = image.select(band).clip(geometry).getThumbURL(params)
            response = requests.get(url, timeout=120)
            response.raise_for_status()
            raw_path = Path(tmp) / f"{index}.png"
            raw_path.write_bytes(response.content)
            with Image.open(raw_path) as raw_image:
                raw = raw_image.convert("RGB")
            if "brutos" in kinds:
                save(raw, f"frames/brutos/{index:03d}.png")
            if "preparados" in kinds:
                save(raw, f"frames/preparados/{index:03d}.png")
            try:
                font = ImageFont.truetype("DejaVuSans.ttf", 13)
            except OSError:
                font = ImageFont.load_default()
            final = Image.new("RGB", (max(raw.width, 500), raw.height + 90 + 22 * len(layer["legend"])), "white")
            final.paste(raw, (0, 40))
            draw = ImageDraw.Draw(final)
            draw.text((10, 10), f"{layer['name']} | {territory['name']} | {band}", fill="black", font=font)
            for i, entry in enumerate(layer["legend"]):
                y = raw.height + 50 + i * 22
                draw.rectangle((10, y, 25, y + 15), fill="#" + entry["color"].lstrip("#"))
                draw.text((32, y), entry["label"], fill="black", font=font)
            finals.append(final)
            if "finais" in kinds:
                save(final, f"frames/finais/{index:03d}.png")
        if "gif" in kinds:
            save(finals[0], "gifs/animacao.gif", save_all=True, append_images=finals[1:], duration=500, loop=0)
        if "colagem" in kinds:
            width = max(frame.width for frame in finals)
            height = max(frame.height for frame in finals)
            columns = min(3, len(finals))
            collage = Image.new("RGB", (width * columns, height * ((len(finals) + columns - 1) // columns)), "white")
            for i, frame in enumerate(finals):
                collage.paste(frame, ((i % columns) * width, (i // columns) * height))
            save(collage, "colagens/colagem.png")
    return produced
