"""Revisão do catálogo: acesso, bandas e miniatura de cada produto.

Para cada produto de `gif_factory/config` (validado, beta e pendente):
  1. carrega a imagem como o pipeline faz (asset ou processador + bands_slice);
  2. lista as bandas/anos disponíveis;
  3. gera uma miniatura do último ano com a paleta configurada.

O resultado é uma folha de contato HTML para revisar todos os produtos de uma
vez, mais um JSON com o status técnico de cada um. Nada é exportado no GEE.

Uso (na raiz do projeto, com o Earth Engine autenticado):
    python validation/catalog_review.py --project mapbiomas-brazil
    python validation/catalog_review.py --only brasil_fire_col5   # filtra por id

Saída: validation/catalog_review/<UTC>/index.html
"""
import argparse
import concurrent.futures
import html
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))

# Recorte da miniatura por iniciativa: [oeste, sul, leste, norte].
REGIONS = {
    "brasil": [-74.0, -34.0, -34.0, 5.5],
    "peru": [-81.5, -18.5, -68.5, 0.0],
    "paraguay": [-62.7, -27.7, -54.2, -19.2],
    "drc": [12.0, -13.5, 31.5, 5.5],
}


def review(layer, project, out_dir, size):
    import ee
    from gif_factory.engine import catalog_image, discover
    from gif_factory.catalog import config_loader
    from mapbiomas_data.core import VisualizationManager

    result = {key: layer[key] for key in ("id", "name", "initiative", "theme", "collection", "status",
                                          "visualization_key", "processor", "asset")}
    try:
        all_bands, _ = discover(layer)
        result["bands"] = len(all_bands)
        result["first_band"], result["last_band"] = all_bands[0], all_bands[-1]
    except Exception as exc:
        result["error"] = f"Bandas: {exc}"
        return result
    try:
        image, product = catalog_image(layer)
        viz = VisualizationManager(config_loader()).get_viz_params(layer["visualization_key"])
        band = image.select(all_bands[-1])
        if layer["initiative_code"] in REGIONS:
            region = ee.Geometry.Rectangle(REGIONS[layer["initiative_code"]])
        else:
            # Iniciativas sem recorte conhecido: a área do próprio asset.
            region = ee.Image(layer["asset"]).geometry().bounds() if layer.get("asset") else \
                ee.Geometry.Rectangle(REGIONS["brasil"])
        if product.get("rgb"):
            year = all_bands[-1][-4:]
            bands = [b for b in all_bands if b.endswith(year)][:3]
            vis = image.select(bands).visualize(min=viz.get("min", 0), max=viz.get("max", 1))
        elif viz.get("random_viz") or not viz.get("palette"):
            vis = band.randomVisualizer().select(["viz-red", "viz-green", "viz-blue"])
        else:
            vis = band.visualize(min=viz.get("min", 0), max=viz.get("max", 1), palette=viz["palette"])
        url = vis.getThumbURL({"region": region, "dimensions": size, "format": "png"})
        thumb = out_dir / "thumbs" / f"{layer['id']}.png"
        for attempt in range(3):  # o servidor de miniaturas às vezes derruba a conexão
            try:
                with urllib.request.urlopen(url, timeout=180) as response:
                    thumb.write_bytes(response.read())
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(5 * (attempt + 1))
        result["thumb"] = f"thumbs/{thumb.name}"
        result["legend"] = layer["legend"]
    except Exception as exc:
        result["error"] = f"Miniatura: {exc}"
    return result


def write_html(results, out_dir, stamp):
    ok = sum(1 for r in results if "error" not in r)
    cards = []
    for r in results:
        swatches = "".join(
            f'<span class="sw" style="background:#{html.escape(str(item["color"]).lstrip("#"))}" '
            f'title="{html.escape(str(item["label"]))}"></span>' for item in r.get("legend", [])[:24])
        body = (f'<img src="{html.escape(r["thumb"])}" loading="lazy">' if r.get("thumb")
                else f'<div class="err">{html.escape(r.get("error", "sem miniatura"))}</div>')
        bands = (f'{r["bands"]} bandas · {html.escape(r["first_band"])} → {html.escape(r["last_band"])}'
                 if r.get("bands") else "")
        cards.append(f'''<article class="card {html.escape(r["status"])}{' fail' if 'error' in r else ''}">
  {body}
  <h3>{html.escape(r["name"])}</h3>
  <p class="meta">{html.escape(r["initiative"])} · {html.escape(r["theme"])} · col. {html.escape(str(r["collection"]))}
     · <b>{html.escape(r["status"])}</b></p>
  <p class="meta">estilo <code>{html.escape(r["visualization_key"])}</code>
     {f'· processador <code>{html.escape(r["processor"])}</code>' if r.get("processor") else ''}</p>
  <p class="meta">{bands}</p>
  <div class="legend">{swatches}</div>
  <p class="id"><code>{html.escape(r["id"])}</code></p>
</article>''')
    page = f'''<!doctype html><meta charset="utf-8"><title>Revisão do catálogo</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;background:#f6f6f4;color:#222}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:16px}}
.card{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px}}
.card.beta{{border-color:#d9a400}} .card.fail{{border-color:#c62828;background:#fff5f5}}
img{{width:100%;background:#eee;border-radius:4px}} h3{{font-size:14px;margin:8px 0 4px}}
.meta{{font-size:12px;margin:2px 0;color:#555}} .id{{font-size:11px;color:#888}}
.err{{font-size:12px;color:#c62828;min-height:80px}} .sw{{display:inline-block;width:14px;height:14px;border:1px solid #999;margin-right:2px}}
</style>
<h1>Revisão do catálogo</h1>
<p>{stamp} · {ok} de {len(results)} produtos com miniatura. Borda amarela = beta; vermelha = falha.</p>
<div class="grid">{"".join(cards)}</div>'''
    (out_dir / "index.html").write_text(page, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default="mapbiomas-brazil")
    parser.add_argument("--only", default="", help="filtra produtos cujo id contém este texto")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    import ee
    ee.Initialize(project=args.project)
    from gif_factory.catalog import catalog_layers

    layers = [layer for layer in catalog_layers(("validado", "beta", "pendente")) if args.only in layer["id"]]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = HERE / "catalog_review" / stamp
    (out_dir / "thumbs").mkdir(parents=True)
    print(f"Revisando {len(layers)} produtos -> {out_dir}")
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(review, layer, args.project, out_dir, args.size): layer for layer in layers}
        results = []
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            mark = "ERRO" if "error" in result else "ok  "
            print(f"[{len(results):3d}/{len(layers)}] {mark} {result['id']} {result.get('error', '')[:120]}")
    order = {layer["id"]: i for i, layer in enumerate(layers)}
    results.sort(key=lambda r: order[r["id"]])
    (out_dir / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    write_html(results, out_dir, stamp)
    failed = sum(1 for r in results if "error" in r)
    print(f"\n{len(results) - failed} ok, {failed} com erro. Abra: {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
