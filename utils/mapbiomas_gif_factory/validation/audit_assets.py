"""Auditoria somente leitura dos assets do inventário no Earth Engine.

Registra, por registro, o tipo EE observado (IMAGE / IMAGE_COLLECTION) ou o
erro de acesso. Não valida bandas, legenda, máscara ou conteúdo: apenas
metadados de existência/tipo do asset.

Não cria tarefas, não exporta e não altera o inventário original.

Uso (na raiz do repo, com o projeto autenticado):
    python mapbiomas_gif_factory/validation/audit_assets.py [--project NAME]

Saída: mapbiomas_gif_factory/validation/asset_audit_<UTC>.json
"""
import argparse
import concurrent.futures
import json
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
INVENTORY = HERE.parent / "src" / "gif_factory" / "reference_inventory.json"


def audit(records, project):
    import ee
    ee.Initialize(project=project)

    assets = list(dict.fromkeys(r["asset"] for r in records if r.get("asset")))

    def check(asset):
        try:
            return {"asset": asset, "ee_type": ee.data.getAsset(asset).get("type", "UNKNOWN"), "error": None}
        except Exception as exc:  # noqa: BLE001 - registrar qualquer falha de acesso
            return {"asset": asset, "ee_type": None, "error": str(exc)}

    lookup = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for start in range(0, len(assets), 40):
            batch = assets[start:start + 40]
            for result in executor.map(check, batch):
                lookup[result["asset"]] = result
            time.sleep(0.3)

    entries = []
    for record in records:
        asset = record.get("asset")
        base = {"initiative": record["initiative"], "theme": record["theme"],
                "collection": record["collection"], "product": record["product"]}
        if not asset:
            entries.append({**base, "asset": None, "ee_type": None,
                            "error": record.get("asset_note", "sem asset")})
        else:
            entries.append({**base, **lookup[asset]})
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="mapbiomas-brazil")
    args = parser.parse_args()

    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    records = inventory["records"]
    entries = audit(records, args.project)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = HERE / f"asset_audit_{stamp}.json"
    payload = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "project": args.project,
        "total": len(entries),
        "records": entries,
        "scope": "existencia/tipo do asset; nao valida bandas, legenda ou conteudo",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    from collections import Counter
    counts = Counter((e["ee_type"], e["error"] is not None) for e in entries)
    print(f"{len(entries)} registros -> {output}")
    print(counts)


if __name__ == "__main__":
    main()
