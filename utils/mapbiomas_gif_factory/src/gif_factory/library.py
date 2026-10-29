"""Inventário conservador da biblioteca: nenhum artefato é sobrescrito."""
from pathlib import Path
import hashlib
import json
import os
import uuid
from .temporal import temporal_key


def variant_for(all_bands, selected):
    if list(all_bands) == list(selected):
        return "completo"
    keys = [temporal_key(b) for b in selected]
    if all(k is not None and k[1:] == (0, 0) for k in keys):
        years = [k[0] for k in keys]
        if years == list(range(years[0], years[-1] + 1)):
            return f"{years[0]}-{years[-1]}"
    digest = hashlib.sha256(json.dumps(selected, ensure_ascii=False).encode()).hexdigest()[:12]
    return f"selecao_{digest}"


def fingerprint(layer, territory, bands, dimension=2000):
    relevant = {k: v for k, v in layer.items() if k not in {"id", "name", "status", "start_year", "end_year", "bands"}}
    value = {"layer": relevant, "territory": territory, "bands": list(bands), "dimension": dimension, "renderer": "cartografico-v1"}
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def checksum(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_path(root, variant):
    return Path(root) / "metadados" / f"{variant}.json"


def load_manifest(root, variant):
    path = manifest_path(root, variant)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != 1 or not isinstance(data.get("artifacts"), dict):
            return None
        for records in data["artifacts"].values():
            if not isinstance(records, list):
                return None
            for record in records:
                if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                    return None
                rel = Path(record["path"])
                if not record["path"] or rel.is_absolute() or ".." in rel.parts:
                    return None
        return data
    except (OSError, ValueError, TypeError):
        return None


def assess(root, variant, signature, kinds):
    root = Path(root)
    manifest = load_manifest(root, variant)
    if manifest is None:
        if manifest_path(root, variant).exists():
            return "verificar"
        tracked = set()
        metadata_dir = root / "metadados"
        if metadata_dir.is_dir():
            for meta in metadata_dir.glob("*.json"):
                other = load_manifest(root, meta.stem)
                if other is None:
                    return "verificar"
                for records in other["artifacts"].values():
                    tracked.update(record["path"] for record in records)
        for folder in ("GIFs", "Colagens", "Frames"):
            directory = root / folder
            if directory.exists():
                for p in directory.rglob("*"):
                    if not p.is_file() or p.suffix.lower() not in {".gif", ".png"}:
                        continue
                    if p.relative_to(root).as_posix() in tracked:
                        continue
                    if variant == "completo":
                        # Existing files without provenance cannot safely be classified.
                        return "verificar"
                    if p.stem.endswith("_" + variant) or variant in p.relative_to(directory).parts:
                        return "verificar"
        return "nao_gerado"
    if manifest.get("fingerprint") != signature:
        return "verificar"
    missing = False
    for kind in kinds:
        records = manifest["artifacts"].get(kind)
        if not records:
            missing = True
            continue
        for record in records:
            rel = Path(record.get("path", ""))
            if rel.is_absolute() or ".." in rel.parts:
                return "verificar"
            path = root / rel
            if not path.is_file():
                missing = True
            elif path.stat().st_size != record.get("size") or checksum(path) != record.get("sha256"):
                return "verificar"
    return "parcial" if missing else "disponivel"


def artifact_kind(relative):
    parts = Path(relative).parts
    if parts[0] == "GIFs":
        return "gif"
    if parts[0] == "Colagens":
        return "colagem"
    if len(parts) > 1 and parts[0] == "Frames":
        return {"Mapa_basico": "brutos", "Mapa_com_ano_e_escala": "preparados", "Mapa_completo": "finais"}[parts[1]]
    raise ValueError(f"Arquivo fora da estrutura esperada: {relative}")


def ingest(staging, destination, variant, signature, bands, layer, territory):
    """Mescla apenas arquivos ausentes; não altera resultados existentes."""
    staging, destination = Path(staging), Path(destination)
    existing = load_manifest(destination, variant)
    if manifest_path(destination, variant).exists() and existing is None:
        raise ValueError("Metadados inválidos; verifique a biblioteca antes de continuar.")
    if existing and existing.get("fingerprint") != signature:
        raise ValueError("Configuração diferente da registrada; nenhum arquivo foi sobrescrito.")
    manifest = existing or {"schema": 1, "fingerprint": signature, "bands": list(bands), "layer": layer, "territory": territory, "artifacts": {}}
    files = sorted(p for p in staging.rglob("*") if p.is_file() and p.suffix.lower() in {".gif", ".png"})
    if not files:
        raise ValueError("Nenhum artefato para publicar.")
    candidates = []
    for source in files:
        relative = source.relative_to(staging)
        target = destination / relative
        digest = checksum(source)
        if target.exists() and (not target.is_file() or checksum(target) != digest):
            raise FileExistsError(f"Resultado diferente já existe: {target}; nenhum arquivo foi sobrescrito.")
        candidates.append((source, target, relative, digest))
    import shutil
    for source, target, relative, digest in candidates:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            with source.open("rb") as src, target.open("xb") as dst:
                try:
                    shutil.copyfileobj(src, dst)
                except OSError:
                    dst.close()
                    target.unlink(missing_ok=True)
                    raise
        kind = artifact_kind(relative)
        records = manifest["artifacts"].setdefault(kind, [])
        record = {"path": relative.as_posix(), "size": target.stat().st_size, "sha256": digest}
        records[:] = [r for r in records if r.get("path") != record["path"]] + [record]
    save_manifest(destination, variant, manifest)
    return [target for _, target, _, _ in candidates]


def save_manifest(root, variant, manifest):
    path = manifest_path(root, variant)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tmp.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def sync_selected(files, source_root, destination_root):
    """Copia seleção local e seus metadados, preservando conjuntos incompletos."""
    import shutil
    source_root, destination_root = Path(source_root).resolve(), Path(destination_root).resolve()
    if source_root == destination_root or source_root in destination_root.parents:
        raise ValueError("A pasta de destino não pode estar dentro da pasta de resultados.")
    groups = {}
    for file in files:
        source = Path(file).resolve()
        source.relative_to(source_root)
        found = None
        for parent in source.parents:
            if parent != source_root and source_root not in parent.parents:
                break
            directory = parent / "metadados"
            if not directory.is_dir():
                continue
            relative = source.relative_to(parent).as_posix()
            for meta in directory.glob("*.json"):
                manifest = load_manifest(parent, meta.stem)
                if manifest:
                    for kind, records in manifest["artifacts"].items():
                        record = next((r for r in records if r.get("path") == relative), None)
                        if record:
                            found = (parent, meta.stem, manifest, kind, record)
                            break
                if found:
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Arquivo sem metadados confiáveis: {source}")
        parent, variant, manifest, kind, record = found
        if not source.is_file() or checksum(source) != record.get("sha256"):
            raise ValueError(f"Arquivo local diferente dos metadados: {source}")
        target_root = destination_root / parent.relative_to(source_root)
        target_manifest = load_manifest(target_root, variant)
        if manifest_path(target_root, variant).exists() and target_manifest is None:
            raise ValueError(f"Metadados inválidos no Drive: {target_root}")
        if target_manifest and target_manifest.get("fingerprint") != manifest.get("fingerprint"):
            raise ValueError(f"Configuração diferente no Drive: {target_root}")
        target = target_root / record["path"]
        if target.exists() and (not target.is_file() or checksum(target) != record["sha256"]):
            raise FileExistsError(f"Arquivo diferente já existe no Drive: {target}")
        key = (parent, variant)
        groups.setdefault(key, {"manifest": manifest, "target_root": target_root, "kinds": set(), "files": []})
        groups[key]["kinds"].add(kind)
        groups[key]["files"].append((source, target))
    results = []
    for (_, variant), group in groups.items():
        target_root, source_manifest = group["target_root"], group["manifest"]
        existing = load_manifest(target_root, variant)
        manifest = existing or {k: v for k, v in source_manifest.items() if k != "artifacts"}
        manifest.setdefault("artifacts", {})
        for kind in group["kinds"]:
            for record in source_manifest["artifacts"][kind]:
                target = target_root / record["path"]
                if target.exists() and (not target.is_file() or checksum(target) != record["sha256"]):
                    raise FileExistsError(f"Arquivo diferente já existe no Drive: {target}")
        for source, target in group["files"]:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                results.append((str(source), "já existe", str(target)))
                continue
            with source.open("rb") as src, target.open("xb") as dst:
                try:
                    shutil.copyfileobj(src, dst)
                except OSError:
                    dst.close()
                    target.unlink(missing_ok=True)
                    raise
            results.append((str(source), "copiado", str(target)))
        for kind in group["kinds"]:
            manifest["artifacts"][kind] = source_manifest["artifacts"][kind]
        save_manifest(target_root, variant, manifest)
    return results
