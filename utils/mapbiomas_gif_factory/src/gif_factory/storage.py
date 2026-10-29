"""Publicação seletiva, sem sobrescrever arquivos existentes."""
from pathlib import Path
import shutil


def list_artifacts(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in {".gif", ".png"}) if root.exists() else []


def copy_artifacts(files, root, destination):
    root, destination = Path(root).resolve(), Path(destination).resolve()
    if destination == root or root in destination.parents:
        raise ValueError("A pasta de destino não pode estar dentro da pasta de resultados.")
    results = []
    for file in files:
        try:
            source = Path(file).resolve()
            relative = source.relative_to(root)
            if not source.is_file():
                raise FileNotFoundError(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation avoids silently replacing a previous result.
            with source.open("rb") as src:
                with target.open("xb") as dst:
                    try:
                        shutil.copyfileobj(src, dst)
                    except OSError:
                        dst.close()
                        target.unlink(missing_ok=True)
                        raise
            results.append((str(source), "copiado", str(target)))
        except FileExistsError:
            results.append((str(file), "já existe; não sobrescrito", ""))
        except (OSError, ValueError) as exc:
            results.append((str(file), "falha", str(exc)))
    return results
