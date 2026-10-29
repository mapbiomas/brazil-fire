import os
import re
from typing import List


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def clean_filename(name: str, keep_extension: bool = True) -> str:
    if keep_extension:
        base, ext = os.path.splitext(name)
        base = re.sub(r"[^\w\s-]", "", base)
        base = re.sub(r"\s+", "_", base.strip())
        return base + ext if ext else base
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name


def list_files(directory: str, extension: str = None) -> List[str]:
    if not os.path.exists(directory):
        return []
    files = os.listdir(directory)
    if extension:
        files = [f for f in files if f.endswith(extension)]
    return sorted(files)
