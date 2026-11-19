"""Build app/repo_map.json: the folder tree of this repository with a short
description of each folder taken from its README.md.

Usage (from the repository root):
    python utils/fire_repo_map/build_repo_map.py

Standard library only. The Pages workflow runs it before publishing, so the
published map always matches `main`.
"""
import json
import os
import re
import subprocess
from datetime import datetime, timezone

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app", "repo_map.json")

SKIP_DIRS = {".git", "__pycache__", ".ipynb_checkpoints", "node_modules", "_site"}
REPO = "mapbiomas/brazil-fire"
BRANCH = "main"
PAGES = "https://mapbiomas.github.io/brazil-fire/"

# Folders published on GitHub Pages (kept in sync with .github/workflows/pages.yml).
PAGES_APPS = {
    "utils/fire_monitor_highlights": "fire_monitor_highlights/",
    "utils/fire_legends": "legends/",
    "utils/fire_dashboard": "dashboard/",
    "global_analises": "global_analises/",
    "utils/fire_repo_map": "repo_map/",
    "utils/fire_showcase": "",
}

KIND_BY_EXT = {
    ".js": "gee", ".py": "python", ".ipynb": "notebook", ".html": "web",
    ".css": "web", ".json": "data", ".csv": "data", ".geojson": "data",
    ".md": "docs", ".png": "image", ".jpg": "image", ".webp": "image",
    ".svg": "image", ".gif": "image", ".zip": "package", ".yml": "config",
    ".yaml": "config", ".toml": "config", ".txt": "docs",
}


def strip_md(text):
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)          # images
    text = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", text)       # links
    text = re.sub(r"[*`]+", "", text)                           # emphasis, code (keep snake_case)
    text = re.sub(r"<[^>]+>", "", text)                         # html
    return re.sub(r"\s+", " ", text).strip(" _")


def read_readme(path):
    """Return (title, description) from a README.md, or (None, None)."""
    try:
        with open(path, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return None, None
    title, desc, in_code = None, [], False
    for line in lines:
        s = line.strip()
        if s.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if title is None and s.startswith("# "):
            title = strip_md(s[2:])
            continue
        if not s:
            if desc:
                break
            continue
        if (s.startswith(("#", "|", "---", "[![", "<", "- ", "* "))
                or s.lower().startswith("developed by")):
            if desc:
                break
            continue
        if s.startswith(">"):
            s = s.lstrip("> ").strip()
            if not s:
                continue
        desc.append(s)
        if len(" ".join(desc)) > 320:
            break
    text = strip_md(" ".join(desc))
    if len(text) > 300:
        text = text[:297].rsplit(" ", 1)[0] + "…"
    return title, text or None


def status_of(rel, title):
    t = (title or "").lower()
    p = rel.lower()
    if "legacy" in t:
        return "legacy"
    if "production" in t or p.endswith("/production"):
        return "production"
    if "active" in t:
        return "active"
    if "pilot" in t or "exploratory" in t or "exploratório" in t:
        return "pilot"
    return None


def walk(abs_dir, rel):
    entries = sorted(os.listdir(abs_dir), key=str.lower)
    node = {"name": os.path.basename(abs_dir) if rel else "brazil-fire", "path": rel,
            "dirs": [], "files": [], "kinds": {}}
    for name in entries:
        full = os.path.join(abs_dir, name)
        child_rel = f"{rel}/{name}" if rel else name
        if os.path.isdir(full):
            if name in SKIP_DIRS or (name.startswith(".") and name != ".github"):
                continue
            node["dirs"].append(walk(full, child_rel))
        else:
            ext = os.path.splitext(name)[1].lower()
            kind = KIND_BY_EXT.get(ext, "other")
            node["files"].append({"name": name, "kind": kind})
            node["kinds"][kind] = node["kinds"].get(kind, 0) + 1

    title, desc = read_readme(os.path.join(abs_dir, "README.md"))
    if title:
        node["title"] = title
    if desc:
        node["desc"] = desc
    st = status_of(rel, title)
    if st:
        node["status"] = st
    if rel in PAGES_APPS:
        node["app"] = PAGES + PAGES_APPS[rel]

    total = len(node["files"])
    kinds = dict(node["kinds"])
    for d in node["dirs"]:
        total += d["total"]
        for k, v in d["kinds_total"].items():
            kinds[k] = kinds.get(k, 0) + v
    node["total"] = total
    node["kinds_total"] = kinds
    return node


def last_commit():
    try:
        out = subprocess.run(["git", "-C", ROOT, "log", "-1", "--format=%h %cI"],
                             capture_output=True, text=True, check=True).stdout.split()
        return {"sha": out[0], "date": out[1]}
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def main():
    tree = walk(ROOT, "")
    data = {
        "repo": REPO,
        "branch": BRANCH,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": last_commit(),
        "tree": tree,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
    print(f"{OUT}: {tree['total']} files, {os.path.getsize(OUT) // 1024} KB")


if __name__ == "__main__":
    main()
