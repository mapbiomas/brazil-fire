"""
M3 - Gateway for the GEE Sample Collection Toolkit
"""

import os

from M_lang import L as Lang


# Canonical GEE reference toolkit (current production; catalog snapshots ship v08).
GEE_TOOLKIT_URL = (
    "https://code.earthengine.google.com/?scriptPath="
    "users%2Fmapbiomasworkspace1%2Fmapbiomas-fire%3A5-Monitor-Fuego%2FM3%2FM3_Toolkit_Monitor_Fuego"
)


def _find_repo_root(start):
    """Walk up until the brazil-fire repository root is found."""
    current = os.path.abspath(start)
    while True:
        if os.path.isdir(os.path.join(current, "network", "catalog")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def _version_label(here):
    """Return production/version_03 from the directory layout."""
    parts = os.path.normpath(here).split(os.sep)
    if "production" in parts:
        return "production"
    for part in reversed(parts):
        if part.startswith("version_"):
            return part
    return "catalog"


def _toolkit_links():
    """Return (GitHub snapshot URL, GEE reference URL) for the current version."""
    here = os.path.dirname(os.path.abspath(__file__))
    toolkit_path = os.path.join(here, "M3_toolkit.js")
    repo_root = _find_repo_root(here)
    if repo_root:
        rel_path = os.path.relpath(toolkit_path, repo_root).replace(os.sep, "/")
        github_link = (
            "https://github.com/mapbiomas/brazil-fire/blob/main/" + rel_path
        )
    else:
        # Non-standard checkout: show the local path instead of a wrong URL.
        github_link = toolkit_path
    return github_link, GEE_TOOLKIT_URL


def show_toolkit_links():
    """
    Simplified function for the M3 stage.
    Sample collection has been migrated to be carried out exclusively through the GEE JavaScript Toolkit.
    """
    github_link, gee_link = _toolkit_links()
    version = _version_label(os.path.dirname(os.path.abspath(__file__)))
    slides_link = "https://docs.google.com/presentation/d/1kQlwA-DAliL07Hp5b67o82PIBtYnsuk2tWjq04D6KYk"

    print("\n" + "="*70)
    print(f"   {Lang.M3_TITLE}")
    print("="*70)
    print(f"\n  {Lang.M3_INTRO_LINE1}")
    print(f"  {Lang.M3_INTRO_LINE2}\n")
    print(f"   1.1. {Lang.M3_SOURCE} [{version} snapshot]:")
    print(f"     {github_link}\n")
    print(f"   1.2. {Lang.M3_EDITOR} [GEE reference, current production]:")
    print(f"     {gee_link}\n")
    print(f"   2. {Lang.M3_SLIDES}:")
    print(f"     {slides_link}")
    print("\n" + "="*70 + "\n")
    return True

# Maintain compatibility with legacy notebooks
def run_collection_toolkit():
    return show_toolkit_links()

def start_sample_extraction(ui_toolkit=None):
    pass
