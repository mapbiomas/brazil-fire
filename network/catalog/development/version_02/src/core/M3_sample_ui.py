"""
M3 - Gateway for the GEE Sample Collection Toolkit
"""

from M_lang import L as Lang

def show_toolkit_links():
    """
    Simplified function for the M3 stage.
    Sample collection has been migrated to be carried out exclusively through the GEE JavaScript Toolkit.
    """
    github_link = "https://github.com/mapbiomas/peru-fire/blob/main/mapbiomas_fire_monitor/version_01/src/core/M3_toolkit.js"
    gee_link = "https://code.earthengine.google.com/?scriptPath=users%2Fmapbiomasworkspace1%2Fmapbiomas-fire%3A5-Monitor-Fuego%2FToolkit_Monitor_Fuego"
    slides_link = "https://docs.google.com/presentation/d/1kQlwA-DAliL07Hp5b67o82PIBtYnsuk2tWjq04D6KYk"

    print("\n" + "="*70)
    print(f"   {Lang.M3_TITLE}")
    print("="*70)
    print(f"\n  {Lang.M3_INTRO_LINE1}")
    print(f"  {Lang.M3_INTRO_LINE2}\n")
    print(f"   1.1. {Lang.M3_SOURCE}:")
    print(f"     {github_link}\n")
    print(f"   1.2. {Lang.M3_EDITOR}:")
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
