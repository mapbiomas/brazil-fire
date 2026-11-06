"""
M_regions — GEE FeatureCollection registry for administrative regions.

Each country maps to the GEE asset path of its region featureCollection.
The actual region names are stored inside each featureCollection (not hardcoded here).
"""

# The property name used to filter regions in GEE FeatureCollections.
REGION_NAME_PROPERTY = 'region_nam'

REGION_ASSETS = {
    'peru':     'projects/mapbiomas-peru/assets/FIRE/AUXILIARY_DATA/regiones_fuego_peru_v1',
    'bolivia':  'projects/mapbiomas-bolivia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_bolivia_v1',
    'chile':    'projects/mapbiomas-chile/assets/FIRE/AUXILIARY_DATA/regiones_fuego_chile_v1',
    'colombia': 'projects/mapbiomas-colombia/assets/FIRE/AUXILIARY_DATA/regiones_fuego_colombia_v1',
    'ecuador':  'projects/mapbiomas-ecuador/assets/FIRE/AUXILIARY_DATA/regiones_fuego_ecuador_v1',
    'guyana':   'projects/mapbiomas-guyana/assets/FIRE/AUXILIARY_DATA/regiones_fuego_guyana_v1',
    'suriname': 'projects/mapbiomas-suriname/assets/FIRE/AUXILIARY_DATA/regiones_fuego_suriname_v1',
    'venezuela':'projects/mapbiomas-venezuela/assets/FIRE/AUXILIARY_DATA/regiones_fuego_venezuela_v1',
    'paraguay': 'projects/mapbiomas-paraguay/assets/FIRE/AUXILIARY_DATA/regiones_fuego_paraguay_v1',
}


def list_countries():
    """Return sorted list of available country codes."""
    return sorted(REGION_ASSETS.keys())


def asset_for(country):
    """Return the GEE featureCollection asset path for a given country."""
    country = country.lower().replace(' ', '_')
    return REGION_ASSETS.get(country)


def country_label(code):
    """Return a human-readable label for a country code."""
    labels = {
        'peru': 'Peru',
        'brazil': 'Brasil',
        'bolivia': 'Bolivia',
        'colombia': 'Colombia',
        'ecuador': 'Ecuador',
        'guyana': 'Guyana',
        'suriname': 'Suriname',
        'french_guiana': 'Guiana Francesa',
        'venezuela': 'Venezuela',
        'paraguay': 'Paraguay',
        'argentina': 'Argentina',
        'chile': 'Chile',
        'uruguay': 'Uruguay',
        'indonesia': 'Indonesia',
    }
    return labels.get(code.lower(), code.title())
