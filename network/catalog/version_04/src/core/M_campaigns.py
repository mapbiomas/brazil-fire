"""Campaign registry for the MapBiomas Fire catalog.

A campaign binds a country + data source + sensor + periodicity. Targets
(what is being predicted) are a separate dimension, resolved by M_targets.
The registry is declarative: adding a country/campaign is a JSON change.

The GCS/GEE layout is intentionally not redefined here. Each campaign
declares its own roots, and the catalog reads them as-is, so the export
layout can never diverge from what the pipeline already produced.
"""

import json
import os
from dataclasses import dataclass, field


CAMPAIGNS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'campaigns')
)
INDEX_FILE = 'index.json'


@dataclass(frozen=True)
class Campaign:
    id: str
    label: str
    country: str
    source: str
    campaign: str
    collection: str | None
    sensors: tuple
    periodicity: str
    mosaic_methods: tuple
    catalog: str
    gcs_root: str
    gee_project: str
    gee_root: str
    asset_regions: str
    region_property: str
    year_start: int
    year_end: int
    engine: str
    targets: tuple = field(default_factory=tuple)
    status: str = 'active'

    @property
    def years(self):
        return list(range(int(self.year_start), int(self.year_end) + 1))

    @classmethod
    def from_dict(cls, data):
        required = (
            'id', 'country', 'source', 'campaign', 'sensors', 'periodicity',
            'gcs_root', 'gee_project', 'gee_root', 'asset_regions',
        )
        missing = [key for key in required if not data.get(key)]
        if missing:
            raise ValueError(
                f"Campaign '{data.get('id', '?')}' is missing: {', '.join(missing)}"
            )
        return cls(
            id=data['id'],
            label=data.get('label', data['id']),
            country=data['country'],
            source=data['source'],
            campaign=data['campaign'],
            collection=data.get('collection'),
            sensors=tuple(data.get('sensors') or ()),
            periodicity=data['periodicity'],
            mosaic_methods=tuple(data.get('mosaic_methods') or ('minnbr',)),
            catalog=data.get('catalog', 'CATALOG_01'),
            gcs_root=data['gcs_root'].rstrip('/'),
            gee_project=data['gee_project'],
            gee_root=data['gee_root'].rstrip('/'),
            asset_regions=data['asset_regions'],
            region_property=data.get('region_property', 'region_nam'),
            year_start=int(data.get('year_start', 2025)),
            year_end=int(data.get('year_end', 2025)),
            engine=data.get('engine', 'monitor'),
            targets=tuple(data.get('targets') or ('burned_area',)),
            status=data.get('status', 'active'),
        )

    def as_dict(self):
        return {
            'id': self.id,
            'label': self.label,
            'country': self.country,
            'source': self.source,
            'campaign': self.campaign,
            'collection': self.collection,
            'sensors': list(self.sensors),
            'periodicity': self.periodicity,
            'mosaic_methods': list(self.mosaic_methods),
            'catalog': self.catalog,
            'gcs_root': self.gcs_root,
            'gee_project': self.gee_project,
            'gee_root': self.gee_root,
            'asset_regions': self.asset_regions,
            'region_property': self.region_property,
            'year_start': self.year_start,
            'year_end': self.year_end,
            'engine': self.engine,
            'targets': list(self.targets),
            'status': self.status,
        }


def load_campaigns(refresh=False):
    """Return {campaign_id: Campaign} from the registry directory."""
    if not refresh and _CACHE:
        return dict(_CACHE)
    campaigns = {}
    if not os.path.isdir(CAMPAIGNS_DIR):
        raise FileNotFoundError(f'Campaign registry not found: {CAMPAIGNS_DIR}')
    for name in sorted(os.listdir(CAMPAIGNS_DIR)):
        if not name.endswith('.json') or name == INDEX_FILE:
            continue
        path = os.path.join(CAMPAIGNS_DIR, name)
        with open(path, 'r', encoding='utf-8') as handle:
            campaign = Campaign.from_dict(json.load(handle))
        campaigns[campaign.id] = campaign
    _CACHE.clear()
    _CACHE.update(campaigns)
    return dict(campaigns)


_CACHE = {}


def get_campaign(campaign_id):
    campaigns = load_campaigns()
    if campaign_id not in campaigns:
        raise KeyError(
            f"Unknown campaign '{campaign_id}'. Available: {', '.join(sorted(campaigns))}"
        )
    return campaigns[campaign_id]


def list_campaigns(country=None, source=None, periodicity=None, status=None):
    result = []
    for campaign in load_campaigns().values():
        if country and campaign.country != country:
            continue
        if source and campaign.source != source:
            continue
        if periodicity and campaign.periodicity != periodicity:
            continue
        if status and campaign.status != status:
            continue
        result.append(campaign)
    return sorted(result, key=lambda item: item.id)


def campaign_paths(campaign):
    """GCS paths owned by a campaign. The layout is the existing one."""
    base = f"{campaign.gcs_root}/{campaign.campaign}"
    return {
        'campaign': base,
        'images': f"{campaign.gcs_root}/LIBRARY_IMAGES",
        'models': f"{base}/LIBRARY_MODELS",
        'samples': f"{base}/LIBRARY_SAMPLES",
        'classifications': f"{base}/LIBRARY_CLASSIFICATIONS",
        'cache': f"{campaign.gcs_root}/.CACHE",
    }


def campaign_gee_paths(campaign):
    base = f"{campaign.gee_root}/{campaign.campaign}"
    return {
        'campaign': base,
        'images': f"{campaign.gee_root}/LIBRARY_IMAGES",
        'models': f"{base}/LIBRARY_MODELS",
        'samples': f"{base}/LIBRARY_SAMPLES",
        'classifications': f"{base}/LIBRARY_CLASSIFICATIONS",
    }


def set_campaign(campaign_id, target_id='burned_area', language='en',
                 personal_task_flag='MONITOR', clean_cache=False):
    """Resolve a campaign + target and configure the pipeline globals.

    Keeps ``set_global_opts`` untouched, so the rest of the M0-M6 code keeps
    working exactly as before. Extra identifiers are stored in CONFIG for the
    target-aware modules and for the catalog.
    """
    from M_targets import get_target, require_supported
    from M0_auth_config import CONFIG, set_global_opts

    campaign = get_campaign(campaign_id)
    target = get_target(target_id)
    require_supported(target)
    if target_id not in campaign.targets:
        raise ValueError(
            f"Campaign '{campaign_id}' does not declare target '{target_id}'. "
            f"Declared: {', '.join(campaign.targets)}"
        )

    set_global_opts(
        country=campaign.country,
        campaign=campaign.campaign,
        sensor=list(campaign.sensors),
        periodicity=campaign.periodicity,
        mosaic_methods=list(campaign.mosaic_methods),
        personal_task_flag=personal_task_flag,
        language=language,
        gcs_bucket='mapbiomas-fire',
        gcs_library_images_prefix=campaign.gcs_root,
        gcs_campaigns_prefix=campaign.gcs_root,
        gee_project=campaign.gee_project,
        gee_library_images_prefix=campaign.gee_root,
        gee_campaigns_prefix=campaign.gee_root,
        asset_regions=campaign.asset_regions,
        clean_cache=clean_cache,
    )

    CONFIG['campaign_id'] = campaign.id
    CONFIG['campaign_source'] = campaign.source
    CONFIG['campaign_collection'] = campaign.collection
    CONFIG['campaign_engine'] = campaign.engine
    CONFIG['target_id'] = target.id
    return CONFIG
