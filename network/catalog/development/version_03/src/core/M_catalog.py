"""Catalog inventory: campaigns x targets and their data status.

This is the "what can I classify?" layer. It never redefines the GCS/GEE
layout: every path comes from the campaign registry. GCS checks are opt-in
because they cost network calls.
"""

import M_campaigns
import M_targets


def matrix():
    """Return one row per (campaign, target) pair."""
    rows = []
    for campaign in M_campaigns.list_campaigns():
        for target_id in campaign.targets:
            target = M_targets.get_target(target_id)
            rows.append({
                'campaign': campaign.id,
                'campaign_label': campaign.label,
                'country': campaign.country,
                'source': campaign.source,
                'campaign_folder': campaign.campaign,
                'periodicity': campaign.periodicity,
                'sensors': list(campaign.sensors),
                'target': target.id,
                'target_label': target.label,
                'target_enabled': target.enabled,
                'status': campaign.status,
            })
    return rows


def _gcs_status(campaign):
    from M_gcs import exists, ls

    paths = M_campaigns.campaign_paths(campaign)

    def _count(folder):
        if not exists(folder):
            return 0
        try:
            return len(ls(folder))
        except Exception:
            return 0

    has_images = exists(paths['images'])
    models = _count(paths['models'])
    samples = _count(paths['samples'])
    classifications = _count(paths['classifications'])

    if not has_images:
        status = 'missing_images'
    elif not models:
        status = 'missing_models'
    elif not samples:
        status = 'missing_samples'
    elif not classifications:
        status = 'classified_pending'
    else:
        status = 'ready'

    return {
        'images': has_images,
        'models': models,
        'samples': samples,
        'classifications': classifications,
        'status': status,
    }


def describe(campaign_id, check_gcs=False):
    campaign = M_campaigns.get_campaign(campaign_id)
    targets = []
    for target_id in campaign.targets:
        target = M_targets.get_target(target_id)
        targets.append({
            'id': target.id,
            'label': target.label,
            'task': target.task,
            'enabled': target.enabled,
            'probability_band': target.probability_band,
            'auxiliary_band': target.auxiliary_band,
            'stats_prefix': target.stats_prefix,
            'asset_suffix': target.asset_suffix,
        })

    data = _gcs_status(campaign) if check_gcs else None
    return {
        'campaign': campaign.as_dict(),
        'targets': targets,
        'paths': M_campaigns.campaign_paths(campaign),
        'gee_paths': M_campaigns.campaign_gee_paths(campaign),
        'data': data,
        'status': data['status'] if data else campaign.status,
    }


def inventory(check_gcs=False, country=None, source=None):
    return [
        describe(campaign.id, check_gcs=check_gcs)
        for campaign in M_campaigns.list_campaigns(country=country, source=source)
    ]


def classify_plan(campaign_id, target_id, periods, regions=None, model=None):
    """Skeleton of M5 jobs for a campaign/target.

    The actual workplan is still written by M5_workplan. This exists so the
    catalog can show "what would be classified" without touching the queue.
    """
    campaign = M_campaigns.get_campaign(campaign_id)
    target = M_targets.get_target(target_id)
    M_targets.require_supported(target)
    if target_id not in campaign.targets:
        raise ValueError(
            f"Campaign '{campaign_id}' does not declare target '{target_id}'."
        )

    periods = [str(period) for period in periods]
    regions = list(regions or [])
    return [
        {
            'campaign': campaign.id,
            'campaign_folder': campaign.campaign,
            'country': campaign.country,
            'target': target.id,
            'model': model,
            'period': period,
            'regions': regions,
            'periodicity': campaign.periodicity,
        }
        for period in periods
    ]


def validate_registry():
    """Cheap consistency checks used by tests and by the UI."""
    problems = []
    seen = set()
    for campaign in M_campaigns.list_campaigns():
        if campaign.id in seen:
            problems.append(f'duplicate campaign id: {campaign.id}')
        seen.add(campaign.id)
        if not campaign.targets:
            problems.append(f'{campaign.id}: no targets')
        for target_id in campaign.targets:
            try:
                M_targets.get_target(target_id)
            except KeyError:
                problems.append(f'{campaign.id}: unknown target {target_id}')
        if campaign.periodicity not in ('monthly', 'yearly'):
            problems.append(f'{campaign.id}: unsupported periodicity {campaign.periodicity}')
    return problems
