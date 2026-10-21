"""Target registry for the catalog.

A campaign says where the data comes from; a target says what is predicted.
Only ``burned_area`` is enabled today. Binary is the only implemented task;
multiclass/regression profiles can be declared later and will fail loudly
instead of silently producing a binary output.
"""

import json
import os
from dataclasses import dataclass


TARGETS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'targets')
)


@dataclass(frozen=True)
class Target:
    id: str
    label: str
    task: str = 'binary'
    label_column: str = 'fire'
    classes: dict = None
    probability_band: str = 'probability'
    auxiliary_band: str | None = 'dayOfYear'
    stats_prefix: str = 'burned'
    asset_suffix: str = 'burned_area'
    post_processing: str = 'fire'
    enabled: bool = False

    @classmethod
    def from_dict(cls, data):
        if not data.get('id'):
            raise ValueError('Target definition is missing id')
        return cls(
            id=data['id'],
            label=data.get('label', data['id']),
            task=data.get('task', 'binary'),
            label_column=data.get('label_column', 'fire'),
            classes=dict(data.get('classes') or {}),
            probability_band=data.get('probability_band', 'probability'),
            auxiliary_band=data.get('auxiliary_band', 'dayOfYear'),
            stats_prefix=data.get('stats_prefix', 'burned'),
            asset_suffix=data.get('asset_suffix', data['id']),
            post_processing=data.get('post_processing', data['id']),
            enabled=bool(data.get('enabled', False)),
        )

    @property
    def positive_class(self):
        return (self.classes or {}).get('positive', 'positive')

    @property
    def negative_class(self):
        return (self.classes or {}).get('negative', 'negative')


def load_targets(refresh=False):
    if not refresh and _CACHE:
        return dict(_CACHE)
    targets = {}
    if not os.path.isdir(TARGETS_DIR):
        raise FileNotFoundError(f'Target registry not found: {TARGETS_DIR}')
    for name in sorted(os.listdir(TARGETS_DIR)):
        if not name.endswith('.json'):
            continue
        with open(os.path.join(TARGETS_DIR, name), 'r', encoding='utf-8') as handle:
            target = Target.from_dict(json.load(handle))
        targets[target.id] = target
    _CACHE.clear()
    _CACHE.update(targets)
    return dict(targets)


_CACHE = {}


def get_target(target_id):
    targets = load_targets()
    if target_id not in targets:
        raise KeyError(
            f"Unknown target '{target_id}'. Available: {', '.join(sorted(targets))}"
        )
    return targets[target_id]


def list_targets(enabled_only=True):
    targets = [t for t in load_targets().values() if t.enabled or not enabled_only]
    return sorted(targets, key=lambda item: item.id)


def require_supported(target):
    """Fail loudly when a declared profile has no implementation yet."""
    if not target.enabled:
        raise ValueError(f"Target '{target.id}' is declared but not enabled.")
    if target.task != 'binary':
        raise NotImplementedError(
            f"Target '{target.id}' uses task='{target.task}'; only 'binary' "
            "is implemented. Add the profile implementation before enabling it."
        )
    return target
