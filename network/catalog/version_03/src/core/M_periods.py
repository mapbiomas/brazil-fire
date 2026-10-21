"""Period definitions and normalization helpers.

Annual and monthly are the enabled periods. The registry already describes
semester and quarter so adding them later is a configuration change, not a
new branch in every module.
"""

from dataclasses import dataclass
from datetime import datetime


PERIODS = {
    'yearly': {
        'folder': 'ANNUALLY',
        'gee_folder': 'YEARLY',
        'periods_per_year': 1,
        'months': 12,
        'key': '{year}',
        'state_suffix': 'annually',
        'asset_type': 'yearly_burned_area',
        'label': 'yearly',
        'enabled': True,
    },
    'monthly': {
        'folder': 'MONTHLY',
        'gee_folder': 'MONTHLY',
        'periods_per_year': 12,
        'months': 1,
        'key': '{year}_{index:02d}',
        'state_suffix': 'monthly',
        'asset_type': 'monthly_burned_area',
        'label': 'monthly',
        'enabled': True,
    },
    'semestral': {
        'folder': 'SEMIANNUAL',
        'gee_folder': 'SEMIANNUAL',
        'periods_per_year': 2,
        'months': 6,
        'key': '{year}_S{index}',
        'state_suffix': 'semestral',
        'asset_type': 'semiannual_burned_area',
        'label': 'semestral',
        'enabled': False,
    },
    'quarterly': {
        'folder': 'QUARTERLY',
        'gee_folder': 'QUARTERLY',
        'periods_per_year': 4,
        'months': 3,
        'key': '{year}_Q{index}',
        'state_suffix': 'quarterly',
        'asset_type': 'quarterly_burned_area',
        'label': 'quarterly',
        'enabled': False,
    },
}

ALL_PERIODICITIES = tuple(PERIODS)
ENABLED_PERIODICITIES = tuple(
    name for name, definition in PERIODS.items() if definition['enabled']
)
FOLDER_TO_PERIODICITY = {
    definition['folder']: name for name, definition in PERIODS.items()
}


@dataclass(frozen=True)
class Period:
    key: str
    year: int
    index: int
    periodicity: str
    folder: str
    start: datetime
    end: datetime
    asset_type: str
    months: int
    label: str


def _add_months(value, months):
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    return datetime(year, month, 1)


def period_key(year, index, periodicity):
    """Build the canonical key: 2025, 2025_06, 2025_S1, 2025_Q1."""
    definition = PERIODS[periodicity]
    return definition['key'].format(year=int(year), index=int(index))


def parse_period(value, periodicity=None):
    """Return a Period for a key. Future period keys are parsed too."""
    if isinstance(value, Period):
        return value

    text = str(value).strip()
    parts = text.split('_')
    if not parts or not parts[0].isdigit() or len(parts[0]) != 4:
        raise ValueError(f"Invalid period '{value}'; expected YYYY, YYYY_MM, YYYY_S# or YYYY_Q#.")
    year = int(parts[0])

    if len(parts) == 1:
        inferred = 'yearly'
        index = 1
    elif len(parts) == 2:
        second = parts[1]
        if second.startswith('S'):
            inferred = 'semestral'
            index = int(second[1:])
        elif second.startswith('Q'):
            inferred = 'quarterly'
            index = int(second[1:])
        else:
            inferred = 'monthly'
            index = int(second)
    else:
        raise ValueError(f"Invalid period '{value}'; too many components.")

    periodicity = (periodicity or inferred).lower()
    if periodicity not in PERIODS:
        raise ValueError(f"Unsupported periodicity '{periodicity}'.")
    if periodicity != inferred:
        raise ValueError(f"Period '{value}' does not match periodicity '{periodicity}'.")

    definition = PERIODS[periodicity]
    if not 1 <= index <= definition['periods_per_year']:
        raise ValueError(
            f"Invalid index {index} for {periodicity}; expected 1..{definition['periods_per_year']}."
        )

    start = datetime(year, (index - 1) * definition['months'] + 1, 1)
    return Period(
        key=period_key(year, index, periodicity),
        year=year,
        index=index,
        periodicity=periodicity,
        folder=definition['folder'],
        start=start,
        end=_add_months(start, definition['months']),
        asset_type=definition['asset_type'],
        months=definition['months'],
        label=definition['label'],
    )


def list_periods(years, periodicity, include_current=False, now=None):
    """Return periods for the given years, newest first.

    include_current=True includes the period in progress (used by M1/M2 for
    yearly). Monthly always excludes the current month until it is complete.
    """
    definition = PERIODS[periodicity]
    now = now or datetime.now()
    result = []
    for year in sorted({int(year) for year in years}, reverse=True):
        for index in range(definition['periods_per_year'], 0, -1):
            period = parse_period(period_key(year, index, periodicity))
            if period.start > now:
                continue
            if include_current or period.end <= now:
                result.append(period)
    return result


def period_state_key(kind, periodicity):
    """Return the cache key suffix, e.g. cogs_monthly / assets_annually."""
    return f"{kind}_{PERIODS[periodicity]['state_suffix']}"


def normalize_period(value):
    """Return YYYY, YYYY_MM or a future period key as text."""
    if value is None:
        return ''
    text = str(value).strip()
    if text.lower() in {'', 'nan', 'nat', 'none', '<na>'}:
        return ''
    if text.endswith('.0') and text[:-2].isdigit():
        text = text[:-2]
    if text.isdigit() and len(text) == 6:
        return f"{text[:4]}_{text[4:]}"
    return text


def period_year(value):
    """Return the integer year from a period value, or None."""
    token = normalize_period(value).split('_')[0]
    try:
        return int(float(token))
    except (TypeError, ValueError):
        return None
