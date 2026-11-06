"""Pin EE asset loads to one snapshot for repeatable tile requests.

The approach follows DatensEE's Apache-2.0 snapshot-pinning design:
https://github.com/michaelfdewitt/datensee/blob/master/cli/src/datensee/pinning.py
"""
import copy

_LOADS = {'Image.load', 'ImageCollection.load', 'Feature.load', 'Collection.loadTable'}
_UNSUPPORTED = {'FeatureCollection.loadBigQuery', 'FeatureCollection.runBigQuery'}


def pin_expression(expression, snapshot_us):
    """Return a pinned copy of cloud-serialized EE JSON, timestamp in µs."""
    if isinstance(snapshot_us, bool) or not isinstance(snapshot_us, int) or not 0 < snapshot_us <= 10**17:
        raise ValueError('Invalid Earth Engine snapshot timestamp')
    result = copy.deepcopy(expression)

    def visit(node):
        if isinstance(node, list):
            for item in node:
                visit(item)
        elif isinstance(node, dict):
            invocation = node.get('functionInvocationValue')
            if isinstance(invocation, dict):
                name = invocation.get('functionName')
                if name in _UNSUPPORTED or (isinstance(name, str) and 'loadGeoTIFF' in name):
                    raise ValueError(f'Snapshot cannot safely pin {name}')
                if name in _LOADS:
                    invocation.setdefault('arguments', {}).setdefault(
                        'version', {'constantValue': snapshot_us})
            for value in node.values():
                visit(value)
    visit(result)
    return result
