"""Regression tests for the session-local country raster cache."""
import json
import os
import sys
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'core'))
import M2_mosaic_logic as m2

POLYGON = {'type': 'Polygon', 'coordinates': [[[0, 0], [2, 0], [2, 4], [0, 4], [0, 0]]]}


class CountryMaskCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.counter = 0
        self.lock = threading.Lock()

    def source(self, name, transform=None, width=4, height=4, crs='EPSG:4326', valid=True):
        path = os.path.join(self.temp.name, name)
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
            path, 'w', driver='GTiff', width=width, height=height, count=1, dtype='uint8',
            crs=crs, transform=transform or from_origin(0.01, 4, 1, 1)
        ) as dst:
            dst.write(np.ones((height, width), dtype='uint8'), 1)
            dst.write_mask(np.full((height, width), 255 if valid else 0, dtype='uint8'))
        return path

    def burn(self, args, label=None):
        with self.lock:
            self.counter += 1
        with open(args[-2], encoding='utf-8') as f:
            features = json.load(f)['features']
        with rasterio.open(args[-1], 'r+') as dst:
            dst.write(rasterize([f['geometry'] for f in features], out_shape=dst.shape,
                                transform=dst.transform, dtype='uint8'), 1)

    def test_reuse_and_grid_geometry_invalidation(self):
        sources = [self.source('a.tif'), self.source('b.tif', valid=False),
                   self.source('c.tif', transform=from_origin(1, 4, 1, 1)),
                   self.source('d.tif', width=5), self.source('e.tif', crs='EPSG:3857'),
                   self.source('f.tif', height=5)]
        with patch.object(m2, 'run_cmd', side_effect=self.burn):
            for i, source in enumerate(sources):
                m2.mask_raster_to_country(source, os.path.join(self.temp.name, f'out{i}.tif'), POLYGON)
            changed = {'type': 'Polygon', 'coordinates': [[[0, 0], [3, 0], [3, 4], [0, 4], [0, 0]]]}
            m2.mask_raster_to_country(sources[0], os.path.join(self.temp.name, 'changed.tif'), changed)
            with patch.object(m2, '_MASK_CACHE_VERSION', m2._MASK_CACHE_VERSION + 1):
                m2.mask_raster_to_country(sources[0], os.path.join(self.temp.name, 'version.tif'), POLYGON)
        self.assertEqual(self.counter, 7)
        with rasterio.open(os.path.join(self.temp.name, 'out1.tif')) as result:
            self.assertFalse(result.dataset_mask().any())

    def test_parallel_same_key_and_retry_after_failure(self):
        source = self.source('src.tif', transform=from_origin(0.125, 4, 1, 1))
        def apply(i):
            return m2.mask_raster_to_country(source, os.path.join(self.temp.name, f'out{i}.tif'), POLYGON)
        with patch.object(m2, 'run_cmd', side_effect=RuntimeError('burn failed')):
            with self.assertRaisesRegex(RuntimeError, 'burn failed'):
                apply('failed')
        with patch.object(m2, 'run_cmd', side_effect=self.burn):
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(apply, range(4)))
        self.assertEqual(self.counter, 1)


if __name__ == '__main__':
    unittest.main()
