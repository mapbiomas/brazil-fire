"""Regression coverage for the isolated M2 country mask path."""
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src' / 'core'))
import M2_mosaic_logic as m2


class M2MaskRegression(unittest.TestCase):
    def test_validate_real_cog_and_reject_plain_geotiff(self):
        import numpy as np
        import rasterio
        from rasterio.shutil import copy
        from rasterio.transform import from_origin

        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'source.tif')
            cog = os.path.join(directory, 'cog.tif')
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
                with rasterio.open(source, 'w', driver='GTiff', width=4, height=4, count=1,
                                   dtype='uint8', crs='EPSG:4326', transform=from_origin(0, 4, 1, 1)) as dst:
                    dst.write(np.ones((4, 4), dtype='uint8'), 1)
                    dst.write_mask(np.array([[0, 255, 255, 255]] * 4, dtype='uint8'))
            copy(source, cog, driver='COG', BIGTIFF='YES')
            m2._validate_masked_cog(source, cog)
            with self.assertRaisesRegex(ValueError, 'not a COG'):
                m2._validate_masked_cog(source, source)

    def test_collections_burn_only_polygon_features_once_and_drop_alpha(self):
        import numpy as np
        import rasterio
        from rasterio.enums import ColorInterp
        from rasterio.features import rasterize
        from rasterio.transform import from_origin

        polygon = {'type': 'Polygon', 'coordinates': [[[0, 0], [2, 0], [2, 4], [0, 4], [0, 0]]]}
        geometry = {'type': 'GeometryCollection', 'geometries': [polygon,
                    {'type': 'LineString', 'coordinates': [[3, 0], [3, 4]]}]}
        calls = []
        def burn(args, label=None):
            calls.append(args)
            with open(args[-2], encoding='utf-8') as f:
                features = json.load(f)['features']
            self.assertEqual([x['geometry']['type'] for x in features], ['Polygon'])
            with rasterio.open(args[-1], 'r+') as target:
                target.write(rasterize([x['geometry'] for x in features], out_shape=target.shape,
                                       transform=target.transform, dtype='uint8'), 1)

        with tempfile.TemporaryDirectory() as directory:
            source, output = (os.path.join(directory, x) for x in ('src.tif', 'out.tif'))
            with rasterio.open(source, 'w', driver='GTiff', width=4, height=4, count=2,
                               dtype='uint8', crs='EPSG:4326', transform=from_origin(0, 4, 1, 1)) as dst:
                dst.write(np.full((4, 4), 9, dtype='uint8'), 1)
                dst.write(np.array([[0, 255, 255, 255]] * 4, dtype='uint8'), 2)
                dst.colorinterp = (ColorInterp.gray, ColorInterp.alpha)
            with patch.object(m2, 'run_cmd', side_effect=burn), \
                 patch.object(m2, '_remote_mask_blob', return_value=None):
                m2.mask_raster_to_country(source, output, geometry, window_size=2)
            self.assertEqual(len(calls), 1)
            with rasterio.open(output) as dst:
                self.assertEqual(dst.count, 1)
                np.testing.assert_array_equal(dst.dataset_mask() > 0, [[False, True, False, False]] * 4)
                self.assertEqual(dst.transform, from_origin(0, 4, 1, 1))

    def test_validation_failure_prevents_replacement(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(m2, 'CONFIG', {'bucket': 'bucket', 'asset_regions': 'regions', 'country': 'testland'}), \
             patch.object(m2, 'period_cog_path', return_value='cogs'), \
             patch.object(m2, 'mosaic_name_for_period', return_value='mosaic_red'), \
             patch('M_gcs.blob_exists', return_value=True), \
             patch.object(m2, 'get_temp_dir', return_value=directory), \
             patch.object(m2, 'check_m2_dependencies', return_value=[]), \
             patch.object(m2, 'download'), patch.object(m2, '_country_geometry_geojson', return_value={}), \
             patch.object(m2, 'mask_raster_to_country'), patch.object(m2, 'run_cmd') as command, \
             patch.object(m2, '_validate_masked_cog', side_effect=ValueError('invalid')), \
             patch.object(m2, 'upload') as upload, \
             patch('M1_update_manifest.complete_gcs_item') as complete:
            with self.assertRaisesRegex(ValueError, 'invalid'):
                m2.mask_existing_country_cog('2025_01', 'red', 'sentinel2', 'minnbr', gdal_threads=3)
            self.assertIn('NUM_THREADS=3', command.call_args.args[0])
            upload.assert_not_called()
            complete.assert_not_called()


if __name__ == '__main__':
    unittest.main()
