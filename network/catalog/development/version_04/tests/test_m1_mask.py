"""Verify the M1 geometry mask is applied without clipping the image."""
import pathlib
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'src' / 'core'))
import M1_export_logic as logic


class GeometryMaskTests(unittest.TestCase):
    def test_paint_and_update_preserve_existing_mask(self):
        image, geometry, feature, collection = (MagicMock() for _ in range(4))
        empty_image, painted, mask = (MagicMock() for _ in range(3))
        empty_image.paint.return_value = painted
        painted.selfMask.return_value = mask
        with patch.object(logic.ee, 'Feature', return_value=feature) as make_feature, \
             patch.object(logic.ee, 'FeatureCollection', return_value=collection) as make_collection, \
             patch.object(logic.ee, 'Image', return_value=empty_image):
            result = logic.mask_to_geometry(image, geometry)
        make_feature.assert_called_once_with(geometry)
        make_collection.assert_called_once_with([feature])
        empty_image.paint.assert_called_once_with(collection, 1)
        image.updateMask.assert_called_once_with(mask)
        image.clip.assert_not_called()
        self.assertIs(result, image.updateMask.return_value)


if __name__ == '__main__':
    unittest.main()
