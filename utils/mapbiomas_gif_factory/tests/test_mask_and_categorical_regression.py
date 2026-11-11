import unittest
from unittest.mock import MagicMock, patch

from gif_factory.legends import _categorical, _vera_style
from mapbiomas_data.core.ee_downloader import EEDownloader


class MaskAndCategoricalRegressionTests(unittest.TestCase):
    def test_vera_explicit_mask_color(self):
        entry = {"representable": True, "classes": [
            {"value": 0, "color": "cc00ff", "labels": {"pt": "Zero"}},
            {"value": 1, "color": "112233", "labels": {"pt": "Um"}},
        ]}
        viz = _vera_style(entry, "pt")["visualization"]
        self.assertEqual(viz["mask_color"], "ffffff")
        self.assertNotIn("nodata_value", viz)

    def test_categorical_labels_do_not_acquire_zero_based_prefixes(self):
        style = _categorical([(1, "Primeiro", "First", "Primero", "112233"),
                              (2, "Segundo", "Second", "Segundo", "445566")])
        self.assertFalse(style["visualization"]["prefix_labels"])
        self.assertEqual([item["label"] for item in style["legend"]], ["Primeiro", "Segundo"])

    def test_mask_color_fills_visualized_mask_not_class_zero(self):
        image = MagicMock()
        band = image.select.return_value
        rendered = band.visualize.return_value
        region = MagicMock()
        downloader = EEDownloader(MagicMock())
        downloader.vertical_dimension = 300
        with patch("mapbiomas_data.core.ee_downloader.ee.Image") as ee_image:
            downloader._get_thumb_url(image, "year", {
                "min": 0, "max": 1, "palette": ["cc00ff", "112233"],
                "mask_color": "#ffffff",
            }, region)
            band.unmask.assert_not_called()
            band.visualize.assert_called_once_with(min=0, max=1, palette=["cc00ff", "112233"], bands=["year"])
            ee_image.constant.assert_called_once_with([255, 255, 255])
            rendered.unmask.assert_called_once_with(ee_image.constant.return_value.toUint8.return_value, False)

    def test_legacy_path_still_unmasks_before_visualization(self):
        image = MagicMock()
        downloader = EEDownloader(MagicMock())
        with patch("mapbiomas_data.core.ee_downloader.ee.Image"):
            downloader._get_thumb_url(image, "year", {"min": 1, "max": 2}, MagicMock())
        image.select.return_value.unmask.assert_called_once_with()
        image.select.return_value.unmask.return_value.visualize.assert_called_once()


if __name__ == "__main__":
    unittest.main()
