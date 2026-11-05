import unittest

from gif_factory.reference_catalog import experimental_layer


class ReferenceCatalogTests(unittest.TestCase):
    def setUp(self):
        self.record = {"initiative": "colombia", "theme": "fire", "collection": "c1",
                       "product": "annual", "asset": "projects/mapbiomas-public/assets/test"}

    def test_explicit_style_is_preserved(self):
        layer = experimental_layer(self.record, "ffffff,#800000", "Sem fogo,Fogo", 0, 1)
        self.assertEqual(layer["visualization"]["palette"], ["ffffff", "800000"])
        self.assertEqual(layer["legend"][1]["label"], "Fogo")
        self.assertEqual(layer["initiative"], "Colômbia")

    def test_invalid_or_unavailable_configuration_is_rejected(self):
        for palette, labels, minimum, maximum in [("fff,000", "a,b", 0, 1),
                                                   ("ffffff,000000", "a", 0, 1),
                                                   ("ffffff,000000", "a,b", 1, 1)]:
            with self.assertRaises(ValueError):
                experimental_layer(self.record, palette, labels, minimum, maximum)
        with self.assertRaises(ValueError):
            experimental_layer({**self.record, "asset": None}, "ffffff,000000", "a,b", 0, 1)
