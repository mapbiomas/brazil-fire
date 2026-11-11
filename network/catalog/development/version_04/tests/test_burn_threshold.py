"""version_04: burned cut is p >= 0.05 and M4 defaults match legacy version_02."""

import ast
import os
import sys
import unittest

import numpy as np

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))
sys.path.insert(0, CORE)

import M_dnn  # noqa: E402


def _source(name):
    with open(os.path.join(CORE, name), encoding='utf-8') as handle:
        return handle.read()


class BurnThresholdTests(unittest.TestCase):
    def test_threshold_is_five_percent(self):
        self.assertEqual(M_dnn.BURN_THRESHOLD, 50)

    def test_low_probabilities_are_not_burned(self):
        probs = np.array([0.0004, 0.0006, 0.01, 0.049, 0.05, 0.08, 0.92])
        burned = np.round(probs * 1000) >= M_dnn.BURN_THRESHOLD
        self.assertEqual(
            burned.tolist(), [False, False, False, False, True, True, True]
        )

    def test_m5_and_m6_use_the_threshold(self):
        for name in ('M5_inference.py', 'M6_publisher.py'):
            src = _source(name)
            self.assertIn('from M_dnn import BURN_THRESHOLD', src, name)
            self.assertNotIn('* 1000) > 0', src, name)
            self.assertNotIn('prob_band > 0', src, name)


class LegacyDefaultsTests(unittest.TestCase):
    def test_trainer_defaults(self):
        tree = ast.parse(_source('M4_algorithms_dnn.py'))
        train = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == 'train'
        )
        names = [a.arg for a in train.args.args]
        defaults = dict(zip(names[-len(train.args.defaults):], train.args.defaults))
        self.assertEqual(ast.literal_eval(defaults['keep_prob']), 1.0)
        self.assertIn("'model_layers', [7, 14, 7, 14, 7]", _source('M4_algorithms_dnn.py'))

    def test_ui_default_layers(self):
        self.assertIn('value="7, 14, 7, 14, 7"', _source('M4_ui.py'))


if __name__ == '__main__':
    unittest.main()
