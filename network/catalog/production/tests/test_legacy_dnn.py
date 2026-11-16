"""version_04: M4 defaults match the legacy version_02 network."""

import ast
import os
import unittest

CORE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'core'))


def _source(name):
    with open(os.path.join(CORE, name), encoding='utf-8') as handle:
        return handle.read()


class LegacyDefaultsTests(unittest.TestCase):
    def test_trainer_defaults(self):
        src = _source('M4_algorithms_dnn.py')
        train = next(
            node for node in ast.walk(ast.parse(src))
            if isinstance(node, ast.FunctionDef) and node.name == 'train'
        )
        names = [a.arg for a in train.args.args]
        defaults = dict(zip(names[-len(train.args.defaults):], train.args.defaults))
        self.assertEqual(ast.literal_eval(defaults['keep_prob']), 1.0)
        self.assertIn("'model_layers', [7, 14, 7, 14, 7]", src)
        self.assertIn("'model_iters', 7000", src)

    def test_ui_default_layers(self):
        self.assertIn('value="7, 14, 7, 14, 7"', _source('M4_ui.py'))


if __name__ == '__main__':
    unittest.main()
