from unittest import TestCase

from ctgan.utils import load_demo


class TestLoadDemo(TestCase):

    def test_load_demo(self):
        data, discrete = load_demo()
        self.assertEqual(data.shape, (32561, 15))
        expected_cols = ['age', 'workclass', 'fnlwgt', 'education',
                         'education-num', 'marital-status', 'occupation',
                         'relationship', 'race', 'sex', 'capital-gain',
                         'capital-loss', 'hours-per-week', 'native-country',
                         'income']
        self.assertEqual(expected_cols, list(data.columns.values))

        expected_discrete = [
            'workclass', 'education', 'marital-status', 'occupation',
            'relationship', 'race', 'sex', 'native-country', 'income']
        self.assertEqual(expected_discrete, discrete)
