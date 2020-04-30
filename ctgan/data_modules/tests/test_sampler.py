import numpy as np
from unittest import TestCase

from ctgan.utils import generate_data, get_test_variables
from ctgan.data_modules import DataSampler, DataTransformer


class TestSampler(TestCase):
    def setUp(self):
        self._vars = get_test_variables()

    def tearDown(self):
        del self._vars

    def test_sample(self):
        np.random.seed(0)
        data, discrete = generate_data(self._vars['batch_size'])

        transformer = DataTransformer()
        transformer.fit(data, discrete)
        train_data = transformer.transform(data)

        sampler = DataSampler(train_data, transformer.output_info)
        output = sampler.sample(1, [0, 0], [0, 0])
        expected_output = np.array(
            [[0.3721639, 1., 1., 0., 0., 0.],
             [-0.31326372, 1., 1., 0., 0., 0.]])
        np.testing.assert_almost_equal(
            output, expected_output, decimal=self._vars['decimal'])

    def test_sample_none(self):
        np.random.seed(0)
        data, discrete = generate_data(self._vars['batch_size'])

        transformer = DataTransformer()
        transformer.fit(data, [])
        train_data = transformer.transform(data)

        sampler = DataSampler(train_data, transformer.output_info)
        output = sampler.sample(1, None, None)
        expected_output = np.array(
            [[0.46909913, 1., 0.08564136, 0., 1.]])
        np.testing.assert_almost_equal(
            output, expected_output, self._vars['decimal'])
