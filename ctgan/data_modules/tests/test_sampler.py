import numpy as np
from unittest import TestCase

from ctgan.utils import generate_data
from ctgan.data_modules import Sampler, DataTransformer


class TestSampler(TestCase):
    def setUp(self):
        self._input_dim = 10
        self._pac = 10
        self._batch_size = 10

    def tearDown(self):
        del self._input_dim
        del self._pac
        del self._batch_size

    def test_sample(self):
        np.random.seed(0)
        data, discrete = generate_data(self._batch_size)

        transformer = DataTransformer()
        transformer.fit(data, discrete)
        train_data = transformer.transform(data)

        sampler = Sampler(train_data, transformer.output_info)
        output = sampler.sample(1, [0, 0], [0, 0])
        expected_output = np.array(
            [[0.3721639, 1., 1., 0., 0., 0.],
             [-0.31326372, 1., 1., 0., 0., 0.]])
        np.testing.assert_almost_equal(output, expected_output)

    def test_sample_none(self):
        np.random.seed(0)
        data, discrete = generate_data(self._batch_size)

        transformer = DataTransformer()
        transformer.fit(data, [])
        train_data = transformer.transform(data)

        sampler = Sampler(train_data, transformer.output_info)
        output = sampler.sample(1, None, None)
        expected_output = np.array(
            [[0.46909913, 1., 0.08564136, 0., 1.]])
        np.testing.assert_almost_equal(output, expected_output)
