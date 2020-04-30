import numpy as np
import tensorflow as tf
import pandas as pd
from unittest import TestCase

from ctgan.utils import generate_data, get_test_variables
from ctgan.data_modules import DataTransformer


class TestDataTransformer(TestCase):
    def setUp(self):
        self._vars = get_test_variables()

    def tearDown(self):
        del self._vars

    def test_fit(self):
        np.random.seed(0)
        data, discrete = generate_data(self._vars['batch_size'])

        transformer = DataTransformer()
        transformer.fit(data, discrete)
        expected_info = [(1, 'tanh', 1), (1, 'softmax', 1), (4, 'softmax', 0)]
        expected_dimensions = 6
        np.testing.assert_equal(transformer.output_info, expected_info)
        np.testing.assert_equal(
            transformer.output_dimensions, expected_dimensions)

    def test_tensors(self):
        np.random.seed(0)
        data, discrete = generate_data(self._vars['batch_size'])

        transformer = DataTransformer()
        transformer.fit(data, discrete)
        transformer.generate_tensors()
        expected_info = [
            tf.constant([0, 1, 0], dtype=tf.int32),
            tf.constant([1, 2, 1], dtype=tf.int32),
            tf.constant([2, 6, 1], dtype=tf.int32)]
        expected_cond = [tf.constant([2, 6, 0, 4, 0], dtype=tf.int32)]
        tf.assert_equal(expected_info, transformer.output_tensor)
        tf.assert_equal(expected_cond, transformer.cond_tensor)

    def test_transform(self):
        np.random.seed(0)
        data, discrete = generate_data(self._vars['batch_size'])

        transformer = DataTransformer()
        transformer.fit(data, discrete)
        transformed_data = transformer.transform(data)
        expected_data = np.array([[-0.09027826, 1., 1., 0., 0., 0.],
                                  [0.1340608, 1., 0., 1., 0., 0.],
                                  [-0.01753295, 1., 0., 1., 0., 0.],
                                  [-0.09557786, 1., 1., 0., 0., 0.],
                                  [-0.25904065, 1., 0., 1., 0., 0.],
                                  [0.04062398, 1., 0., 0., 0., 1.],
                                  [-0.24025436, 1., 0., 0., 1., 0.],
                                  [0.3721639, 1., 1., 0., 0., 0.],
                                  [0.46909913, 1., 0., 0., 1., 0.],
                                  [-0.31326372, 1., 1., 0., 0., 0.]])
        np.testing.assert_almost_equal(
            transformed_data, expected_data, decimal=self._vars['decimal'])

    def test_inverse_transform(self):
        np.random.seed(0)
        data, discrete = generate_data(self._vars['batch_size'])

        transformer = DataTransformer()
        transformer.fit(data, discrete)
        transformed_data = transformer.transform(data)
        inverse_data = transformer.inverse_transform(transformed_data)
        pd.testing.assert_frame_equal(data, inverse_data)
