import tensorflow as tf
import numpy as np
from unittest import TestCase

from ctgan.layers import ResidualLayer


class TestResidualLayer(TestCase):
    def setUp(self):
        self._input_dim = 5
        self._output_dim = 5
        self._batch_size = 1

    def tearDown(self):
        del self._input_dim
        del self._output_dim
        del self._batch_size

    def test_residual_layer(self):
        tf.random.set_seed(0)
        inputs = tf.random.uniform([self._batch_size, self._input_dim])
        residual_layer = ResidualLayer(
            self._input_dim,
            self._output_dim)

        outputs = residual_layer(inputs)
        expected_outputs = tf.constant(
            [[0.3448831, 0.1013758, 0., 0.4421106, 0., 0.2919751,
              0.2065665, 0.5353907, 0.5612575, 0.4166745]],
            dtype=tf.float32)
        np.testing.assert_almost_equal(outputs.numpy(), expected_outputs.numpy())
