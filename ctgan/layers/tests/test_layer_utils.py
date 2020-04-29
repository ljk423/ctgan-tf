import tensorflow as tf
import numpy as np
from functools import partial
from unittest import TestCase

from ctgan.layers import init_bounded


class TestLayerUtils(TestCase):
    def setUp(self):
        self._input_dim = 5
        self._output_dim = 5

    def tearDown(self):
        del self._input_dim
        del self._output_dim

    def test_no_args(self):
        self.assertRaises(
            AttributeError,
            init_bounded,
            (self._input_dim, self._output_dim))
        self.assertRaises(
            AttributeError,
            init_bounded,
            (self._input_dim, self._output_dim),
            dim=self._input_dim)

    def test_init_layer(self):
        fc = tf.keras.layers.Dense(
            self._output_dim, input_dim=(self._input_dim,),
            kernel_initializer=partial(init_bounded, dim=self._input_dim),
            bias_initializer=partial(init_bounded, dim=self._input_dim))
        fc.build((self._input_dim,))
        bound = 1 / np.sqrt(self._input_dim)
        self.assertTrue(np.min(fc.get_weights()[0]) >= -bound)
        self.assertTrue(np.max(fc.get_weights()[0]) <= bound)

        self.assertTrue(np.min(fc.get_weights()[1]) >= -bound)
        self.assertTrue(np.max(fc.get_weights()[1]) <= bound)
