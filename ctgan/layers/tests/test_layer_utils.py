import tensorflow as tf
import numpy as np
from functools import partial
from unittest import TestCase

from ctgan.utils import get_test_variables
from ctgan.layers import init_bounded


class TestLayerUtils(TestCase):
    def setUp(self):
        self._vars = get_test_variables()

    def tearDown(self):
        del self._vars

    def test_no_args(self):
        self.assertRaises(
            AttributeError,
            init_bounded,
            (self._vars['input_dim'], self._vars['output_dim']))
        self.assertRaises(
            AttributeError,
            init_bounded,
            (self._vars['input_dim'], self._vars['output_dim']),
            dim=self._vars['input_dim'])

    def test_init_layer(self):
        fc = tf.keras.layers.Dense(
            self._vars['output_dim'], input_dim=(self._vars['input_dim'],),
            kernel_initializer=partial(
                init_bounded, dim=self._vars['input_dim']),
            bias_initializer=partial(
                init_bounded, dim=self._vars['input_dim']))
        fc.build((self._vars['input_dim'],))
        bound = 1 / np.sqrt(self._vars['input_dim'])
        self.assertTrue(np.min(fc.get_weights()[0]) >= -bound)
        self.assertTrue(np.max(fc.get_weights()[0]) <= bound)

        self.assertTrue(np.min(fc.get_weights()[1]) >= -bound)
        self.assertTrue(np.max(fc.get_weights()[1]) <= bound)
