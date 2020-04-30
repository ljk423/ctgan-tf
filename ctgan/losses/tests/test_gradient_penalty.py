import tensorflow as tf
from unittest import TestCase

from ctgan.utils import get_test_variables
from ctgan.losses import gradient_penalty


class TestGradientPenalty(TestCase):
    def setUp(self):
        self._vars = get_test_variables()

    def tearDown(self):
        del self._vars

    def test_gradient_penalty(self):
        tf.random.set_seed(0)
        real = tf.random.uniform(
            [self._vars['batch_size'], self._vars['input_dim']])
        fake = tf.random.uniform([self._vars['batch_size'], self._vars['input_dim']])

        gp = gradient_penalty(
            lambda x: x**2, real, fake,
            pac=self._vars['pac'], gp_lambda=self._vars['gp_lambda'])
        expected_output = tf.constant(1002.7697, dtype=tf.float32)
        tf.assert_equal(gp, expected_output)
