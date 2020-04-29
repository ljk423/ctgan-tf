import tensorflow as tf
from unittest import TestCase

from ctgan.losses import gradient_penalty


class TestGradientPenalty(TestCase):
    def setUp(self):
        self._input_dim = 10
        self._pac = 10
        self._batch_size = 10
        self._gp_lambda = 10.0

    def tearDown(self):
        del self._input_dim
        del self._pac
        del self._batch_size
        del self._gp_lambda

    def test_gradient_penalty(self):
        tf.random.set_seed(0)
        real = tf.random.uniform([self._batch_size, self._input_dim])
        fake = tf.random.uniform([self._batch_size, self._input_dim])

        gp = gradient_penalty(
            lambda x: x**2, real, fake, pac=self._pac, gp_lambda=self._gp_lambda)
        expected_output = tf.constant(1002.7697, dtype=tf.float32)
        tf.assert_equal(gp, expected_output)
