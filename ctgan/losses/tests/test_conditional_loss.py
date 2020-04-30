import tensorflow as tf
from unittest import TestCase

from ctgan.utils import get_test_variables
from ctgan.losses import conditional_loss


class TestConditionalLoss(TestCase):
    def setUp(self):
        self._vars = get_test_variables()
        self._cond_tensor = [
            tf.constant([0, 2, 0, 2], dtype=tf.int32),
            tf.constant([2, 4, 2, 4], dtype=tf.int32)
        ]

    def tearDown(self):
        del self._vars
        del self._cond_tensor

    def test_gradient_penalty(self):
        tf.random.set_seed(0)
        fake = tf.random.uniform(
            [self._vars['batch_size'], self._vars['n_opt']])
        cond = tf.random.uniform(
            [self._vars['batch_size'], self._vars['n_opt']])
        mask = tf.random.uniform(
            [self._vars['batch_size'], self._vars['n_col']])
        cond_loss = conditional_loss(self._cond_tensor, fake, cond, mask)
        expected_output = tf.constant(0.7187628, dtype=tf.float32)
        tf.assert_equal(cond_loss, expected_output)
