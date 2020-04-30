import tensorflow as tf
from unittest import TestCase

from ctgan.utils import get_test_variables
from ctgan.models import Critic


class TestCritic(TestCase):
    def setUp(self):
        self._vars = get_test_variables()

    def tearDown(self):
        del self._vars

    def test_build_model(self):
        critic = Critic(
            self._vars['input_dim'],
            self._vars['layer_dims'],
            self._vars['pac'])
        critic.build((self._vars['batch_size'], self._vars['input_dim']))
        self.assertIsNotNone(critic)
        self.assertEqual(
            len(critic.layers), len(self._vars['layer_dims'])*3 + 1)

    def test_call_model(self):
        tf.random.set_seed(0)
        inputs = tf.random.uniform(
            [self._vars['batch_size'], self._vars['input_dim']])

        critic = Critic(
            self._vars['input_dim'],
            self._vars['layer_dims'],
            self._vars['pac'])
        critic.build((self._vars['batch_size'], self._vars['input_dim']))

        outputs = critic(inputs)
        expected_output = tf.constant([[-0.08817893]], dtype=tf.float32)
        tf.assert_equal(outputs, expected_output)
