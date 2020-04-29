import tensorflow as tf
from unittest import TestCase

from ctgan.models import Critic


class TestCritic(TestCase):
    def setUp(self):
        self._input_dim = 10
        self._layer_dims = [256, 256]
        self._pac = 10
        self._batch_size = 10

    def tearDown(self):
        del self._input_dim
        del self._layer_dims
        del self._pac

    def test_build_model(self):
        critic = Critic(
            self._input_dim,
            self._layer_dims,
            self._pac)
        critic.build((self._batch_size, self._input_dim))
        self.assertIsNotNone(critic)
        self.assertEqual(len(critic.layers), len(self._layer_dims)*3 + 1)

    def test_call_model(self):
        tf.random.set_seed(0)
        inputs = tf.random.uniform([self._batch_size, self._input_dim])

        critic = Critic(
            self._input_dim,
            self._layer_dims,
            self._pac)
        critic.build((self._batch_size, self._input_dim))

        outputs = critic(inputs)
        expected_output = tf.constant([[-0.08817893]], dtype=tf.float32)
        tf.assert_equal(outputs, expected_output)
