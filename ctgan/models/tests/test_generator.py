import tensorflow as tf
import numpy as np
from unittest import TestCase

from ctgan.models import Generator


class TestGenerator(TestCase):
    def setUp(self):
        self._input_dim = 5
        self._layer_dims = [256, 256]
        self._output_dim = 5
        self._output_tensor = [
            tf.constant([0, 1, 0], dtype=tf.int32),
            tf.constant([1, 5, 0], dtype=tf.int32)
        ]
        self._tau = 0.2
        self._batch_size = 1

    def tearDown(self):
        del self._input_dim
        del self._layer_dims
        del self._output_dim
        del self._output_tensor
        del self._tau
        del self._batch_size

    def test_build_model(self):
        generator = Generator(
            self._input_dim,
            self._layer_dims,
            self._output_dim,
            self._output_tensor,
            self._tau)
        generator.build((self._batch_size, self._input_dim))
        self.assertIsNotNone(generator)
        self.assertEqual(len(generator.layers), len(self._layer_dims) + 1)

    def test_call_model(self):
        tf.random.set_seed(0)
        inputs = tf.random.uniform([self._batch_size, self._input_dim])

        generator = Generator(
            self._input_dim,
            self._layer_dims,
            self._output_dim,
            self._output_tensor,
            self._tau)
        generator.build((self._batch_size, self._input_dim))

        outputs, outputs_act = generator(inputs)
        print(outputs)
        expected_outputs = tf.constant(
            [[-0.07589048, -0.20733598, 0.01827493, -0.12580258, 0.09565035]],
            dtype=tf.float32)
        expected_outputs_act = tf.constant(
            [[-0.07574511, -0.20441517, 0.01827289, -0.12514308, 0.0953597]],
            dtype=tf.float32)

        np.testing.assert_almost_equal(
            outputs.numpy(), expected_outputs.numpy())
        np.testing.assert_almost_equal(
            outputs_act.numpy(), expected_outputs_act.numpy())
