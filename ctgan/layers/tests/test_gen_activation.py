import tensorflow as tf
import numpy as np
from unittest import TestCase

from ctgan.layers import GenActivation


class TestGenActLayer(TestCase):
    def setUp(self):
        self._input_dim = 5
        self._output_dim = 5
        self._output_tensor = [
            tf.constant([0, 1, 0], dtype=tf.int32),
            tf.constant([1, 5, 1], dtype=tf.int32)
        ]
        self._tau = 0.2
        self._batch_size = 1

    def tearDown(self):
        del self._input_dim
        del self._output_dim
        del self._output_tensor
        del self._tau
        del self._batch_size

    def test_gumbel_softmax(self):
        tf.random.set_seed(0)
        inputs = tf.random.uniform([self._batch_size, self._input_dim])
        gen_act_layer = GenActivation(
            self._input_dim,
            self._output_dim,
            self._output_tensor,
            self._tau)
        outputs = gen_act_layer._gumbel_softmax(inputs, tau=self._tau)
        expected_outputs = tf.constant(
            [[2.1834146e-02, 8.1468701e-01, 1.4715219e-01,
              1.6876719e-07, 1.6326491e-02]],
            dtype=tf.float32)
        np.testing.assert_almost_equal(
            outputs.numpy(), expected_outputs.numpy())

        outputs = gen_act_layer._gumbel_softmax(
            inputs, tau=self._tau, hard=True)
        expected_outputs = tf.constant(
            [[0., 1., 0., 0., 0.]], dtype=tf.float32)
        np.testing.assert_almost_equal(
            outputs.numpy(), expected_outputs.numpy())

    def test_gen_act_layer(self):
        tf.random.set_seed(0)
        inputs = tf.random.uniform([self._batch_size, self._input_dim])
        gen_act_layer = GenActivation(
            self._input_dim,
            self._output_dim,
            self._output_tensor,
            self._tau)

        outputs, outputs_act = gen_act_layer(inputs)
        expected_outputs = tf.constant(
            [[0.3448848, 0.10137627, -0.16654477, 0.4421128, -0.44433516]],
            dtype=tf.float32)
        expected_outputs_act = tf.constant(
            [[3.3183137e-01, 3.7113253e-02, 5.5598521e-01,
              4.0690151e-01, 4.8746678e-09]],
            dtype=tf.float32)

        np.testing.assert_almost_equal(
            outputs.numpy(), expected_outputs.numpy())
        np.testing.assert_almost_equal(
            outputs_act.numpy(), expected_outputs_act.numpy())
