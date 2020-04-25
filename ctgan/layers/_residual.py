"""
Residual Layer definition.
"""

from ._layer_utils import init_bounded

import tensorflow as tf
from functools import partial


class ResidualLayer(tf.keras.layers.Layer):
    """
    Residual Layer used on the Generator.
    This layer applies the following operations to the input:

    .. math:: \\text{output} = ReLU(BN_{\\epsilon=10^{-5}, \\text{momentum}=0.9}(FC_{\\text{input_dim} \\to \\text{output_dim}}(\\text{input}))) \oplus \\text{input}

    Parameters
    ----------
    input_dim: int
        Fully Connected layer input dimension.
    output_dim: int
        Fully Connected layer output dimension.
    """
    def __init__(self, input_dim, output_dim):
        super(ResidualLayer, self).__init__()
        self._output_dim = output_dim
        self._fc = tf.keras.layers.Dense(
            self._output_dim, input_dim=(input_dim,),
            kernel_initializer=partial(init_bounded, dim=input_dim),
            bias_initializer=partial(init_bounded, dim=input_dim))
        self._bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self._relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        """
        Performs a feedforward pass of the Residual Layer on the provided
        input batch.

        Parameters
        ----------
        inputs: tf.Tensor
            Batch of data.
        kwargs: dict
            (training=bool).

        Returns
        -------
        tuple(tf.Tensor, tf.Tensor)
            A tensor resultant from concatenating the outputs and inputs of
            the layer.
        """
        outputs = self._fc(inputs, **kwargs)
        outputs = self._bn(outputs, **kwargs)
        outputs = self._relu(outputs, **kwargs)
        return tf.concat([outputs, inputs], axis=1)
