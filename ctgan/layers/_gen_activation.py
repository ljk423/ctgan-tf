"""
Conditional Generator activation layer definition.
"""
from functools import partial
import tensorflow as tf
import tensorflow_probability as tfp

from ._layer_utils import init_bounded


class GenActivation(tf.keras.layers.Layer):
    """Conditional Generator activation layer.

    Besides computing the usual Fully Connected operation :math:`y = xA^T + b`,
    this layer also applies a mix of activation functions. The scalar values
    :math:`{\\alpha}_i` are generated by tanh, while the mode indicator
    :math:`{\\beta}_i` and the discrete values :math:`d_i` are generated by
    Gumbel-Softmax.

    For more detailed information, please refer to sections 4.2-4.4 of the
    authors paper :cite:`xu2019modeling`.

    Parameters
    ----------
    input_dim: int
        Fully Connected layer input dimension.

    output_dim: int
        Fully Connected layer output dimension.

    transformer_info: list[tf.Tensor]
        List of tensors containing information regarding the
        activation functions of each data columns.

    tau: float
        Gumbel-Softmax non-negative scalar temperature.
    """
    # pylint: disable=too-few-public-methods

    def __init__(self, input_dim, output_dim, transformer_info, tau):
        super(GenActivation, self).__init__()
        self._output_dim = output_dim
        self._transformer_info = transformer_info
        self._tau = tau
        self._fc = tf.keras.layers.Dense(
            output_dim, input_dim=(input_dim,),
            kernel_initializer=partial(init_bounded, dim=input_dim),
            bias_initializer=partial(init_bounded, dim=input_dim))

    def call(self, inputs, **kwargs):
        """Performs a feedforward pass of the activation functions on the
        provided input batch.

        Parameters
        ----------
        inputs: tf.Tensor
            Batch of data.

        kwargs: dict
            Dictionary with functions options. Mainly used to set
            ``training=True`` when training the layer weights.

        Returns
        -------
        tuple(tf.Tensor, tf.Tensor)
            A tuple containing the output of the FC layer, and the output
            of the FC layer and the activation layers.
        """
        outputs = self._fc(inputs, **kwargs)
        data_t = tf.zeros(tf.shape(outputs))
        for idx in self._transformer_info:
            act = tf.where(
                idx[2] == 0,
                tf.math.tanh(outputs[:, idx[0]:idx[1]]),
                self._gumbel_softmax(outputs[:, idx[0]:idx[1]], tau=self._tau))
            data_t = tf.concat(
                [data_t[:, :idx[0]], act, data_t[:, idx[1]:]], axis=1)
        return outputs, data_t

    @tf.function(experimental_relax_shapes=True)
    def _gumbel_softmax(self, logits, tau=1.0, hard=False, dim=-1):
        """Samples from the Gumbel-Softmax distribution
        :cite:`maddison2016concrete`, :cite:`jang2016categorical` and
        optionally discretizes.

        Parameters
        ----------
        logits: tf.Tensor
            Un-normalized log probabilities.

        tau: float, default=1.0
            Non-negative scalar temperature.

        hard: bool, default=False
            If ``True``, the returned samples will be discretized as
            one-hot vectors, but will be differentiated as soft samples.

        dim: int, default=1
            The dimension along which softmax will be computed.

        Returns
        -------
        tf.Tensor
            Sampled tensor of same shape as ``logits`` from the
            Gumbel-Softmax distribution. If ``hard=True``, the returned samples
            will be one-hot, otherwise they will be probability distributions
            that sum to 1 across ``dim``.
        """

        gumbel_dist = tfp.distributions.Gumbel(loc=0, scale=1)
        gumbels = gumbel_dist.sample(tf.shape(logits))
        gumbels = (logits + gumbels) / tau
        output = tf.nn.softmax(gumbels, dim)

        if hard:
            index = tf.math.reduce_max(output, 1, keepdims=True)
            output_hard = tf.cast(tf.equal(output, index), output.dtype)
            output = tf.stop_gradient(output_hard - output) + output
        return output
