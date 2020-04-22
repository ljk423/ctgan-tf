import tensorflow as tf
import tensorflow_probability as tfp
import math
from functools import partial


def init_bounded(shape, **kwargs):
    dim = kwargs['dim']
    dtype = kwargs['dtype']
    bound = 1 / math.sqrt(dim)
    return tf.random.uniform(shape=shape, minval=-bound, maxval=bound, dtype=dtype)


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_outputs):
        super(ResidualLayer, self).__init__()
        self._num_outputs = num_outputs
        self._fc = tf.keras.layers.Dense(
            self._num_outputs, input_dim=(input_dim,),
            kernel_initializer=partial(init_bounded, dim=input_dim),
            bias_initializer=partial(init_bounded, dim=input_dim))
        self._bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self._relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        outputs = self._fc(inputs, **kwargs)
        outputs = self._bn(outputs, **kwargs)
        outputs = self._relu(outputs, **kwargs)
        return tf.concat([outputs, inputs], axis=1)


class GenActivation(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_outputs, transformer_info, tau):
        super(GenActivation, self).__init__()
        self._num_outputs = num_outputs
        self._transformer_info = transformer_info
        self._tau = tau
        self._fc = tf.keras.layers.Dense(
            num_outputs, input_dim=(input_dim,),
            kernel_initializer=partial(init_bounded, dim=input_dim),
            bias_initializer=partial(init_bounded, dim=input_dim))

    def call(self, inputs, **kwargs):
        outputs = self._fc(inputs, **kwargs)
        data_t = tf.zeros(tf.shape(outputs))
        for idx in self._transformer_info:
            act = tf.where(idx[2] == 0,
                           tf.math.tanh(outputs[:, idx[0]:idx[1]]),
                           self._gumbel_softmax(outputs[:, idx[0]:idx[1]], tau=self._tau))
            data_t = tf.concat([data_t[:, :idx[0]], act, data_t[:, idx[1]:]], axis=1)
        return outputs, data_t

    @tf.function(experimental_relax_shapes=True)
    def _gumbel_softmax(self, logits, tau=1.0, hard=False, dim=-1):
        r"""
        Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.

        Args:
          logits: `[..., num_features]` unnormalized log probabilities
          tau: non-negative scalar temperature
          hard: if ``True``, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
          dim (int): A dimension along which softmax will be computed. Default: -1.

        Returns:
          Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
          If ``hard=True``, the returned samples will be one-hot, otherwise they will
          be probability distributions that sum to 1 across `dim`.

        .. note::
          The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`

          It achieves two things:
          - makes the output value exactly one-hot
          (since we add then subtract y_soft value)
          - makes the gradient equal to y_soft gradient
          (since we strip all other gradients)

        .. _Link 1:
            https://arxiv.org/abs/1611.00712
        .. _Link 2:
            https://arxiv.org/abs/1611.01144
        """

        gumbel_dist = tfp.distributions.Gumbel(loc=0, scale=1)
        gumbels = gumbel_dist.sample(tf.shape(logits))
        gumbels = (logits + gumbels) / tau
        y = tf.nn.softmax(gumbels, dim)

        if hard:
            # Straight through.
            index = tf.math.reduce_max(y, 1, keep_dims=True)
            y_hard = tf.cast(tf.equal(y, index), y.dtype)
            y = tf.stop_gradient(y_hard - y) + y
        return y

