import tensorflow as tf
import tensorflow_probability as tfp


class GenActLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, transformer_info, tau):
        super(GenActLayer, self).__init__()
        self.num_outputs = num_outputs
        self.transformer_info = transformer_info
        self.tau = tau
        self.kernel = None

    def build(self, input_shape):
        self.kernel = self.add_weight(
            'kernel', shape=[input_shape[1], self.num_outputs])

    def call(self, inputs, **kwargs):
        data = tf.matmul(inputs, self.kernel)
        data_t = []
        for idx in self.transformer_info:
            data_t += [tf.where(idx[5] == 0,
                                tf.math.tanh(data[:, idx[0]:idx[1]]),
                                self._gumbel_softmax(data[:, idx[0]:idx[1]], tau=self.tau))]
        return tf.concat(data_t, axis=1)

    @tf.function
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

        #exp_dist = tfp.distributions.Exponential(tf.constant([1], dtype=tf.float32))
        #gumbels = -tf.math.log(exp_dist.sample(tf.shape(logits)))
        #gumbels = tf.reshape(gumbels, tf.shape(logits))
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

@tf.function
def _apply_activate(data, transformer_output):
    data_t = []
    st = 0
    for item in transformer_output:
        if item[1] == 'tanh':
            ed = st + item[0]
            data_t.append(tf.math.tanh(data[:, st:ed]))
            st = ed
        elif item[1] == 'softmax':
            ed = st + item[0]
            data_t.append(gumbel_softmax(data[:, st:ed], tau=0.2))
            st = ed
        else:
            assert 0

    return tf.concat(data_t, axis=1)


def gumbel_softmax(logits, tau=1.0, hard=False, dim=-1):
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
