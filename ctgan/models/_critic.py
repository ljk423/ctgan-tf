"""
Critic model definition.
"""

import tensorflow as tf
from functools import partial

from ..layers import init_bounded


class Critic(tf.keras.Model):
    """
    Critic model.

    It uses a PacGAN framework :cite:`lin2018pacgan` with ``pac`` pacs, thus
    it applies a reshape function before passing the data through the
    remaining layers.
    It applies the following operations for every layer declared in
    ``dis_dims``:

    .. math:: \\text{output} = Dropout_{0.5}(LeakyReLU_{0.2}(
            FC_{\\text{prev_layer_dim} \\to \\text{layer_dim}}(\\text{input})))

    The last layer of the network is a FC layer with a single output.

    For further details of the network architecture, please check section 4.4
    of :cite:`xu2019modeling`.

    Parameters
    ----------
    input_dim: int
        Dimension of the input data.

    dis_dims: list[int]
        List of the number of neurons per layer.

    pac: int
        Size of the Pac framework. For more details consult the original
        paper :cite:`xu2019modeling` and the PacGAN framework paper
        :cite:`lin2018pacgan`.

    References
    ----------
    .. bibliography:: ../bibtex/refs.bib
    """
    def __init__(self, input_dim, dis_dims, pac):
        super(Critic, self).__init__()
        self._pac = pac
        self._input_dim = input_dim

        self._model = [self._reshape_func]
        dim = input_dim * self._pac
        for layer_dim in list(dis_dims):
            self._model += [
                tf.keras.layers.Dense(
                    layer_dim, input_dim=(dim,),
                    kernel_initializer=partial(init_bounded, dim=dim),
                    bias_initializer=partial(init_bounded, dim=dim)),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(0.5)]
            dim = layer_dim

        layer_dim = 1
        self._model += [tf.keras.layers.Dense(
            layer_dim, input_dim=(dim,),
            kernel_initializer=partial(init_bounded, dim=dim),
            bias_initializer=partial(init_bounded, dim=dim))]

    def _reshape_func(self, inputs, **kwargs):
        """
        Method that reshapes the input tensor according the Pac size.
        For more details consult the original CTGAN paper
        :cite:`xu2019modeling` (section 4.4) and the PacGAN framework paper
        :cite:`lin2018pacgan`.

        Parameters
        ----------
        inputs: tf.Tensor
            Input tensor.

        Returns
        -------
        tf.Tensor
            Reshaped tensor in the form of
            ``[-1, inputs.shape[1] * self._pac]``.
        """
        dims = inputs.get_shape().as_list()
        return tf.reshape(inputs, [-1, dims[1] * self._pac])

    def call(self, inputs, **kwargs):
        """Feedforward pass on the network layers.

        Parameters
        ----------
        inputs: tf.Tensor
            Input data tensor.

        kwargs: dict
            Dictionary with functions options. Mainly used to set
            ``training=True`` when training the network layers.

        Returns
        -------
        tf.Tensor
            The output tensor resultant from the feedforward pass.
        """
        outputs = inputs
        for layer in self._model:
            outputs = layer(outputs, **kwargs)
        return outputs
