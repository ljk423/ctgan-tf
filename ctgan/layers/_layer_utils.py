"""
Utilities for layer initialization.
"""
import math
import tensorflow as tf


def init_bounded(shape, **kwargs):
    """
    Initializes the weights or bias of a fully connected layer according to the
    ``dim`` passed in ``kwargs``. They are computed according to
    :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
    where :math:`k=\\frac{1}{\\text{dim}}`.

    Parameters
    ----------
    shape: tf.Shape
        Shape of the tensor to initialize.
    kwargs: dict
        Dictionary of parameters, containing ``dim`` and ``dtype``.

    Returns
    -------
    tf.Tensor
        Tensor with the initialiazed weights of shape ``shape``, sampled
        from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
        where :math:`k=\\frac{1}{\\text{dim}}`.
    """

    if 'dim' not in kwargs:
        raise AttributeError('dim not passed as input')
    if 'dtype' not in kwargs:
        raise AttributeError('dtype not passed as input')

    dim = kwargs['dim']
    d_type = kwargs['dtype']
    bound = 1 / math.sqrt(dim)
    return tf.random.uniform(shape=shape, minval=-bound, maxval=bound, dtype=d_type)
