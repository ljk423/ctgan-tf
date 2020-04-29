"""
Methods to compute gradient penalty.
"""
import tensorflow as tf


def gradient_penalty(f, real, fake, pac=10, gp_lambda=10.0):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.

    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term
    to the loss function that penalizes the network if the gradient norm moves
    away from 1. However, it is impossible to evaluate this function at all
    points in the input space. The compromise used in the paper is to choose
    random points on the lines between real and generated samples, and check
    the gradients at these points. Note that it is the gradient w.r.t. the
    input averaged samples, not the weights of the discriminator,
    that we're penalizing!

    In order to evaluate the gradients, we must first run samples through the
    generator and evaluate the loss. Then we get the gradients of the
    discriminator w.r.t. the input averaged samples. The L2 norm and penalty
    can then be calculated for this gradient.

    For more information check the WGAN-GP paper :cite:`gulrajani2017improved`.

    Parameters
    ----------
    f: function
        The gradient penalty will be computed over the output of this
        method.

    real: tf.Tensor
        Real sample batch.

    fake: tf.Tensor
        Fake sample batch.

    pac: int, default=10
        Size of the Pac framework. For more details consult the original
        paper and the PacGAN framework paper :cite:`lin2018pacgan`.

    gp_lambda: float, default=10.0
        Gradient Penalty lambda.

    Returns
    -------
    tf.Tensor
        The gradient penalty of the interpolated sample batch.
    """
    alpha = tf.random.uniform([real.shape[0] // pac, 1, 1], 0., 1.)
    alpha = tf.tile(alpha, tf.constant([1, pac, real.shape[1]], tf.int32))
    alpha = tf.reshape(alpha, [-1, real.shape[1]])

    interpolates = alpha * real + ((1 - alpha) * fake)
    with tf.GradientTape() as t:
        t.watch(interpolates)
        pred = f(interpolates)
    grad = t.gradient(pred, [interpolates])[0]
    grad = tf.reshape(grad, tf.constant([-1, pac * real.shape[1]], tf.int32))

    slopes = tf.math.reduce_euclidean_norm(grad, axis=1)
    gp = tf.reduce_mean((slopes - 1.) ** 2) * gp_lambda
    return gp
