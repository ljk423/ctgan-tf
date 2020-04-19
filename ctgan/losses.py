from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


@tf.function
def cond_loss(transformer_info, data, c, m):
    s = tf.shape(m)
    loss = tf.zeros(s)

    for item in transformer_info:
        data_logsoftmax = data[:, item[0]:item[1]]
        c_argmax = tf.math.argmax(c[:, item[2]:item[3]], axis=1)
        l = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(
            c_argmax, data_logsoftmax), [-1, 1])
        loss = tf.concat([loss[:, :item[-1]], l, loss[:, item[-1]+1:]], axis=1)

    return tf.reduce_sum(loss * m) / tf.cast(s[0], dtype=tf.float32)


#@tf.function(experimental_relax_shapes=True)
def gradient_penalty(f, real, fake, pac=10, gp_lambda=10.0):
    """Calculates the gradient penalty loss for a batch of "averaged" samples.
    In Improved WGANs, the 1-Lipschitz constraint is enforced by adding a term to the
    loss function that penalizes the network if the gradient norm moves away from 1.
    However, it is impossible to evaluate this function at all points in the input
    space. The compromise used in the paper is to choose random points on the lines
    between real and generated samples, and check the gradients at these points. Note
    that it is the gradient w.r.t. the input averaged samples, not the weights of the
    discriminator, that we're penalizing!
    In order to evaluate the gradients, we must first run samples through the generator
    and evaluate the loss. Then we get the gradients of the discriminator w.r.t. the
    input averaged samples. The l2 norm and penalty can then be calculated for this
    gradient.
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

