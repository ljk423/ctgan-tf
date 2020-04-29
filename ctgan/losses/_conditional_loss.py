"""
Method to compute data_modules losses.
"""
import tensorflow as tf


def conditional_loss(transformer_info, data, c, m):
    """Computes the loss for conditional vectors.

    The goal is to force the Generator to produce samples that match the
    original sampled conditional vector.

    For further details consult *Generator loss* under section 4.3 of
    :cite:`xu2019modeling`.

    Parameters
    ----------
    transformer_info: tf.Tensor
        Transformer output information tensor, indicating
        which indexes of the ``data`` tensor correspond to discrete variables,
        and the corresponding indexes of the ``c`` tensor.

    data: tf.Tensor
        Batch of data outputted by the Generator.

    c: tf.Tensor
        Original sampled data_modules vector.

    m: tf.Tensor
        Original sampled mask vector.

    Returns
    -------
    tf.Tensor
        The conditional loss.
    """
    shape = tf.shape(m)
    c_loss = tf.zeros(shape)

    for item in transformer_info:
        data_logsoftmax = data[:, item[0]:item[1]]
        cond_vec = tf.math.argmax(c[:, item[2]:item[3]], axis=1)
        loss = tf.reshape(tf.nn.sparse_softmax_cross_entropy_with_logits(
            cond_vec, data_logsoftmax), [-1, 1])
        c_loss = tf.concat(
            [c_loss[:, :item[-1]], loss, c_loss[:, item[-1]+1:]], axis=1)

    return tf.reduce_sum(c_loss * m) / tf.cast(shape[0], dtype=tf.float32)
