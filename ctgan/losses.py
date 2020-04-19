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

