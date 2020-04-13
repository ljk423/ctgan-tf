from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


@tf.function
def d_loss_fn(fake_logit, real_logit):
    fake_loss = tf.math.reduce_mean(fake_logit)
    real_loss = tf.math.reduce_mean(real_logit)
    return fake_loss - real_loss

@tf.function
def g_loss_fn(fake_logit):
    f_loss = -tf.math.reduce_mean(fake_logit)
    return f_loss


@tf.function
def cond_loss(transformer_info, data, c, m):
    loss = []

    for item in transformer_info:
        st, ed, st_c, ed_c, is_continuous, is_softmax = item
        if is_continuous == 0 and is_softmax == 1:
            data_logsoftmax = data[:, st:ed]
            c_argmax = tf.math.argmax(c[:, st_c:ed_c], axis=1)
            loss += [tf.nn.sparse_softmax_cross_entropy_with_logits(c_argmax, data_logsoftmax)]

    loss = tf.stack(loss, axis=1)
    return tf.reduce_sum(loss * m) / data.shape[0]


def _cond_loss(transformer_info, data, c, m):
    loss = []
    st = 0
    st_c = 0
    skip = False
    for item in transformer_info:
        if item[1] == 'tanh':
            st += item[0]
            skip = True

        elif item[1] == 'softmax':
            if skip:
                skip = False
                st += item[0]
                continue

            ed = st + item[0]
            ed_c = st_c + item[0]

            tmp = tf.nn.sparse_softmax_cross_entropy_with_logits(
                tf.math.argmax(c[:, st_c:ed_c], axis=1),
                data[:, st:ed],
            )
            loss.append(tmp)
            st = ed
            st_c = ed_c

        else:
            assert 0

    loss = tf.stack(loss, axis=1)

    return tf.reduce_sum(loss * m) / data.shape[0]