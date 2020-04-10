# MIT License
#
# Copyright (c) 2019 Drew Szurko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
            #print("item:", item, "|st:", st, "|ed:", ed, "|st_c:", st_c, "|ed_c:", ed_c)
            #print("c:", c[:, st_c:ed_c].shape, c[:, st_c:ed_c])
            #print("argmax:", torch.argmax(c[:, st_c:ed_c], dim=1).shape, torch.argmax(c[:, st_c:ed_c], dim=1))
            #print("data:", data[:, st:ed].shape, data[:, st:ed])
            #print()

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