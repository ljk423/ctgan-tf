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
def g_loss_fn(fake_logit, x_fake, transformer_info, c, m):
    f_loss = -tf.math.reduce_mean(fake_logit)
    if c is not None:
        f_loss += _cond_loss(transformer_info, x_fake, c, m)
    return f_loss


@tf.function
def _cond_loss(transformer_info, data, c, m):
    loss = []

    for item in transformer_info:
        st, ed, st_c, ed_c, is_continuous, is_softmax = item
        if is_continuous == 0 and is_softmax == 1:
            index = tf.reduce_max(c[:, st_c:ed_c], axis=1, keepdims=True)
            cond = tf.cast(tf.equal(c[:, st_c:ed_c], index), tf.float32)
            loss += [tf.losses.categorical_crossentropy(data[:, st:ed], cond)]

    loss = tf.stack(loss, axis=1)
    return tf.reduce_sum(loss * m) / data.shape[0]

