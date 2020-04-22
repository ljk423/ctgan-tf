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

"""Implementation of WGANGP model.

Details available at https://arxiv.org/abs/1704.00028.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import datetime
from functools import partial
import numpy as np
import os
import copy
import joblib
import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2

from ctgan.transformer import DataTransformer
from ctgan.sampler import Sampler
from ctgan.conditional import ConditionalGenerator
from ctgan.models import Generator, Critic
from ctgan import losses
from ctgan.utils import pbar

AUTOTUNE = tf.data.experimental.AUTOTUNE


class CTGANSynthesizer:
    def __init__(self, file_path=None, z_dim=128, pac=10, gen_dim=(256, 256),
                 dis_dim=(256, 256), l2scale=1e-6, batch_size=500,
                 g_penalty_lambda=10.0, tau=0.2):
        if file_path is not None:
            self._load(file_path)
            return

        assert batch_size % 2 == 0
        assert batch_size % pac == 0

        self._z_dim = z_dim
        self._pac = pac
        self._pac_dim = None
        self._l2scale = l2scale
        self._batch_size = batch_size
        self._grad_penalty_lambda = g_penalty_lambda
        self._tau = tau
        self._gen_dim = gen_dim
        self._dis_dim = dis_dim

        self._g_opt = tf.keras.optimizers.Adam(
            learning_rate=2e-4, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        self._c_opt = tf.keras.optimizers.Adam(
            learning_rate=2e-4, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        self._transformer = DataTransformer()
        self._data_sampler = None
        self._cond_generator = None
        self._generator = None
        self._critic = None

    def train(self, train_data, discrete_columns=tuple(), epochs=300, log_frequency=True, verbose=True):
        # Initialize DataTransformer and ConditionalGenerator based on input data
        # and discrete columns info
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        self._transformer.generate_tensors()

        self._data_sampler = Sampler(train_data, self._transformer.output_info)
        data_dim = self._transformer.output_dimensions
        self._cond_generator = ConditionalGenerator(
            train_data, self._transformer.output_info, log_frequency)
        self._generator = Generator(
            self._z_dim + self._cond_generator.n_opt, self._gen_dim, data_dim,
            self._transformer.output_tensor, self._tau)
        self._critic = Critic(
            data_dim + self._cond_generator.n_opt, self._dis_dim, self._pac)
        g_train_loss = tf.metrics.Mean()
        d_train_loss = tf.metrics.Mean()
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        #with train_summary_writer.as_default():
            # Get concrete graph for some given inputs
            #func_graph = self.train_g.get_concrete_function().graph
            # Write the graph
            #summary_ops_v2.graph(func_graph.as_graph_def(), step=0)

        stats = list()
        d_grads = list()
        g_grads = list()
        d_weights = list()
        g_weights = list()

        self._generator.build((self._batch_size, self._generator._input_dim))
        self._critic.build((self._batch_size, self._critic._input_dim))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for epoch in range(epochs):
            bar = pbar(len(train_data), self._batch_size, epoch, epochs)
            for _ in range(steps_per_epoch):
                d_weights.append([np.mean(g.numpy()) for g in self._critic.trainable_weights])
                g_weights.append([np.mean(g.numpy()) for g in self._generator.trainable_weights])
                # Train discriminator with real and fake samples
                d_loss, gp, d_g = self._train_d()
                d_train_loss(d_loss)

                g_loss, cond_loss, g_g = self._train_g()
                g_train_loss(g_loss)

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.postfix['cond_loss'] = f'{cond_loss:6.3f}'
                bar.postfix['gp'] = f'{gp:6.3f}'
                bar.update(self._batch_size)

                stats.append([d_loss.numpy(), gp.numpy(), g_loss.numpy(), cond_loss])
                d_grads.append([np.mean(g.numpy()) for g in d_g])
                g_grads.append([np.mean(g.numpy()) for g in g_g])

            with train_summary_writer.as_default():
                tf.summary.scalar('g_loss', g_train_loss.result(), step=epoch)
                tf.summary.scalar('d_loss', d_train_loss.result(), step=epoch)
            g_train_loss.reset_states()
            d_train_loss.reset_states()

            bar.close()
            del bar

        stats = [stats, d_grads, g_grads, d_weights, g_weights]
        joblib.dump(stats, 'notebooks/tf.stats')

    @tf.function
    def train_d_step(self, fake_cat, real_cat):
        with tf.GradientTape() as t:
            y_fake = self._critic(fake_cat, training=True)
            y_real = self._critic(real_cat, training=True)

            gp = losses.gradient_penalty(
                partial(self._critic, training=True), real_cat, fake_cat,
                self._pac, self._grad_penalty_lambda)
            loss = -(tf.reduce_mean(y_real) - tf.reduce_mean(y_fake))
            d_loss = loss + gp

        grad = t.gradient(d_loss, self._critic.trainable_variables)
        self._c_opt.apply_gradients(zip(grad, self._critic.trainable_variables))
        return loss, gp, grad

    def _train_d(self):
        fake_z = tf.random.normal([self._batch_size, self._z_dim])

        # Generate conditional vector
        cond_vec = self._cond_generator.sample(self._batch_size)
        if cond_vec is None:
            c1, m1, col, opt = None, None, None, None
            c1 = tf.constant(-1)
            real = self._data_sampler.sample(self._batch_size, col, opt)
        else:
            c1, m1, col, opt = cond_vec
            c1 = tf.convert_to_tensor(c1)
            fake_z = tf.concat([fake_z, c1], axis=1)

            perm = np.arange(self._batch_size)
            np.random.shuffle(perm)
            real = self._data_sampler.sample(self._batch_size, col[perm], opt[perm])
            c2 = tf.gather(c1, perm)

        fake, fake_act = self._generator(fake_z, training=True)
        real = tf.convert_to_tensor(real.astype('float32'))

        if c1 is not None:
            fake_cat = tf.concat([fake_act, c1], axis=1)
            real_cat = tf.concat([real, c2], axis=1)
        else:
            fake_cat = fake
            real_cat = real

        return self.train_d_step(fake_cat, real_cat)

    @tf.function
    def train_g_step(self, fake_z):
        with tf.GradientTape() as t:
            fake, fake_act = self._generator(fake_z, training=True)
            y_fake = self._critic(fake_act, training=True)
            g_loss = -tf.reduce_mean(y_fake)

        weights = self._generator.trainable_variables
        grad = t.gradient(g_loss, weights)
        grad = [grad[i] + self._l2scale * weights[i] for i in range(len(grad))]
        self._g_opt.apply_gradients(zip(grad, self._generator.trainable_variables))
        return g_loss, 0, grad

    @tf.function
    def train_g_cond_step(self, fake_z, c1, m1, cond_info):
        with tf.GradientTape() as t:
            fake, fake_act = self._generator(fake_z, training=True)
            y_fake = self._critic(tf.concat([fake_act, c1], axis=1), training=True)
            cond_loss = losses.cond_loss(cond_info, fake, c1, m1)
            g_loss = -tf.reduce_mean(y_fake) + cond_loss

        weights = self._generator.trainable_variables
        grad = t.gradient(g_loss, weights)
        grad = [grad[i] + self._l2scale * weights[i] for i in range(len(grad))]
        self._g_opt.apply_gradients(zip(grad, self._generator.trainable_variables))
        return g_loss, cond_loss, grad

    def _train_g(self):
        fake_z = tf.random.normal([self._batch_size, self._z_dim])
        cond_vec = self._cond_generator.sample(self._batch_size)

        if cond_vec is None:
            return self.train_g_step(fake_z)

        c1, m1, col, opt = cond_vec
        c1 = tf.convert_to_tensor(c1, name="c1")
        m1 = tf.convert_to_tensor(m1, name="m1")
        fake_z = tf.concat([fake_z, c1], axis=1, name="fake_z")
        return self.train_g_cond_step(fake_z, c1, m1, self._transformer.cond_tensor)

    def sample(self, n):
        """Sample data similar to the training data.

        Args:
            n (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        steps = n // self._batch_size + 1
        data = []
        for _ in tf.range(steps):
            fake_z = tf.random.normal([self._batch_size, self._z_dim])
            cond_vec = self._cond_generator.sample_zero(self._batch_size)
            if cond_vec is not None:
                c1 = tf.constant(cond_vec)
                fake_z = tf.concat([fake_z, c1], axis=1)

            fake = self._generator(fake_z)[1]
            data.append(fake.numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]
        return self._transformer.inverse_transform(data, None)

    def dump(self, file_path, overwrite=False):
        """
        Export model to disk.
        It will create multiple files, including a folder with all BGM and OHE models.
        :param file_path: Path to main file of the class.
        :param overwrite: Overwrite existing files.
        """
        if file_path is None or len(file_path) == 0:
            raise NameError
        dir_name = os.path.dirname(file_path)
        if len(dir_name) and not os.path.exists(os.path.dirname(file_path)):
            raise NotADirectoryError
        if not overwrite and os.path.exists(file_path):
            raise FileExistsError

        # Create a copy of class dict as we are about to change the dictionary
        class_dict = {k: v for k, v in self.__dict__.items() if type(v) in [int, float, tuple]}
        class_dict['_cond_generator'] = self._cond_generator.__dict__
        class_dict['_transformer'] = self._transformer.__dict__
        class_dict['_gen_weights'] = self._generator.get_weights()

        # Dump dictionary to file
        joblib.dump(class_dict, file_path)
        del class_dict

    def _load(self, file_path):
        """
        Load existent dumped files to the current instance of the class.
        :param file_path: Path to the main file of the class.
        """
        if file_path is None or len(file_path) == 0:
            raise NameError
        if not os.path.exists(file_path):
            raise FileNotFoundError

        # Load class attributes
        class_dict = joblib.load(file_path)
        if class_dict is None:
            raise AttributeError

        # Load class attributes
        for key, value in class_dict.items():
            if type(value) in [int, float, tuple]:
                setattr(self, key, value)

        # Load binary models/encoders to class dict
        self._transformer = DataTransformer.from_dict(class_dict['_transformer'])
        self._cond_generator = ConditionalGenerator.from_dict(class_dict['_cond_generator'])

        # Load Generator instance
        self._generator = Generator(
            self._z_dim + self._cond_generator.n_opt,
            self._gen_dim,
            self._transformer.output_dimensions,
            self._transformer.output_tensor,
            self._tau)
        self._generator.build((self._batch_size, self._generator._input_dim))
        self._generator.set_weights(class_dict['_gen_weights'])

