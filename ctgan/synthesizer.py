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

import joblib
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.ops import summary_ops_v2
from absl import flags

from ctgan.transformer import DataTransformer
from ctgan.sampler import Sampler
from ctgan.conditional import ConditionalGenerator
from ctgan.models import Generator, Critic
from ctgan.layers import _apply_activate
from ctgan import losses
from ctgan.utils import pbar

AUTOTUNE = tf.data.experimental.AUTOTUNE
FLAGS = flags.FLAGS


class CTGANSynthesizer:
    def __init__(self, z_dim=128, pac=10, gen_dim=(256, 256),
                 dis_dim=(256, 256), l2scale=1e-6, batch_size=500, n_critic=1,
                 g_penalty_lambda=10.0, tau=0.2):
        assert batch_size % 2 == 0
        assert batch_size % pac == 0

        self.z_dim = z_dim
        self.pac = pac
        self.pac_dim = None
        self.l2scale = l2scale
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.grad_penalty_lambda = g_penalty_lambda
        self.tau = tau
        self.gen_dim = gen_dim
        self.dis_dim = dis_dim

        self.g_opt = tf.keras.optimizers.Adam(
            learning_rate=2e-4, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        self.c_opt = tf.keras.optimizers.Adam(
            learning_rate=2e-4, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        self.transformer = DataTransformer()
        self.data_sampler = None
        self.cond_generator = None
        self.generator = None
        self.critic = None

    def train(self, train_data, discrete_columns=tuple(), epochs=300, log_frequency=True, verbose=True):
        # Initialize DataTransformer and ConditionalGenerator based on input data
        # and discrete columns info
        self.transformer.fit(train_data, discrete_columns)
        train_data = self.transformer.transform(train_data)

        self.data_sampler = Sampler(train_data, self.transformer.output_info)
        data_dim = self.transformer.output_dimensions
        self.cond_generator = ConditionalGenerator(
            train_data, self.transformer.output_info, log_frequency)
        self.generator = Generator(
            self.z_dim + self.cond_generator.n_opt, self.gen_dim, data_dim)
        self.critic = Critic(
            data_dim + self.cond_generator.n_opt, self.dis_dim, self.pac)

        #if verbose:
        #    self.generator.summary()
        #    self.critic.summary()

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

        self.generator.build((self.batch_size, self.generator.input_dim))
        self.critic.build((self.batch_size, self.critic.input_dim))

        print([np.mean(g) for g in self.critic.get_weights()])
        print([np.mean(g) for g in self.generator.get_weights()])
        print([g.shape for g in self.generator.get_weights()])
        print("gen fc 0:", self.generator.get_weights()[0])

        print(np.random.rand())
        steps_per_epoch = max(len(train_data) // self.batch_size, 1)
        for epoch in range(epochs):
            bar = pbar(len(train_data), self.batch_size, epoch, epochs)
            for _ in range(steps_per_epoch):
                d_weights.append([np.mean(g.numpy()) for g in self.critic.trainable_weights])
                g_weights.append([np.mean(g.numpy()) for g in self.generator.trainable_weights])
                # Train discriminator with real and fake samples
                d_loss, gp, d_g = self.train_d()
                d_train_loss(d_loss)

                g_loss, cond_loss, g_g = self.train_g()
                g_train_loss(g_loss)

                bar.postfix['g_loss'] = f'{g_train_loss.result():6.3f}'
                bar.postfix['d_loss'] = f'{d_train_loss.result():6.3f}'
                bar.postfix['cond_loss'] = f'{cond_loss:6.3f}'
                bar.postfix['gp'] = f'{gp:6.3f}'
                bar.update(self.batch_size)

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
        joblib.dump(stats, 'tf.stats')

    #@tf.function
    def train_d(self):
        with tf.GradientTape() as t:
            fake_z = tf.random.normal([self.batch_size, self.z_dim])
            #fake_z = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.z_dim)).astype(np.float32))

            #tf.print("fake_z_orig:", tf.reduce_mean(fake_z))
            # Generate conditional vector
            cond_vec = self.cond_generator.sample(self.batch_size)
            #tf.print("Cond vec:", cond_vec)
            if cond_vec is None:
                c1, m1, col, opt = None, None, None, None
                real = self.data_sampler.sample(self.batch_size, col, opt)
            else:
                c1, m1, col, opt = cond_vec
                c1 = tf.convert_to_tensor(c1)
                m1 = tf.convert_to_tensor(m1)
                fake_z = tf.concat([fake_z, c1], axis=1)

                perm = np.arange(self.batch_size)
                np.random.shuffle(perm)
                real = self.data_sampler.sample(self.batch_size, col[perm], opt[perm])
                c2 = tf.gather(c1, perm)

            fake = self.generator(fake_z, training=True)
            #fake_act = fake
            fake_act = _apply_activate(fake, self.transformer.output_info)
            real = tf.convert_to_tensor(real.astype('float32'))
            #tf.print("fake_z:", tf.reduce_mean(fake_z))
            #tf.print("fake_z:", fake_z)
            #tf.print("fake:", tf.reduce_mean(fake))
            #tf.print("real:", tf.reduce_mean(real))

            if c1 is not None:
                fake_cat = tf.concat([fake_act, c1], axis=1)
                real_cat = tf.concat([real, c2], axis=1)
            else:
                fake_cat = fake
                real_cat = real

            y_fake = self.critic(fake_cat, training=True)
            y_real = self.critic(real_cat, training=True)
            #tf.print("fake:", tf.reduce_mean(y_fake))
            #tf.print("real:", tf.reduce_mean(y_real))
            #tf.print("loss:", tf.reduce_mean(y_fake) - tf.reduce_mean(y_real))
            #tf.print()

            gp = self.gradient_penalty(
                partial(self.critic, training=True), real_cat, fake_cat)
            #gp = 0
            loss = -(tf.reduce_mean(y_real) - tf.reduce_mean(y_fake))
            #losses.d_loss_fn(fake_logits, real_logits)

            #tf.print("d_loss:", loss)
            #tf.print("gp:", gp)

            d_loss = loss + gp
        grad = t.gradient(d_loss, self.critic.trainable_variables)
        self.c_opt.apply_gradients(zip(grad, self.critic.trainable_variables))
        return loss, gp, grad

    #@tf.function
    def train_g(self):
        with tf.GradientTape() as t:
            fake_z = tf.random.normal([self.batch_size, self.z_dim])
            #fake_z = tf.convert_to_tensor(np.random.normal(size=(self.batch_size, self.z_dim)).astype(np.float32))
            cond_vec = self.cond_generator.sample(self.batch_size)

            if cond_vec is None:
                c1, m1, col, opt = None, None, None, None
            else:
                c1, m1, col, opt = cond_vec
                c1 = tf.convert_to_tensor(c1)
                m1 = tf.convert_to_tensor(m1)
                fake_z = tf.concat([fake_z, c1], axis=1)

            fake = self.generator(fake_z, training=True)
            fake_act = _apply_activate(fake, self.transformer.output_info)
            #fake_act = fake

            if c1 is not None:
                y_fake = self.critic(tf.concat([fake_act, c1], axis=1), training=True)
            else:
                y_fake = self.critic(fake_act, training=True)

            if cond_vec is None:
                cond_loss = 0
            else:
                cond_loss = losses._cond_loss(
                    self.transformer.output_info, fake, c1, m1)
                #cond_loss = losses.cond_loss(
                #    self.transformer.output_info_tensor(), fake, c1, m1)
            #loss = losses.g_loss_fn(y_fake)
            g_loss = -tf.reduce_mean(y_fake) + cond_loss
            #tf.print("g_loss:", g_loss)
            #tf.print("cond_loss:", cond_loss)

        weights = self.generator.trainable_variables
        grad = t.gradient(g_loss, weights)
        grad = [grad[i] + self.l2scale * weights[i] for i in range(len(grad))]
        self.g_opt.apply_gradients(zip(grad, self.generator.trainable_variables))
        return g_loss, cond_loss, grad

    #@tf.function
    def gradient_penalty(self, f, real, fake):
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
        alpha = tf.random.uniform([real.shape[0] // self.pac, 1, 1], 0., 1.)
        #alpha = tf.convert_to_tensor(np.random.rand(real.shape[0] // self.pac, 1, 1).astype(np.float32))
        alpha = tf.tile(alpha, tf.constant([1, self.pac, real.shape[1]], tf.int32))
        alpha = tf.reshape(alpha, [-1, real.shape[1]])

        interpolates = alpha * real + ((1 - alpha) * fake)
        with tf.GradientTape() as t:
            t.watch(interpolates)
            pred = f(interpolates)
        grad = t.gradient(pred, [interpolates])[0]
        grad = tf.reshape(grad, tf.constant([-1, self.pac * real.shape[1]], tf.int32))

        slopes = tf.math.reduce_euclidean_norm(grad, axis=1)
        gp = tf.reduce_mean((slopes - 1.) ** 2) * self.grad_penalty_lambda

        return gp

    #@tf.function
    def sample(self, n):
        """Sample data similar to the training data.

        Args:
            n (int):
                Number of rows to sample.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """

        steps = n // self.batch_size + 1
        data = []
        for _ in tf.range(steps):
            fake_z = tf.random.normal([self.batch_size, self.z_dim])
            cond_vec = self.cond_generator.sample_zero(self.batch_size)
            if cond_vec is not None:
                c1 = tf.constant(cond_vec)
                fake_z = tf.concat([fake_z, c1], axis=1)

            fake = self.generator(fake_z)
            fake = _apply_activate(fake, self.transformer.output_info)
            data.append(fake.numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self.transformer.inverse_transform(data, None)





