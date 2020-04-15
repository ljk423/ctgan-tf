import tensorflow as tf
from functools import partial
from ctgan.layers import *


class Critic(tf.keras.Model):
    def __init__(self, input_dim, dis_dims, pac):
        super(Critic, self).__init__()
        self.pac = pac
        self.input_dim = input_dim

        self.model = [self._reshape_func]
        dim = input_dim * self.pac
        for layer_dim in list(dis_dims):
            self.model += [
                tf.keras.layers.Dense(
                    layer_dim, input_dim=(dim,),
                    kernel_initializer=partial(init_bounded, dim=dim),
                    bias_initializer=partial(init_bounded, dim=dim)),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(0.5)]
            dim = layer_dim

        layer_dim = 1
        self.model += [tf.keras.layers.Dense(
            layer_dim, input_dim=(dim,),
            kernel_initializer=partial(init_bounded, dim=dim),
            bias_initializer=partial(init_bounded, dim=dim))]

    def _reshape_func(self, x, **kwargs):
        dims = x.get_shape().as_list()
        return tf.reshape(x, [-1, dims[1] * self.pac])

    def call(self, x, **kwargs):
        out = x
        for layer in self.model:
            out = layer(out, **kwargs)
        return out


class Generator(tf.keras.Model):
    def __init__(self, input_dim, gen_dims, data_dim, transformer_info, tau):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.model = list()
        dim = input_dim
        for layer_dim in list(gen_dims):
            self.model += [ResidualLayer(dim, layer_dim)]
            dim += layer_dim

        #self.model += [tf.keras.layers.Dense(
        #    data_dim, input_dim=(dim,),
        #    kernel_initializer=partial(init_bounded, dim=dim),
        #    bias_initializer=partial(init_bounded, dim=dim))]
        self.model += [GenActLayer(dim, data_dim, transformer_info, tau)]

    def call(self, x, **kwargs):
        out = x
        for layer in self.model:
            out = layer(out, **kwargs)
        return out
