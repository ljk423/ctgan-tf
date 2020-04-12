import tensorflow as tf
import math
import numpy as np
from functools import partial


def init_bounded(shape, **kwargs):
    dim = kwargs['dim']
    dtype = kwargs['dtype']
    bound = 1 / math.sqrt(dim)
    #return tf.cast(tf.convert_to_tensor(np.random.uniform(-bound, bound, shape)), dtype)
    return tf.random.uniform(shape=shape, minval=-bound, maxval=bound, dtype=dtype)


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


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_outputs):
        super(ResidualLayer, self).__init__()
        self.num_outputs = num_outputs
        self.fc = tf.keras.layers.Dense(
            self.num_outputs, input_dim=(input_dim,),
            kernel_initializer=partial(init_bounded, dim=input_dim),
            bias_initializer=partial(init_bounded, dim=input_dim))
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        outputs = self.fc(inputs, **kwargs)
        outputs = self.bn(outputs, **kwargs)
        outputs = self.relu(outputs, **kwargs)
        return tf.concat([outputs, inputs], axis=1)


class Generator(tf.keras.Model):
    def __init__(self, input_dim, gen_dims, data_dim):
        super(Generator, self).__init__()

        self.input_dim = input_dim
        self.model = list()
        dim = input_dim
        for layer_dim in list(gen_dims):
            self.model += [ResidualLayer(dim, layer_dim)]
            dim += layer_dim

        self.model += [tf.keras.layers.Dense(
            data_dim, input_dim=(dim,),
            kernel_initializer=partial(init_bounded, dim=dim),
            bias_initializer=partial(init_bounded, dim=dim))]

    def call(self, x, **kwargs):
        out = x
        for layer in self.model:
            out = layer(out, **kwargs)
        return out
