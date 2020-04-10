import tensorflow as tf
import math


def init_bounded(shape, dtype=None):
    bound = 1 / math.sqrt(shape[0])
    return tf.random.uniform(shape=shape, minval=-bound, maxval=bound, dtype=dtype)


class Critic(tf.keras.Model):
    def __init__(self, dis_dims, pac):
        super(Critic, self).__init__()
        self.pac = pac

        self.model = [self._reshape_func]
        for layer_dim in list(dis_dims):
            self.model += [
                tf.keras.layers.Dense(
                    layer_dim,
                    kernel_initializer=init_bounded,
                    bias_initializer=init_bounded),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(0.5)]

        self.model += [tf.keras.layers.Dense(
            1, kernel_initializer=init_bounded, bias_initializer=init_bounded)]

    def _reshape_func(self, x, **kwargs):
        dims = x.get_shape().as_list()
        return tf.reshape(x, [-1, dims[1] * self.pac])

    def call(self, x, **kwargs):
        out = x
        for layer in self.model:
            out = layer(out, **kwargs)
        return out


class ResidualLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(ResidualLayer, self).__init__()
        self.num_outputs = num_outputs
        self.fc = tf.keras.layers.Dense(
            self.num_outputs,
            kernel_initializer=init_bounded,
            bias_initializer=init_bounded)
        self.bn = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.9)
        self.relu = tf.keras.layers.ReLU()

    def call(self, inputs, **kwargs):
        outputs = self.fc(inputs, **kwargs)
        outputs = self.bn(outputs, **kwargs)
        outputs = self.relu(outputs, **kwargs)
        return tf.concat([outputs, inputs], axis=1)


class Generator(tf.keras.Model):
    def __init__(self, gen_dims, data_dim):
        super(Generator, self).__init__()

        self.model = list()
        for layer_dim in list(gen_dims):
            self.model += [ResidualLayer(layer_dim)]

        self.model += [tf.keras.layers.Dense(
            data_dim, kernel_initializer=init_bounded, bias_initializer=init_bounded)]

    def call(self, x, **kwargs):
        out = x
        for layer in self.model:
            out = layer(out, **kwargs)
        return out
