import tensorflow as tf
import math


def init_bounded(shape, dtype=None):
    bound = math.sqrt(1 / shape[0])
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
        for layer in self.model:
            x = layer(x, **kwargs)
        return x


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
        outputs = self.fc(inputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)
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
        for layer in self.model:
            x = layer(x, **kwargs)
        return x
