from ctgan.layers import *


class Critic(tf.keras.Model):
    def __init__(self, input_dim, dis_dims, pac):
        super(Critic, self).__init__()
        self._pac = pac
        self._input_dim = input_dim

        self._model = [self.__reshape_func]
        dim = input_dim * self._pac
        for layer_dim in list(dis_dims):
            self._model += [
                tf.keras.layers.Dense(
                    layer_dim, input_dim=(dim,),
                    kernel_initializer=partial(init_bounded, dim=dim),
                    bias_initializer=partial(init_bounded, dim=dim)),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Dropout(0.5)]
            dim = layer_dim

        layer_dim = 1
        self._model += [tf.keras.layers.Dense(
            layer_dim, input_dim=(dim,),
            kernel_initializer=partial(init_bounded, dim=dim),
            bias_initializer=partial(init_bounded, dim=dim))]

    def __reshape_func(self, x, **kwargs):
        dims = x.get_shape().as_list()
        return tf.reshape(x, [-1, dims[1] * self._pac])

    def call(self, x, **kwargs):
        out = x
        for layer in self._model:
            out = layer(out, **kwargs)
        return out


class Generator(tf.keras.Model):
    def __init__(self, input_dim, gen_dims, data_dim, transformer_info, tau):
        super(Generator, self).__init__()

        self._input_dim = input_dim
        self._model = list()
        dim = input_dim
        for layer_dim in list(gen_dims):
            self._model += [ResidualLayer(dim, layer_dim)]
            dim += layer_dim

        self._model += [GenActivation(dim, data_dim, transformer_info, tau)]

    def call(self, x, **kwargs):
        out = x
        for layer in self._model:
            out = layer(out, **kwargs)
        return out
