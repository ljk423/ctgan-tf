"""
Conditional Tabular Generative Adversarial Network Synthesizer module.

It contains ``CTGANSynthesizer``, a class containing a Tensorflow
implementation of the work published in `Modeling Tabular data using
Conditional GAN <https://arxiv.org/abs/1907.00503>`_. The original PyTorch
implementation can be found in the authors' `GitHub repository
<https://github.com/sdv-dev/CTGAN>`_.
"""

import datetime
import os
import joblib
import numpy as np
import tensorflow as tf
from functools import partial

from ..data_modules import ConditionalGenerator, Sampler, DataTransformer
from ..models import Generator, Critic
from ..losses import conditional_loss, gradient_penalty
from ._bar_utils import ProgressBar


class CTGANSynthesizer:
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different
    components are orchestrated together.
    For more details about the process, please check the original PyTorch
    solution (designed by the authors) which is available on their
    `GitHub <https://github.com/sdv-dev/CTGAN>`_, and their Modeling Tabular
    data using Conditional GAN :cite:`xu2019modeling` paper.

    Parameters
    ----------
    file_path: str, default=None
        File path to where a Synthesizer model is stored. If this
        value is ``None``, it will create a new instance of this class.

    log_dir: str, default=None
        Directory path to where the log files (metrics) will be
        stored. If this value is ``None``, no logs will be saved.

    z_dim: int, default=128
        Embedding/latent dimension of fake samples passed to the
        Generator.

    pac: int, default=10
        Size of the Pac framework. For more details, consult the original
        paper :cite:`xu2019modeling`, and the PacGAN framework paper
        :cite:`lin2018pacgan`.

    gen_dim: tuple[int], or list[int], default=(256, 256)
        Size of the output samples for each one of the Residuals.
        A Residual Layer will be created for each one of the values provided.

    crt_dim: tuple[int], or list[int], default=(256, 256)
        Size of the output samples for each one of the Critic layers.
        A Fully Connected layer will be created for each one of the
        provided values.

    l2_scale: float, default=1e-6
        L2 regularization (:math:`\\lambda`) added to the Generator optimizer.
        This is computed by adding the weights scaled by :math:`\\lambda` to
        the computed gradients :cite:`2018fastadam`:
        :math:`\\Delta_{\\mathcal{w}}' = \\Delta_{\\mathcal{W}} +
                \\lambda * \\mathcal{W}`

    batch_size: int, default=500
        Number of samples to process at each training step.
        This value must be even and divisible by ``pac``.

    gp_lambda: float, default=10.0
        Gradient Penalty lambda. For more information, refer to the WGAN-GP
        paper :cite:`gulrajani2017improved`.

    tau: float, default=0.2
        Gumbel-Softmax non-negative scalar temperature
        :cite:`maddison2016concrete`, :cite:`jang2016categorical`.

    Raises
    ------
    IsDirectoryError
        If ``log_dir`` is not None and the path does not exist.

    ValueError
        If ``batch_size`` is not an even value or divisible by ``pac``.

    See Also
    --------
    ctgan.data_modules.DataTransformer : Transforms the input dataset by
        applying mode-specific normalization and OneHot encoding.

    ctgan.data_modules.ConditionalGenerator : Samples conditional and mask
        vectors according to the set of available discrete variables.

    ctgan.data_modules.Sampler : Samples real data from the original dataset.

    ctgan.models.Generator : TensorFlow model that attempts to replicate
        the original data, given some random noise and conditional vectors.

    ctgan.models.Critic : TensorFlow model that attempts to tell apart real
        from fake samples, outputted by the Generator.

    References
    ----------
    .. bibliography:: ../bibtex/refs.bib

    Examples
    --------
    >>> from ctgan.cli import load_demo
    >>> data, discrete = load_demo()
    >>> data.head(5)
       age          workclass  fnlwgt   education  ...  income
    0   39          State-gov   77516   Bachelors  ...   <=50K
    1   50   Self-emp-not-inc   83311   Bachelors  ...   <=50K
    2   38            Private  215646     HS-grad  ...   <=50K
    3   53            Private  234721        11th  ...   <=50K
    4   28            Private  338409   Bachelors  ...   <=50K
    >>> from ctgan.synthesizer import CTGANSynthesizer
    >>> model = CTGANSynthesizer()
    >>> model.train(data, discrete, epochs=1)
    Epoch 1/1
    32500/32500 |██| 3354.54samples/s  ETA: 00:00  Elapsed Time: 00:09
            g_loss: 2.065  cond_loss: 2.122  c_loss:-0.526  gp: 1.089
    >>> s = model.sample(5)
    >>> s.head(5)
       age workclass  fnlwgt      education  ...  income
    0   74   Private  168809   Some-college  ...    >50K
    1   51   Private  157349        HS-grad  ...    >50K
    2   20   Private  101469           11th  ...    >50K
    3   42   Private  225237           11th  ...    >50K
    4   67   Private  117769           11th  ...   <=50K

    """

    def __init__(self,
                 file_path=None,
                 log_dir=None,
                 z_dim=128,
                 pac=10,
                 gen_dim=(256, 256),
                 crt_dim=(256, 256),
                 l2_scale=1e-6,
                 batch_size=500,
                 gp_lambda=10.0,
                 tau=0.2):
        if file_path is not None:
            self._load(file_path)
            return
        if log_dir is not None and os.path.exists(log_dir):
            raise IsADirectoryError("Log directory does not exist.")
        if batch_size % 2 != 0 or batch_size % pac != 0:
            raise ValueError(
                "batch_size needs to be an even value divisible by pac.")

        self._log_dir = log_dir
        self._z_dim = z_dim
        self._pac = pac
        self._pac_dim = None
        self._l2_scale = l2_scale
        self._batch_size = batch_size
        self._gp_lambda = gp_lambda
        self._tau = tau
        self._gen_dim = tuple(gen_dim)
        self._crt_dim = tuple(crt_dim)
        self._g_opt = tf.keras.optimizers.Adam(
            learning_rate=2e-4, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        self._c_opt = tf.keras.optimizers.Adam(
            learning_rate=2e-4, beta_1=0.5, beta_2=0.9, epsilon=1e-08)
        self._transformer = DataTransformer()
        self._data_sampler = None
        self._cond_generator = None
        self._generator = None
        self._critic = None

    def train(self,
              train_data,
              discrete_columns=tuple(),
              epochs=300,
              log_frequency=True):
        """Fit the CTGAN according to the provided train_data.

        Parameters
        ----------
        train_data: pandas.DataFrame, or np.ndarray
            Training data. It must be a 2-dimensional ``np.ndarray``,
            or a ``pandas.DataFrame``.

        discrete_columns: list
            List of discrete columns to be used to generate the Conditional
            Vector. If ``train_data`` is a ``np.ndarray``, this list should
            contain the integer indices of the columns. Otherwise, if it is a
            ``pandas.DataFrame``, this list should contain the column names.

        epochs: int, default=300
            Number of training epochs.

        log_frequency: bool, default=True
            Whether to use log frequency of categorical levels in conditional
            sampling.

        """
        # Initialize DataTransformer and ConditionalGenerator based on the
        # input data and discrete columns info
        self._transformer.fit(train_data, discrete_columns)
        train_data = self._transformer.transform(train_data)
        self._transformer.generate_tensors()

        self._data_sampler = Sampler(train_data, self._transformer.output_info)
        data_dim = self._transformer.output_dimensions
        self._cond_generator = ConditionalGenerator(
            train_data,
            self._transformer.output_info,
            log_frequency)
        self._generator = Generator(
            self._z_dim + self._cond_generator.n_opt,
            self._gen_dim,
            data_dim,
            self._transformer.output_tensor,
            self._tau)
        self._critic = Critic(
            data_dim + self._cond_generator.n_opt,
            self._crt_dim,
            self._pac)

        # Create TF metrics
        metrics = {
            'g_loss': tf.metrics.Mean(),
            'cond_loss': tf.metrics.Mean(),
            'c_loss': tf.metrics.Mean(),
            'gp': tf.metrics.Mean(),
        }
        if self._log_dir is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = \
                self._log_dir + '/gradient_tape/' + current_time + '/train'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        # Build model graphs
        self._generator.build((self._batch_size, self._generator._input_dim))
        self._critic.build((self._batch_size, self._critic._input_dim))

        # Train models
        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for epoch in range(epochs):
            bar = ProgressBar(
                len(train_data), self._batch_size, epoch, epochs, metrics)
            for _ in range(steps_per_epoch):
                c_loss, gp = self._train_c()
                metrics['c_loss'](c_loss)
                metrics['gp'](gp)
                g_loss, cond_loss = self._train_g()
                metrics['g_loss'](g_loss)
                metrics['cond_loss'](cond_loss)
                bar.update(metrics)

            if self._log_dir is not None:
                with train_summary_writer.as_default():
                    for m in metrics:
                        tf.summary.scalar(m, metrics[m].result(), step=epoch)
                        metrics[m].reset_states()
            bar.close()
            del bar

    @tf.function
    def train_c_step(self, fake_cat, real_cat):
        """Critic training step.

        Computes the loss, gradient penalty, gradients and it uses the
        optimizer to update the Critic weights.

        Parameters
        ----------
        fake_cat: tf.Tensor
            Fake sample outputted by the Generator.

        real_cat: tf.Tensor
            Real sample extracted from the training set.

        Returns
        -------
        tuple(tf.Tensor, tf.Tensor)
            Tuple containing (critic_loss, gradient_penalty).

        """
        with tf.GradientTape() as t:
            y_fake = self._critic(fake_cat, training=True)
            y_real = self._critic(real_cat, training=True)

            gp = gradient_penalty(
                partial(self._critic, training=True), real_cat, fake_cat,
                self._pac, self._gp_lambda)
            loss = -(tf.reduce_mean(y_real) - tf.reduce_mean(y_fake))
            c_loss = loss + gp

        grad = t.gradient(c_loss, self._critic.trainable_variables)
        self._c_opt.apply_gradients(
            zip(grad, self._critic.trainable_variables))
        return loss, gp

    def _train_c(self):
        """Critic training method.

        It samples uniformly random data, generates a conditional vector, and
        samples real data from the dataset. It then passes the fake data to
        the generator and invokes ``_train_c_step`` to compute and
        update the gradients of the Critic network.

        Returns
        -------
        tuple(tf.Tensor, tf.Tensor)
            Tuple containing (critic_loss, gradient_penalty).

        """
        fake_z = tf.random.normal([self._batch_size, self._z_dim])

        # Generate data_modules vector
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
            real = self._data_sampler.sample(
                self._batch_size, col[perm], opt[perm])
            c2 = tf.gather(c1, perm)

        fake, fake_act = self._generator(fake_z, training=True)
        real = tf.convert_to_tensor(real.astype('float32'))

        if c1 is not None:
            fake_cat = tf.concat([fake_act, c1], axis=1)
            real_cat = tf.concat([real, c2], axis=1)
        else:
            fake_cat = fake
            real_cat = real

        return self.train_c_step(fake_cat, real_cat)

    @tf.function
    def train_g_step(self, fake_z):
        """Generator training step for datasets that do not have discrete
        variables, thus, those that do not sample conditional vectors.

        It computes the loss, gradients and it uses the optimizer to update
        the Generator weights.

        Parameters
        ----------
        fake_z: tf.Tensor
            Randomly sample noise.

        Returns
        -------
        tuple(tf.Tensor, tf.Tensor)
            Tuple containing (generator_loss, conditional_loss = 0).

        """
        with tf.GradientTape() as t:
            fake, fake_act = self._generator(fake_z, training=True)
            y_fake = self._critic(fake_act, training=True)
            g_loss = -tf.reduce_mean(y_fake)

        weights = self._generator.trainable_variables
        grad = t.gradient(g_loss, weights)
        grad = [grad[i] + self._l2_scale * weights[i]
                for i in range(len(grad))]
        self._g_opt.apply_gradients(
            zip(grad, self._generator.trainable_variables))
        return g_loss, tf.constant(0, dtype=tf.float32)

    @tf.function
    def train_g_cond_step(self, fake_z, c1, m1, cond_info):
        """Generator training step for datasets that contain discrete
        variables, therefore, we need to compute the conditional loss.

        Additionally, it computes the loss, gradients and it uses the
        optimizer to update the Generator weights.

        Parameters
        ----------
        fake_z: tf.Tensor
            Randomly sample noise, concatenated with a cond vector.

        c1: tf.Tensor
            Conditional vector.

        m1: tf.Tensor
            Mask vector.

        cond_info: tf.Tensor
            Transformer data_modules information tensor.

        Returns
        -------
        tuple(tf.Tensor, tf.Tensor)
            Tuple containing (generator_loss, conditional_loss).

        """
        with tf.GradientTape() as t:
            fake, fake_act = self._generator(fake_z, training=True)
            y_fake = self._critic(
                tf.concat([fake_act, c1], axis=1), training=True)
            cond_loss = conditional_loss(cond_info, fake, c1, m1)
            g_loss = -tf.reduce_mean(y_fake) + cond_loss

        weights = self._generator.trainable_variables
        grad = t.gradient(g_loss, weights)
        grad = [grad[i] + self._l2_scale * weights[i]
                for i in range(len(grad))]
        self._g_opt.apply_gradients(
            zip(grad, self._generator.trainable_variables))
        return g_loss, cond_loss

    def _train_g(self):
        """Generator training method.

        It uniformly samples random data, generates a conditional vector, and
        it invokes ``_train_g_step``, if no conditional vectors exist,
        or ``_train_g_cond_step``, if they do, to compute and update
        the gradients of the Generator network.

        Returns
        -------
        (tf.Tensor, tf.Tensor)
            Tuple containing (generator_loss, conditional_loss).

        """
        fake_z = tf.random.normal([self._batch_size, self._z_dim])
        cond_vec = self._cond_generator.sample(self._batch_size)

        if cond_vec is None:
            return self.train_g_step(fake_z)

        c1, m1, col, opt = cond_vec
        c1 = tf.convert_to_tensor(c1, name="c1")
        m1 = tf.convert_to_tensor(m1, name="m1")
        fake_z = tf.concat([fake_z, c1], axis=1, name="fake_z")
        return self.train_g_cond_step(
            fake_z, c1, m1, self._transformer.cond_tensor)

    def sample(self, n):
        """Sample data similar to the training data.

        Parameters
        ----------
        n: int
            Number of rows of the output sample.

        Returns
        -------
        np.ndarray, or pandas.DataFrame
            The synthesized sample data.

        Raises
        ------
        ValueError
            If ``n`` is equal or less than 0.
        """
        if n <= 0:
            raise ValueError("Invalid number of samples.")

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
        """Export model to disk.

        This operation also involves transforming and converting the
        ``ConditionalGenerator`` and the ``DataTransformer`` and dumping all
        to the the same file.

        Parameters
        ----------
        file_path: str
            Path to where the model will be stored.

        overwrite: bool, default=False
            Overwrite existing files.

        Raises
        ------
        NameError
            If ``file_path`` is an invalid string.

        NotADirectoryError
            If the ``file_path`` directory does not exist.

        FileExistsError
            If ``overwrite=False`` and the provided ``file_path``
            already exists.
        """
        if file_path is None or len(file_path) == 0:
            raise NameError("Invalid file_path.")
        dir_name = os.path.dirname(file_path)
        if len(dir_name) and not os.path.exists(os.path.dirname(file_path)):
            raise NotADirectoryError("The file directory does not exist.")
        if not overwrite and os.path.exists(file_path):
            raise FileExistsError(
                "File already exists. If you wish to replace it,"
                " use overwrite=True")

        # Create a copy of class dict as we are about to change the dictionary
        class_dict = {k: v for k, v in self.__dict__.items()
                      if type(v) in [int, float, tuple]}
        class_dict['_cond_generator'] = self._cond_generator.__dict__
        class_dict['_transformer'] = self._transformer.__dict__
        class_dict['_gen_weights'] = self._generator.get_weights()

        # Dump dictionary to file
        joblib.dump(class_dict, file_path)
        del class_dict

    def _load(self, file_path):
        """Load an existent dumped model to the current instance of the class.

        Parameters
        ----------
        file_path: str
            Path where the model is stored.

        Raises
        ------
        NameError
            If ``file_path`` is an invalid string.

        FileNotFoundError
            If the provided path does not exist.
        """
        if file_path is None or len(file_path) == 0:
            raise NameError("Invalid file_path.")
        if not os.path.exists(file_path):
            raise FileNotFoundError("The provided file_path does not exist.")

        # Load class attributes
        class_dict = joblib.load(file_path)
        if class_dict is None:
            raise AttributeError

        # Load class attributes
        for key, value in class_dict.items():
            if type(value) in [int, float, tuple]:
                setattr(self, key, value)

        # Load binary models/encoders to class dict
        self._transformer = DataTransformer.from_dict(
            class_dict['_transformer'])
        self._cond_generator = ConditionalGenerator.from_dict(
            class_dict['_cond_generator'])

        # Load Generator instance
        self._generator = Generator(
            self._z_dim + self._cond_generator.n_opt,
            self._gen_dim,
            self._transformer.output_dimensions,
            self._transformer.output_tensor,
            self._tau)
        self._generator.build((self._batch_size, self._generator._input_dim))
        self._generator.set_weights(class_dict['_gen_weights'])
