"""
This module contains the definition of the ConditionalGenerator.
"""
import numpy as np


class ConditionalGenerator(object):
    """Conditional Generator.

    This generator is used along with the model defined in
    :class:`ctgan.models.Generator`, to sample conditional vectors.

    Parameters
    ----------
    data: np.ndarray, default=None
        Transformed input data by :class:`ctgan.data_modules.DataTransformer`.

    output_info: list[tuple], default=None
        List containing metadata about the data columns of the original
        dataset, namely the number of columns where to apply a given
        activation function.

    log_frequency: bool, default=None
        Whether to use log frequency of categorical levels in conditional
        sampling.

    See Also
    --------
    DataTransformer : Transforms the input dataset by applying mode-specific
        normalization and OneHot encoding.

    Attributes
    ----------
    n_opt: int
        Number of generated features by each conditional vector.
    """

    @classmethod
    def from_dict(cls, in_dict):
        """Create a new instance of this class by loading data from an
        external class dictionary.

        Parameters
        ----------
        in_dict: dict
            External class dictionary.

        Returns
        -------
        ConditionalGenerator
            A new instance with the same internal data as the one
            provided by `in_dict`.
        """
        dt = ConditionalGenerator()
        dt.__dict__ = in_dict
        return dt

    def __init__(self, data=None, output_info=None, log_frequency=None):
        if data is None or output_info is None or log_frequency is None:
            return

        self._model = []

        start = 0
        skip = False
        max_interval = 0
        counter = 0
        for item in output_info:
            if item[1] == 'tanh':
                start += item[0]
                skip = True
                continue

            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    start += item[0]
                    continue

                end = start + item[0]
                max_interval = max(max_interval, end - start)
                counter += 1
                self._model.append(np.argmax(data[:, start:end], axis=-1))
                start = end

            else:
                assert 0

        assert start == data.shape[1]

        self._interval = []
        self._n_col = 0
        self.n_opt = 0
        skip = False
        start = 0
        self._p = np.zeros((counter, max_interval))
        for item in output_info:
            if item[1] == 'tanh':
                skip = True
                start += item[0]
                continue
            elif item[1] == 'softmax':
                if skip:
                    start += item[0]
                    skip = False
                    continue
                end = start + item[0]
                tmp = np.sum(data[:, start:end], axis=0)
                if log_frequency:
                    tmp = np.log(tmp + 1)
                tmp = tmp / np.sum(tmp)
                self._p[self._n_col, :item[0]] = tmp
                self._interval.append((self.n_opt, item[0]))
                self.n_opt += item[0]
                self._n_col += 1
                start = end
            else:
                assert 0

        self._interval = np.asarray(self._interval)

    def _random_choice_prob_index(self, idx):
        a = self._p[idx]
        r = np.expand_dims(np.random.rand(a.shape[0]), axis=1)
        return (a.cumsum(axis=1) > r).argmax(axis=1)

    def sample(self, batch_size):
        """Sample a conditional vector for the given batch of data.

        For a more detailed implementation of this method, consult section 4.3
        of :cite:`ci2019modeling`.

        Parameters
        ----------
        batch_size: int
            Size of the data batch.

        Returns
        -------
        None, or tuple(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
            None, if the training data did not contain any discrete columns.
            A tuple containing:

            - `vec`: conditional vector
            - `mask`: mask vector
            - `idx`: index of the mask vector which is set to 1
            - `opt_prime`: index of the column vector which is set to 1

        """
        if self._n_col == 0:
            return None

        idx = np.random.choice(np.arange(self._n_col), batch_size)

        vec = np.zeros((batch_size, self.n_opt), dtype='float32')
        mask = np.zeros((batch_size, self._n_col), dtype='float32')

        mask[np.arange(batch_size), idx] = 1
        opt_prime = self._random_choice_prob_index(idx)
        opt = self._interval[idx, 0] + opt_prime
        vec[np.arange(batch_size), opt] = 1
        return vec, mask, idx, opt_prime

    def sample_zero(self, batch_size):
        """Sample a conditional vector for the given batch of data.

        For a more detailed implementation of this method, consult section 4.3
        of :cite:`ci2019modeling`.

        Parameters
        ----------
        batch_size: int
            Size of the data batch.

        Returns
        -------
        None, or np.ndarray
            None, if the training data did not contain any discrete columns.
            The conditional vector, otherwise.

        """
        if self._n_col == 0:
            return None

        vec = np.zeros((batch_size, self.n_opt), dtype='float32')
        idx = np.random.choice(np.arange(self._n_col), batch_size)
        for i in range(batch_size):
            col = idx[i]
            pick = int(np.random.choice(self._model[col]))
            vec[i, pick + self._interval[col, 0]] = 1

        return vec
