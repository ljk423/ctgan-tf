"""
This module contains the definition of the Sampler.
"""
import numpy as np


class DataSampler:
    """Data Sampler.

    It samples real data from the original dataset.

    Parameters
    ----------
    data: np.ndarray
        Training data.

    output_info: list[tuple]
        List containing metadata about the data columns of the original
        dataset, namely the number of columns where to apply a given
        activation function.

    See Also
    --------
    DataTransformer : Transforms the input dataset by applying mode-specific
        normalization and OneHot encoding.

    """
    # pylint: disable=too-few-public-methods

    def __init__(self, data, output_info):
        super(DataSampler, self).__init__()
        self._data = data
        self._model = []
        self._n = len(data)

        st_idx = 0
        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                st_idx += item[0]
                skip = True
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st_idx += item[0]
                    continue

                ed_idx = st_idx + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st_idx + j])[0])

                self._model.append(tmp)
                st_idx = ed_idx
            else:
                assert 0

        assert st_idx == data.shape[1]

    def sample(self, n_samples, col_idx, opt_idx):
        """Sample a batch of training data.

        Parameters
        ----------
        n_samples: int
            Size of the batch.

        col_idx: np.ndarray
            If `col` is None, then there won't be any restrictions to which
            data can be sampled. Otherwise, assuming that discrete variables
            are OneHot encoded, it samples data that has the value of the
            index `opt` on column `col` set to 1.

        opt_idx: np.ndarray
            If `col` is None, then there won't be any restrictions to which
            data can be sampled. Otherwise, assuming that discrete variables
            are OneHot encoded, it samples data that has the value of the
            index `opt` on column `col` set to 1.

        Returns
        -------
            Real training data.
        """
        if col_idx is None:
            idx = np.random.choice(np.arange(self._n), n_samples)
            return self._data[idx]

        idx = []
        for col, opt in zip(col_idx, opt_idx):
            idx.append(np.random.choice(self._model[col][opt]))

        return self._data[idx]
