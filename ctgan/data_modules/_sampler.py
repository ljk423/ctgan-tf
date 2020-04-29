"""
This module contains the definition of the Sampler.
"""
import numpy as np


class Sampler(object):
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

    def __init__(self, data, output_info):
        super(Sampler, self).__init__()
        self._data = data
        self._model = []
        self._n = len(data)

        st = 0
        skip = False
        for item in output_info:
            if item[1] == 'tanh':
                st += item[0]
                skip = True
            elif item[1] == 'softmax':
                if skip:
                    skip = False
                    st += item[0]
                    continue

                ed = st + item[0]
                tmp = []
                for j in range(item[0]):
                    tmp.append(np.nonzero(data[:, st + j])[0])

                self._model.append(tmp)
                st = ed
            else:
                assert 0

        assert st == data.shape[1]

    def sample(self, n, col, opt):
        """Sample a batch of training data.

        Parameters
        ----------
        n: int
            Size of the batch.

        col: int
            If `col` is None, then there won't be any restrictions to which
            data can be sampled. Otherwise, assuming that discrete variables
            are OneHot encoded, it samples data that has the value of the
            index `opt` on column `col` set to 1.

        opt: int
            If `col` is None, then there won't be any restrictions to which
            data can be sampled. Otherwise, assuming that discrete variables
            are OneHot encoded, it samples data that has the value of the
            index `opt` on column `col` set to 1.

        Returns
        -------
            Real training data.
        """
        if col is None:
            idx = np.random.choice(np.arange(self._n), n)
            return self._data[idx]

        idx = []
        for c, o in zip(col, opt):
            idx.append(np.random.choice(self._model[c][o]))

        return self._data[idx]
