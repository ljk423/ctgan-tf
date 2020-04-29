"""
Module with methods for testing.
"""
import numpy as np
import pandas as pd
import tensorflow as tf


def generate_data(batch_size, seed=0):
    """Generate a batch of random data, to use as dataset.

    Parameters
    ----------
    batch_size: int
        Number of rows of the batch.

    seed: int, default=0
        See for the RNG.

    Returns
    -------
    tuple(pandas.DataFrame, list[str])
        A tuple containing the dataframe with the random dataset and
        the list of discrete variables.
    """
    np.random.seed(seed)
    data = np.concatenate((
        np.random.random((batch_size, 1)),
        np.random.randint(0, 5, size=(batch_size, 1))), axis=1)

    dataframe = pd.DataFrame(data, columns=['col1', 'col2'])
    discrete = ['col2']
    return dataframe, discrete
