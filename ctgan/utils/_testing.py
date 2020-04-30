"""
Module with methods for testing.
"""
import numpy as np
import pandas as pd


def get_test_variables():
    """Method that returns variables used across all unit tests.

    Returns
    -------
    dict
        A dictionary containing the training variables.

    """
    return {
        'decimal': 4,
        'input_dim': 10,
        'output_dim': 10,
        'pac': 10,
        'batch_size': 10,
        'gp_lambda': 10.0,
        'n_opt': 10,
        'n_col': 5,
        'layer_dims': [256, 256],
        'tau': 0.2
    }


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
