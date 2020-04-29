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
        A tuple containing the dataframe with the random dataset and the list of
        discrete variables.
    """
    np.random.seed(seed)
    data = np.concatenate((
        np.random.random((batch_size, 1)),
        np.random.randint(0, 5, size=(batch_size, 1))), axis=1)

    df = pd.DataFrame(data, columns=['col1', 'col2'])
    discrete = ['col2']
    return df, discrete


def compare_objects(desired, other):
    """Compare two objects.

    Parameters
    ----------
    desired: object
        Object to compare to.

    other: object
        Other object.

    Returns
    -------
    bool
        True, if the objects are equal. False, otherwise.
    """
    if not isinstance(other, type(desired)):
        return False
    try:
        for attr, value in desired.__dict__.items():
            if attr not in other.__dict__:
                return False
            if _compare_values(value, other.__dict__[attr]) is False:
                return False
        return True
    except AttributeError:
        return _compare_values(desired, other)


def _compare_values(obj, other):
    try:
        if isinstance(obj, pd.Series):
            pd.testing.assert_series_equal(obj, other)
        elif isinstance(obj, pd.DataFrame):
            pd.testing.assert_frame_equal(obj, other)
        elif isinstance(obj, tf.Tensor):
            tf.assert_equal(obj, other)
        elif issubclass(type(obj), tf.keras.models.Model):
            np.testing.assert_equal(obj.get_weights(), other.get_weights())
        elif isinstance(obj, np.ndarray) and 'int' in str(obj.dtype):
            np.testing.assert_equal(obj, other)
        else:
            np.testing.assert_almost_equal(obj, other)
        return True
    except AssertionError:
        return False