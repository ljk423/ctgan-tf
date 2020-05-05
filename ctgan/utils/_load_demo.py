"""
Module that provides a method for loading an external dataset, for testing.
"""
import pandas as pd


def load_demo():
    """Loads an external dataset, for testing purposes.

    Returns
    -------
    tuple(pandas.DataFrame, list[str])
        DataFrame of the loaded CSV file, and a list with the corresponding
        discrete variables.

    Examples
    --------
    >>> from ctgan.utils import load_demo
    >>> data, discrete = load_demo()
    >>> data.head(5)
       age          workclass  fnlwgt   education  ...  income
    0   39          State-gov   77516   Bachelors  ...   <=50K
    1   50   Self-emp-not-inc   83311   Bachelors  ...   <=50K
    2   38            Private  215646     HS-grad  ...   <=50K
    3   53            Private  234721        11th  ...   <=50K
    4   28            Private  338409   Bachelors  ...   <=50K
    >>> discrete
    ['workclass', 'education', 'marital-status', 'occupation',
    'relationship', 'race', 'sex', 'native-country', 'income']

    """
    demo_url = 'http://ctgan-data.s3.amazonaws.com/census.csv.gz'
    discrete_columns = [
        'workclass',
        'education',
        'marital-status',
        'occupation',
        'relationship',
        'race',
        'sex',
        'native-country',
        'income'
    ]
    return pd.read_csv(demo_url, compression='gzip'), discrete_columns
