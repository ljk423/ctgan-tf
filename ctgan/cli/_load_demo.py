"""
Module that provides a method for loading an external dataset, for testing.
"""
import pandas as pd


def load_demo():
    """Loads an external dataset, for testing purposes.

    Returns
    -------
    pandas.DataFrame
        DataFrame of the loaded CSV file.
    """
    demo_url = 'http://ctgan-data.s3.amazonaws.com/census.csv.gz'
    return pd.read_csv(demo_url, compression='gzip')