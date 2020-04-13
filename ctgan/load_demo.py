import pandas as pd


def load_demo():
    demo_url = 'http://ctgan-data.s3.amazonaws.com/census.csv.gz'
    return pd.read_csv(demo_url, compression='gzip')