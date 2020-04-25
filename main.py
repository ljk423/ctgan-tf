from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
from absl import app
import tensorflow as tf
import numpy as np
import pandas as pd
from ctgan.synthesizer import CTGANSynthesizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(argv):
    del argv
    DEMO_URL = 'http://ctgan-data.s3.amazonaws.com/census.csv.gz'
    data = pd.read_csv(DEMO_URL, compression='gzip')
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

    np.random.seed(0)
    tf.random.set_seed(0)

    model = CTGANSynthesizer()
    model.train(data, discrete_columns, 300)
    model.dump('model.joblib', overwrite=True)
    model = CTGANSynthesizer('model.joblib')
    sampled = model.sample(data.shape[0])
    sampled.to_csv('tests/tensorflow.csv', index=False)


if __name__ == '__main__':
    app.run(main)
