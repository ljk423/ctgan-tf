# MIT License
#
# Copyright (c) 2019 Drew Szurko
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

from absl import app
import tensorflow as tf
from tensorflow import keras

import numpy as np
import pandas as pd
from ctgan.synthesizer import CTGANSynthesizer

keras.backend.clear_session()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def main(argv):
    del argv
    df = pd.read_csv('aug-training.csv')
    df.set_index('bookingid')
    df = df[['total_cost', 'nights']]
    discrete_columns = ['package']

    tf.random.set_seed(42)
    np.random.seed(42)

    #discrete_columns = ['package', 'nights', 'booking_date_day',
    #   'booking_date_month', 'booking_date_year', 'checkin_date_day',
    #   'checkin_date_month', 'checkin_date_year']

    ctgan = CTGANSynthesizer()
    ctgan.train(df, discrete_columns, epochs=3)

    print(ctgan.sample(1000))


def main2(argv):
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

    model = CTGANSynthesizer()
    model.train(data, discrete_columns, 300)
    sampled = model.sample(data.shape[0])

    sampled.to_csv('tests/tensorflow.csv', index=False)


if __name__ == '__main__':
    app.run(main2)
