<p align="center">
    <img src="https://i.imgur.com/mbY9pvC.png" width="30%">
</p>
<h1 align="center">TensorFlow CTGAN</h1>
<p align="center">TensorFlow 2.1 implementation of Conditional Tabular GAN.</p>


<p align="center">
    <a href="./LICENSE.md">
        <img src="https://img.shields.io/badge/license-MIT-blue.svg">
    </a>
    <a href="https://pypi.python.org/pypi/ctgan-tf">
        <img alt="PyPI Shield" src="https://img.shields.io/pypi/v/ctgan-tf.svg">
    </a>
    <a href="https://travis-ci.com/pbmartins/ctgan-tf">
        <img alt="Build Status" src="https://travis-ci.com/pbmartins/ctgan-tf.svg?branch=master">
    </a>
    <a href="https://codecov.io/gh/pbmartins/ctgan-tf">
        <img alt="Coverage Status" src="https://codecov.io/gh/pbmartins/ctgan-tf/branch/master/graph/badge.svg">
    </a>
</p>

Tensorflow 2.1 implementation of a Conditional Tabular Generative Adversarial 
Network. CTGAN is a GAN-based data synthesizer that can "generate synthetic 
tabular data with high fidelity".

This model was originally designed by the *Data to AI Lab at MIT* team, 
and it was published in their NeurIPS paper 
[Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503).

For more information regarding this work, and to access the original PyTorch 
implementation provided by the authors, 
please refer to their GitHub repository and their documentation:

* **Documentation**: https://pbmartins.github.io/ctgan-tf
* **Original PyTorch Documentation**: https://sdv-dev.github.io/CTGAN
* **Original PyTorch repository**: https://github.com/sdv-dev/CTGAN

# Install

## Requirements

As of this moment, **CTGAN** has been solely tested tested on 
[Python 3.7](https://www.python.org/downloads/), 
and [TensorFlow 2.1](https://www.tensorflow.org/install).


* `tensorflow` (<2.2,>=2.1.0)
* `tensorflow-probability` (<1.0,>=0.9.0)
* `scikit-learn` (<0.23,>=0.21)
* `numpy` (<2,>=1.17.4)
* `pandas` (<1.0.2,>=1.0)
* `tqdm` (<4.44,>=4.43)

## Install

You can either install `ctgan-tf` through the PyPI package:

```shell script
pip3 install ctgan-tf
```

Or by cloning this repository and copying the `ctgan` folder to your 
project folder, or simply run:

```shell script
make install
```

# Data Format

**CTGAN** expects the input data to be a table given as either a `numpy.ndarray` 
or a `pandas.DataFrame` object with two types of columns:

* **Continuous Columns**: Columns that contain numerical values and which can 
  take any value.
* **Discrete columns**: Columns that only contain a finite number of possible 
  values, whether these are string values or not.

# Quickstart

Before being able to use CTGAN you will need to prepare your data as 
specified above.

For this example, we will be loading some data using the `ctgan.load_demo` 
function.

```python
from ctgan.utils import load_demo

data, discrete_columns = load_demo()
```

Even though the provided example already contains a list of discrete values, aside from the data itself, you will need to create a list with the names of 
the discrete variables:

```python
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
```

Once you have the data ready, you need to import and create an instance of the 
`CTGANSynthesizer` class and fit it passing your data and the list of 
discrete columns.

```python
from ctgan.synthesizer import CTGANSynthesizer

ctgan = CTGANSynthesizer()
ctgan.train(data, discrete_columns)
```

Once the process has finished, all you need to do is call the `sample` method
of your `CTGANSynthesizer` instance indicating the number of rows that you 
want to generate.

```python
samples = ctgan.sample(1000)
```

The output will be a table with the exact same format as the input and filled with the synthetic
data generated by the model.

For a more in-depth guide and API specification, check our documentation 
[here](https://pbmartins.github.io/ctgan-tf).