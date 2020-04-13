
<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“sdv-dev” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![Development Status](https://img.shields.io/badge/Development%20Status-2%20--%20Pre--Alpha-yellow)](https://pypi.org/search/?c=Development+Status+%3A%3A+2+-+Pre-Alpha)
[![PyPI Shield](https://img.shields.io/pypi/v/ctgan.svg)](https://pypi.python.org/pypi/ctgan)
[![Travis CI Shield](https://travis-ci.org/sdv-dev/CTGAN.svg?branch=master)](https://travis-ci.org/sdv-dev/CTGAN)
[![Downloads](https://pepy.tech/badge/ctgan)](https://pepy.tech/project/ctgan)
[![Coverage Status](https://codecov.io/gh/sdv-dev/CTGAN/branch/master/graph/badge.svg)](https://codecov.io/gh/sdv-dev/CTGAN)

# Tensorflow CTGAN

Tensorflow 2.0 implementation of a Conditional Tabular Generative Adversarial Network. 
CTGAN is a GAN-based data synthesizer that can generate synthetic tabular data with high fidelity.

This model was originally designed by the *Data to AI Lab at MIT* team, and published in their NeurIPS paper
 [Modeling Tabular data using Conditional GAN](https://arxiv.org/abs/1907.00503).

For more information regarding this work, and to access the original PyTorch implementation provided by the authors, 
please refer to their GitHub repository and their documentation:

* Documentation: https://sdv-dev.github.io/CTGAN
* Homepage: https://github.com/sdv-dev/CTGAN

# Install

## Requirements

As of this moment, **CTGAN** has been solely tested tested on [Python 3.7](https://www.python.org/downloads/), 
and [TensorFlow 2.1](https://www.tensorflow.org/install).

## Install

At the moment, we still do not provided a PyPI package. Therefore, to use this model, simply clone this repository and 
copy the `ctgan` folder to your project folder.

# Data Format

**CTGAN** expects the input data to be a table given as either a `numpy.ndarray` or a
`pandas.DataFrame` object with two types of columns:

* **Continuous Columns**: Columns that contain numerical values and which can take any value.
* **Discrete columns**: Columns that only contain a finite number of possible values, wether
these are string values or not.

This is an example of a table with 4 columns:

* A continuous column with float values
* A continuous column with integer values
* A discrete column with string values
* A discrete column with integer values

|   | A    | B   | C   | D |
|---|------|-----|-----|---|
| 0 | 0.1  | 100 | 'a' | 1 |
| 1 | -1.3 | 28  | 'b' | 2 |
| 2 | 0.3  | 14  | 'a' | 2 |
| 3 | 1.4  | 87  | 'a' | 3 |
| 4 | -0.1 | 69  | 'b' | 2 |


**NOTE**: CTGAN does not distinguish between float and integer columns, which means that it will
sample float values in all cases. If integer values are required, the outputted float values
must be rounded to integers in a later step, outside of CTGAN.

# Python Quickstart

In this short tutorial we will guide you through a series of steps that will help you
getting started with **CTGAN**.

## 1. Model the data

### Step 1: Prepare your data

Before being able to use CTGAN you will need to prepare your data as specified above.

For this example, we will be loading some data using the `ctgan.load_demo` function.

```python
from ctgan import load_demo

data = load_demo()
```

This will download a copy of the [Adult Census Dataset](https://archive.ics.uci.edu/ml/datasets/adult) as a dataframe:

|   age | workclass        |   fnlwgt | ... |   hours-per-week | native-country   | income   |
|-------|------------------|----------|-----|------------------|------------------|----------|
|    39 | State-gov        |    77516 | ... |               40 | United-States    | <=50K    |
|    50 | Self-emp-not-inc |    83311 | ... |               13 | United-States    | <=50K    |
|    38 | Private          |   215646 | ... |               40 | United-States    | <=50K    |
|    53 | Private          |   234721 | ... |               40 | United-States    | <=50K    |
|    28 | Private          |   338409 | ... |               40 | Cuba             | <=50K    |
|   ... | ...              |      ... | ... |              ... | ...              | ...      |


Aside from the table itself, you will need to create a list with the names of the discrete
variables.

For this example:

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

### Step 2: Fit CTGAN to your data

Once you have the data ready, you need to import and create an instance of the `CTGANSynthesizer`
class and fit it passing your data and the list of discrete columns.

```python
from ctgan import CTGANSynthesizer

ctgan = CTGANSynthesizer()
ctgan.train(data, discrete_columns)
```

This process is likely to take a long time to run.
If you want to make the process shorter, or longer, you can control the number of training epochs
that the model will be performing by adding it to the `fit` call:

```python
ctgan.train(data, discrete_columns, epochs=5)
```

## 2. Generate synthetic data

Once the process has finished, all you need to do is call the `sample` method of your
`CTGANSynthesizer` instance indicating the number of rows that you want to generate.

```python
samples = ctgan.sample(1000)
```

The output will be a table with the exact same format as the input and filled with the synthetic
data generated by the model.

|     age | workclass    |    fnlwgt | ... |   hours-per-week | native-country   | income   |
|---------|--------------|-----------|-----|------------------|------------------|----------|
| 26.3191 | Private      | 124079    | ... |          40.1557 | United-States    | <=50K    |
| 39.8558 | Private      | 133996    | ... |          40.2507 | United-States    | <=50K    |
| 38.2477 | Self-emp-inc | 135955    | ... |          40.1124 | Ecuador          | <=50K    |
| 29.6468 | Private      |   3331.86 | ... |          27.012  | United-States    | <=50K    |
| 20.9853 | Private      | 120637    | ... |          40.0238 | United-States    | <=50K    |
|     ... | ...          |       ... | ... |              ... | ...              | ...      |


