----------
User Guide
----------

TensorFlow 2.X implementation of a Conditional Tabular Generative Adversarial
Network.

**CTGAN** is a GAN-based data synthesizer that can generate synthetic tabular
data with high fidelity.

Data Format
-----------

The :class:`ctgan.synthesizer.CTGANSynthesizer` expects the input data to be
a table given as either a :class:`numpy.ndarray` or a :class:`pandas.DataFrame`
object with two types of columns:

* **Continuous Columns**: Columns that contain numerical values and which
  can take any value.
* **Discrete columns**: Columns that only contain a finite number of possible
  values, whether these are string values or not.

This is an example of a table with 4 columns:

* A continuous column with float values
* A continuous column with integer values
* A discrete column with string values
* A discrete column with integer values

+---+------+-----+-----+---+
|   |  A   |  B  |  C  | D |
+===+======+=====+=====+===+
| 0 | 0.1  | 100 | 'a' | 1 |
+---+------+-----+-----+---+
| 1 | -1.3 | 28  | 'b' | 2 |
+---+------+-----+-----+---+
| 2 | 0.3  | 14  | 'a' | 2 |
+---+------+-----+-----+---+
| 3 | 1.4  | 87  | 'a' | 3 |
+---+------+-----+-----+---+
| 4 | -0.1 | 69  | 'b' | 2 |
+---+------+-----+-----+---+

.. note:: CTGAN does not distinguish between float and integer columns, which
          means that it will sample float values in all cases.

          If integer values
          are required, the outputted float values must be rounded to integers
          in a later step, outside of CTGAN.

Quickstart guide
-----------------

In this short tutorial we will guide you through a series of steps that will
help you getting started with **CTGAN**.

Before being able to use the synthesizer you will need to prepare your data as
specified above. If you just want to run a test example, load the dataset
provided when invoking the :class:`ctgan.utils.load_demo` function.

    >>> from ctgan.utils import load_demo
    >>> data, discrete_columns = load_demo()

This will download a copy of the `Adult Census Dataset
<https://archive.ics.uci.edu/ml/datasets/adult>`_ as a dataframe:

+-----+------------------+--------+-----+----------------+----------------+--------+
| age |    workclass     | fnlwgt | ... | hours-per-week | native-country | income |
+=====+==================+========+=====+================+================+========+
| 39  | State-gov        | 77516  | ... | 40             | United-States  | <=50K  |
+-----+------------------+--------+-----+----------------+----------------+--------+
| 50  | Self-emp-not-inc | 83311  | ... | 13             | United-States  | <=50K  |
+-----+------------------+--------+-----+----------------+----------------+--------+
| 38  | Private          | 215646 | ... | 40             | United-States  | <=50K  |
+-----+------------------+--------+-----+----------------+----------------+--------+
| 53  | Private          | 234721 | ... | 40             | United-States  | <=50K  |
+-----+------------------+--------+-----+----------------+----------------+--------+
| 28  | Private          | 338409 | ... | 40             | Cuba           | <=50K  |
+-----+------------------+--------+-----+----------------+----------------+--------+
| ... | ...              | ...    | ... | ...            | ...            | ...    |
+-----+------------------+--------+-----+----------------+----------------+--------+

Even though the provided example data already contains a list of discrete
values to consider, you will need to create a list with the names of
the discrete variables.

    >>> discrete_columns = [
    ...    'workclass',
    ...    'education',
    ...    'marital-status',
    ...    'occupation',
    ...    'relationship',
    ...    'race',
    ...    'sex',
    ...    'native-country',
    ...    'income'
    >>> ]

Once you have the data ready, you need to import and create an instance of
:class:`ctgan.synthesizer.CTGANSynthesizer` class and fit it passing your data
and the list of discrete columns.

    >>> from ctgan.synthesizer import CTGANSynthesizer
    >>> ctgan = CTGANSynthesizer(file_path=None, log_dir=None, z_dim=128,
    ...                          pac=10, gen_dim=(256, 256), crt_dim=(256, 256),
    ...                          l2_scale=1e-6, batch_size=500, gp_lambda=10.0,
    ...                          tau=0.2)
    >>> ctgan.train(data, discrete_columns)

This process is likely to take a long time to run.
If you want to make the process shorter, or longer, you can control the number
of training epochs that the model will be performing by adding it to the
:meth:`ctgan.synthesizer.CTGANSynthesizer.train` call.

    >>> ctgan.train(data, discrete_columns, epochs=5)

Once the process has finished, all you need to do is call the
:meth:`ctgan.synthesizer.CTGANSynthesizer.sample` method of your model instance,
indicating the number of rows that you want to generate.

    >>> n_samples = 1000
    >>> samples = ctgan.sample(n_samples)

The output will be a table with the exact same format as the input and filled
with the synthetic data generated by the model.

+---------+--------------+---------+-----+----------------+----------------+--------+
|   age   |   workclass  |  fnlwgt | ... | hours-per-week | native-country | income |
+=========+==============+=========+=====+================+================+========+
| 26.3191 | Private      | 124079  | ... | 40.1557        | United-States  | <=50K  |
+---------+--------------+---------+-----+----------------+----------------+--------+
| 39.8558 | Private      | 133996  | ... | 40.2507        | United-States  | <=50K  |
+---------+--------------+---------+-----+----------------+----------------+--------+
| 38.2477 | Self-emp-inc | 135955  | ... | 40.1124        | Ecuador        | <=50K  |
+---------+--------------+---------+-----+----------------+----------------+--------+
| 29.6468 | Private      | 3331.86 | ... | 27.012         | United-States  | <=50K  |
+---------+--------------+---------+-----+----------------+----------------+--------+
| 20.9853 | Private      | 120637  | ... | 40.0238        | United-States  | <=50K  |
+---------+--------------+---------+-----+----------------+----------------+--------+
| ...     | ...          | ...     | ... | ...            | ...            | ...    |
+---------+--------------+---------+-----+----------------+----------------+--------+

After fitting your synthesizer, you may want to save it to deploy into a
production system or further testing.

    >>> model_path = '/path/to/your/model/file.joblib'
    >>> ctgan.dump(model_path, overwrite=True)

To load a previously dumped model, just call the
:class:`ctgan.synthesizer.CTGANSynthesizer` constructor with the path pointing
to the model file. Remember that you cannot train the models after loading them
from the disk, as the Critic network is not saved.

    >>> from ctgan.synthesizer import CTGANSynthesizer
    >>> model_path = '/path/to/your/model/file.joblib'
    >>> ctgan = CTGANSynthesizer(file_path=model_path)
    >>> samples = ctgan.sample(1000)

Command-line interface
----------------------

**CTGAN** comes with a command line interface that allows modeling and sampling
data without the need to write any Python code. This is available under the
`ctgan-tf` command that will have been set up in your system upon installing
the `ctgan-tf` Python package::

    $ ctgan-tf -h
    usage: ctgan-tf [-h] [-f FILE_PATH] [-l LOG_DIR] [-z Z_DIM] [-p PAC]
                    [-g GEN_DIMS] [-c CRT_DIMS] [-s L2_SCALE] [-b BATCH_SIZE]
                    [-w GP_LAMBDA] [-t TAU] [-e EPOCHS] [-d DISCRETE_COLUMNS]
                    [-i TRAINING_DATA] [-o OUTPUT_MODEL]
                    num_samples output_data

    CTGAN Command Line Interface

    positional arguments:
      num_samples           Number of rows to sample.
      output_data           Path of the synthetic output file

    optional arguments:
      -h, --help            show this help message and exit
      -f FILE_PATH, --file_path FILE_PATH
                            File path to where a CTGAN Synthesizer model is
                            stored. If this value is None, a new instance will be
                            created.
      -l LOG_DIR, --log_dir LOG_DIR
                            Directory to where log files will be stored.
      -z Z_DIM, --z_dim Z_DIM
                            Embedding dimension of fake samples.
      -p PAC, --pac PAC     Size of Pac framework.
      -g GEN_DIMS, --gen_dims GEN_DIMS
                            Comma separated list of Generator layer sizes.
      -c CRT_DIMS, --crt_dims CRT_DIMS
                            Comma separated list of Critic layer sizes.
      -s L2_SCALE, --l2_scale L2_SCALE
                            L2 regularization of the Generator optimizer.
      -b BATCH_SIZE, --batch_size BATCH_SIZE
                            Training batch size.
      -w GP_LAMBDA, --gp_lambda GP_LAMBDA
                            Gradient Penalty lambda.
      -t TAU, --tau TAU     Gumbel-Softmax non-negative scalar temperature.
      -e EPOCHS, --epochs EPOCHS
                            Number of training epochs
      -d DISCRETE_COLUMNS, --discrete_columns DISCRETE_COLUMNS
                            Comma separated list of discrete columns, no
                            whitespaces
      -i TRAINING_DATA, --training_data TRAINING_DATA
                            Path to training data
      -o OUTPUT_MODEL, --output_model OUTPUT_MODEL
                            Path of the model output file

.. note:: Recall that the input data should always be in CSV format, and the
          discrete columns are an argument of the CLI.