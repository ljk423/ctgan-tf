
.. image:: images/tf-ctgan.png
    :width: 200px
    :align: center


----------------
TensorFlow CTGAN
----------------

.. raw:: html

    <p align="center">
        <a class="reference external image-reference" href="https://github.com/pbmartins/ctgan-tf/blob/master/LICENSE">
            <img alt="MIT License" src="https://img.shields.io/pypi/v/ctgan-tf.svg">
        </a>
        <a class="reference external image-reference" href="https://pypi.python.org/pypi/ctgan-tf">
            <img alt="PyPI Shield" src="https://img.shields.io/pypi/v/ctgan-tf.svg">
        </a>
        <a class="reference external image-reference" href="https://travis-ci.com/pbmartins/ctgan-tf">
            <img alt="Build Status" src="https://travis-ci.com/pbmartins/ctgan-tf.svg?token=ES61mh8SK9WT5Hr1iCs7&branch=master">
        </a>
        <a class="reference external image-reference" href="https://codecov.io/gh/pbmartins/ctgan-tf">
            <img alt="Coverage Status" src="https://codecov.io/gh/pbmartins/ctgan-tf/branch/stable/graph/badge.svg?token=BXT0G35Y9Q">
        </a>
    </p>


Welcome to the `ctgan-tf` documentation.

---------------
Getting started
---------------

Tensorflow 2.1 implementation of a Conditional Tabular Generative Adversarial
Network. CTGAN is a GAN-based data synthesizer that can "generate synthetic
tabular data with high fidelity".

This model was originally designed by the *Data to AI Lab at MIT* team, and
it was published in their NeurIPS paper `Modeling Tabular data using
Conditional GAN <https://arxiv.org/abs/1907.00503>`_.

For more information regarding this work, and to access the original PyTorch
implementation provided by the authors,
please refer to their GitHub repository and their documentation:

* Documentation: https://sdv-dev.github.io/CTGAN
* Homepage: https://github.com/sdv-dev/CTGAN

Install
-------

Requirements
^^^^^^^^^^^^

* `tensorflow` (<2.2,>=2.1.0)
* `tensorflow-probability` (<1.0,>=0.9.0)
* `scikit-learn` (<0.23,>=0.21)
* `numpy` (<2,>=1.17.4)
* `pandas` (<1.0.2,>=1.0)
* `tqdm` (<4.44,>=4.43)

Install
^^^^^^^

You can either install `ctgan-tf` through the PyPI package::

    $ pip3 install ctgan-tf

Or by cloning this repository and copying the `ctgan` folder to your project
folder.

Please check the `Contributing Guide <contributing.html>`_ if you want to install
the development version.

`User Guide <user_guide.html>`_
-------------------------------

This user guide will help you start using this toolbox as fast as possible.


`API Documentation <api.html>`_
-------------------------------

The exact API of all functions and classes, as given in the
doctring. The API documents expected types and allowed features for
all functions, and all parameters available for the algorithms.


`History <history.html>`_
------------------------------

Log of the ctgan-tf history.

`About ctgan-tf <about.html>`_
--------------------------------------

Just to know about this project came to life.

See the `README <https://github.com/pbmartins/ctgan-tf/blob/master/README.md>`_
for more information.

