-----
About
-----

.. raw:: html

    <p align="center">
        <img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“sdv-dev” />
        <i>An open source project from Data to AI Lab at MIT.</i>
    </p>

This project is based on the PyTorch implementation of the **Conditional
Tabular Generative Adversarial Network**, publicly available on this GitHub
`repository <https://github.com/sdv-dev/CTGAN>`_.

The CTGAN was originally presented in a NeurIPS paper *Modeling Tabular data
using Conditional GAN* :cite:`xu2019modeling`, to try to "generate synthetic
tabular data with high fidelity", and it was a research conducted by the
*Data to AI Lab* at MIT.

Since there was already a R port of the original PyTorch code, it was missing
a TensorFlow implementation of the latter. That's the rationale behind this
project.

.. note:: Most of the code, specially from :mod:`ctgan.data_modules`,
          was reused from the original repository. The goal was to only migrate
          what was necessary to make it run on TensorFlow 2.1. Additionally,
          some of the infrastructure code was also heavily inspired the
          `imbalanced-learn <https://imbalanced-learn.org/>`_ project.