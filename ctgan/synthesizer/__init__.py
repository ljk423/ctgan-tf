"""
The :mod:`ctgan.synthesizer` module contains the definition of the CTGAN
Synthesizer - the "main" class for training a Conditional Tabular GAN, as well
as helper methods to control the execution of the training steps.

It contains the Tensorflow 2 implementation of the work published in
*Modeling Tabular data using Conditional GAN* :cite:`xu2019modeling`.
The original PyTorch implementation can be found in the authors'
`GitHub repository <https://github.com/sdv-dev/CTGAN>`_.
"""

from ._synthesizer import CTGANSynthesizer

__author__ = 'Pedro Martins'
__email__ = 'pbmartins@ua.pt'
__version__ = '1.0.0'

__all__ = [
    'CTGANSynthesizer'
]
