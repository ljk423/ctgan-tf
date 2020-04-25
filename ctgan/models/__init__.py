"""
The :mod:`ctgan.models` module contains the definition of the neural networks
used as part of the Generative Adversarial Network built in
:class:`ctgan.synthesizer.CTGANSynthesizer`.

They are built using TensorFlow 2, and make use of the custom layers available
in :mod:`ctgan.layers`.

For further details, please consult sections 4.4 of :cite:`xu2019modeling`.
"""

from ._critic import Critic
from ._generator import Generator

__all__ = [
    'Critic',
    'Generator'
]
