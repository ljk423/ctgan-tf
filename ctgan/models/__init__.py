"""
The :mod:`ctgan.models` module contains the definition of the neural networks
used as part of the Generative Adversarial Network built in
:class:`ctgan.synthesizer.CTGANSynthesizer`.

They are built using TensorFlow 2, and make use of the custom layers available
in :mod:`ctgan.layers`.

For further details, please consult sections 4.4 of :cite:`xu2019modeling`.

This module also contains the extension of some :mod:`sklearn` models, used in
:class:`ctgan.data_modules.DataTransformer`.
"""

from ._critic import Critic
from ._generator import Generator
from ._bgm import BGM
from ._ohe import OHE

__all__ = [
    'Critic',
    'Generator',
    'BGM',
    'OHE'
]
