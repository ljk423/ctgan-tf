"""
The :mod:`ctgan.data_modules` module contains the definition of classes used by
:class:`ctgan.synthesizer.CTGANSynthesizer` during training.

These classes mainly provide data transformation and sampling. For further
details, please consult sections 4.2-4.3 of :cite:`xu2019modeling`.
"""

from ._conditional import ConditionalGenerator
from ._sampler import Sampler
from ._transformer import DataTransformer

__all__ = [
    'ConditionalGenerator',
    'Sampler',
    'DataTransformer'
]
