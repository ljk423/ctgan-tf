"""
The :mod:`ctgan.layers` module contains the definition of custom neural networks
layers, and helper methods to initialize their weights and bias.

For further details, please consult sections 4.4 of :cite:`xu2019modeling`.
"""

from ._layer_utils import init_bounded
from ._residual import ResidualLayer
from ._gen_activation import GenActivation

__all__ = [
    'init_bounded',
    'ResidualLayer',
    'GenActivation'
]
