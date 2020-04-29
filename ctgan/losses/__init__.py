"""
The :mod:`ctgan.losses` module contains the definition of custom loss
functions.

For further details, please consult sections 4.3 of :cite:`xu2019modeling`, and
WGAN-GP paper :cite:`gulrajani2017improved`.
"""

from ._conditional_loss import conditional_loss
from ._gradient_penalty import gradient_penalty

__all__ = [
    'conditional_loss',
    'gradient_penalty'
]
