"""
Top-level module for CTGAN.
"""
from . import data_modules
from . import layers
from . import losses
from . import models
from . import synthesizer
from . import cli

__version__ = '1.0.0'
__release__ = '1.0.0'
__all__ = [
    'data_modules',
    'layers',
    'losses',
    'models',
    'synthesizer',
    'cli'
]
