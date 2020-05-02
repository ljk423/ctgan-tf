"""
Top-level module for CTGAN.
"""
from . import data_modules
from . import layers
from . import losses
from . import models
from . import synthesizer
from . import cli
from . import utils
from ._version import __version__, __release__

__all__ = [
    'data_modules',
    'layers',
    'losses',
    'models',
    'synthesizer',
    'cli',
    'utils',
    '__version__',
    '__release__'
]
