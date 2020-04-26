"""
The :mod:`ctgan.cli` provides a command-line interface for using the `ctgan-tf`
toolbox.
"""

from ._load_demo import load_demo
from ._cli import cli

__all__ = [
    'load_demo',
    'cli'
]
