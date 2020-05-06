#! /usr/bin/env python
"""Toolbox for TensorFlow 2.1 CTGAN implementation."""

import os

from setuptools import find_packages, setup

# get __version__ from _version.py
ver_file = os.path.join('ctgan', '_version.py')
with open(ver_file) as f:
    exec(f.read())

DISTNAME = 'ctgan-tf'
DESCRIPTION = 'TensorFlow 2.1 implementation of Conditional Tabular GAN.'
with open('README.md') as readme_file:
    LONG_DESCRIPTION = readme_file.read()
MAINTAINER = 'Pedro Martins'
MAINTAINER_EMAIL = 'pbmartins@ua.pt'
URL = 'https://github.com/pbmartins/ctgan-tf'
LICENSE = 'MIT'
DOWNLOAD_URL = 'https://github.com/pbmartins/ctgan-tf'
VERSION = __version__
CLASSIFIERS = ['Intended Audience :: Science/Research',
               'Intended Audience :: Developers',
               'License :: OSI Approved',
               'Programming Language :: Python',
               'Topic :: Software Development',
               'Topic :: Scientific/Engineering',
               'Operating System :: Microsoft :: Windows',
               'Operating System :: POSIX',
               'Operating System :: Unix',
               'Operating System :: MacOS',
               'Programming Language :: Python :: 3.7']
INSTALL_REQUIRES = [
    'tensorflow<2.2,>=2.1.0',
    'tensorflow-probability<1.0,>=0.9.0',
    'scikit-learn<0.23,>=0.21',
    'numpy<2,>=1.17.4',
    'pandas<1.0.2,>=1.0',
    'tqdm<4.44,>=4.43'
]
EXTRAS_REQUIRE = {
    'tests': [
        'pytest>=5.4.0',
        'pytest-cov>=2.8.0'],
    'dev': [
        # general
        'bump2version>=1.0.0',
        'pip>=20.0.0',

        # style check
        'flake8>=3.7.9',
        'pylint-fail-under>=0.3.0',

        # tests
        'pytest>=5.4.0',
        'pytest-cov>=2.8.0',

        # distribute on PyPI
        'twine>=3.1.1',
        'wheel>=0.30.0',

        # Advanced testing
        'coverage>=5.1',
    ],
    'docs': [
        'm2r>=0.2.0',
        'Sphinx<3.0.0,>=2.4.4',
        'sphinx_rtd_theme>=0.4.3',
        'autodocsumm>=0.1.10',
        'numpydoc<1.0.0,>=0.9.2',
        'sphinxcontrib-bibtex==1.0.0'
    ]
}
ENTRY_POINTS = {
    'console_scripts': ['ctgan-tf=ctgan.__main__:cli.cli']
}

setup(
    name=DISTNAME,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    download_url=DOWNLOAD_URL,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    zip_safe=False,  # the package can run out of an .egg file
    classifiers=CLASSIFIERS,
    entry_points=ENTRY_POINTS,
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    tests_requires=EXTRAS_REQUIRE['tests'],
    extras_require=EXTRAS_REQUIRE,
    python_requires='>=3.7',
)
