######################
ctgan-tf API
######################

This is the full API documentation of the `ctgan-tf` toolbox.

.. _synthesizer_ref:

:mod:`ctgan.synthesizer`: CTGAN Synthesizer
======================================================

.. automodule:: ctgan.synthesizer
    :no-members:
    :no-inherited-members:

.. currentmodule:: ctgan

.. autosummary::
   :toctree: generated/
   :template: class.rst

   synthesizer.CTGANSynthesizer

.. _data_modules_ref:

:mod:`ctgan.data_modules`: Data modules
========================================================================

.. automodule:: ctgan.data_modules
   :no-members:
   :no-inherited-members:

.. currentmodule:: ctgan

.. autosummary::
   :toctree: generated/
   :template: class.rst

   data_modules.ConditionalGenerator
   data_modules.DataTransformer
   data_modules.DataSampler

.. _models_ref:

:mod:`ctgan.models`: Neural Network and Transformer Models
==========================================================

.. automodule:: ctgan.models
    :no-members:
    :no-inherited-members:

.. currentmodule:: ctgan

Neural Networks
---------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   models.Critic
   models.Generator

Transformer models
------------------
.. autosummary::
   :toctree: generated/
   :template: class.rst

   models.BGM
   models.OHE

.. _layers_ref:

:mod:`ctgan.layers`: Neural Network Layers
====================================================

.. automodule:: ctgan.layers
    :no-members:
    :no-inherited-members:

.. currentmodule:: ctgan

.. autosummary::
   :toctree: generated/
   :template: class.rst

   layers.ResidualLayer
   layers.GenActivation

Utilities
---------

.. autosummary::
   :toctree: generated/
   :template: function.rst

   layers.init_bounded

.. _losses_ref:

:mod:`ctgan.losses`: Model losses
====================================================

.. automodule:: ctgan.losses
    :no-members:
    :no-inherited-members:

.. currentmodule:: ctgan

.. autosummary::
   :toctree: generated/
   :template: function.rst

   losses.conditional_loss
   losses.gradient_penalty

.. _utils_ref:

:mod:`ctgan.cli`: Command line interface
========================================================================

.. automodule:: ctgan.cli
   :no-members:
   :no-inherited-members:

.. currentmodule:: ctgan

.. autosummary::
   :toctree: generated/
   :template: function.rst

    cli.load_demo
    cli.cli

:mod:`ctgan.utils`: Utils
========================================================================

.. automodule:: ctgan.utils
   :no-members:
   :no-inherited-members:

.. currentmodule:: ctgan

.. autosummary::
   :toctree: generated/
   :template: class.rst

    utils.ProgressBar

.. autosummary::
   :toctree: generated/
   :template: function.rst

    utils.generate_data
