"""
Bayesian Gaussian Mixture model definition.
"""
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


class BGM(BayesianGaussianMixture):
    """Bayesian Gaussian Mixture model.

    Extends :class:`sklearn.mixture.BayesianGaussianMixture`.
    """

    def __eq__(self, other):
        try:
            np.testing.assert_equal(self.__dict__, other.__dict__)
            return True
        except AssertionError:
            return False
