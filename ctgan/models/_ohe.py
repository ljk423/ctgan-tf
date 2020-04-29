"""
One-Hot encoder definition.
"""
import numpy as np
from sklearn.preprocessing import OneHotEncoder


class OHE(OneHotEncoder):
    """One-Hot encoder.

    Extends :class:`sklearn.preprocessing.OneHotEncoder`.
    """

    def __eq__(self, other):
        try:
            np.testing.assert_equal(self.__dict__, other.__dict__)
            return True
        except AssertionError:
            return False
