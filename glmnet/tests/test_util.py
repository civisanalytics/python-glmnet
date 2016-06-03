import unittest

import numpy as np
from numpy.testing import assert_warns

from glmnet.util import _interpolate_model


class TestUtils(unittest.TestCase):

    def test_interpolate_model_intercept_only(self):
        lambda_path = np.array((0.99,))
        coef_path = np.random.random(size=(5, 1))
        intercept_path = np.random.random(size=(1,))

        # would be nice to use assertWarnsRegex to check the message, but this
        # fails due to http://bugs.python.org/issue20484
        assert_warns(RuntimeWarning, _interpolate_model, lambda_path,
                     coef_path, intercept_path, 0.99)


if __name__ == "__main__":
    unittest.main()
