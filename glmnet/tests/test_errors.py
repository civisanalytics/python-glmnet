import unittest

from glmnet.errors import _check_glmnet_error_flag


class TestErrors(unittest.TestCase):

    def test_zero_jerr(self):
        # This should not raise any warnings or exceptions.
        _check_glmnet_error_flag(0, n_lambda=100)

    def test_convergence_err(self):
        msg = ("Model did not converge for smaller values of lambda, "
               "returning solution for the largest 75 values.")
        with self.assertWarns(RuntimeWarning, msg=msg):
            _check_glmnet_error_flag(-76, n_lambda=100)

    def test_zero_var_err(self):
        msg = "All predictors have zero variance (glmnet error no. 7777)."
        with self.assertRaises(ValueError, msg=msg):
            _check_glmnet_error_flag(7777, n_lambda=100)

    def test_all_negative_rel_penalty(self):
        msg = ("At least one value of relative_penalties must be positive, "
               "(glmnet error no. 10000).")
        with self.assertRaises(ValueError, msg=msg):
            _check_glmnet_error_flag(10000, n_lambda=100)

    def test_memory_allocation_err(self):
        msg = "Memory allocation error (glmnet error no. 1234)."
        with self.assertRaises(RuntimeError, msg=msg):
            _check_glmnet_error_flag(1234, n_lambda=100)

    def test_other_fatal_err(self):
        msg = "Fatal glmnet error no. 8888."
        with self.assertRaises(RuntimeError, msg=msg):
            _check_glmnet_error_flag(8888, msg)
