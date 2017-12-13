import unittest

from glmnet.errors import _check_error_flag


class TestErrors(unittest.TestCase):

    def test_zero_jerr(self):
        # This should not raise any warnings or exceptions.
        _check_error_flag(0)

    def test_convergence_err(self):
        msg = ("Model did not converge for smaller values of lambda, "
               "returning solution for the largest 75 values.")
        with self.assertWarns(RuntimeWarning, msg=msg):
            _check_error_flag(-76)

    def test_zero_var_err(self):
        msg = "All predictors have zero variance (glmnet error no. 7777)."
        with self.assertRaises(ValueError, msg=msg):
            _check_error_flag(7777)

    def test_all_negative_rel_penalty(self):
        msg = ("At least one value of relative_penalties must be positive, "
               "(glmnet error no. 10000).")
        with self.assertRaises(ValueError, msg=msg):
            _check_error_flag(10000)

    def test_memory_allocation_err(self):
        msg = "Memory allocation error (glmnet error no. 1234)."
        with self.assertRaises(RuntimeError, msg=msg):
            _check_error_flag(1234)

    def test_other_fatal_err(self):
        msg = "Fatal glmnet error no. 7778."
        with self.assertRaises(RuntimeError, msg=msg):
            _check_error_flag(7778)

    def test_class_prob_close_to_1(self):
        msg = "Probability for class 2 close to 0."
        with self.assertRaises(ValueError, msg=msg):
            _check_error_flag(8002)

    def test_class_prob_close_to_0(self):
        msg = "Probability for class 4 close to 0."
        with self.assertRaises(ValueError, msg=msg):
            _check_error_flag(8004)

    def test_predicted_class_close_to_0_or_1(self):
        msg = "Predicted probability close to 0 or 1 for lambda no. 7."
        with self.assertWarns(RuntimeWarning, msg=msg):
            _check_error_flag(-20007)

    def test_did_not_converge(self):
        msg = "Solver did not converge (glmnet error no. 90000)."
        with self.assertRaises(ValueError, msg=msg):
            _check_error_flag(90000)

