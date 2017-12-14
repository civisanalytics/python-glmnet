import unittest

import numpy as np

from scipy.sparse import csr_matrix

from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
from sklearn.utils import estimator_checks
from sklearn.utils.testing import ignore_warnings

from util import sanity_check_regression

from glmnet import ElasticNet




class TestElasticNet(unittest.TestCase):

    def setUp(self):
        np.random.seed(488881)
        x, y = make_regression(n_samples=1000, random_state=561)
        x_sparse = csr_matrix(x)

        x_wide, y_wide = make_regression(n_samples=100, n_features=150,
                                         random_state=1105)
        x_wide_sparse = csr_matrix(x_wide)

        self.inputs = [(x,y), (x_sparse, y), (x_wide, y_wide),
                       (x_wide_sparse, y_wide)]
        self.alphas = [0., 0.25, 0.50, 0.75, 1.]
        self.n_splits = [-1, 0, 5]
        self.scoring = [
            "r2",
            "mean_squared_error",
            "mean_absolute_error",
            "median_absolute_error",
        ]

    @ignore_warnings(RuntimeWarning)
    def test_estimator_interface(self):
        estimator_checks.check_estimator(ElasticNet)

    def test_with_defaults(self):
        m = ElasticNet(random_state=2821)
        for x, y in self.inputs:
            m = m.fit(x, y)
            sanity_check_regression(m, x)

            # check selection of lambda_best
            self.assertTrue(m.lambda_best_inx_ <= m.lambda_max_inx_)

            # check full path predict
            p = m.predict(x, lamb=m.lambda_path_)
            self.assertEqual(p.shape[-1], m.lambda_path_.size)

    def test_with_single_var(self):
        x = np.random.rand(500,1)
        y = (1.3 * x).ravel()

        m = ElasticNet(random_state=449065)
        m = m.fit(x, y)
        self.check_r2_score(y, m.predict(x), 0.90)

    def test_with_no_predictor_variance(self):
        x = np.ones((500, 1))
        y = np.random.rand(500)

        m = ElasticNet(random_state=561)
        msg = "All predictors have zero variance (glmnet error no. 7777)."
        with self.assertRaises(ValueError, msg=msg):
            m.fit(x, y)

    def test_relative_penalties(self):
        m1 = ElasticNet(random_state=4328)
        m2 = ElasticNet(random_state=4328)
        for x, y in self.inputs:
            p = x.shape[1]

            # m1 no relative penalties applied
            m1.fit(x, y)

            # find the nonzero indices from LASSO
            nonzero = np.nonzero(m1.coef_)

            # unpenalize those nonzero coefs
            penalty = np.repeat(1, p)
            penalty[nonzero] = 0

            # refit the model with the unpenalized coefs
            m2.fit(x, y, relative_penalties=penalty)

            # verify that the unpenalized coef ests exceed the penalized ones
            # in absolute value
            assert(np.all(np.abs(m1.coef_) <= np.abs(m2.coef_)))

    def test_alphas(self):
        x, y = self.inputs[0]
        for alpha in self.alphas:
            m = ElasticNet(alpha=alpha, random_state=2465)
            m = m.fit(x, y)
            self.check_r2_score(y, m.predict(x), 0.90, alpha=alpha)

    def test_n_splits(self):
        x, y = self.inputs[0]
        for n in self.n_splits:
            m = ElasticNet(n_splits=n, random_state=6601)
            if n > 0 and n < 3:
                with self.assertRaisesRegexp(ValueError,
                                             "n_splits must be at least 3"):
                    m = m.fit(x, y)
            else:
                m = m.fit(x, y)
                sanity_check_regression(m, x)

    def test_cv_scoring(self):
        x, y = self.inputs[0]
        for method in self.scoring:
            m = ElasticNet(scoring=method, random_state=1729)
            m = m.fit(x, y)
            self.check_r2_score(y, m.predict(x), 0.90, scoring=method)

    def test_predict_without_cv(self):
        x, y = self.inputs[0]
        m = ElasticNet(n_splits=0, random_state=340561)
        m = m.fit(x, y)

        # should not make prediction unless value is passed for lambda
        with self.assertRaises(ValueError):
            m.predict(x)

    def test_coef_interpolation(self):
        x, y = self.inputs[0]
        m = ElasticNet(n_splits=0, random_state=1729)
        m = m.fit(x, y)

        # predict for a value of lambda between two values on the computed path
        lamb_lo = m.lambda_path_[1]
        lamb_hi = m.lambda_path_[2]

        # a value not equal to one on the computed path
        lamb_mid = (lamb_lo + lamb_hi) / 2.0

        pred_lo = m.predict(x, lamb=lamb_lo)
        pred_hi = m.predict(x, lamb=lamb_hi)
        pred_mid = m.predict(x, lamb=lamb_mid)

        self.assertFalse(np.allclose(pred_lo, pred_mid))
        self.assertFalse(np.allclose(pred_hi, pred_mid))

    def test_lambda_clip_warning(self):
        x, y = self.inputs[0]
        m = ElasticNet(n_splits=0, random_state=1729)
        m = m.fit(x, y)

        # we should get a warning when we ask for predictions at values of
        # lambda outside the range of lambda_path_
        with self.assertWarns(RuntimeWarning):
            # note, lambda_path_ is in decreasing order
            m.predict(x, lamb=m.lambda_path_[0] + 1)

        with self.assertWarns(RuntimeWarning):
            m.predict(x, lamb=m.lambda_path_[-1] - 1)

    def check_r2_score(self, y_true, y, at_least, **other_params):
        score = r2_score(y_true, y)
        msg = "expected r2 of {}, got: {}, with: {}".format(at_least, score, other_params)
        self.assertTrue(score > at_least, msg)

    def test_random_state_cv(self):
        random_state = 133
        m = ElasticNet(random_state=random_state)
        x, y = self.inputs[0]
        m.fit(x, y)
        print(dir(m.cv))
        assert m.cv.random_state == random_state


if __name__ == "__main__":
    unittest.main()
