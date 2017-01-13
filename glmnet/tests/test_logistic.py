import itertools
import unittest

from nose.tools import ok_, eq_

import numpy as np

from scipy.sparse import csr_matrix

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.utils import estimator_checks
from sklearn.utils.testing import ignore_warnings

from util import sanity_check_logistic

from glmnet import LogitNet


class TestLogitNet(unittest.TestCase):

    def setUp(self):
        np.random.seed(488881)
        # binomial
        x, y = make_classification(n_samples=300, random_state=6601)
        x_sparse = csr_matrix(x)

        x_wide, y_wide = make_classification(n_samples=100, n_features=150,
                                             random_state=8911)
        x_wide_sparse = csr_matrix(x_wide)
        self.binomial = [(x,y), (x_sparse, y), (x_wide, y_wide),
                         (x_wide_sparse, y_wide)]

        # multinomial
        x, y = make_classification(n_samples=400, n_classes=3, n_informative=15,
                                   n_features=25, random_state=10585)
        x_sparse = csr_matrix(x)

        x_wide, y_wide = make_classification(n_samples=400, n_classes=3,
                                             n_informative=15, n_features=500,
                                             random_state=15841)
        x_wide_sparse = csr_matrix(x_wide)
        self.multinomial = [(x, y), (x_sparse, y), (x_wide, y_wide),
                            (x_wide_sparse, y_wide)]

        self.alphas = [0., 0.25, 0.50, 0.75, 1.]
        self.n_splits = [-1, 0, 5]
        self.scoring = [
            "accuracy",
            "roc_auc",
            "average_precision",
            "log_loss",
            "precision_macro",
            "precision_micro",
            "precision_weighted",
            "f1_micro",
            "f1_macro",
            "f1_weighted",
        ]
        self.multinomial_scoring = [
            "accuracy",
            "log_loss",
            "precision_macro",
            "precision_micro",
            "precision_weighted",
            "f1_micro",
            "f1_macro",
            "f1_weighted"
        ]

    @ignore_warnings(RuntimeWarning)  # ignore convergence warnings from glmnet
    def test_estimator_interface(self):
        estimator_checks.check_estimator(LogitNet)

    def test_with_defaults(self):
        m = LogitNet(random_state=29341)
        for x, y in itertools.chain(self.binomial, self.multinomial):
            m = m.fit(x, y)
            sanity_check_logistic(m, x)

            # check selection of lambda_best
            ok_(m.lambda_best_inx_ <= m.lambda_max_inx_)

            # check full path predict
            p = m.predict(x, lamb=m.lambda_path_)
            eq_(p.shape[-1], m.lambda_path_.size)

    def test_alphas(self):
        x, y = self.binomial[0]
        for alpha in self.alphas:
            m = LogitNet(alpha=alpha, random_state=41041)
            m = m.fit(x, y)
            check_accuracy(y, m.predict(x), 0.85, alpha=alpha)

    def test_relative_penalties(self):
        x, y = self.binomial[0]
        p = x.shape[1]

        # m1 no relative penalties applied
        m1 = LogitNet(alpha=1)
        m1.fit(x, y)

        # find the nonzero indices from LASSO
        nonzero = np.nonzero(m1.coef_[0])

        # unpenalize those nonzero coefs
        penalty = np.repeat(1, p)
        penalty[nonzero] = 0

        # refit the model with the unpenalized coefs
        m2 = LogitNet(alpha=1)
        m2.fit(x, y, relative_penalties=penalty)

        # verify that the unpenalized coef ests exceed the penalized ones
        # in absolute value
        assert(np.all(np.abs(m1.coef_[0]) <= np.abs(m2.coef_[0])))

    def test_n_splits(self):
        x, y = self.binomial[0]
        for n in self.n_splits:
            m = LogitNet(n_splits=n, random_state=46657)
            if n > 0 and n < 3:
                with self.assertRaisesRegexp(ValueError,
                                             "n_splits must be at least 3"):
                    m = m.fit(x, y)
            else:
                m = m.fit(x, y)
                sanity_check_logistic(m, x)

    def test_cv_scoring(self):
        x, y = self.binomial[0]
        for method in self.scoring:
            m = LogitNet(scoring=method, random_state=52633)
            m = m.fit(x, y)
            check_accuracy(y, m.predict(x), 0.85, scoring=method)

    def test_cv_scoring_multinomial(self):
        x, y = self.multinomial[0]
        for method in self.scoring:
            m = LogitNet(scoring=method, random_state=488881)

            if method in self.multinomial_scoring:
                m = m.fit(x, y)
                check_accuracy(y, m.predict(x), 0.65, scoring=method)
            else:
                with self.assertRaises(ValueError):
                    m.fit(x, y)

    def test_predict_without_cv(self):
        x, y = self.binomial[0]
        m = LogitNet(n_splits=0, random_state=399001)
        m = m.fit(x, y)

        # should not make prediction unless value is passed for lambda
        with self.assertRaises(ValueError):
            m.predict(x)

    def test_coef_interpolation(self):
        x, y = self.binomial[0]
        m = LogitNet(n_splits=0, random_state=561)
        m = m.fit(x, y)

        # predict for a value of lambda between two values on the computed path
        lamb_lo = m.lambda_path_[1]
        lamb_hi = m.lambda_path_[2]

        # a value not equal to one on the computed path
        lamb_mid = (lamb_lo + lamb_hi) / 2.0

        pred_lo = m.predict_proba(x, lamb=lamb_lo)
        pred_hi = m.predict_proba(x, lamb=lamb_hi)
        pred_mid = m.predict_proba(x, lamb=lamb_mid)

        self.assertFalse(np.allclose(pred_lo, pred_mid))
        self.assertFalse(np.allclose(pred_hi, pred_mid))

    def test_lambda_clip_warning(self):
        x, y = self.binomial[0]
        m = LogitNet(n_splits=0, random_state=1729)
        m = m.fit(x, y)

        with self.assertWarns(RuntimeWarning):
            m.predict(x, lamb=m.lambda_path_[0] + 1)

        with self.assertWarns(RuntimeWarning):
            m.predict(x, lamb=m.lambda_path_[-1] - 1)

    def test_single_class_exception(self):
        x, y = self.binomial[0]
        y = np.ones_like(y)
        m = LogitNet()

        with self.assertRaises(ValueError) as e:
            m.fit(x, y)

        self.assertEqual("Training data need to contain at least 2 classes.",
                         str(e.exception))


def check_accuracy(y, y_hat, at_least, **other_params):
    score = accuracy_score(y, y_hat)
    msg = "expected accuracy of {}, got: {} with {}".format(at_least, score, other_params)
    ok_(score > at_least, msg)


if __name__ == "__main__":
    unittest.main()
