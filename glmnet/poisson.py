import numpy as np

from scipy.sparse import issparse, csc_matrix
from scipy import stats

from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets

from _glmnet import spfishnet, fishnet, solns
from .util import (_fix_lambda_path,
                   _check_glmnet_error_flag,
                   _check_user_lambda,
                   _interpolate_model,
                   _score_lambda_path)


class PoissonNet(BaseEstimator):
    """Poisson regression with elastic net penalty.

    This is a wrapper for the glmnet function fishnet.

    Parameters
    ----------
    alpha : float, default 1
        The alpha parameter, 0 <= alpha <= 1, 0 for ridge, 1 for lasso

    n_lambda : int, default 100
        Maximum number of lambda values to compute

    min_lambda_ratio : float, default 1e-4
        In combination with n_lambda, the ratio of the smallest and largest
        values of lambda computed.

    lambda_path : array, default None
        Instead of supplying n_lambda, provide an array of specific values to
        compute. The specified values must be in decreasing order. When None,
        the path of lambda values will be determined automatically. A maximum
        of `n_lambda` values will be computed.

    standardize : bool, default True
        Standardize input features prior to fitting. The final coefficients
        will be on the scale of the original data regardless of the value of
        standardize.

    fit_intercept : bool, default True
        Include an intercept term in the model

    cut_point : float, default 1
        The cut point to use for selecting lambda_best.
            arg_max lambda cv_score(lambda) >= \
                cv_score(lambda_max) - cut_point * standard_error(lambda_max)

    n_splits : int, default 3
        Number of cross validation folds for computing performance metrics and
        determining `lambda_best_` and `lambda_max_`. If non-zero, must be
        at least 3.

    scoring : string, callable, or None, default None
        Scoring method for model selection during cross validation. When None,
        defaults to deviance. Alternatively, supply a function or callable
        object with the following signature ``scorer(estimator, X, y)``.
        Note, the scoring function affects the selection of `lambda_best_` and
        `lambda_max_`, fitting the same data with different scoring methods
        will result in the selection of different models.

    n_jobs : int, default 1
        Maximum number of threads for computing cross validation metrics.

    tol : float, default 1e-7
        Convergence tolerance

    max_iter : int, default 100000
        Maximum passes over the data.

    random_state : number, default None
        Seed for the random number generator. The glmnet solver is not
        deterministic, this seed is used for determining the cv folds.

    verbose : bool, default False
        When Truem some warnings and log messages are suppressed.

    Attributes
    ----------
    n_lambda_ : int
        The number of lambda values found by glmnet. Note, this may be less
        than the number specified via n_lambda.

    lambda_path_ : array, shape (n_lambda_,)
        The values of lambda found by glmnet, in decreasing order.

    coef_path_ : array, shape (n_features, n_lambda_)
        The set of coefficients for each value of lambda in lambda_path_

    coef_ : array, shape (n_features,)
        The coefficients corresponding to lambda_best_

    intercept_ : float
        The intercept corresponding to lambda_best_

    intercept_path_ : array, shape (n_lambda_,)
        The intercept for each value of lambda in lambda_path_

    cv_mean_score_ : array, shape (n_lambda_,)
        The mean cv score for each value of lambda. This will be set by fit_cv

    cv_standard_error_ : array, shape (n_lambda,)
        The standard error of the mean cv score for each value of lambda, this
        will be set by fit_cv.

    lambda_max_ : float
        The value of lambda that gives the best performs in cross validation

    lambda_best_ : float
        The largest value of lambda which is greater than lambda_max_ and
        performs within cut_point * standard error of lambda_max_
    """
    def __init__(self, alpha=1, n_lambda=100, min_lambda_ratio=1e-4,
                 lambda_path=None, standardize=True, fit_intercept=True,
                 cut_point=1.0, n_splits=3, scoring=True, n_jobs=1, tol=1e-7,
                 max_iter=100000, random_state=None, verbose=False):

        self.alpha = alpha
        self.n_lambda = n_lambda
        self.min_lambda_ratio = min_lambda_ratio
        self.lambda_path = lambda_path
        self.standardize = standardize
        self.fit_intercept = fit_intercept
        self.cut_point = cut_point
        self.n_splits = n_splits
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, relative_penalties=None):
        """Fit the model to training data. If n_splits > 1 also run n-fold cross
        validation on all values in lambda_path.

        The model will be fit n+1 times. On the first pass, the lambda_path
        will be determined, on the remaining passes, the model performance for
        each value of lambda. After cross validation, the attribute
        `cv_mean_score_` will contain the mean score over all folds for each
        value of lambda, and `cv_standard_error_` will contain the standard
        error of `cv_mean_score_` for each value of lambda. The value of lambda
        which achieves the best performance in cross validation will be saved
        to `lambda_max_` additionally, the largest value of lambda s.t.:
            cv_score(l) >= cv_score(lambda_max_) -\
                           cut_point * standard_error(lambda_max_)
        will be saved to `lambda_best_`.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input features

        Y : array, shape (n_samples,)
            Target values

        sample_weight : array, shape (n_samples,)
            Optional weight vector for observations

        relative_penalties: array, shape (n_features,)
            Optional relative weight vector for penalty.
            0 entries remove penalty.

        Returns
        -------
        self : object
            Returns self.
        """
        if self.alpha > 1 or self.alpha < 0:
            raise ValueError("alpha must be between 0 and 1")

        X, y = check_X_y(X, y, accept_sparse='csr', ensure_min_samples=2)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])

        self._fit(X, y, sample_weight, relative_penalties)

        # score each model on the path of lambda values found by glmnet and
        # select the best scoring
        if self.n_splits >= 3:
            cv_scores = _score_lambda_path(self, X, y, sample_weight,
                                           relative_penalties, self.n_splits,
                                           self.scoring, classifier=True,
                                           n_jobs=self.n_jobs,
                                           verbose=self.verbose)

            self.cv_mean_score_ = np.atleast_1d(np.mean(cv_scores, axis=0))
            self.cv_standard_error_ = np.atleast_1d(stats.sem(cv_scores))

            self.lambda_max_inx_ = np.argmax(self.cv_mean_score_)
            self.lambda_max_ = self.lambda_path_[self.lambda_max_inx_]

            target_score = self.cv_mean_score_[self.lambda_max_inx_] -\
                self.cut_point * self.cv_standard_error_[self.lambda_max_inx_]

            self.lambda_best_inx_ = np.argwhere(
                self.cv_mean_score_ >= target_score)[0]
            self.lambda_best_ = self.lambda_path_[self.lambda_best_inx_]

            self.coef_ = self.coef_path_[..., self.lambda_best_inx_]
            self.coef_ = self.coef_.squeeze(axis=self.coef_.ndim-1)
            self.intercept_ = \
                self.intercept_path_[..., self.lambda_best_inx_].squeeze()
            if self.intercept_.shape == ():  # convert 0d array to scalar
                self.intercept_ = float(self.intercept_)

        return self

    def _fit(self, X, y, sample_weight, relative_penalties):
        if self.lambda_path is not None:
            n_lambda = len(self.lambda_path)
            min_lambda_ratio = 1.0
        else:
            n_lambda = self.n_lambda
            min_lambda_ratio = self.min_lambda_ratio

        check_classification_targets(y)
        if len(np.unique(y)) < 2:
            raise ValueError("prediction target must have at least two "
                             "distinct values.")

        _y = y.astype(dtype=np.float64, order='F', copy=True)

        _sample_weight = sample_weight.astype(dtype=np.float64, order='F',
                                              copy=True)

        exclude_vars = 0

        if relative_penalties is None:
            relative_penalties = np.ones(X.shape[1], dtype=np.float64,
                                         order='F')

        coef_bounds = np.empty((2, X.shape[1]), dtype=np.float64, order='F')
        coef_bounds[0, :] = -np.inf
        coef_bounds[1, :] = np.inf

        offset = np.zeros(X.shape[0])

        max_features = X.shape[1] + 1

        if issparse(X):
            _x = csc_matrix(X, dtype=np.float64, copy=True)

            (self.n_lambda_,
             self.intercept_path_,
             ca,
             ia,
             nin,
             _,  # dev0
             _,  # dev
             self.lambda_path_,
             _,  # nlp
             jerr) = spfishnet(self.alpha,
                               _x.shape[0],
                               _x.shape[1],
                               _x.data,
                               _x.indptr + 1,  # Fortran uses 1-based indexing
                               _x.indices + 1,
                               _y,
                               offset,
                               _sample_weight,
                               exclude_vars,
                               relative_penalties,
                               coef_bounds,
                               max_features,
                               max_features - 1,
                               min_lambda_ratio,
                               self.lambda_path_,
                               self.tol,
                               n_lambda,
                               self.standardize,
                               self.fit_intercept,
                               self.max_iter)
        else:  # not sparse
            _x = X.astype(dtype=np.float64, order='F', copy=True)

            (self.n_lambda_,
             self.intercept_path_,
             ca,
             ia,
             nin,
             _,  # dev0
             _,  # dev
             self.lambda_path_,
             _,  # nlp
             jerr) = fishnet(self.alpha,
                             _x,
                             _y,
                             offset,
                             sample_weight,
                             exclude_vars,
                             relative_penalties,
                             coef_bounds,
                             max_features,
                             min_lambda_ratio,
                             self.lambda_path,
                             self.tol,
                             max_features - 1,
                             n_lambda,
                             self.standardize,
                             self.fit_intercept,
                             self.max_iter)

        # raises RuntimeError if jerr is nonzero
        _check_glmnet_error_flag(jerr)

        self.lambda_path_ = self.lambda_path_[:self.n_lambda_]
        self.lambda_path_ = _fix_lambda_path(self.lambda_path_)
        self.intercept_path_ = self.intercept_path_[:self.n_lambda_]
        ca = ca[:, :self.n_lambda_]
        nin = nin[:self.n_lambda_]
        self.coef_path_ = solns(_x.shape[1], ca, ia, nin)

        return self

    def decision_function(self, X, lamb=None):
        lambda_best = None
        if hasattr(self, 'lambda_best_'):
            lambda_best = self.lambda_best_

        lamb = _check_user_lambda(self.lambda_path_, lambda_best, lamb)
        coef, intercept = _interpolate_model(self.lambda_path_,
                                             self.coef_path_,
                                             self.intercept_path_, lamb)

        X = check_array(X, accept_sparse='csr')
        z = X.dot(coef) + intercept

        # drop last dimension (lambda path) when we are predicting for a
        # single value of lambda
        return z.squeeze()

    def predict(self, X, lamb=None):
        """Predict the response Y for each sample in X

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        lamb : array, shape (n_lambda,)
            Values of lambda from lambda_path_ from which to make predictions.
            If no values are provided, the returned predictions will be those
            corresponding to lambda_best_. The values of lamb must also be in
            the range of lambda_path_, values greater than max(lambda_path_)
            or less than  min(lambda_path_) will be clipped.

        Returns
        -------
        C : array, shape (n_samples,) or (n_samples, n_lambda)
            Predicted response value for each sample given each value of lambda
        """
        return np.exp(self.decision_function(X, lamb))

    def score(self, X, y, lamb=None):
        """Returns the nagative deviance for each value of lambda.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Test samples

        y : array, shape (n_samples,)
            True values for X

        lamb : array, shape (n_lambda,)
            Values from lambda_path_ for which to score predictions.

        Returns
        -------
        scores : array, shape (n_lambda,)
            Negative deviance of predictions for each lambda.
        """
        p = y.reshape(-1, 1) / self.predict(X, lamb=lamb)
        # can't have any 0 below
        p = p.clip(np.finfo(float).eps, np.inf)

        return -2 * np.sum(y.reshape(-1, 1) * np.log(p), axis=0)
