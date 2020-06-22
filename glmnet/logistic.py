import numpy as np

from scipy.special import expit
from scipy.sparse import issparse, csc_matrix
from scipy import stats

from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.utils import check_array, check_X_y
from sklearn.utils.multiclass import check_classification_targets

from .errors import _check_error_flag
from _glmnet import lognet, splognet, lsolns
from .util import (_fix_lambda_path,
                   _check_user_lambda,
                   _interpolate_model,
                   _score_lambda_path)


class LogitNet(BaseEstimator):
    """Logistic Regression with elastic net penalty.

    This is a wrapper for the glmnet function lognet.

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
        In place of supplying n_lambda, provide an array of specific values
        to compute. The specified values must be in decreasing order. When
        None, the path of lambda values will be determined automatically. A
        maximum of `n_lambda` values will be computed.

    standardize : bool, default True
        Standardize input features prior to fitting. The final coefficients
        will be on the scale of the original data regardless of the value
        of standardize.

    fit_intercept : bool, default True
        Include an intercept term in the model.

    lower_limits : array, (shape n_features,) default -infinity
        Array of lower limits for each coefficient, must be non-positive.
        Can be a single value (which is then replicated), else an array
        corresponding to the number of features.

    upper_limits : array, (shape n_features,) default +infinity
        Array of upper limits for each coefficient, must be positive.
        See lower_limits.

    cut_point : float, default 1
        The cut point to use for selecting lambda_best.
            arg_max lambda  cv_score(lambda) >= cv_score(lambda_max) - cut_point * standard_error(lambda_max)

    n_splits : int, default 3
        Number of cross validation folds for computing performance metrics and
        determining `lambda_best_` and `lambda_max_`. If non-zero, must be
        at least 3.

    scoring : string, callable or None, default None
        Scoring method for model selection during cross validation. When None,
        defaults to classification score. Valid options are `accuracy`,
        `roc_auc`, `average_precision`, `precision`, `recall`. Alternatively,
        supply a function or callable object with the following signature
        ``scorer(estimator, X, y)``. Note, the scoring function affects the
        selection of `lambda_best_` and `lambda_max_`, fitting the same data
        with different scoring methods will result in the selection of
        different models.

    n_jobs : int, default 1
        Maximum number of threads for computing cross validation metrics.

    tol : float, default 1e-7
        Convergence tolerance.

    max_iter : int, default 100000
        Maximum passes over the data.

    random_state : number, default None
        Seed for the random number generator. The glmnet solver is not
        deterministic, this seed is used for determining the cv folds.

    max_features : int
        Optional maximum number of features with nonzero coefficients after
        regularization. If not set, defaults to X.shape[1] during fit
        Note, this will be ignored if the user specifies lambda_path.

    verbose : bool, default False
        When True some warnings and log messages are suppressed.

    Attributes
    ----------
    classes_ : array, shape(n_classes,)
        The distinct classes/labels found in y.

    n_lambda_ : int
        The number of lambda values found by glmnet. Note, this may be less
        than the number specified via n_lambda.

    lambda_path_ : array, shape (n_lambda_,)
        The values of lambda found by glmnet, in decreasing order.

    coef_path_ : array, shape (n_classes, n_features, n_lambda_)
        The set of coefficients for each value of lambda in lambda_path_.

    coef_ : array, shape (n_clases, n_features)
        The coefficients corresponding to lambda_best_.

    intercept_ : array, shape (n_classes,)
        The intercept corresponding to lambda_best_.

    intercept_path_ : array, shape (n_classes, n_lambda_)
        The set of intercepts for each value of lambda in lambda_path_.

    cv_mean_score_ : array, shape (n_lambda_,)
        The mean cv score for each value of lambda. This will be set by fit_cv.

    cv_standard_error_ : array, shape (n_lambda_,)
        The standard error of the mean cv score for each value of lambda, this
        will be set by fit_cv.

    lambda_max_ : float
        The value of lambda that gives the best performance in cross
        validation.

    lambda_best_ : float
        The largest value of lambda which is greater than lambda_max_ and
        performs within cut_point * standard error of lambda_max_.
    """

    def __init__(self, alpha=1, n_lambda=100, min_lambda_ratio=1e-4,
                 lambda_path=None, standardize=True, fit_intercept=True,
                 lower_limits=-np.inf, upper_limits=np.inf,
                 cut_point=1.0, n_splits=3, scoring=None, n_jobs=1, tol=1e-7,
                 max_iter=100000, random_state=None, max_features=None, verbose=False):

        self.alpha = alpha
        self.n_lambda = n_lambda
        self.min_lambda_ratio = min_lambda_ratio
        self.lambda_path = lambda_path
        self.standardize = standardize
        self.lower_limits = lower_limits
        self.upper_limits = upper_limits
        self.fit_intercept = fit_intercept
        self.cut_point = cut_point
        self.n_splits = n_splits
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.max_features = max_features
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, relative_penalties=None, groups=None):
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

        y : array, shape (n_samples,)
            Target values

        sample_weight : array, shape (n_samples,)
            Optional weight vector for observations

        relative_penalties: array, shape (n_features,)
            Optional relative weight vector for penalty.
            0 entries remove penalty.

        groups: array, shape (n_samples,)
            Group labels for the samples used while splitting the dataset into train/test set.
            If the groups are specified, the groups will be passed to sklearn.model_selection.GroupKFold.
            If None, then data will be split randomly for K-fold cross-validation via sklearn.model_selection.KFold.

        Returns
        -------
        self : object
            Returns self.
        """
        X, y = check_X_y(X, y, accept_sparse='csr', ensure_min_samples=2)
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        else:
            sample_weight = np.asarray(sample_weight)

            if y.shape != sample_weight.shape:
                raise ValueError('the shape of weights is not the same with the shape of y')

        if not np.isscalar(self.lower_limits):
            self.lower_limits = np.asarray(self.lower_limits)
            if len(self.lower_limits) != X.shape[1]:
                raise ValueError("lower_limits must equal number of features")

        if not np.isscalar(self.upper_limits):
            self.upper_limits = np.asarray(self.upper_limits)
            if len(self.upper_limits) != X.shape[1]:
                raise ValueError("upper_limits must equal number of features")

        if any(self.lower_limits > 0) if isinstance(self.lower_limits, np.ndarray) else self.lower_limits > 0:
            raise ValueError("lower_limits must be non-positive")

        if any(self.upper_limits < 0) if isinstance(self.upper_limits, np.ndarray) else self.upper_limits < 0:
            raise ValueError("upper_limits must be positive")

        if self.alpha > 1 or self.alpha < 0:
            raise ValueError("alpha must be between 0 and 1")

        # fit the model
        self._fit(X, y, sample_weight, relative_penalties)

        # score each model on the path of lambda values found by glmnet and
        # select the best scoring
        if self.n_splits >= 3:
            if groups is None:
                self._cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            else:
                self._cv = GroupKFold(n_splits=self.n_splits)

            cv_scores = _score_lambda_path(self, X, y, groups,
                                           sample_weight,
                                           relative_penalties,
                                           self.scoring,
                                           n_jobs=self.n_jobs,
                                           verbose=self.verbose)

            self.cv_mean_score_ = np.atleast_1d(np.mean(cv_scores, axis=0))
            self.cv_standard_error_ = np.atleast_1d(stats.sem(cv_scores))

            self.lambda_max_inx_ = np.argmax(self.cv_mean_score_)
            self.lambda_max_ = self.lambda_path_[self.lambda_max_inx_]

            target_score = self.cv_mean_score_[self.lambda_max_inx_] -\
                self.cut_point * self.cv_standard_error_[self.lambda_max_inx_]

            self.lambda_best_inx_ = np.argwhere(self.cv_mean_score_ >= target_score)[0]
            self.lambda_best_ = self.lambda_path_[self.lambda_best_inx_]

            self.coef_ = self.coef_path_[..., self.lambda_best_inx_]
            self.coef_ = self.coef_.squeeze(axis=self.coef_.ndim-1)
            self.intercept_ = self.intercept_path_[..., self.lambda_best_inx_].squeeze()
            if self.intercept_.shape == ():  # convert 0d array to scalar
                self.intercept_ = float(self.intercept_)

        return self

    def _fit(self, X, y, sample_weight=None, relative_penalties=None):
        if self.lambda_path is not None:
            n_lambda = len(self.lambda_path)
            min_lambda_ratio = 1.0
        else:
            n_lambda = self.n_lambda
            min_lambda_ratio = self.min_lambda_ratio

        check_classification_targets(y)
        self.classes_ = np.unique(y)  # the output of np.unique is sorted
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError("Training data need to contain at least 2 "
                             "classes.")

        # glmnet requires the labels a one-hot-encoded array of
        # (n_samples, n_classes)
        if n_classes == 2:
            # Normally we use 1/0 for the positive and negative classes. Since
            # np.unique sorts the output, the negative class will be in the 0th
            # column. We want a model predicting the positive class, not the
            # negative class, so we flip the columns here (the != condition).
            #
            # Broadcast comparison of self.classes_ to all rows of y. See the
            # numpy rules on broadcasting for more info, essentially this
            # "reshapes" y to (n_samples, n_classes) and self.classes_ to
            # (n_samples, n_classes) and performs an element-wise comparison
            # resulting in _y with shape (n_samples, n_classes).
            _y = (y[:, None] != self.classes_).astype(np.float64, order='F')
        else:
            # multinomial case, glmnet uses the entire array so we can
            # keep the original order.
            _y = (y[:, None] == self.classes_).astype(np.float64, order='F')

        # use sample weights, making sure all weights are positive
        # this is inspired by the R wrapper for glmnet, in lognet.R
        if sample_weight is not None:
            weight_gt_0 = sample_weight > 0
            sample_weight = sample_weight[weight_gt_0]
            _y = _y[weight_gt_0, :]
            X = X[weight_gt_0, :]
            _y = _y * np.expand_dims(sample_weight, 1)

        # we need some sort of "offset" array for glmnet
        # an array of shape (n_examples, n_classes)
        offset = np.zeros((X.shape[0], n_classes), dtype=np.float64,
                          order='F')

        # You should have thought of that before you got here.
        exclude_vars = 0

        # how much each feature should be penalized relative to the others
        # this may be useful to expose to the caller if there are vars that
        # must be included in the final model or there is some prior knowledge
        # about how important some vars are relative to others, see the glmnet
        # vignette:
        # http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html
        if relative_penalties is None:
            relative_penalties = np.ones(X.shape[1], dtype=np.float64,
                                         order='F')

        coef_bounds = np.empty((2, X.shape[1]), dtype=np.float64, order='F')
        coef_bounds[0, :] = self.lower_limits
        coef_bounds[1, :] = self.upper_limits

        if n_classes == 2:
            # binomial, tell glmnet there is only one class
            # otherwise we will get a coef matrix with two dimensions
            # where each pair are equal in magnitude and opposite in sign
            # also since the magnitudes are constrained to sum to one, the
            # returned coefficients would be one half of the proper values
            n_classes = 1


        # This is a stopping criterion (nx)
        # R defaults to nx = num_features, and ne = num_features + 1
        if self.max_features is None:
            max_features = X.shape[1]
        else:
            max_features = self.max_features

        # for documentation on the glmnet function lognet, see doc.py
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
             jerr) = splognet(self.alpha,
                              _x.shape[0],
                              _x.shape[1],
                              n_classes,
                              _x.data,
                              _x.indptr + 1,  # Fortran uses 1-based indexing
                              _x.indices + 1,
                              _y,
                              offset,
                              exclude_vars,
                              relative_penalties,
                              coef_bounds,
                              max_features,
                              X.shape[1] + 1,
                              min_lambda_ratio,
                              self.lambda_path,
                              self.tol,
                              n_lambda,
                              self.standardize,
                              self.fit_intercept,
                              self.max_iter,
                              0)
        else:  # not sparse
            # some notes: glmnet requires both x and y to be float64, the two
            # arrays
            # may also be overwritten during the fitting process, so they need
            # to be copied prior to calling lognet. The fortran wrapper will
            # copy any arrays passed to a wrapped function if they are not in
            # the fortran layout, to avoid making extra copies, ensure x and y
            # are `F_CONTIGUOUS` prior to calling lognet.
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
             jerr) = lognet(self.alpha,
                            n_classes,
                            _x,
                            _y,
                            offset,
                            exclude_vars,
                            relative_penalties,
                            coef_bounds,
                            X.shape[1] + 1,
                            min_lambda_ratio,
                            self.lambda_path,
                            self.tol,
                            max_features,
                            n_lambda,
                            self.standardize,
                            self.fit_intercept,
                            self.max_iter,
                            0)

        # raises RuntimeError if self.jerr_ is nonzero
        self.jerr_ = jerr
        _check_error_flag(self.jerr_)

        # glmnet may not return the requested number of lambda values, so we
        # need to trim the trailing zeros from the returned path so
        # len(lambda_path_) is equal to n_lambda_
        self.lambda_path_ = self.lambda_path_[:self.n_lambda_]
        # also fix the first value of lambda
        self.lambda_path_ = _fix_lambda_path(self.lambda_path_)
        self.intercept_path_ = self.intercept_path_[:, :self.n_lambda_]
        # also trim the compressed coefficient matrix
        ca = ca[:, :, :self.n_lambda_]
        # and trim the array of n_coef per lambda (may or may not be non-zero)
        nin = nin[:self.n_lambda_]
        # decompress the coefficients returned by glmnet, see doc.py
        self.coef_path_ = lsolns(X.shape[1], ca, ia, nin)
        # coef_path_ has shape (n_features, n_classes, n_lambda), we should
        # match shape for scikit-learn models:
        # (n_classes, n_features, n_lambda)
        self.coef_path_ = np.transpose(self.coef_path_, axes=(1, 0, 2))

        return self

    def decision_function(self, X, lamb=None):
        lambda_best = None
        if hasattr(self, 'lambda_best_'):
            lambda_best = self.lambda_best_

        lamb = _check_user_lambda(self.lambda_path_, lambda_best, lamb)
        coef, intercept = _interpolate_model(self.lambda_path_,
                                             self.coef_path_,
                                             self.intercept_path_, lamb)

        # coef must be (n_classes, n_features, n_lambda)
        if coef.ndim != 3:
            # we must be working with an intercept only model
            coef = coef[:, :, np.newaxis]
        # intercept must be (n_classes, n_lambda)
        if intercept.ndim != 2:
            intercept = intercept[:, np.newaxis]

        X = check_array(X, accept_sparse='csr')
        # return (n_samples, n_classes, n_lambda)
        z = np.empty((X.shape[0], coef.shape[0], coef.shape[-1]))
        # well... sometimes we just need a for loop
        for c in range(coef.shape[0]):  # all classes
            for l in range(coef.shape[-1]):  # all values of lambda
                z[:, c, l] = X.dot(coef[c, :, l])
        z += intercept

        # drop the last dimension (lambda) when we are predicting for a single
        # value of lambda, and drop the middle dimension (class) when we are
        # predicting from a binomial model (for consistency with scikit-learn)
        return z.squeeze()

    def predict_proba(self, X, lamb=None):
        """Probability estimates for each class given X.

        The returned estimates are in the same order as the values in
        classes_.

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
        T : array, shape (n_samples, n_classes) or (n_samples, n_classes, n_lambda)
        """
        z = self.decision_function(X, lamb)
        expit(z, z)

        # reshape z to (n_samples, n_classes, n_lambda)
        n_lambda = len(np.atleast_1d(lamb))
        z = z.reshape(X.shape[0], -1, n_lambda)

        if z.shape[1] == 1:
            # binomial, for consistency and to match scikit-learn, add the
            # complement so z has shape (n_samples, 2, n_lambda)
            z = np.concatenate((1-z, z), axis=1)
        else:
            # normalize for multinomial
            z /= np.expand_dims(z.sum(axis=1), axis=1)

        if n_lambda == 1:
            z = z.squeeze(axis=-1)
        return z

    def predict(self, X, lamb=None):
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)

        lamb : array, shape (n_lambda,)
            Values of lambda from lambda_path_ from which to make predictions.
            If no values are provided for lamb, the returned predictions will
            be those corresponding to lambda_best_. The values of lamb must
            also be in the range of lambda_path_, values greater than
            max(lambda_path_) or less than  min(lambda_path_) will be clipped.

        Returns
        -------
        T : array, shape (n_samples,) or (n_samples, n_lambda)
            Predicted class labels for each sample given each value of lambda
        """

        scores = self.predict_proba(X, lamb)
        indices = scores.argmax(axis=1)

        return self.classes_[indices]

    def score(self, X, y, lamb=None):
        """Returns the mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Test samples

        y : array, shape (n_samples,)
            True labels for X

        lamb : array, shape (n_lambda,)
            Values from lambda_path_ for which to score predictions.

        Returns
        -------
        score : array, shape (n_lambda,)
            Mean accuracy for each value of lambda.
        """
        pred = self.predict(X, lamb=lamb)
        return np.apply_along_axis(accuracy_score, 0, pred, y)
