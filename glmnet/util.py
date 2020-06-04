import math
import warnings

import numpy as np

from scipy.interpolate import interp1d

from sklearn.base import clone
from sklearn.exceptions import UndefinedMetricWarning
from joblib import Parallel, delayed

from .scorer import check_scoring


def _score_lambda_path(est, X, y, groups, sample_weight, relative_penalties,
                       scoring, n_jobs, verbose):
    """Score each model found by glmnet using cross validation.

    Parameters
    ----------
    est : estimator
        The previously fitted estimator.

    X : array, shape (n_samples, n_features)
        Input features

    y : array, shape (n_samples,)
        Target values.

    groups: array, shape (n_samples,)
        Group labels for the samples used while splitting the dataset into train/test set.

    sample_weight : array, shape (n_samples,)
        Weight of each row in X.

    relative_penalties: array, shape (n_features,)
        Relative weight vector for penalty.
        0 entries remove penalty.

    scoring : string, callable or None
        Scoring method to apply to each model.

    n_jobs: int
        Maximum number of threads to use for scoring models.

    verbose : bool
        Emit logging data and warnings when True.

    Returns
    -------
    scores : array, shape (n_lambda,)
        Scores for each value of lambda over all cv folds.
    """
    scorer = check_scoring(est, scoring)
    cv_split = est._cv.split(X, y, groups)

    # We score the model for every value of lambda, for classification
    # models, this will be an intercept-only model, meaning it predicts
    # the same class regardless of the input. Obviously, this makes some of
    # the scikit-learn metrics unhappy, so we are silencing these warnings.
    # Also note, catch_warnings is not thread safe.
    with warnings.catch_warnings():
        action = 'always' if verbose else 'ignore'
        warnings.simplefilter(action, UndefinedMetricWarning)

        scores = Parallel(n_jobs=n_jobs, verbose=verbose, backend='threading')(
            delayed(_fit_and_score)(est, scorer, X, y, sample_weight, relative_penalties,
                                    est.lambda_path_, train_idx, test_idx)
            for (train_idx, test_idx) in cv_split)

    return scores


def _fit_and_score(est, scorer, X, y, sample_weight, relative_penalties,
                   score_lambda_path, train_inx, test_inx):
    """Fit and score a single model.

    Parameters
    ----------
    est : estimator
        The previously fitted estimator.

    scorer : callable
        The scoring function to apply to each model.

    X : array, shape (n_samples, n_features)
        Input features

    y : array, shape (n_samples,)
        Target values.

    sample_weight : array, shape (n_samples,)
        Weight of each row in X.

    relative_penalties: array, shape (n_features,)
        Relative weight vector for penalty.
        0 entries remove penalty.

    score_lambda_path : array, shape (n_lambda,)
        The lambda values to evaluate/score.

    train_inx : array, shape (n_train,)
        Array of integers indicating which rows from X, y are in the training
        set for this fold.

    test_inx : array, shape (n_test,)
        Array of integers indicating which rows from X, y are in the test
        set for this fold.

    Returns
    -------
    scores : array, shape (n_lambda,)
        Scores for each value of lambda for a single cv fold.
    """
    m = clone(est)
    m = m._fit(X[train_inx, :], y[train_inx], sample_weight[train_inx], relative_penalties)

    lamb = np.clip(score_lambda_path, m.lambda_path_[-1], m.lambda_path_[0])
    return scorer(m, X[test_inx, :], y[test_inx], lamb=lamb)


def _fix_lambda_path(lambda_path):
    """Replace the first value in lambda_path (+inf) with something more
    reasonable. The method below matches what is done in the R/glmnent wrapper."""
    if lambda_path.shape[0] > 2:
        lambda_0 = math.exp(2 * math.log(lambda_path[1]) - math.log(lambda_path[2]))
        lambda_path[0] = lambda_0
    return lambda_path


def _check_user_lambda(lambda_path, lambda_best=None, lamb=None):
    """Verify the user-provided value of lambda is acceptable and ensure this
    is a 1-d array.

    Parameters
    ----------
    lambda_path : array, shape (n_lambda,)
        The path of lambda values as found by glmnet. This must be in
        decreasing order.

    lambda_best : float, optional
        The value of lambda producing the highest scoring model (found via
        cross validation).

    lamb : float, array, or None
        The value(s) of lambda for which predictions are desired. This must
        be provided if `lambda_best` is None, meaning the model was fit with
        cv_folds < 1.

    Returns
    -------
    lamb : array, shape (n_lambda,)
        The value(s) of lambda, potentially clipped to the range of values in
        lambda_path.
    """

    if lamb is None:
        if lambda_best is None:
            raise ValueError("You must specify a value for lambda or run "
                             "with cv_folds > 1 to select a value "
                             "automatically.")
        lamb = lambda_best

    # ensure numpy math works later
    lamb = np.array(lamb, ndmin=1)
    if np.any(lamb < lambda_path[-1]) or np.any(lamb > lambda_path[0]):
        warnings.warn("Some values of lamb are outside the range of "
                      "lambda_path_ [{}, {}]".format(lambda_path[-1],
                                                     lambda_path[0]),
                      RuntimeWarning)
    np.clip(lamb, lambda_path[-1], lambda_path[0], lamb)

    return lamb


def _interpolate_model(lambda_path, coef_path, intercept_path, lamb):
    """Interpolate coefficients and intercept between values of lambda.

    Parameters
    ----------
    lambda_path : array, shape (n_lambda,)
        The path of lambda values as found by glmnet. This must be in
        decreasing order.

    coef_path : array, shape (n_features, n_lambda) or
        (n_classes, n_features, n_lambda)
        The path of coefficients as found by glmnet.

    intercept_path : array, shape (n_lambda,) or (n_classes, n_lambda)
        The path of intercepts as found by glmnet.

    Returns
    -------
    coef : array, shape (n_features, n_lambda) or
        (n_classes, n_features, n_lambda)
        The interpolated path of coefficients.

    intercept : array, shape (n_lambda,) or (n_classes, n_lambda)
        The interpolated path of intercepts.
    """
    if lambda_path.shape[0] == 1:
        warnings.warn("lambda_path has a single value, this may be an "
                      "intercept-only model.", RuntimeWarning)
        coef = np.take(coef_path, 0, axis=-1)
        intercept = np.take(intercept_path, 0, axis=-1)
    else:
        coef = interp1d(lambda_path, coef_path)(lamb)
        intercept = interp1d(lambda_path, intercept_path)(lamb)

    return coef, intercept
