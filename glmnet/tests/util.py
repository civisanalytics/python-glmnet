import numpy as np
from nose.tools import ok_, eq_


def sanity_check_logistic(m, x):
    sanity_check_model_attributes(m)
    sanity_check_cv_attrs(m, is_clf=True)

    ok_(m.classes_ is not None)
    ok_(m.coef_path_.ndim == 3, "wrong number of dimensions for coef_path_")

    n_classes = len(m.classes_)
    if len(m.classes_) == 2:  # binomial is a special case
        n_classes = 1
    ok_(m.coef_path_.shape[0] == n_classes, "wrong size for coef_path_")

    ok_(m.intercept_path_.ndim == 2, "wrong number of dimensions for intercept_path_")

    # check preds at random value of lambda
    l = np.random.choice(m.lambda_path_)
    p = m.predict(x, lamb=l)
    check_logistic_predict(m, x, p)

    p = m.predict_proba(x, lamb=l)
    check_logistic_predict_proba(m, x, p)

    # if cv ran, check default behavior of predict and predict_proba
    if m.n_splits >= 3:
        p = m.predict(x)
        check_logistic_predict(m, x, p)

        p = m.predict_proba(x)
        check_logistic_predict_proba(m, x, p)


def check_logistic_predict(m, x, p):
    eq_(p.shape[0], x.shape[0])
    ok_(np.all(np.in1d(np.unique(p),m.classes_)))


def check_logistic_predict_proba(m, x, p):
    eq_(p.shape[0], x.shape[0])
    eq_(p.shape[1], len(m.classes_))
    ok_(np.all(p >= 0) and np.all(p <= 1.), "predict_proba values outside [0,1]")


def sanity_check_regression(m, x):
    sanity_check_model_attributes(m)
    sanity_check_cv_attrs(m)

    ok_(m.coef_path_.ndim == 2, "wrong number of dimensions for coef_path_")
    ok_(m.intercept_path_.ndim == 1, "wrong number of dimensions for intercept_path_")

    # check predict at random value of lambda
    l = np.random.choice(m.lambda_path_)
    p = m.predict(x, lamb=l)
    eq_(p.shape[0], x.shape[0])

    # if cv ran, check default behavior of predict
    if m.n_splits >= 3:
        p = m.predict(x)
        eq_(p.shape[0], x.shape[0])


def sanity_check_model_attributes(m):
    ok_(m.n_lambda_ > 0, "n_lambda_ is not set")
    ok_(m.lambda_path_.size == m.n_lambda_, "lambda_path_ does not have length n_lambda_")
    ok_(m.coef_path_.shape[-1] == m.n_lambda_, "wrong size for coef_path_")
    ok_(m.intercept_path_.shape[-1] == m.n_lambda_, "wrong size for intercept_path_")
    ok_(m.jerr_ == 0, "jerr is non-zero")


def sanity_check_cv_attrs(m, is_clf=False):
    if m.n_splits >= 3:
        if is_clf:
            ok_(m.coef_.shape[-1] == m.coef_path_.shape[1], "wrong size for coef_")
        else:
            ok_(m.coef_.size == m.coef_path_.shape[0], "wrong size for coef_")
        ok_(m.intercept_ is not None, "intercept_ is not set")
        ok_(m.cv_mean_score_.size == m.n_lambda_, "wrong size for cv_mean_score_")
        ok_(m.cv_standard_error_.size == m.n_lambda_, "wrong size for cv_standard_error_")
        ok_(m.lambda_max_ is not None, "lambda_max_ is not set")
        ok_(m.lambda_max_inx_ >= 0 and m.lambda_max_inx_ < m.n_lambda_,
            "lambda_max_inx_ is outside bounds of lambda_path_")
        ok_(m.lambda_best_ is not None, "lambda_best_ is not set")
        ok_(m.lambda_best_inx_ >= 0 and m.lambda_best_inx_ < m.n_lambda_,
            "lambda_best_inx_ is outside bounds of lambda_path_")
