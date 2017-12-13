import warnings


def _check_error_flag(jerr):
    """Check the glmnet solver error flag and issue warnings or raise
    exceptions as appropriate.

    The codes break down roughly as follows:

        jerr == 0: everything is fine
        jerr > 0: fatal errors such as memory allocation problems
        jerr < 0: non-fatal errors such as convergence warnings
    """
    if jerr == 0:
        return

    if jerr > 0:
        _fatal_errors(jerr)

    if jerr < 0:
        _convergence_errors(jerr)


def _fatal_errors(jerr):
    if jerr == 7777:
        raise ValueError("All predictors have zero variance "
                         "(glmnet error no. 7777).")

    if jerr == 10000:
        raise ValueError("At least one value of relative_penalties must be "
                         "positive (glmnet error no. 10000).")

    if jerr == 90000:
        raise ValueError("Solver did not converge (glmnet error no. 90000).")

    if jerr < 7777:
        raise RuntimeError("Memory allocation error (glmnet error no. {})."
                           .format(jerr))

    if jerr > 8000 and jerr < 9000:
        k = jerr - 8000
        raise ValueError("Probability for class {} close to 0.".format(k))

    if jerr > 9000:
        k = jerr - 9000
        raise ValueError("Probability for class {} close to 1.".format(k))

    else:
        raise RuntimeError("Fatal glmnet error no. {}.".format(jerr))


def _convergence_errors(jerr):
    if jerr < -20000:
        k = abs(20000 + jerr)
        warnings.warn("Predicted probability close to 0 or 1 for "
                      "lambda no. {}.".format(k), RuntimeWarning)

    if jerr > -20000 and jerr < -10000:
        # This indicates the number of non-zero coefficients in a model
        # exceeded a user-specified bound. We don't expose this parameter to
        # the user, so there is not much sense in exposing the error either.
        warnings.warn("Non-fatal glmnet error no. {}.".format(jerr),
                      RuntimeWarning)

    if jerr > -10000:
        warnings.warn("Model did not converge for smaller values of lambda, "
                      "returning solution for the largest {} values."
                      .format(-1 * (jerr - 1)), RuntimeWarning)
