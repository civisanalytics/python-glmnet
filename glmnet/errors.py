import warnings


def _check_glmnet_error_flag(jerr, n_lambda):
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
        _fatal_errors(jerr, n_lambda)

    if jerr < 0:
        _convergence_errors(jerr, n_lambda)


def _fatal_errors(jerr, n_lambda):
    if jerr == 7777:
        raise ValueError("All predictors have zero variance "
                         "(glmnet error no. 7777).")

    if jerr == 10000:
        raise ValueError("At least one value of relative_penalties must be "
                         "positive (glmnet error no. 10000).")

    if jerr < 7777:
        raise RuntimeError("Memory allocation error (glmnet error no. {})."
                           .format(jerr))

    else:
        raise RuntimeError("Fatal glmnet error no. {}.".format(jerr))


def _convergence_errors(jerr, n_lambda):
    if abs(jerr) <= n_lambda:
        warnings.warn("Model did not converge for smaller values of lambda, "
                      "returning solution for the largest {} values."
                      .format(-1 * (jerr - 1)), RuntimeWarning)
    else:
        warnings.warn("Non-fatal glmnet error no. {}.".format(jerr),
                      RuntimeWarning)

