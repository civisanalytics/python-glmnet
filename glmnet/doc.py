"""For convenience, documentation for the wrapped functions from glmnet:

Notes:
    In general, numpy will handle sending the correct array representation to
    the wrapped functions. If the array is not already 'F Contiguous', a copy
    will be made and passed to the glmnet. The glmnet documentation suggests that
    x and y may be overwritten during the fitting process, so these arrays should
    be copied anyway (with order='F').

--------------------------------------------------------------------------------
lognet: binomial/multinomial logistic elastic net

call:
    lmu,a0,ca,ia,nin,dev0,dev,alm,nlp,jerr = lognet(parm,nc,x,y,g,jd,vp,cl,nx,flmin,ulam,thr,[ne,nlam,isd,intr,maxit,kopt])

Parameters
----------
parm : float
    The elasticnet mixing parameter, 0 <= param <= 1. The penalty is defined
    as (1 - param) / 2 ||B|| + param |B|. param=1 is the lasso penalty and
    param=0 is the ridge penalty.

nc : int
    The number of distinct classes in y. Note, for the binomial case this
    value should be 1.

x : rank-2 array('float') with bounds (no,ni)
    Input matrix with shape [n_samples, n_features]

y : rank-2 array('float') with bounds (no,max(2,nc))
    Response variable with shape [n_samples, n_classes]. Note for the binomial
    case, this must be [n_samples, 2].

g : rank-2 array('float') with bounds (no,shape(y,1))
    Offsets of shape [n_samples, n_classes]. Unless you know what why just
    pass np.zeros((n_samples, n_classes)).

jd : rank-1 array('int') with bounds (*)
    Feature deletion flag, equivalent to applying an infinite penalty. To
    include all features in the model, jd=0. To exclude the ith and jth feature:
        jd=[1, i, j]
    Note fortran uses 1-based indexing so the 0th feature is 1. If you are
    excluding features, the first element of jd must be a 1 to signal glmnet.

vp : rank-1 array('float') with bounds (ni)
    Relative penalty to apply to each feature, use np.ones(n_features) to
    uniformily apply the elasticnet penalty.

cl : rank-2 array('float') with bounds (2,ni)
    Interval constraints for the model coefficients. vp[0, :] are lower bounds and
    vp[1, :] are the upper bounds.

nx : int
    The maximum number of variables allowed to enter all models along the path
    of param. If ne (see below) is also supplied, nx > ne. This should typically
    be set to n_features.

flmin : float
    Smallest value for lambda as a fraction of lambda_max (the lambda for which
    all coefficients are zero). If n_samples > n_features, this value should be
    1e-4, for n_features > n_samples use 1e-2. Note, if the lambda path is explicitly
    provided (see ulam below), flmin will be ignored, but it must be > 1.

ulam : rank-1 array('float') with bounds (nlam)
    User supplied lambda sequence. Note glmnet typically computes its own
    sequence of lambda values (when ulam = None). If a specific sequence of
    lambdas is desired, they should be passed in decreasing order.

thr : float
    Convergence threshold for coordinate descent, a good value is 1e-7.

ne : int, Default: min(shape(x, 1), nx)
    The maximum number of variables allowed to enter the largest model (stopping
    criterion), if provided, nx > ne.

nlam : int, Default: len(ulam)
    Maximum number of lambda values. If ulam is not supplied, nlam must be
    provided, 100 is a good value.

isd : int, Default: 1/True
    Standardize predictor variables prior to fitting model. Note, output coefficients
    will always reference the original variable locations and scales.

intr : int, Default: 1/True
    Include an intercept term in the model.

maxit : int, Default: 100000
    Maximum number of passes ofer the data for all lambda values.

kopt : int, Default: 0 (Newton-Raphson)
    Optimization algorithm:
        0: Newton-Raphson (exact hessian)
        1: Modified Newton-Raphson (upper-bounded hessian, sometimes faster)
        2: Non-zero coefficients same for each class (exact behavior is not documented)

Returns
-------
lmu : int
    Actual number of lambda values used, may not be equal to nlam.

a0 : rank-2 array('float') with bounds (nc,nlam)
    Intercept values for each class at each value of lambda.

ca : rank-3 array('float') with bounds (nx,nc,nlam)
    Compressed coefficients for each class at each value of lambda. Suggest
    using lsolns to convert to a usable layout [n_features, n_classes, n_lambda].
    Note in the binomial case, there will be one set of coefficients instead
    of two.

ia : rank-1 array('int') with bounds (nx)
    Pointers to compressed coefficients, used by lsolns to decompress the ca
    array.

nin : rank-1 array('int') with bounds (nlam)
    Number of compressed coefficients for each value of lambda, used by lsolns
    to decompress the ca array.

dev0 : rank-1 array('float') with bounds (nlam)
    Null deviance (intercept only model) for each value of lambda.

dev : rank-1 array('float') with bounds (nlam)
    Fraction of deviance explained for each value of lambda.

alm : rank-1 array('float') with bounds (nlam)
    Actual lambda values corresponding to each solution.

nlp : int
    Number of passes over the data for all lambda values.

jerr : int
    Error flag:
        = 0: no error
        > 0: fatal error - no output returned
            < 7777: memory allocation error
            = 7777: all used predictors have zero variance
            = 8000 + k: null probability < 1e-5 for class k
            = 9000 + k: null probability for class k > 1 - 1e-5
            = 10000: maxval(vp) <= 0
            = 90000: bounds adjustment non convergence
        < 0: non fatal error - partial output

--------------------------------------------------------------------------------
lsolns: uncompress coefficient vectors for all solutions

call:
    b = lsolns(ni, ca, ia, nin)

Parameters
----------
ni : int
    Number of input features.

ca : rank-3 array('float') with bounds (nx,nc,lmu)
    Compressed coefficient array, as returned by lognet.

ia : rank-1 array('int') with bounds (nx)
    Pointers to compressed coefficients, as returned by lognet.

nin : rank-1 array('int') with bounds (lmu)
    Number of compressed coefficients for each solution, as returned by
    lognet.

Returns
-------
b : rank-3 array('float') with bounds (ni,nc,lmu)
    Dense coefficient array referencing original variables.

--------------------------------------------------------------------------------
elnet: regression with elastic net penalty and squared-error loss

call:
    lmu,a0,ca,ia,nin,rsq,alm,nlp,jerr = elnet(ka,parm,x,y,w,jd,vp,cl,nx,flmin,ulam,thr,[ne,nlam,isd,intr,maxit])

Parameters
----------
ka : int
    Algorithm flag, 1 for covariance updating algorithm, 2 for naive algorithm.
    When n_features (ni) < 500, the covariance updating algorithm should be used
    as it saves all inner-products ever computed, making it much faster than the
    naive algorithm which needs to loop through each sample every time an inner-
    product is computed. In the case of n_features >> n_samples, the reverse
    is true w.r.t. efficiency.

parm : float
    The elasticnet mixing parameter, 0 <= param <= 1. The penalty is defined
    as (1 - param) / 2 ||B|| + param |B|. param=1 is the lasso penalty and
    param=0 is the ridge penalty.

x : rank-2 array('float') with bounds (no,ni)
    Input matrix with shape [n_samples, n_features]

y : rank-1 array('f') with bounds (no)
    Response variable

w : rank-1 array('f') with bounds (no)
    Observation weights

jd : rank-1 array('int') with bounds (*)
    Feature deletion flag, equivalent to applying an infinite penalty. To
    include all features in the model, jd=0. To exclude the ith and jth feature:
        jd=[1, i, j]
    Note fortran uses 1-based indexing so the 0th feature is 1. If you are
    excluding features, the first element of jd must be a 1 to signal glmnet.

vp : rank-1 array('float') with bounds (ni)
    Relative penalty to apply to each feature, use np.ones(n_features) to
    uniformily apply the elasticnet penalty.

cl : rank-2 array('float') with bounds (2,ni)
    Interval constraints for the model coefficients. vp[0, :] are lower bounds and
    vp[1, :] are the upper bounds.

nx : int
    The maximum number of variables allowed to enter all models along the path
    of param. If ne (see below) is also supplied, nx > ne. This should typically
    be set to n_features.

flmin : float
    Smallest value for lambda as a fraction of lambda_max (the lambda for which
    all coefficients are zero). If n_samples > n_features, this value should be
    1e-4, for n_features > n_samples use 1e-2. Note, if the lambda path is explicitly
    provided (see ulam below), flmin will be ignored, but it must be > 1.

ulam : rank-1 array('float') with bounds (nlam)
    User supplied lambda sequence. Note glmnet typically computes its own
    sequence of lambda values (when ulam = None). If a specific sequence of
    lambdas is desired, they should be passed in decreasing order.

thr : float
    Convergence threshold for coordinate descent, a good value is 1e-7.

ne : int, Default: min(shape(x, 1), nx)
    The maximum number of variables allowed to enter the largest model (stopping
    criterion), if provided, nx > ne.

nlam : int, Default: len(ulam)
    Maximum number of lambda values. If ulam is not supplied, nlam must be
    provided, 100 is a good value.

isd : int, Default: 1/True
    Standardize predictor variables prior to fitting model. Note, output coefficients
    will always reference the original variable locations and scales.

intr : int, Default: 1/True
    Include an intercept term in the model.

maxit : int, Default: 100000
    Maximum number of passes ofer the data for all lambda values.

Returns
-------
lmu : int
    Actual number of lambda values used, may not be equal to nlam.

a0 : rank-2 array('float') with bounds (nc,nlam)
    Intercept values for each class at each value of lambda.

ca : rank-2 array('float') with bounds (nx,nlam)
    Compressed coefficients at each value of lambda. Suggest using solns to
    convert to a usable layout.

ia : rank-1 array('int') with bounds (nx)
    Pointers to compressed coefficients, used by solns to decompress the ca
    array.

nin : rank-1 array('int') with bounds (nlam)
    Number of compressed coefficients for each value of lambda, used by solns
    to decompress the ca array.

rsq : rank-1 array('float') with bounds (nlam)
    R^2 values for each lambda.

alm : rank-1 array('float') with bounds (nlam)
    Actual lambda values corresponding to each solution.

nlp : int
    Number of passes over the data for all lambda values.

jerr : int
    Error flag:
        = 0: no error
        > 0: fatal error - no output returned
        < 0: non fatal error - partial output

--------------------------------------------------------------------------------
solns: uncompress coefficient vectors for all solutions

call:
    b = solns(ni,ca,ia,nin)

Parameters
----------
ni : int
    Number of input features.

ca : rank-2 array('float') with bounds (nx,lmu)
    Compressed coefficient array, as returned by elnet.

ia : input rank-1 array('int') with bounds (nx)
    Pointers to compressed coefficients, as returned by elnet.

nin : input rank-1 array('int') with bounds (lmu)
    Number of compressed coefficients for each solution, as returned by elnet.

Returns
-------
b : rank-2 array('float') with bounds (ni,lmu)
    Dense coefficient array referencing original variables.
"""
