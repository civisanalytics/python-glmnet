Python GLMNET
=============

|CircleCI| |Conda| |PyPI| |PyVersions|

.. |CircleCI| image:: https://circleci.com/gh/civisanalytics/python-glmnet.svg?style=svg
    :target: https://circleci.com/gh/civisanalytics/python-glmnet
    :alt: Build status

.. |Conda| image:: https://anaconda.org/conda-forge/glmnet/badges/version.svg
   :target: https://anaconda.org/conda-forge/glmnet
   :alt: Latest version on conda forge

.. |PyPI| image:: https://img.shields.io/pypi/v/glmnet.svg
   :target: https://pypi.org/project/glmnet/
   :alt: Latest version on PyPI

.. |PyVersions| image:: https://img.shields.io/pypi/pyversions/glmnet.svg
   :target: https://pypi.org/project/glmnet/
   :alt: Supported python versions for python-glmnet

This is a Python wrapper for the fortran library used in the R package
`glmnet <http://web.stanford.edu/~hastie/glmnet/glmnet_alpha.html>`__.
While the library includes linear, logistic, Cox, Poisson, and
multiple-response Gaussian, only linear and logistic are implemented in
this package.

The API follows the conventions of
`Scikit-Learn <http://scikit-learn.org/stable/>`__, so it is expected to
work with tools from that ecosystem.

Installation
------------

requirements
~~~~~~~~~~~~

``python-glmnet`` requires Python version >= 3.6, ``scikit-learn``, ``numpy``,
and ``scipy``. Installation from source or via ``pip`` requires a Fortran compiler.

conda
~~~~~

.. code:: bash

    conda install -c conda-forge glmnet


pip
~~~

.. code:: bash

    pip install glmnet


source
~~~~~~

``glmnet`` depends on numpy, scikit-learn and scipy.
A working Fortran compiler is also required to build the package.
For Mac users, ``brew install gcc`` will take care of this requirement.

.. code:: bash

    git clone git@github.com:civisanalytics/python-glmnet.git
    cd python-glmnet
    python setup.py install

Usage
-----

General
~~~~~~~

By default, ``LogitNet`` and ``ElasticNet`` fit a series of models using
the lasso penalty (α = 1) and up to 100 values for λ (determined by the
algorithm). In addition, after computing the path of λ values,
performance metrics for each value of λ are computed using 3-fold cross
validation. The value of λ corresponding to the best performing model is
saved as the ``lambda_max_`` attribute and the largest value of λ such
that the model performance is within ``cut_point * standard_error`` of
the best scoring model is saved as the ``lambda_best_`` attribute.

The ``predict`` and ``predict_proba`` methods accept an optional
parameter ``lamb`` which is used to select which model(s) will be used
to make predictions. If ``lamb`` is omitted, ``lambda_best_`` is used.

Both models will accept dense or sparse arrays.

Regularized Logistic Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from glmnet import LogitNet

    m = LogitNet()
    m = m.fit(x, y)

Prediction is similar to Scikit-Learn:

.. code:: python

    # predict labels
    p = m.predict(x)
    # or probability estimates
    p = m.predict_proba(x)

Regularized Linear Regression
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

    from glmnet import ElasticNet

    m = ElasticNet()
    m = m.fit(x, y)

Predict:

.. code:: python

    p = m.predict(x)
