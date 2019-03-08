# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased


## 2.1.0 - 2019-03-11

### Added
* [#29](https://github.com/civisanalytics/python-glmnet/pull/29)
  Provide understandable error messages for more glmnet solver errors.
* [#31](https://github.com/civisanalytics/python-glmnet/pull/31)
  Expose `max_features` parameter in `ElasticNet` and `LogitNet`.
* [#34](https://github.com/civisanalytics/python-glmnet/pull/34)
  Use sample weights in `LogitNet`.
* [#41](https://github.com/civisanalytics/python-glmnet/pull/41)
  Add `lower_limits` and `upper_limits` parameters to `ElasticNet`
  and `LogitNet`, allowing users to restrict the range of fitted coefficients.

### Changed
* [#44](https://github.com/civisanalytics/python-glmnet/pull/44)
  Change CircleCI configuration file from v1 to v2, switch to pytest,
  and test in Python versions 3.4 - 3.7.
* [#36](https://github.com/civisanalytics/python-glmnet/pull/36)
  Convert README to .rst format for better display on PyPI (#35).

### Fixed
* [#24](https://github.com/civisanalytics/python-glmnet/pull/24)
  Use shuffled splits (controlled by input seed) for cross validation (#23).
* [#47](https://github.com/civisanalytics/python-glmnet/pull/47)
  Remove inappropriate `__init__.py` from the root path (#46).
* [#51](https://github.com/civisanalytics/python-glmnet/pull/51)
  Satisfy scikit-learn estimator checks. Includes:
  Allow one-sample predictions; allow list inputs for sample weights;
  Ensure scikit-learn Estimator compatibility.

## 2.0.0 - 2017-03-01

### API Changes
* [#10](https://github.com/civisanalytics/python-glmnet/pull/10) the parameter `n_folds` in the constructors of `LogitNet` and `ElasticNet` has been changed to `n_splits` for consistency with Scikit-Learn.

### Added
* [#6](https://github.com/civisanalytics/python-glmnet/pull/6) expose relative penalty

### Changed
* [#10](https://github.com/civisanalytics/python-glmnet/pull/10) update Scikit-Learn to 0.18

### Fixed
* [#3](https://github.com/civisanalytics/python-glmnet/pull/3) ensure license and readme are included in sdist
* [#8](https://github.com/civisanalytics/python-glmnet/pull/8) fix readme encoding
* [#14](https://github.com/civisanalytics/python-glmnet/pull/14) fix reference to `lambda_best_` in docs
* [#16](https://github.com/civisanalytics/python-glmnet/pull/16) fix import path for UndefinedMetricWarning

## 1.0.0 - 2016-06-03
### Added
- Initial release
