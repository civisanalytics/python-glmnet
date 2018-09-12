# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

## Unreleased

### Changed
* [#44] (https://github.com/civisanalytics/python-glmnet/pull/44) Change CircleCI configuration file from v1 to v2.

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
