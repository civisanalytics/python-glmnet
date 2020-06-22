import pkg_resources

from .logistic import LogitNet
from .linear import ElasticNet

__all__ = ['LogitNet', 'ElasticNet']

__version__ = pkg_resources.get_distribution("glmnet").version
