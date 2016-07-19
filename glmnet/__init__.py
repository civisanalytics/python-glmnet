from .logistic import LogitNet
from .linear import ElasticNet
from ._version import get_versions

__all__ = ['LogitNet', 'ElasticNet']


__version__ = get_versions()['version']
del get_versions
