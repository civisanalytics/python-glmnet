from .logistic import LogitNet
from .linear import ElasticNet
from .poisson import PoissonNet
from ._version import get_versions

__all__ = ['LogitNet', 'ElasticNet', 'PoissonNet']


__version__ = get_versions()['version']
del get_versions
