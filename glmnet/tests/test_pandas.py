import unittest

from sklearn.datasets import make_regression, make_classification
from glmnet import LogitNet, ElasticNet

from glmnet.tests.util import sanity_check_logistic, sanity_check_regression

pd = None
try:
    import pandas as pd
except:
    pass


class TestElasticNetPandas(unittest.TestCase):

    @unittest.skipUnless(pd, "pandas not available")
    def test_with_pandas_df(self):
        x, y = make_regression(random_state=561)
        df = pd.DataFrame(x)
        df['y'] = y

        m = ElasticNet(n_splits=3, random_state=123)
        m = m.fit(df.drop(['y'], axis=1), df.y)
        sanity_check_regression(m, x)


class TestLogitNetPandas(unittest.TestCase):

    @unittest.skipUnless(pd, "pandas not available")
    def test_with_pandas_df(self):
        x, y = make_classification(random_state=1105)
        df = pd.DataFrame(x)
        df['y'] = y

        m = LogitNet(n_splits=3, random_state=123)
        m = m.fit(df.drop(['y'], axis=1), df.y)
        sanity_check_logistic(m, x)


if __name__ == "__main__":
    unittest.main()
