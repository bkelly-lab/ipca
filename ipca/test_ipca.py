import pytest
import numpy as np
from sklearn.utils.testing import assert_raises


from ipca import IPCARegressor
# Test Construction Errors
@pytest.mark.fast_test
def test_construction_errors():
    assert_raises(ValueError, IPCARegressor, n_factors=0)
    assert_raises(NotImplementedError, IPCARegressor, intercept='jabberwocky')
    assert_raises(ValueError, IPCARegressor, iter_tol=2)


# Create test data and run package
from statsmodels.datasets import grunfeld
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
data.firm = data.firm.apply(lambda x: x.decode('utf-8'))
# Establish unique IDs to conform with package
N = len(np.unique(data.firm))
ID = dict(zip(np.unique(data.firm).tolist(),np.arange(1,N+1)))
data.firm = data.firm.apply(lambda x: ID[x])
# Ensure that ordering of the data is correct
data = data[['firm','year','invest','value','capital']]
# Convert to numpy
data = data.to_numpy()

regr = IPCARegressor(n_factors=1, intercept=False)
Gamma_New, Factor_New = regr.fit(data=data)
