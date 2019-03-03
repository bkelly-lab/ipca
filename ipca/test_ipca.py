import pytest
import numpy as np
import pandas as pd
from sklearn.utils.testing import assert_raises
from sklearn.utils import estimator_checks

import h5py



from ipca import IPCARegressor



@pytest.mark.fast_test
def test_construction_errors():
    assert_raises(ValueError, IPCARegressor, n_factors=0)
    assert_raises(NotImplementedError, IPCARegressor, intercept='jabberwocky')
    assert_raises(ValueError, IPCARegressor, iter_tol=2)


# Create test data and run package
P = pd.read_csv('../../../TESTDATA/IPCADATA_AsPanel.csv', delimiter=',',header=None)
P = P.values
regr = IPCARegressor(n_factors=5, intercept=False)
Gamma_New, Factor_New =regr.fit(data=P)
print('Gamma', Gamma_New)
print('Factor', Factor_New)
