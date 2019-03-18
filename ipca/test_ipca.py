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
PSF = np.random.randn(len(np.unique(data[:, 1])),1)
PSF = PSF.reshape((1,-1))

# Fit IPCARegressor
regr = IPCARegressor(n_factors=1, intercept=True)
Gamma_New, Factor_New = regr.fit(data=data)
# Obtain Goodness of fit
print('R2total',regr.r2_total)
print('R2pred',regr.r2_pred)
# Use the fitted regressor to predict
data_x = np.delete(data,2,axis=1)
Ypred = regr.predict(data=data_x)
# Test refitting the IPCARegressor
data_refit = data[data[:,1]!=1954,:]
Gamma_New, Factor_New = regr.fit(data=data_refit)

# Test nan observations
regr = IPCARegressor(n_factors=1, intercept=True)
data_nan = data.copy()
data_nan[10:30, 2:] = np.nan
Gamma_New, Factor_New = regr.fit(data=data_nan)


# Test missing observations
regr = IPCARegressor(n_factors=1, intercept=True)
data_missing = data.copy()
data_missing  = data_missing[:-10,:]
Gamma_New, Factor_New = regr.fit(data=data_missing)

# Simulate OOS experiment
# In-sample data excludes observations during last available date
data_IS = data[data[:,1]!=1954,:]
# Out-of-sample consists only of observation at last available date
data_OOS = data[data[:,1]==1954,:]
# Re-fit the regressor
regr.fit(data=data_IS)

Ypred = regr.predictOOS(data=data_OOS, mean_factor=True)
