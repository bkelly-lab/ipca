import pytest
import numpy as np
from sklearn.utils.testing import assert_raises
from statsmodels.datasets import grunfeld


from ipca import IPCARegressor


# Test Construction Errors
@pytest.mark.fast_test
def test_construction_errors():
    assert_raises(ValueError, IPCARegressor, n_factors=0)
    assert_raises(NotImplementedError, IPCARegressor, intercept='jabberwocky')
    assert_raises(ValueError, IPCARegressor, iter_tol=2)


# Create test data and run package
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
data.firm = data.firm.apply(lambda x: x.decode('utf-8'))
# Establish unique IDs to conform with package
N = len(np.unique(data.firm))
ID = dict(zip(np.unique(data.firm).tolist(), np.arange(1, N+1)+5))
data.firm = data.firm.apply(lambda x: ID[x])
# Ensure that ordering of the data is correct
data = data[['firm', 'year', 'invest', 'value', 'capital']]
# Convert to numpy
data = data.to_numpy()
PSF = np.random.randn(len(np.unique(data[:, 1])), 1)
PSF = PSF.reshape((1, -1))

# Test IPCARegressor
regr = IPCARegressor(n_factors=1, intercept=False)
Gamma_New, Factor_New = regr.fit(P=data)
print('R2total', regr.r2_total)
print('R2pred', regr.r2_pred)
print('R2total_x', regr.r2_total_x)
print('R2pred_x', regr.r2_pred_x)
print(Gamma_New)
print(Factor_New)

# Test IPCARegressor with intercept
regr = IPCARegressor(n_factors=1, intercept=True)
Gamma_New, Factor_New = regr.fit(P=data)
print('R2total', regr.r2_total)
print('R2pred', regr.r2_pred)
print('R2total_x', regr.r2_total_x)
print('R2pred_x', regr.r2_pred_x)


# Use the fitted regressor to predict
data_x = np.delete(data, 2, axis=1)
Ypred = regr.predict(P=data_x)

# Test refitting the IPCARegressor with previous data but different n_factors
regr.n_factors = 2
regr.intercept = False
Gamma_New, Factor_New = regr.fit(P=data, refit=True)

# Test refitting the IPCARegressor on new data
data_refit = data[data[:, 1] != 1954, :]
Gamma_New, Factor_New = regr.fit(P=data_refit, refit=False)

# Test nan observations
regr = IPCARegressor(n_factors=1, intercept=True)
data_nan = data.copy()
data_nan[10:30, 2:] = np.nan
Gamma_New, Factor_New = regr.fit(P=data_nan)

# Test missing observations
regr = IPCARegressor(n_factors=1, intercept=True)
data_missing = data.copy()
data_missing = data_missing[:-10, :]
Gamma_New, Factor_New = regr.fit(P=data_missing)

# Simulate OOS experiment
# In-sample data excludes observations during last available date
data_IS = data[data[:, 1] != 1954, :]
# Out-of-sample consists only of observation at last available date
data_OOS = data[data[:, 1] == 1954, :]
# Re-fit the regressor
regr.fit(P=data_IS)
Ypred = regr.predictOOS(P=data_OOS, mean_factor=True)


# Test Walpha Bootstrap
regr = IPCARegressor(n_factors=1, intercept=True)
Gamma_New, Factor_New = regr.fit(P=data)
pval = regr.BS_Walpha(ndraws=1000)
print('p-value', pval)
