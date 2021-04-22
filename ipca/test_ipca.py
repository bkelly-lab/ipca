import pytest
import numpy as np
from statsmodels.datasets import grunfeld
import time
from datetime import datetime

from ipca import InstrumentedPCA


# Test Construction Errors
def test_construction_errors():
  with pytest.raises(ValueError):
    InstrumentedPCA(n_factors=0)
  with pytest.raises(NotImplementedError):
    InstrumentedPCA(intercept='jabberwocky')
  with pytest.raises(ValueError):
    InstrumentedPCA(iter_tol=2)


# Create test data and run package
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
#data.firm = data.firm.apply(lambda x: x.decode('utf-8'))
# Establish unique IDs to conform with package
N = len(np.unique(data.firm))
ID = dict(zip(np.unique(data.firm).tolist(), np.arange(1, N+1)+5))
data.firm = data.firm.apply(lambda x: ID[x])
# Ensure that ordering of the data is correct
data = data[['firm', 'year', 'invest', 'value', 'capital']]
# prep PSF test vars test vars
PSF1 = np.random.randn(len(np.unique(data.loc[:, 'year'])), 1)
PSF1 = PSF1.reshape((1, -1))
PSF2 = np.random.randn(len(np.unique(data.loc[:, 'year'])), 2)
PSF2 = PSF2.reshape((2, -1))
data = data.set_index(['firm', 'year'])
data_y = data['invest']
data_x = data.drop('invest', axis=1)

t0 = datetime.now()

# Test InstrumentedPCA
regr = InstrumentedPCA(n_factors=1, intercept=False)
regr = regr.fit(X=data_x, y=data_y)
print("R2total", regr.score(X=data_x, y=data_y))
print("R2pred", regr.score(X=data_x, y=data_y, mean_factor=True))
print("R2total_x", regr.score(X=data_x, y=data_y, data_type="portfolio"))
print("R2pred_x", regr.score(X=data_x, y=data_y, mean_factor=True,
                             data_type="portfolio"))
print(regr.Gamma)
print(regr.Factors)

# test indices
regr = InstrumentedPCA(n_factors=1, intercept=False)
regr = regr.fit(X=data_x.values, y=data_y.values, indices=data_x.index)
print("R2total", regr.score(X=data_x.values, y=data_y.values,
                            indices=data_x.index))
print("R2pred", regr.score(X=data_x.values, y=data_y.values,
                           indices=data_x.index, mean_factor=True))
print("R2total_x", regr.score(X=data_x.values, y=data_y.values,
                              indices=data_x.index,
                              data_type="portfolio"))
print("R2pred_x", regr.score(X=data_x.values, y=data_y.values,
                             indices=data_x.index,
                             mean_factor=True, data_type="portfolio"))
print(regr.Gamma)
print(regr.Factors)
Gamma_df, Factors_df = regr.get_factors(label_ind=True)
print(Gamma_df)
print(Factors_df)

# Test InstrumentedPCA with intercept
regr = InstrumentedPCA(n_factors=1, intercept=True)
regr = regr.fit(X=data_x, y=data_y)
print("R2total", regr.score(X=data_x, y=data_y))
print("R2pred", regr.score(X=data_x, y=data_y, mean_factor=True))
print("R2total_x", regr.score(X=data_x, y=data_y, data_type="portfolio"))
print("R2pred_x", regr.score(X=data_x, y=data_y, mean_factor=True,
                             data_type="portfolio"))

# Use the fitted regressor to predict
Ypred = regr.predict(X=data_x)

# Test refitting the InstrumentedPCA with previous data but different n_factors
regr.n_factors = 2
regr.intercept = False
regr = regr.fit(X=data_x, y=data_y)

# Test different data-type fits
regr = InstrumentedPCA(n_factors=1, intercept=True)
regr = regr.fit(X=data_x, y=data_y, data_type="panel")
regr = InstrumentedPCA(n_factors=1, intercept=True)
regr = regr.fit(X=data_x, y=data_y, data_type="portfolio")

# Test PSF - one additional factor estimated
regr = InstrumentedPCA(n_factors=2, intercept=False)
regr = regr.fit(X=data_x, y=data_y, PSF=PSF1)

# Test PSF - no additional factors estimated
regr = InstrumentedPCA(n_factors=2, intercept=False)
regr = regr.fit(X=data_x, y=data_y, PSF=PSF2)

# Test nan observations
regr = InstrumentedPCA(n_factors=1, intercept=True)
data_nan = data_x.copy()
data_nan.iloc[10:30, :] = np.nan
regr = regr.fit(X=data_nan, y=data_y)

# Test missing observations
regr = InstrumentedPCA(n_factors=1, intercept=True)
data_missing = data_x.copy()
data_missing = data_missing.iloc[:-10, :]
regr = regr.fit(X=data_x, y=data_y)

# Simulate OOS experiment
y_IS = data_y[data_x.index.get_level_values("year") != 1954]
y_OOS = data_y[data_x.index.get_level_values("year") == 1954]
data_IS = data_x[data_x.index.get_level_values("year") != 1954]
data_OOS = data_x[data_x.index.get_level_values("year") == 1954]
# Re-fit the regressor
regr = regr.fit(X=data_IS, y=y_IS)
Ypred = regr.predictOOS(X=data_OOS, y=y_OOS, mean_factor=True)

# Test Walpha Bootstrap
regr = InstrumentedPCA(n_factors=1, intercept=True)
regr = regr.fit(X=data_x, y=data_y)
pval = regr.BS_Walpha(ndraws=10, n_jobs=-1)
print('p-value', pval)

# Test Wbeta Bootstrap
regr = InstrumentedPCA(n_factors=1, intercept=False)
regr = regr.fit(X=data_x, y=data_y)
pval = regr.BS_Wbeta([0, 1], ndraws=10, n_jobs=-1)
print('p-value', pval)

# Test with regularization
regr = InstrumentedPCA(n_factors=2, alpha=0.5, intercept=False)
regr = regr.fit(X=data_x, y=data_y)
regr = InstrumentedPCA(n_factors=2, alpha=0.5, l1_ratio=0.5, intercept=False)
regr = regr.fit(X=data_x, y=data_y)
regr = InstrumentedPCA(n_factors=2, alpha=0.5, intercept=False)
regr = regr.fit(X=data_x, y=data_y, PSF=PSF1)
regr = InstrumentedPCA(n_factors=1, intercept=True)
regr = regr.fit(X=data_x, y=data_y)

# Test regularization path
regr = InstrumentedPCA(n_factors=2)
cvmse = regr.fit_path(X=data_x, y=data_y)
cvmse = regr.fit_path(X=data_x, y=data_y, alpha_l=np.array([0., 0.5, 1.]))
cvmse = regr.fit_path(X=data_x, y=data_y, PSF=PSF2)
cvmse = regr.fit_path(X=data_x, y=data_y, PSF=PSF2,
                      alpha_l=np.array([0., 0.5, 1.]))
print(datetime.now() - t0)
