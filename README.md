# Instrumented Principal Components Analysis
[![Build Status](https://travis-ci.org/bkelly-lab/ipca.svg?branch=master)](https://travis-ci.org/bkelly-lab/ipca)

This is a Python implementation of the Instrumtented Principal Components Analysis framework by Kelly, Pruitt, Su (2017).


## Usage

Exemplary use of the ipca package. The data is the seminal Grunfeld data set as provided on [statsmodels](http://www.statsmodels.org). Note, the `fit` method
takes a panel of data, `X`, with the following columns:

1. entity id (numeric)
2. time (numeric)
3. and following columns contain characteristics.

as well as a series of dependent variables, `y`, of the same length as `X`.

```python
import numpy as np
from statsmodels.datasets import grunfeld
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)

# Establish unique IDs to conform with package
N = len(np.unique(data.firm))
ID = dict(zip(np.unique(data.firm).tolist(),np.arange(1,N+1)))
data.firm = data.firm.apply(lambda x: ID[x])

# use multi-index for panel groups
data = data.set_index(['firm', 'year'])
y = data['invest']
X = data.drop('invest', axis=1)

# Call ipca
from ipca import InstrumentedPCA
regr = InstrumentedPCA(n_factors=1, intercept=False)
regr = regr.fit(X=X, y=y)
Gamma, Factors = regr.get_factors(label_ind=True)
```

## Installing

The latest release can be installed using pip

```bash
pip install ipca
```

The master branch can be installed by cloning the repo and running setup

```bash
git clone https://github.com/bkelly-lab/ipca.git
cd ipca
python setup.py install
```

## Documenation
The lastest documenation is published [HERE](https://bkelly-lab.github.io/ipca/).

## Requirements

### Running

With the exception of Python 3.6+, which is a hard requirement, the
others are the version that are being used in the test environment.  It
is possible that older versions work.

* **Python 3.6+**:
* NumPy (1.15+)
* SciPy (1.1+)
* Numba (0.42+)
* progressbar (2.5+)
* joblib (0.13+)

### Testing

* pandas (0.24+)
* scikit-learn (0.20+)
* pytest (4.3+)
* statsmodels (0.9+)

## Acknowledgements
The implementation is inspired by the MATLAB code for IPCA made available on [Seth Pruitt's](https://sethpruitt.net/research/) website.

## References

1. Kelly, Pruitt, Su (2017). "Instrumented Principal Components Analysis" [SSRN](https://ssrn.com/abstract=2983919)

-----

The package is still in the development phase, hence please share your comments and suggestions with us.

Contributions welcome!

\- **Matthias Buechner, Leland Bybee**
