# Instrumented Principal Components Analysis
[![Build Status](https://travis-ci.org/bkelly-lab/ipca.svg?branch=master)](https://travis-ci.org/bkelly-lab/ipca)

This is a Python implementation of the Instrumtented Principal Components Analysis framework by Kelly, Pruitt, Su (2017).


## Usage

Exemplary use of the ipca package. The data is the seminal Grunfeld data set as provided on [statsmodels](http://www.statsmodels.org). Note, the package
requires the panel of input data columns to be ordered in the following way:

1. entity id (numeric)
2. time (numeric)
3. dependent variable (numeric),
4. and following columns contain characteristics.

```python
import numpy as np
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

# Call ipca
from ipca import IPCARegressor
regr = IPCARegressor(n_factors=1, intercept=False)
Gamma_New, Factor_New = regr.fit(data=data)
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

With the exception of Python 3.5+, which is a hard requirement, the
others are the version that are being used in the test environment.  It
is possible that older versions work.

* **Python 3.5+**:
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

The package is still in the development phase, hence please share your comments and suggestions with me.

Contributions welcome!

\- **Matthias Buechner**
