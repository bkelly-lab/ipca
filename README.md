# Instrumented Principal Components Analysis
This is a Python implementation of the Instrumtented Principal Components Analysis framework by Kelly, Pruitt, Su (2017). All remaining errors are my own.





## Usage

Like [statsmodels](http://www.statsmodels.org) to include, supports
[patsy](https://patsy.readthedocs.io/en/latest/) formulas for
specifying models. For example, the classic Grunfeld regression can be
specified

```python
import numpy as np
from statsmodels.datasets import grunfeld
data = grunfeld.load_pandas().data
data.year = data.year.astype(np.int64)
# MultiIndex, entity - time
data = data.set_index(['firm','year'])
from ipca import IPCARegressor
regr = IPCARegressor(n_factors=1, intercept=False)
Gamma_New, Factor_New = regr.fit(data=P)
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

## Requirements

### Running

With the exception of Python 3.5+, which is a hard requirement, the
others are the version that are being used in the test environment.  It
is possible that older versions work.

* **Python 3.5+**:
* NumPy (1.15+)
* SciPy (1.1+)
* progressbar (2.5+)


### Testing

* pandas (0.24+)
* sklearn (0.20+)
* pytest (4.3+)
* statsmodels (0.9+)

## References

1. Kelly, Pruitt, Su (2017). "Instrumented Principal Components Analysis" [SSRN](https://ssrn.com/abstract=2983919)
