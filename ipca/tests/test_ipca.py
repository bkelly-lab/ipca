import numpy as np
from ipca import IPCARegressor
import scipy.io as sio


TestData = sio.loadmat('../AUX/IPCA_Pruitt/TestData.mat')
Z = TestData['Z']
Y = TestData['Y']
PSF = TestData['PSF']



IPCA_1 = IPCARegressor(n_factors=5)
IPCA_1.fit(Z=Z,Y=Y)
