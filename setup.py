from setuptools import setup

setup(name='ipca',
      version='0.6.7',
      description='Implements the IPCA method of Kelly, Pruitt, Su (2017)',
      url='https://github.com/bkelly-lab/ipca',
      author='Matthias Buechner, Leland Bybee',
      author_email='mat.buechner@gmail.com, leland.bybee@gmail.com',
      license='MIT',
      keywords=['ipca', 'regression', 'IV', 'instrumental variable'],
      packages=['ipca'],
      install_requires=[
          'numpy','progressbar','numba','scipy','joblib','scikit-learn'
      ],
      zip_safe=False)
