from setuptools import setup

setup(name='ipca',
      version='0.5.4',
      description='Implements the IPCA method of Kelly, Pruitt, Su (2017)',
      url='https://github.com/bkelly-lab/ipca',
      author='Matthias Buechner',
      author_email='matthias.buechner.16@mail.wbs.ac.uk',
      license='MIT',
      keywords=['ipca', 'regression', 'IV', 'instrumental variable'],
      packages=['ipca'],
      install_requires=[
          'numpy','progressbar','numba'
      ],
      zip_safe=False)
