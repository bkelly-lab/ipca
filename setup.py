from setuptools import setup

setup(name='ipca',
      version='0.1',
      description='Implements the IPCA method of Kelly, Pruitt, Su (2017)',
      url='https://github.com/matbuechner/ipca',
      author='Matthias Buechner',
      author_email='',
      license='MIT',
      packages=['ipca'],
      install_requires=[
          'numpy',
      ],
      zip_safe=False)
