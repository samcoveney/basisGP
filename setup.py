from setuptools import setup

setup(name = 'basisgp',
      version = '0.1',
      description = 'Gaussian processes with explicit basis functions',
      url = 'https://github.com/samcoveney/basisgp',
      author = 'Sam Coveney',
      author_email = 'coveney.sam@gmail.com',
      license = 'GPL-3.0+',
      packages = ['basisgp'],
      install_requires = [
          'numpy',
          'scipy',
          'future',
          'jax',
          'jaxlib',
      ],
      zip_safe = False)

