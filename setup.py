from setuptools import setup

setup(name='starrotate',
      version='0.1rc0',
      description='A package for measuring stellar rotation periods',
      url='http://github.com/RuthAngus/starrotate',
      author='Ruth Angus',
      author_email='ruthangus@gmail.com',
      license='MIT',
      packages=['starrotate'],
      install_requires=['numpy', 'pandas', 'h5py', 'tqdm', 'emcee', 'exoplanet', 'pymc3', 'theano', 'astropy', 'matplotlib', 'scipy', 'kplr'],
      zip_safe=False)
