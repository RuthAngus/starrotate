Installation
============

Currently the best way to install *starrotate* is from github.

From source:

.. code-block:: bash

    git clone https://github.com/RuthAngus/starrotate.git
    cd starrotate
    python setup.py install

Dependencies
------------

The dependencies of *starrotate* are
`NumPy <http://www.numpy.org/>`_,
`pandas <https://pandas.pydata.org/>`_,
`h5py <https://www.h5py.org/>`_,
`tqdm <https://tqdm.github.io/>`_,
`emcee <http://dfm.io/emcee/current/>`_,
`exoplanet <https://exoplanet.readthedocs.io/en/stable/>`_,
`astropy <http://www.astropy.org/>`_,
`matplotlib <https://matplotlib.org/>`_,
`scipy <https://www.scipy.org/>`_, and
`kplr <http://dfm.io/kplr/>`_.

These can be installed using pip:

.. code-block:: bash

    conda install numpy pandas h5py tqdm emcee exoplanet astropy matplotlib
    scipy kplr

or

.. code-block:: bash

    pip install numpy pandas h5py tqdm emcee exoplanet astropy matplotlib
    scipy kplr
