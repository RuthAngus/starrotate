.. starrotate documentation master file, created by
   sphinx-quickstart on Sat Nov  3 16:17:18 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

starrotate
====================================

*starrotate* is a tool for measuring stellar rotation periods using
Lomb-Scargle (LS) periodograms, autocorrelation functions (ACFs) and Gaussian
processes (GPs).
It uses the `astropy <http://www.astropy.org/>`_ implementation of
`Lomb-Scargle periodograms
<http://docs.astropy.org/en/stable/stats/lombscargle.html>`_, and the
`exoplanet <https://exoplanet.dfm.io/en/stable/>`_ implementation of
fast `celerite <https://celerite.readthedocs.io/en/latest/?badge=latest>`_
Gaussian processes.

*starrotate* is compatible with any light curve with time, flux and flux
uncertainty measurements, including Kepler, K2 and TESS light curves.
If your light curve is has evenly-spaced (or close to evenly-spaced)
observations, all three of these methods: LS periodograms, ACFs and GPs will
be applicable.
For unevenly spaced light curves like those from the Gaia, or ground-based
observatories, LS periodograms and GPs are preferable to ACFs.

Example usage
-------------
::

    import starrotate as sr

    rotate = sr.RotationModel(time, flux, flux_err)
    lomb_scargle_period = rotate.LS_rotation()
    acf_period = rotate.ACF_rotation()
    gp_period = rotate.GP_rotation()

.. Contents:

User Guide
----------

.. toctree::
   :maxdepth: 2

   user/install
   user/api


Tutorials
---------

.. toctree::
   :maxdepth: 2

   tutorials/Tutorial


License & attribution
---------------------

Copyright 2018, Ruth Angus.

The source code is made available under the terms of the MIT license.

If you make use of this code, please cite this package and its dependencies.
You can find more information about how and what to cite in the
:ref:`citation` documentation.

* :ref:`search`

