profiletools: Classes for working with profile data of arbitrary dimension
==========================================================================

Source home: https://github.com/markchil/profiletools

Overview
--------

:py:mod:`profiletools` is a Python package that provides a convenient, powerful
and extensible way of working with multivariate data, particularly profile data
from magnetic plasma confinement devices. :py:mod:`profiletools` features deep
integration with :py:mod:`gptools` to support Gaussian process regression (GPR).

Notes
-----

:py:mod:`profiletools` has been developed and tested on Python 2.7 and scipy
0.14.0. It may work just as well on other versions, but has not been tested.

:py:mod:`profiletools` uses the module :py:mod:`gptools` for GPR. You can find
the source at https://github.com/markchil/gptools/ and the documentation at
http://gptools.readthedocs.org/

:py:mod:`profiletools` uses the module :py:mod:`eqtools` for tokamak coordinate
transformations. You can find the source at https://github.com/PSFCPlasmaTools/eqtools/
and the documentation at http://eqtools.readthedocs.org/

If you find this software useful, please be sure to cite it:

M.A. Chilenski (2014). profiletools: Classes for working with profile data of
arbitrary dimension, GNU General Public License. github.com/markchil/profiletools

Once I put together a formal publication on this software and its applications,
this readme will be updated with the relevant citation.

Contents
--------

.. toctree::
   :maxdepth: 4
   
   datamodel
   tokamakdata
   cmoddata
   patterns
   profiletools

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

