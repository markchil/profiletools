profiletools: Classes for working with profile data of arbitrary dimension
==========================================================================

Source home: https://github.com/markchil/profiletools

Overview
--------

:py:mod:`profiletools` is a Python package that provides a convenient, powerful
and extensible way of working with multivariate data, particularly profile data
from magnetic plasma confinement devices. :py:mod:`profiletools` features deep
integration with :py:mod:`gptools` to support Gaussian process regression (GPR).

The :py:mod:`profiletools` data model
-------------------------------------

The core class of :py:mod:`profiletools` is the :py:class:`~profiletools.core.Profile`.
This class is designed primarily to hold point measurements of some quantity,
which may depend on an arbitrary number of variables and can be sampled at
arbitrary locations -- there is no implicit assumption that observations lie on
an orderly grid. Internally, a :py:class:`~profiletools.core.Profile` instance
stores the independent variables in attribute `X`. `X` is an array with shape
(`M`, `X_dim`), where `M` is the number of observations and `X_dim` is the
number of independent variables. The observations themselves are stored in the
attribute `y`, which is an array of shape (`M`,). This is essentially how a
sparse matrix is stored and is how :py:mod:`profiletools` can be so flexible
about how many independent variables there are and where they are sampled. There
can be uncertainties on both the independent variables (stored in the attribute
`err_X`) and on the dependent variable (stored in the attribute `err_y`).

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
   
   profiletools

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

