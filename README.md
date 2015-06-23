profiletools
============

Classes for working with profile data of arbitrary dimension

`profiletools` is a Python package that provides a convenient way of accessing plasma profile data, combining data from multiple sources and applying Gaussian process regression (GPR) to the combined data sets.

`profiletools` is designed to work with data of arbitrary dimension (x, y, z, t, ...), and is not restricted to plasma data. That being said, most of the development work has focussed on creating tools to work with data from the Alcator C-Mod tokamak. These classes were written in such a way to make it straightforward to implement classes to access data from other plasma experiments.

Documentation for `profiletools` is located at http://profiletools.readthedocs.org/

`profiletools` uses the module `gptools` for GPR. You can find the source at https://github.com/markchil/gptools/ and the documentation at http://gptools.readthedocs.org/

`profiletools` uses the module `eqtools` for tokamak coordinate transformations. You can find the source at https://github.com/PSFCPlasmaTools/eqtools/ and the documentation at http://eqtools.readthedocs.org/

If you find this software useful, please be sure to cite it:

M.A. Chilenski (2014). profiletools: Classes for working with profile data of arbitrary dimension, GNU General Public License. github.com/markchil/profiletools

Once I put together a formal publication on this software and its applications, this readme will be updated with the relevant citation.
