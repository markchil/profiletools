The :py:mod:`profiletools` data model
=====================================

The :py:class:`~profiletools.core.Profile` class
------------------------------------------------

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

Channels
--------

:py:mod:`profiletools` understands that particular data should be treated as a
unit during averaging and so forth. Such a unit could correspond to all of the
points taken at a given time, or all of the points taken by a given instrument.
The attribute `channels` is an array with shape (`M`, `X_dim`). By default this
array is just a copy of `X` such that measurements at the exact same locations
are grouped together. But, suppose you have sensors at different locations
taking time-resolved measurements. Hence, `X_dim` is two: the first column of
`X` is the time and the second is the spatial coordinate of the sensor. But say
each sensor has a coordinate that varies slightly in time: just using the
default choice for `channels` will cause each individual measurement from each
sensor to be treated as an independent channel, and time averaging will not have
the desired effect. Instead, the second column of `channels` can be set such
that all measurements from a given sensor have the same value and are hence
treated together when averaging data.

Linearly transformed quantities
-------------------------------

:py:class:`~profiletools.core.Profile` objects can also incorporate quantities
which are linear transformations of the underlying point measurements stored in
`X` and `y`. Each channel of a transformed sensor is stored in a
:py:class:`~profiletools.core.Channel` object. This object stores the data
values in attribute `y` which has shape (`M`,) along with the associated
uncertainty `err_y`. Each measurement :math:`y` is taken to be a linear
transformation :math:`y=Tf(X)` where :math:`X` is a collection of `N` points and
:math:`f(X)` refers to the latent variables (i.e., what is stored as `y` in the
:py:class:`~profiletools.core.Profile` itself). The transformation matrices associated with each of
the observations in `y` are stored in the attribute `T` which is an array with
shape (`M`, `N`). The locations used are stored in the attribute `X` which has
shape (`M`, `N`, `X_dim`), with the associated uncertainties stored in `err_X`.
The :py:class:`~profiletools.core.Channel` instances associated with a given
:py:class:`~profiletools.core.Profile` instance are stored in the attribute
`transformed`.

Averaging data
--------------

Many different techniques for averaging the data and computing the associated
uncertainties are supported, refer to :py:func:`~profiletools.core.average_points`
for more details. By carrying out all averaging within a given channel using
this function, it is straightforward to add additional capabilities as needed.

:py:mod:`gptools` integration
-----------------------------

:py:mod:`profiletools` features very tight integration with the :py:mod:`gptools`
package (https://github.com/markchil/gptools/, http://gptools.readthedocs.org/)
to perform Gaussian process fits. Creating a Gaussian process (GP) for data of
arbitrary `X_dim` is as simple as calling the
:py:meth:`~profiletools.core.Profile.create_gp` method of the
:py:class:`~profiletools.core.Profile` instance. The GP can then be trained by
calling :py:meth:`~profiletools.core.Profile.find_gp_MAP_estimate`. Once this is
complete, the smoothed curve can be obtained using
:py:meth:`~profiletools.core.Profile.smooth`. If additional adjustments to the
:py:class:`~gptools.gaussian_process.GaussianProcess` instance are needed, it is
kept in the `gp` attribute of the :py:class:`~profiletools.core.Profile`
instance.
