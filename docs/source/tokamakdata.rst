Plasma profile data
===================

:py:mod:`profiletools` is primarily designed for working with profile data from
magnetic confinement fusion devices, namely the Alcator C-Mod tokamak at MIT.
The :py:class:`~profiletools.CMod.BivariatePlasmaProfile` class is an extension
of :py:class:`~profiletools.core.Profile` designed for this particular use case.

Data model
----------

Plasma profile data are functions of space (1, 2 or 3 coordinates) and time
(hence the term "bivariate" even when `X_dim` is greater than 2).
Time is always the first column in `X`, with the remaining spatial coordinates
forming the other columns.

Tokamak coordinate systems
--------------------------

:py:class:`~profiletools.CMod.BivariatePlasmaProfile` uses :py:mod:`eqtools`
(https://github.com/PSFCPlasmaTools/eqtools/, http://eqtools.readthedocs.org/)
to support the myriad coordinate systems used in tokamak research. Coordinate
transforms are handled using the
:py:meth:`~profiletools.CMod.BivariatePlasmaProfile.convert_abscissa` method.

Constraints for Gaussian process regression
-------------------------------------------

:py:class:`~profiletools.CMod.BivariatePlasmaProfile` provides two methods for
adding constraints to the Gaussian process created with
:py:meth:`~profiletools.CMod.BivariatePlasmaProfile.create_gp`:
:py:meth:`~profiletools.CMod.BivariatePlasmaProfile.constrain_slope_on_axis`
applies a zero slope constraint at the magnetic axis and
:py:meth:`~profiletools.CMod.BivariatePlasmaProfile.constrain_at_limiter`
applies approximate zero slope and value constraints at the location of the
limiter.
