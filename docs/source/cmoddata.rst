Accessing Alcator C-Mod data
============================

:py:mod:`profiletools` provides a collection of functions to access Alcator
C-Mod data. This prevents the user from having to remember the diverse set of
:py:mod:`MDSplus` calls needed to load the data from the tree and delivers the
data in the standard :py:class:`~profiletools.CMod.BivariatePlasmaProfile` class.
Notice that each of these are implemented as a function and not a class -- that
way all of the instances for a given quantity are the same class.

Example
-------

To load the electron density profile from a given shot, simply call the
:py:func:`~profiletools.CMod.ne` function::
    
    p = ne(shot, include=['CTS', 'ETS'])

The optional keyword `include` specifies which signal are included -- in this
case core and edge Thomson scattering. If you want the data expressed in a
specific coordinate system, use the `abscissa` keyword::
    
    p = ne(shot, include=['CTS', 'ETS'], abscissa='r/a')

Or, call :py:meth:`~profiletools.CMod.BivariatePlasmaProfile.convert_abscissa`::
    
    p = ne(shot)
    p.convert_abscissa('r/a')

To request data only from a certain time window, use the `t_min` and `t_max`
keywords. For instance, to get the data from 1.0s to 1.5s, you would type::
    
    p = ne(shot, t_min=1.0, t_max=1.5)

If you want to remove points after having created the
:py:class:`~profiletools.CMod.BivariatePlasmaProfile`, then you can use the
:py:meth:`~profiletools.core.Profile.remove_points` method::
    
    p.remove_points((p.X[:, 0] < t_min) & (p.X[:, 0] > t_max))

If you want to only keep points at specific times (such as points at a specific
sawtooth phase), you can use the
:py:meth:`~profiletools.CMod.BivariatePlasmaProfile:keep_times` method. For each
time point designated, this will find the point in the profile which is closest.
If there are many missing datapoints, blindly applying this technique can result
in data far from the desired point being included. Hence, the `tol` keyword will
cause :py:meth:`~profiletools.CMod.BivariatePlasmaProfile:keep_times` to only
keep points that are within `tol` of the target. So, to keep the points within
1ms of 1.0s, 1.1s and 1.3s, you would type::
    
    p.keep_times([1.0, 1.1, 1.3], tol=1e-3)

Once the data are loaded and confined to the desired window, you can
time-average them. Thomson scattering data have computed uncertainties in the
tree, so you can (and should) use a weighted average::
    
    p.time_average(weighted=True)

There are a wide variety of options for how the data are averaging depending on
the specific application -- see :py:func:`~profiletools.core.average_points` for
more details.

If instead you want to keep all of the points within the designated time window,
you can simply drop that axis from `X`. Recall that time is always the first
column, so you would call::
    
    p.drop_axis(0)

You can plot the data simply by calling
:py:meth:`~profiletools.core.Profile.plot_data`.

Once you have picked the slices you want and/or time-averaged the data, you can
fit a Gaussian process with the following steps::
    
    p.create_gp()
    p.find_gp_MAP_estimate()
    p.plot_gp(ax='gca')

This will plot the smoothed profile on a somewhat sensible grid on the axis
created in the previous call to :py:meth:`~profiletools.core.Profile.plot_data`.

Therefore, the complete example to load and plot the electron density data as a
function of r/a from a given shot averaged over 1.0s to 1.5s is::
    
    p = ne(shot, t_min=1.0, t_max=1.5, abscissa='r/a')
    p.time_average()
    p.plot_data()
    p.create_gp()
    p.find_gp_MAP_estimate()
    p.plot_gp(ax='gca')

Electron density
----------------

The following diagnostics are supported:

* :py:func:`~profiletools.CMod.neCTS`: Core Thomson scattering.
* :py:func:`~profiletools.CMod.neETS`: Edge Thomson scattering.
* :py:func:`~profiletools.CMod.neTCI`: Two-color interferometer. This is a line-
  integrated diagnostic. Loading the data is rather slow because the quadrature
  weights must be computed. Fitting the data is rather slow because of the
  computational cost of including all of the quadrature points in the Gaussian
  process. There are several parameters that let you adjust the tradeoff between
  computational time and accuracy, see the documentation for more details.
* :py:func:`~profiletools.CMod.neReflect`: Scape-off layer reflectometer.
  Because of how these data are stored and processed you need to be very careful
  about how you include them in your fits.

Electron temperature
--------------------

The following diagnostics are supported:

* :py:func:`~profiletools.CMod.TeCTS`: Core Thomson scattering.
* :py:func:`~profiletools.CMod.TeETS`: Edge Thomson scattering.
* :py:func:`~profiletools.CMod.TeFRCECE`: High spatial resolution ECE system.
* :py:func:`~profiletools.CMod.TeGPC`: Grating polychromator ECE system.
* :py:func:`~profiletools.CMod.TeGPC2`: Second grating polychromator ECE system.
* :py:func:`~profiletools.CMod.TeMic`: Michelson interferometer. High frequency
  space resolution but low temporal resolution.

X-ray emissivity
----------------

You must be careful when interpreting the uncertainties on these fits since they
are already inverted/smoothed. This is mostly useful for getting a rough look at
the results of combining the two AXUV systems.

:py:func:`~profiletools.CMod.emissAX` supports both AXA and AXJ through use of
the required `system` argument.
