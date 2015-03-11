Additional patterns and examples
================================

Weighted versus unweighted averaging
------------------------------------

Diagnostics like CTS and ETS have computed uncertainties that can be used to
weight the data during averaging to give a better representation of the sample
statistics. But, the other diagnostics do not: an assumed value (typically 10%)
is used when the data are loaded. This should be replaced with the unweighted
sample standard deviation when the data are averaged in order to give an honest
assessment of the variability in the quantity. To combine weighted and
unweighted averaging, you should create the profiles separately::
    
    p = Te(1101014006, include=['CTS', 'ETS'], abscissa='r/a', t_min=1.0, t_max=1.5)
    p.time_average(weighted=True)
    p_ECE = Te(1101014006, include=['GPC', 'GPC2', 'FRCECE'], abscissa='r/a', t_min=1.0, t_max=1.5)
    p_ECE.time_average(weighted=False)
    p.add_profile(p_ECE)

This example uses the
:py:meth:`~profiletools.CMod.BivariatePlasmaProfile.add_profile` method to merge
the data from `p_ECE` into `p`.

Multiple time slices
--------------------

There is considerable overhead associated with loading the data from the tree
and performing coordinate conversions. Since time averaging mutates the
:py:class:`~profiletools.CMod.BivariatePlasmaProfile` instance in place, it is
necessary to keep a copy of the master profile with all of the data. This is
accomplished using :py:func:`copy.deepcopy`::
    
    p_master = ne(1101014006, include=['CTS', 'ETS'], abscissa='r/a')
    windows = [(1.0, 1.1), (1.1, 1.2)]
    for w in windows:
        p = copy.deepcopy(p_master)
        p.remove_points((p.X[:, 0] < w[0]) | (p.X[:, 0] > w[1]))
        p.time_average(weighted=True)
        p.find_gp_MAP_estimate()
        mean, std = p.smooth(roa)

Unless the plasma is changing rapidly you can probably save some time by setting
the optimal hyperparameters from one time slice as the initial guess for the
next time slice and setting `random_starts` to zero.