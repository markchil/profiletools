# Copyright 2014 Mark Chilenski
# This program is distributed under the terms of the GNU General Purpose License (GPL).
# Refer to http://www.gnu.org/licenses/gpl.txt
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Provides classes for working with Alcator C-Mod data via MDSplus.
"""

from __future__ import division

from .core import Profile

import MDSplus
import scipy
import scipy.interpolate
import eqtools
import warnings

_X_label_mapping = {'psinorm': r'$\psi_n$',
                    'phinorm': r'$\phi_n$',
                    'volnorm': '$V_n$',
                    'Rmid': '$R_{mid}$',
                    'r/a': '$r/a$',
                    'sqrtpsinorm': r'$\sqrt{\psi_n}$',
                    'sqrtphinorm': r'$\sqrt{\phi_n}$',
                    'sqrtvolnorm': r'$\sqrt{V_n}$',
                    'sqrtr/a': r'$\sqrt{r/a}$'}
_X_unit_mapping = {'psinorm': '',
                   'phinorm': '',
                   'volnorm': '',
                   'Rmid': 'm',
                   'r/a': '',
                   'sqrtpsinorm': '',
                   'sqrtphinorm': '',
                   'sqrtvolnorm': '',
                   'sqrtr/a': ''}

class BivariatePlasmaProfile(Profile):
    """Class to represent bivariate (y=f(t, psi)) plasma data.

    The first column of `X` is always time. If the abscissa is 'RZ', then the
    second column is `R` and the third is `Z`. Otherwise the second column is
    the desired abscissa (psinorm, etc.).
    """
    def convert_abscissa(self, new_abscissa):
        """Convert the internal representation of the abscissa to new coordinates.

        Right now, only limited mappings are supported, and must be performed
        BEFORE any time averaging has been carried out.

        Supported original abscissae are:

            ====  =====================================================
            RZ    (R, Z) ordered pairs in physical machine coordinates.
            Rmid  Mapped midplane major radius
            ====  =====================================================

        The target abcissae are what are supported by `rz2rho` and `rmid2rho`
        from the `eqtools` package. Namely,

            ======= ========================
            psinorm Normalized poloidal flux
            phinorm Normalized toroidal flux
            volnorm Normalized volume
            Rmid    Midplane major radius
            r/a     Normalized minor radius
            ======= ========================
                
        Additionally, each valid option may be prepended with 'sqrt'
        to return the square root of the desired normalized unit.

        Parameters
        ----------
        new_abscissa : str
            The new abscissa to convert to. Valid options are defined above.
        """
        # TODO: This assumes the data haven't been averaged along t yet!
        # TODO: NEEDS A LOT OF WORK!
        if self.X_labels[0] != '$t$':
            raise ValueError("Can't convert abscissa after time-averaging at this point!")
        if self.abscissa == new_abscissa:
            return
        elif self.abscissa == 'RZ':
            new_rho = self.efit_tree.rz2rho(new_abscissa, self.X[:, 1], self.X[:, 2], self.X[:, 0], each_t=False)
            self.channels = self.channels[:, 0:2]                
            self.X_dim = 2
        elif self.abscissa == 'Rmid':
            new_rho = self.efit_tree.rmid2rho(new_abscissa, self.X[:, 1], self.X[:, 0], each_t=False)
        else:
            raise NotImplementedError("Conversion from that abscissa is not supported!")
        self.X = scipy.hstack((self.X[:, 0], new_rho))
        self.X_labels = [self.X_labels[0], _X_label_mapping[new_abscissa]]
        self.X_units = [self.X_units[0], _X_unit_mapping[new_abscissa]]
        self.err_X = scipy.hstack((self.err_X[:, 0], scipy.zeros_like(self.X[:, 0])))
        self.abscissa = new_abscissa

    def time_average(self, **kwargs):
        """Compute the time average of the quantity.

        Stores the original bounds of `t` to `self.t_min` and `self.t_max`.

        All parameters are passed to :py:meth:`average_data`.
        """
        self.t_min = self.X[:, 0].min()
        self.t_max = self.X[:, 0].max()
        if self.abscissa == 'RZ':
            self.drop_axis(1)
        self.average_data(axis=0, **kwargs)
    
    def add_profile(self, other):
        """Absorbs the data from one profile object.

        Parameters
        ----------
        other : :py:class:`Profile`
            :py:class:`Profile` to absorb.
        """
        # Warn about merging profiles from different shots:
        if self.shot != other.shot:
            warnings.warn("Merging data from two different shots: %d and %d" % (self.shot, other.shot,))
        other.convert_abscissa(self.abscissa)
        # Split off the diagnostic description when merging profiles:
        super(BivariatePlasmaProfile, self).add_profile(other)
        if self.y_label != other.y_label:
            self.y_label = self.y_label.split(', ')[0]

def neCTS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None):
    """Returns a profile representing electron density from the core Thomson scattering system.

    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'RZ'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    """
    p = BivariatePlasmaProfile(X_dim=3,
                               X_units=['s', 'm', 'm'],
                               y_units='$10^{20}$ m$^{-3}$',
                               X_labels=['$t$', '$R$', '$Z$'],
                               y_label='$n_e$, CTS')

    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)

    N_ne_TS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:ne_rz')

    t_ne_TS = N_ne_TS.dim_of().data()
    ne_TS = N_ne_TS.data() / 1e20
    dev_ne_TS = electrons.getNode(r'yag_new.results.profiles:ne_err').data() / 1e20
    
    Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted').data()
    R_CTS = (electrons.getNode(r'yag.results.param:r').data() *
             scipy.ones_like(Z_CTS))
    channels = range(0, len(Z_CTS))
    
    t_grid, Z_grid = scipy.meshgrid(t_ne_TS, Z_CTS)
    t_grid, R_grid = scipy.meshgrid(t_ne_TS, R_CTS)
    t_grid, channel_grid = scipy.meshgrid(t_ne_TS, channels)
    
    ne = ne_TS.flatten()
    err_ne = dev_ne_TS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())
    
    X = scipy.hstack((t.T, R.T, Z.T))
    
    p.shot = shot
    if efit_tree is None:
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, ne, err_y=err_ne, channels={1: channels})
    # Remove flagged points:
    p.remove_points(scipy.isnan(p.err_y) | (p.err_y == 0.0))
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    return p

def neETS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None):
    """Returns a profile representing electron density from the edge Thomson scattering system.

    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'RZ'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    """
    p = BivariatePlasmaProfile(X_dim=3,
                               X_units=['s', 'm', 'm'],
                               y_units='$10^{20}$ m$^{-3}$',
                               X_labels=['$t$', '$R$', '$Z$'],
                               y_label='$n_e$, ETS')

    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)
    
    N_ne_ETS = electrons.getNode(r'yag_edgets.results:ne')
    
    t_ne_ETS = N_ne_ETS.dim_of().data()
    ne_ETS = N_ne_ETS.data() / 1e20
    dev_ne_ETS = electrons.getNode(r'yag_edgets.results:ne:error').data() / 1e20
    
    Z_ETS = electrons.getNode(r'yag_edgets.data:fiber_z').data()
    R_ETS = (electrons.getNode(r'yag.results.param:R').data() *
             scipy.ones_like(Z_ETS))
    channels = range(0, len(Z_ETS))
    
    t_grid, Z_grid = scipy.meshgrid(t_ne_ETS, Z_ETS)
    t_grid, R_grid = scipy.meshgrid(t_ne_ETS, R_ETS)
    t_grid, channel_grid = scipy.meshgrid(t_ne_ETS, channels)
    
    ne = ne_ETS.flatten()
    err_ne = dev_ne_ETS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())
    
    X = scipy.hstack((t.T, R.T, Z.T))
    
    p.shot = shot
    if efit_tree is None:
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, ne, err_y=err_ne, channels={1: channels})
    # Remove flagged points:
    p.remove_points(scipy.isnan(p.err_y) | (p.err_y == 0.0) |
                    ((p.y == 0.0) & (p.err_y == 2)))
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    return p

def ne(shot, include=['CTS', 'ETS'], **kwargs):
    """Returns a profile representing electron density from both the core and edge Thomson scattering systems.

    Parameters
    ----------
    shot : int
        The shot number to load.
    include : list of str, optional
        The data sources to include. Valid options are:

            ===  =======================
            CTS  Core Thomson scattering
            ETS  Edge Thomson scattering
            ===  =======================

        The default is to include all data sources.
    **kwargs
        All remaining parameters are passed to the individual loading methods.
    """
    if 'electrons' not in kwargs:
        kwargs['electrons'] = MDSplus.Tree('electrons', shot)
    if 'efit_tree' not in kwargs:
        kwargs['efit_tree'] = eqtools.CModEFITTree(shot)
    p_list = []
    for system in include:
        if system == 'CTS':
            p_list.append(neCTS(shot, **kwargs))
        elif system == 'ETS':
            p_list.append(neETS(shot, **kwargs))
        else:
            raise ValueError("Unknown profile '%s'." % (system,))
    
    p = p_list.pop()
    for p_other in p_list:
        p.add_profile(p_other)
    
    return p

def TeCTS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None):
    """Returns a profile representing electron temperature from the core Thomson scattering system.

    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'RZ'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    """
    p = BivariatePlasmaProfile(X_dim=3,
                               X_units=['s', 'm', 'm'],
                               y_units='keV',
                               X_labels=['$t$', '$R$', '$Z$'],
                               y_label='$T_e$, CTS')

    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)

    N_Te_TS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:Te_rz')

    t_Te_TS = N_Te_TS.dim_of().data()
    Te_TS = N_Te_TS.data()
    dev_Te_TS = electrons.getNode(r'yag_new.results.profiles:Te_err').data()
    
    Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted').data()
    R_CTS = (electrons.getNode(r'yag.results.param:r').data() *
             scipy.ones_like(Z_CTS))
    channels = range(0, len(Z_CTS))
    
    t_grid, Z_grid = scipy.meshgrid(t_Te_TS, Z_CTS)
    t_grid, R_grid = scipy.meshgrid(t_Te_TS, R_CTS)
    t_grid, channel_grid = scipy.meshgrid(t_Te_TS, channels)
    
    Te = Te_TS.flatten()
    err_Te = dev_Te_TS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())
    
    X = scipy.hstack((t.T, R.T, Z.T))
    
    p.shot = shot
    if efit_tree is None:
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, Te, err_y=err_Te, channels={1: channels})
    # Remove flagged points:
    p.remove_points(scipy.isnan(p.err_y) | (p.err_y == 0.0))
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    return p

def TeETS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None):
    """Returns a profile representing electron temperature from the edge Thomson scattering system.

    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'RZ'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    """
    p = BivariatePlasmaProfile(X_dim=3,
                               X_units=['s', 'm', 'm'],
                               y_units='keV',
                               X_labels=['$t$', '$R$', '$Z$'],
                               y_label='$T_e$, ETS')

    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)

    N_Te_TS = electrons.getNode(r'yag_edgets.results:te')

    t_Te_TS = N_Te_TS.dim_of().data()
    Te_TS = N_Te_TS.data() / 1e3
    dev_Te_TS = electrons.getNode(r'yag_edgets.results:te:error').data() / 1e3
    
    Z_CTS = electrons.getNode(r'yag_edgets.data:fiber_z').data()
    R_CTS = (electrons.getNode(r'yag.results.param:r').data() *
             scipy.ones_like(Z_CTS))
    channels = range(0, len(Z_CTS))
    
    t_grid, Z_grid = scipy.meshgrid(t_Te_TS, Z_CTS)
    t_grid, R_grid = scipy.meshgrid(t_Te_TS, R_CTS)
    t_grid, channel_grid = scipy.meshgrid(t_Te_TS, channels)
    
    Te = Te_TS.flatten()
    err_Te = dev_Te_TS.flatten()
    Z = scipy.atleast_2d(Z_grid.flatten())
    R = scipy.atleast_2d(R_grid.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())
    
    X = scipy.hstack((t.T, R.T, Z.T))
    
    p.shot = shot
    if efit_tree is None:
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    p.abscissa = 'RZ'
    
    p.add_data(X, Te, err_y=err_Te, channels={1: channels})
    # Remove flagged points:
    p.remove_points(scipy.isnan(p.err_y) | (p.err_y == 0.0) |
                    ((p.y == 0) & (p.err_y == 1)))
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    return p

def TeFRCECE(shot, rate='s', cutoff=0.15, abscissa='Rmid', t_min=None, t_max=None,
             electrons=None, efit_tree=None):
    """Returns a profile representing electron temperature from the FRCECE system.
    
    Parameters
    ----------
    shot : int
        The shot number to load.
    rate : {'s', 'f'}, optional
        Which timebase to use -- the fast or slow data. Default is 's' (slow).
    cutoff : float, optional
        The cutoff value for eliminating cut-off points. All points with values
        less than this will be discarded. Default is 0.15.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'Rmid'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    """
    p = BivariatePlasmaProfile(X_dim=2,
                               X_units=['s', 'm'],
                               y_units='keV',
                               X_labels=['$t$', '$R_{mid}$'],
                               y_label='$T_e$, FRCECE (%s)' % (rate,))

    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)
    if efit_tree is None:
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree

    Te_FRC = []
    R_mid_FRC = []
    t_FRC = []
    channels = []
    for k in xrange(0, 32):
        N = electrons.getNode(r'frcece.data.ece%s%02d' % (rate, k + 1,))
        Te = N.data()
        Te_FRC.extend(Te)
        # There appears to consistently be an extra point. Lacking a better
        # explanation, I will knock off the last point:
        t = N.dim_of().data()[:-1]
        t_FRC.extend(t)
        
        N_R = electrons.getNode(r'frcece.data.rmid_%02d' % (k + 1,))
        R_mid = N_R.data().flatten()
        t_R_FRC = N_R.dim_of().data()
        R_mid_FRC.extend(
            scipy.interpolate.InterpolatedUnivariateSpline(t_R_FRC, R_mid)(t)
        )
        
        channels.extend([k + 1] * len(Te))
    
    Te = scipy.asarray(Te_FRC)
    t = scipy.atleast_2d(scipy.asarray(t_FRC))
    R_mid = scipy.atleast_2d(scipy.asarray(R_mid_FRC))
    
    X = scipy.hstack((t.T, R_mid.T))
    
    p.shot = shot
    p.abscissa = 'Rmid'
    
    p.add_data(X, Te, channels={1: scipy.asarray(channels)})
    # Remove flagged points:
    # I think these are cut off channels, but I am not sure...
    p.remove_points(p.y < cutoff)
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    return p

def TeGPC2(shot, abscissa='Rmid', t_min=None, t_max=None, electrons=None,
           efit_tree=None):
    """Returns a profile representing electron temperature from the GPC2 system.

    Parameters
    ----------
    shot : int
        The shot number to load.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'Rmid'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    """
    p = BivariatePlasmaProfile(X_dim=2,
                               X_units=['s', 'm'],
                               y_units='keV',
                               X_labels=['$t$', '$R_{mid}$'],
                               y_label='$T_e$, GPC2')
    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)
    if efit_tree is None:
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree

    N_GPC2 = electrons.getNode('gpc_2.results.gpc2_te')
    Te_GPC2 = N_GPC2.data()
    t_GPC2 = N_GPC2.dim_of().data()
    
    N_R = electrons.getNode('gpc_2.results.radii')
    R_mid_GPC2 = N_R.data()
    t_R_GPC2 = N_R.dim_of().data()
    
    channels = range(0, Te_GPC2.shape[0])

    t_grid, channel_grid = scipy.meshgrid(t_GPC2, channels)

    R_GPC2 = scipy.zeros_like(t_grid)
    for k in channels:
        R_GPC2[k, :] = scipy.interpolate.InterpolatedUnivariateSpline(
            t_R_GPC2, R_mid_GPC2[k, :]
        )(t_GPC2)

    Te = Te_GPC2.flatten()
    R = scipy.atleast_2d(R_GPC2.flatten())
    channels = channel_grid.flatten()
    t = scipy.atleast_2d(t_grid.flatten())

    X = scipy.hstack((t.T, R.T))

    p.shot = shot
    p.abscissa = 'Rmid'

    p.add_data(X, Te, channels={1: channels})
    
    # Remove flagged points:
    p.remove_points(p.y == 0)
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)

    p.convert_abscissa(abscissa)

    return p

def TeGPC(shot, cutoff=0.15, abscissa='Rmid', t_min=None, t_max=None, electrons=None,
          efit_tree=None):
    """Returns a profile representing electron temperature from the GPC system.

    Parameters
    ----------
    shot : int
        The shot number to load.
    cutoff : float, optional
        The cutoff value for eliminating cut-off points. All points with values
        less than this will be discarded. Default is 0.15.
    abscissa : str, optional
        The abscissa to use for the data. The default is 'Rmid'.
    t_min : float, optional
        The smallest time to include. Default is None (no lower bound).
    t_max : float, optional
        The largest time to include. Default is None (no upper bound).
    electrons : MDSplus.Tree, optional
        An MDSplus.Tree object open to the electrons tree of the correct shot.
        The shot of the given tree is not checked! Default is None (open tree).
    efit_tree : eqtools.CModEFITTree, optional
        An eqtools.CModEFITTree object open to the correct shot. The shot of the
        given tree is not checked! Default is None (open tree).
    """
    p = BivariatePlasmaProfile(X_dim=2,
                               X_units=['s', 'm'],
                               y_units='keV',
                               X_labels=['$t$', '$R_{mid}$'],
                               y_label='$T_e$, GPC2')
    if electrons is None:
        electrons = MDSplus.Tree('electrons', shot)
    if efit_tree is None:
        p.efit_tree = eqtools.CModEFITTree(shot)
    else:
        p.efit_tree = efit_tree
    
    Te_GPC = []
    R_mid_GPC = []
    t_GPC = []
    channels = []
    for k in xrange(0, 9):
        N = electrons.getNode(r'ece.gpc_results.te.te%d' % (k + 1,))
        Te = N.data()
        Te_GPC.extend(Te)
        t = N.dim_of().data()
        t_GPC.extend(t)
        
        N_R = electrons.getNode(r'ece.gpc_results.rad.r%d' % (k + 1,))
        R_mid = N_R.data()
        t_R_mid = N_R.dim_of().data()
        R_mid_GPC.extend(
            scipy.interpolate.InterpolatedUnivariateSpline(t_R_mid, R_mid)(t)
        )
        
        channels.extend([k + 1] * len(Te))

    Te = scipy.asarray(Te_GPC)
    t = scipy.atleast_2d(scipy.asarray(t_GPC))
    R_mid = scipy.atleast_2d(scipy.asarray(R_mid_GPC))
    
    X = scipy.hstack((t.T, R_mid.T))
    
    p.shot = shot
    p.abscissa = 'Rmid'
    
    p.add_data(X, Te, channels={1: scipy.asarray(channels)})
    
    # Remove flagged points:
    # I think these are cut off channels, but I am not sure...
    p.remove_points(p.y < cutoff)
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)

    p.convert_abscissa(abscissa)

    return p

def Te(shot, include=['CTS', 'ETS', 'FRCECE', 'GPC2', 'GPC'], FRCECE_rate='s',
       FRCECE_cutoff=0.15, GPC_cutoff=0.15, **kwargs):
    """Returns a profile representing electron temperature from the Thomson scattering and ECE systems.

    Parameters
    ----------
    shot : int
        The shot number to load.
    include : list of str, optional
        The data sources to include. Valid options are:

            ======  ===============================
            CTS     Core Thomson scattering
            ETS     Edge Thomson scattering
            FRCECE  FRC electron cyclotron emission
            GPC     Grating polychromator
            GPC2    Grating polychromator 2
            ======  ===============================

        The default is to include all data sources.
    FRCECE_rate : {'s', 'f'}, optional
        Which timebase to use for FRCECE -- the fast or slow data. Default is
        's' (slow).
    FRCECE_cutoff : float, optional
        The cutoff value for eliminating cut-off points from FRCECE. All points
        with values less than this will be discarded. Default is 0.15.
    GPC_cutoff : float, optional
        The cutoff value for eliminating cut-off points from GPC. All points
        with values less than this will be discarded. Default is 0.15.
    **kwargs
        All remaining parameters are passed to the individual loading methods.
    """
    if 'electrons' not in kwargs:
        kwargs['electrons'] = MDSplus.Tree('electrons', shot)
    if 'efit_tree' not in kwargs:
        kwargs['efit_tree'] = eqtools.CModEFITTree(shot)
    p_list = []
    for system in include:
        if system == 'CTS':
            p_list.append(TeCTS(shot, **kwargs))
        elif system == 'ETS':
            p_list.append(TeETS(shot, **kwargs))
        elif system == 'FRCECE':
            p_list.append(TeFRCECE(shot, rate=FRCECE_rate, cutoff=FRCECE_cutoff, **kwargs))
        elif system == 'GPC2':
            p_list.append(TeGPC2(shot, **kwargs))
        elif system == 'GPC':
            p_list.append(TeGPC(shot, cutoff=GPC_cutoff, **kwargs))
        else:
            raise ValueError("Unknown profile '%s'." % (system,))
    
    p = p_list.pop()
    for p_other in p_list:
        p.add_profile(p_other)
    
    return p
