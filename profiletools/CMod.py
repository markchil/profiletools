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
import eqtools

class BivariatePlasmaProfile(Profile):
    """Class to represent bivariate (y=f(t, psi)) plasma data.

    The first column of `X` is always time. If the abscissa is 'RZ', then the
    second column is `R` and the third is `Z`. Otherwise the second column is
    the desired abscissa (psi_norm, etc.).
    """
    def convert_abscissa(self, new_abscissa):
        # TODO: This assumes the data haven't been averaged along t yet!
        # TODO: NEEDS A LOT OF WORK!
        if self.abscissa == new_abscissa:
            return
        if self.abscissa == 'RZ':
            if new_abscissa == 'psi_norm':
                psin = self.efit_tree.rz2psinorm(self.X[:, 1], self.X[:, 2], self.X[:, 0], each_t=False)
                self.X = scipy.hstack((self.X[:, 0], psin))
                self.channels = self.channels[:, 0:2]
                self.X_labels = [self.X_labels[0], r'$\psi_n$']
                self.X_units = [self.X_units[0], '']
                self.err_X = scipy.hstack((self.err_X[:, 0], scipy.zeros_like(self.X[:, 0])))
                
                self.X_dim = 2
                
                self.abscissa = new_abscissa
            else:
                raise NotImplementedError("Conversion to that abscissa is not supported!")
        else:
            raise NotImplementedError("Conversion from that abscissa is not supported!")

    def time_average(self, ddof=1):
        """Compute the time average of the quantity.

        Parameters
        ----------
        ddof : int, optional
            The degree of freedom correction used in computing the standard
            deviation. The default is 1, the standard Bessel correction to
            give an unbiased estimate of the variance.
        """
        if self.abscissa == 'RZ':
            self.drop_axis(1)
        self.average_data(axis=0, ddof=ddof)

def neCTS(shot, abscissa='RZ', t_min=None, t_max=None):
    """Returns a profile representing electron density from the core Thomson scattering system.
    """
    p = BivariatePlasmaProfile(X_dim=3,
                               X_units=['s', 'm', 'm'],
                               y_units='$10^{20}$ m$^-3$',
                               X_labels=['$t$', '$R$', '$Z$'],
                               y_label='$n_e$, CTS')

    electrons = MDSplus.Tree('electrons', shot)

    N_ne_TS = electrons.getNode(r'\electrons::top.yag_new.results.profiles:ne_rz')

    t_ne_TS = N_ne_TS.dim_of().data()
    ne_TS = N_ne_TS.data() / 1e20
    dev_ne_TS = electrons.getNode(r'yag_new.results.profiles:ne_err').data() / 1e20
    
    Z_CTS = electrons.getNode(r'yag_new.results.profiles:z_sorted').data()
    R_CTS = electrons.getNode(r'yag.results.param:r').data() * scipy.ones_like(Z_CTS)
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
    p.efit_tree = eqtools.CModEFITTree(shot)
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

