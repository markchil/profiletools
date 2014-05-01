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

class neCTS(Profile):
    """Object to represent data from the core Thomson scattering system.
    """
    def __init__(self, shot, abscissa='RZ'):
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
        
        self.shot = shot
        self.efit_tree = eqtools.CModEFITTree(shot)
        self.abscissa = 'RZ'
        
        super(neCTS, self).__init__(X_dim=3,
                                    X_units=['s', 'm', 'm'],
                                    y_units='1e20 m^-3',
                                    X_labels=['t', 'R', 'Z'],
                                    y_label='n_e CTS')
        self.add_data(X, ne, err_y=err_ne, channels={1: channels})
        self.remove_points(scipy.isnan(self.err_y) | (self.err_y == 0.0))
        self.convert_abscissa(abscissa)

