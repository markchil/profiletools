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

from .core import Profile, read_csv, read_NetCDF

import MDSplus
import scipy
import scipy.interpolate
import scipy.stats
import eqtools
import gptools
import warnings
import matplotlib.pyplot as plt

_X_label_mapping = {'psinorm': r'$\psi_n$',
                    'phinorm': r'$\phi_n$',
                    'volnorm': '$V_n$',
                    'Rmid': '$R_{mid}$',
                    'r/a': '$r/a$',
                    'sqrtpsinorm': r'$\sqrt{\psi_n}$',
                    'sqrtphinorm': r'$\sqrt{\phi_n}$',
                    'sqrtvolnorm': r'$\sqrt{V_n}$',
                    'sqrtr/a': r'$\sqrt{r/a}$'}
_abscissa_mapping = {y:x for x, y in _X_label_mapping.iteritems()}
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
    def convert_abscissa(self, new_abscissa, drop_nan=True, ddof=1):
        """Convert the internal representation of the abscissa to new coordinates.
        
        The target abcissae are what are supported by `rho2rho` from the
        `eqtools` package. Namely,

            ======= ========================
            psinorm Normalized poloidal flux
            phinorm Normalized toroidal flux
            volnorm Normalized volume
            Rmid    Midplane major radius
            r/a     Normalized minor radius
            ======= ========================
                
        Additionally, each valid option may be prepended with 'sqrt' to return
        the square root of the desired normalized unit.

        Parameters
        ----------
        new_abscissa : str
            The new abscissa to convert to. Valid options are defined above.
        drop_nan : bool, optional
            Set this to True to drop any elements whose value is NaN following
            the conversion. Default is True (drop NaN elements).
        ddof : int, optional
            Degree of freedom correction to use when time-averaging a conversion.
        """
        if self.abscissa == new_abscissa:
            return
        elif self.X_dim == 1 or (self.X_dim == 2 and self.abscissa == 'RZ'):
            # TODO: Handle the sqrt cases elegantly!
            if self.abscissa.startswith('sqrt') and self.abscissa[4:] == new_abscissa:
                new_rho = scipy.power(self.X[:, 0], 2)
                # Approximate form from uncertainty propagation:
                err_new_rho = self.err_X[:, 0] * 2 * self.X[:, 0]
            elif new_abscissa.startswith('sqrt') and self.abscissa == new_abscissa[4:]:
                new_rho = scipy.power(self.X[:, 0], 0.5)
                # Approximate form from uncertainty propagation:
                err_new_rho = self.err_X[:, 0] / (2 * scipy.sqrt(self.X[:, 0]))
            else:
                times = self._get_efit_times_to_average()
            
                if self.abscissa == 'RZ':
                    new_rhos = self.efit_tree.rz2rho(
                        new_abscissa,
                        self.X[:, 0],
                        self.X[:, 1],
                        times,
                        each_t=True
                    )
                    self.X_dim = 1
                else:
                    new_rhos = self.efit_tree.rho2rho(
                        self.abscissa,
                        new_abscissa,
                        self.X[:, 0],
                        times,
                        each_t=True
                    )
                new_rho = scipy.mean(new_rhos, axis=0)
                err_new_rho = scipy.std(new_rhos, axis=0, ddof=ddof)
                err_new_rho[scipy.isnan(err_new_rho)] = 0
            
            self.X = scipy.atleast_2d(new_rho).T
            self.X_labels = [_X_label_mapping[new_abscissa]]
            self.X_units = [_X_unit_mapping[new_abscissa]]
            self.err_X = scipy.atleast_2d(err_new_rho).T
        else:
            if self.abscissa.startswith('sqrt') and self.abscissa[4:] == new_abscissa:
                new_rho = scipy.power(self.X[:, 1], 2)
            elif new_abscissa.startswith('sqrt') and self.abscissa == new_abscissa[4:]:
                new_rho = scipy.power(self.X[:, 1], 0.5)
            elif self.abscissa == 'RZ':
                # Need to handle this case separately because of the extra column:
                new_rho = self.efit_tree.rz2rho(new_abscissa,
                                                self.X[:, 1],
                                                self.X[:, 2],
                                                self.X[:, 0],
                                                each_t=False)
                self.channels = self.channels[:, 0:2]                
                self.X_dim = 2
            else:
                new_rho = self.efit_tree.rho2rho(
                    self.abscissa,
                    new_abscissa,
                    self.X[:, 1],
                    self.X[:, 0],
                    each_t=False
                )
            err_new_rho = scipy.zeros_like(self.X[:, 0])
        
            self.X = scipy.hstack((scipy.atleast_2d(self.X[:, 0]).T, scipy.atleast_2d(new_rho).T))
            self.X_labels = [self.X_labels[0], _X_label_mapping[new_abscissa]]
            self.X_units = [self.X_units[0], _X_unit_mapping[new_abscissa]]
            self.err_X = scipy.hstack(
                (
                    scipy.atleast_2d(self.err_X[:, 0]).T,
                    scipy.atleast_2d(err_new_rho).T
                )
            )
        self.abscissa = new_abscissa
        if drop_nan:
            self.remove_points(scipy.isnan(self.X).any(axis=1))

    def time_average(self, **kwargs):
        """Compute the time average of the quantity.

        Stores the original bounds of `t` to `self.t_min` and `self.t_max`.

        All parameters are passed to :py:meth:`average_data`.
        """
        self.t_min = self.X[:, 0].min()
        self.t_max = self.X[:, 0].max()
        # if self.abscissa == 'RZ':
        #     self.drop_axis(1)
        self.average_data(axis=0, **kwargs)
    
    def drop_axis(self, axis):
        if self.X_labels[axis] == '$t$':
            self.t_min = self.X[:, 0].min()
            self.t_max = self.X[:, 0].max()
        super(BivariatePlasmaProfile, self).drop_axis(axis)
    
    def keep_times(self, times):
        if self.X_labels[0] != '$t$':
            raise ValueError("Cannot keep specific time slices after time-averaging!")
        try:
            iter(times)
        except TypeError:
            times = [times]
        self.times = times
        self.keep_slices(0, times)
    
    def add_profile(self, other):
        """Absorbs the data from one profile object.

        Parameters
        ----------
        other : :py:class:`Profile`
            :py:class:`Profile` to absorb.
        """
        # Warn about merging profiles from different shots:
        if self.shot != other.shot:
            warnings.warn("Merging data from two different shots: %d and %d"
                          % (self.shot, other.shot,))
        other.convert_abscissa(self.abscissa)
        # Split off the diagnostic description when merging profiles:
        super(BivariatePlasmaProfile, self).add_profile(other)
        if self.y_label != other.y_label:
            self.y_label = self.y_label.split(', ')[0]

    def remove_edge_points(self, allow_conversion=True):
        """Removes points that are outside the LCFS, including those with NaN for the radial coordinate.

        Must be called when the abscissa is a normalized coordinate. Assumes
        that the last column of `self.X` is space: so it will do the wrong
        thing if you have already taken an average along space.
        
        Parameters
        ----------
        allow_conversion : bool, optional
            If True and self.abscissa is 'RZ', then the profile will be
            converted to psinorm and the points will be dropped. Default is True
            (allow conversion).
        """
        # TODO: This needs a lot more work!
        if self.abscissa == 'RZ':
            if allow_conversion:
                warnings.warn("Removal of edge points not supported with abscissa RZ. Will convert to psinorm.")
                self.convert_abscissa('psinorm')
            else:
                raise ValueError("Removal of edge points not supported with abscissa RZ!")
        if self.abscissa == 'r/a' or 'norm' in self.abscissa:
            x_out = 1.0
        elif self.abscissa == 'Rmid':
            if self.X_dim == 1:
                if hasattr(self, 'times'):
                    x_out = scipy.mean(self.efit_tree.getRmidOutSpline()(self.times))
                else:
                    t_EFIT = self.efit_tree.getTimeBase()
                    
                    if self.t_min != self.t_max:
                        t_EFIT = t_EFIT[(t_EFIT >= self.t_min) & (t_EFIT <= self.t_max)]
                    else:
                        idx = self.efit_tree._getNearestIdx(self.t_min, t_EFIT)
                        t_EFIT = t_EFIT[idx]
                    x_out = scipy.mean(self.efit_tree.getRmidOutSpline()(t_EFIT))
            else:
                assert self.X_dim == 2
                x_out = self.efit_tree.getRmidOutSpline()(scipy.asarray(self.X[:, 0]).flatten())
        else:
            raise ValueError("Removal of edge points not supported with abscissa %s!" % (self.abscissa,))
        self.remove_points((scipy.asarray(self.X[:, -1]).flatten() >= x_out) |
                           scipy.asarray(scipy.isnan(self.X[:, -1])).flatten())
    
    def constrain_slope_on_axis(self, err=0, times=None):
        """Constrains the slope at the magnetic axis of this Profile's Gaussian process to be zero.
        
        Note that this is accomplished approximately for bivariate data by
        specifying the slope to be zero at the magnetic axis for a number of
        points in time.
        
        It is assumed that the Gaussian process has already been created with
        a call to :py:meth:`create_gp`.
        
        It is required that the abscissa be either Rmid or one of the
        normalized coordinates.
        
        Parameters
        ----------
        err : float, optional
            The uncertainty to place on the slope constraint. The default is 0
            (slope constraint is exact). This could also potentially be an
            array for bivariate data where you wish to have the uncertainty
            vary in time.
        times : array-like, optional
            The times to impose the constraint at. Default is to use the
            unique time values in `X[:, 0]`.
        """
        if self.X_dim == 1:
            if self.abscissa == 'Rmid':
                t_EFIT = self.efit_tree.getTimeBase()
                if hasattr(self, 'times'):
                    idx = self.efit_tree._getNearestIdx(self.times, t_EFIT)
                    x0 = scipy.mean(self.efit_tree.getMagR()[idx])
                elif self.t_min != self.t_max:
                    x0 = scipy.mean(self.efit_tree.getMagR()[(t_EFIT >= self.t_min) &
                                                             (t_EFIT <= self.t_max)])
                else:
                    idx = self.efit_tree._getNearestIdx(self.t_min, t_EFIT)
                    x0 = self.efit_tree.getMagR()[idx]
            elif 'norm' in self.abscissa or 'r/a' in self.abscissa:
                x0 = 0
            else:
                raise ValueError("Magnetic axis slope constraint is not "
                                 "supported for abscissa '%s'. Convert to a "
                                 "normalized coordinate or Rmid to use this "
                                 "constraint." % (self.abscissa,))
            self.gp.add_data(x0, 0, err_y=err, n=1)
        elif self.X_dim == 2:
            if times is None:
                times = scipy.unique(scipy.asarray(self.X[:, 0]).ravel())
            if self.abscissa == 'Rmid':
                x0 = scipy.interpolate.interp1d(self.efit_tree.getTimeBase(),
                                                self.efit_tree.getMagR(),
                                                kind='nearest' if not self.efit_tree._tricubic else 'cubic')(times)
            elif 'norm' in self.abscissa or 'r/a' in self.abscissa:
                x0 = scipy.zeros_like(times)
            else:
                raise ValueError("Magnetic axis slope constraint is not "
                                 "supported for abscissa '%s'. Convert to a "
                                 "normalized coordinate or Rmid to use this "
                                 "constraint." % (self.abscissa,))
            y = scipy.zeros_like(x0)
            X = scipy.hstack((scipy.atleast_2d(times).T, scipy.atleast_2d(x0).T))
            n = scipy.tile([0, 1], (len(y), 1))
            self.gp.add_data(X, y, err_y=err, n=n)
        else:
            raise ValueError("Magnetic axis slope constraint is not supported "
                             "for X_dim=%d, abscissa '%s'. Convert to a "
                             "normalized coordinate or Rmid to use this "
                             "constraint." % (self.X_dim, self.abscissa,))
    
    def constrain_at_limiter(self, err_y=0.01, err_dy=0.1, times=None, n_pts=4, expansion=1.25):
        """Constrains the slope and value of this Profile's Gaussian process to be zero at the GH limiter.
        
        The specific value of `X` coordinate to impose this constraint at is
        determined by finding the point of the GH limiter which has the
        smallest mapped coordinate.
        
        If the limiter location is not found in the tree, the system will
        instead use R=0.91m, Z=0.0m as the limiter location. This is a bit
        outside of where the limiter is, but will act as a conservative
        approximation for cases where the exact position is not available.
        
        Note that this is accomplished approximately for bivariate data by
        specifying the slope and value to be zero at the limiter for a number
        of points in time.
        
        It is assumed that the Gaussian process has already been created with
        a call to :py:meth:`create_gp`.
        
        The abscissa cannot be 'Z' or 'RZ'.
        
        Parameters
        ----------
        err_y : float, optional
            The uncertainty to place on the value constraint. The default is
            0.01. This could also potentially be an array for bivariate data
            where you wish to have the uncertainty vary in time.
        err_dy : float, optional
            The uncertainty to place on the slope constraint. The default is
            0.1. This could also potentially be an array for bivariate data
            where you wish to have the uncertainty vary in time.
        times : array-like, optional
            The times to impose the constraint at. Default is to use the
            unique time values in `X[:, 0]`.
        n_pts : int, optional
            The number of points outside of the limiter to use. It helps to use
            three or more points outside the plasma to ensure appropriate
            behavior. The constraint is applied at `n_pts` linearly spaced
            points between the limiter location (computed as discussed above)
            and the limiter location times `expansion`. If you set this to one
            it will only impose the constraint at the limiter. Default is 4.
        expansion : float, optional
            The factor by which the coordinate of the limiter location is
            multiplied to get the outer limit of the `n_pts` constraint points.
            Default is 1.25.
        """
        if self.abscissa in ['RZ', 'Z']:
            raise ValueError("Limiter constraint is not supported for abscissa "
                             "'%s'. Convert to a normalized coordinate or Rmid "
                             "to use this constraint." % (self.abscissa,))
        # Fail back to a conservative position if the limiter data are not in
        # the tree:
        try:
            analysis = MDSplus.Tree('analysis', self.shot)
            Z_lim = analysis.getNode('.limiters.gh_limiter.z').getData().data()
            R_lim = analysis.getNode('.limiters.gh_limiter.r').getData().data()
        except MDSplus.TreeException:
            print("No limiter data, defaulting to R=0.91, Z=0.0!")
            Z_lim = [0.0]
            R_lim = [0.91]
        if self.X_dim == 1:
            t_EFIT = self.efit_tree.getTimeBase()
            
            if hasattr(self, 'times'):
                idx = self.efit_tree._getNearestIdx(self.times, t_EFIT)
                t_EFIT = t_EFIT[idx]
            elif self.t_min != self.t_max:
                t_EFIT = t_EFIT[(t_EFIT >= self.t_min) & (t_EFIT <= self.t_max)]
            else:
                idx = self.efit_tree._getNearestIdx(self.t_min, t_EFIT)
                t_EFIT = t_EFIT[idx]
            rho_lim = scipy.mean(
                self.efit_tree.rz2rho(
                    self.abscissa, R_lim, Z_lim, t_EFIT, each_t=True
                ), axis=0
            )
            xa = rho_lim.min()
            x_pts = scipy.linspace(xa, xa * expansion, n_pts)
            y = scipy.zeros_like(x_pts)
            self.gp.add_data(x_pts, y, err_y=err_y, n=0)
            self.gp.add_data(x_pts, y, err_y=err_dy, n=1)
        elif self.X_dim == 2:
            if times is None:
                times = scipy.unique(scipy.asarray(self.X[:, 0]).ravel())
            rho_lim = self.efit_tree.rz2rho(self.abscissa, R_lim, Z_lim, times, each_t=True)
            xa = rho_lim.max(axis=1)
            x_pts = scipy.asarray([scipy.linspace(x, x * expansion, n_pts) for x in xa]).flatten()
            times = scipy.tile(times, n_pts)
            X = scipy.hstack((scipy.atleast_2d(times).T, scipy.atleast_2d(x_pts).T))
            y = scipy.zeros_like(x_pts)
            n = scipy.tile([0, 1], (len(y), 1))
            self.gp.add_data(X, y, err_y=err_y, n=0)
            self.gp.add_data(X, y, err_y=err_dy, n=n)
        else:
            raise ValueError("Limiter constraint is not supported for X_dim=%d, "
                             "abscissa '%s'. Convert to a normalized "
                             "coordinate or Rmid to use this constraint."
                             % (self.X_dim, self.abscissa,))

    
    def create_gp(self, constrain_slope_on_axis=True, constrain_at_limiter=True,
                  axis_constraint_kwargs={}, limiter_constraint_kwargs={}, **kwargs):
        """Create a Gaussian process to handle the data.
        
        Calls :py:meth:`Profile.create_gp`, then imposes constraints as
        requested.
        
        Defaults to using a squared exponential kernel in two dimensions or a
        Gibbs kernel with tanh warping in one dimension.
        
        Parameters
        ----------
        constrain_slope_on_axis : bool, optional
            If True, a zero slope constraint at the magnetic axis will be
            imposed after creating the gp. Default is True (constrain slope).
        constrain_at_limiter : bool, optional
            If True, a zero slope and value constraint at the GH limiter will
            be imposed after creating the gp. Default is True (constrain at
            axis).
        axis_constraint_kwargs : dict, optional
            The contents of this dictionary are passed as kwargs to
            :py:meth:`constrain_slope_on_axis`.
        limiter_constraint_kwargs : dict, optional
            The contents of this dictionary are passed as kwargs to
            :py:meth:`constrain_at_limiter`.
        **kwargs : optional kwargs
            All remaining kwargs are passed to :py:meth:`Profile.create_gp`.
        """
        # Increase the diagonal factor for multivariate data -- I was having
        # issues with the default level when using slope constraints.
        if self.X_dim > 1 and 'diag_factor' not in kwargs:
            kwargs['diag_factor'] = 1e4
        if 'k' not in kwargs and self.X_dim == 1:
            kwargs['k'] = 'gibbstanh'
        if kwargs.get('k', None) == 'gibbstanh':
            # Set the bound on x0 intelligently according to the abscissa:
            if 'x0_bounds' not in kwargs:
                kwargs['x0_bounds'] = (0.87, 0.915) if self.abscissa == 'Rmid' else (0.94, 1.1)
        super(BivariatePlasmaProfile, self).create_gp(**kwargs)
        if constrain_slope_on_axis:
            self.constrain_slope_on_axis(**axis_constraint_kwargs)
        if constrain_at_limiter:
            self.constrain_at_limiter(**limiter_constraint_kwargs)
    
    def compute_a_over_L(self, X, force_update=False, plot=False,
                         gp_kwargs={}, MAP_kwargs={}, plot_kwargs={},
                         return_prediction=False, special_vals=0,
                         special_X_vals=0, **predict_kwargs):
        """Compute the normalized inverse gradient scale length.
        
        Only works on data that have already been time-averaged at the moment.
        
        Parameters
        ----------
        X : array-like
            The points to evaluate a/L at.
        force_update : bool, optional
            If True, a new Gaussian process will be created even if one already
            exists. Set this if you have added data or constraints since you
            created the Gaussian process. Default is False (use current Gaussian
            process if it exists).
        plot : bool, optional
            If True, a plot of a/L is produced. Default is False (no plot).
        gp_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`create_gp` if it gets called. Default is {}.
        MAP_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`find_gp_MAP_estimate` if it gets called. Default is {}.
        plot_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to `plot` when
            plotting the mean of a/L. Default is {}.
        return_prediction : bool, optional
            If True, the full prediction of the value and gradient are returned
            in a dictionary. Default is False (just return value and stddev of
            a/L).
        special_vals : int, optional
            The number of special return values incorporated into
            `output_transform` that should be dropped before computing a/L. This
            is used so that things like volume averages can be efficiently
            computed at the same time as a/L. Default is 0 (no extra values).
        special_X_vals : int, optional
            The number of special points included in the abscissa that should
            not be included in the evaluation of a/L. Default is 0 (no extra
            values).
        **predict_kwargs : optional parameters
            All other parameters are passed to the Gaussian process'
            :py:meth:`predict` method.
        """
        # TODO: Add ability to just compute value.
        # TODO: Make finer-grained control over what to return.
        if force_update or self.gp is None:
            self.create_gp(**gp_kwargs)
            if not predict_kwargs.get('use_MCMC', False):
                self.find_gp_MAP_estimate(**MAP_kwargs)
        if self.X_dim == 1:
            # Get GP fit:
            XX = scipy.concatenate((X, X[special_X_vals:]))
            n = scipy.concatenate((scipy.zeros_like(X), scipy.ones_like(X[special_X_vals:])))
            out = self.gp.predict(XX, n=n, full_output=True,
                                  **predict_kwargs)
            mean = out['mean']
            cov = out['cov']
            
            # Ditch the special values:
            special_mean = mean[:special_vals]
            special_cov = cov[:special_vals, :special_vals]
            X = X[special_X_vals:]
            
            cov = cov[special_vals:, special_vals:]
            mean = mean[special_vals:]
            
            var = scipy.diagonal(cov)
            mean_val = mean[:len(X)]
            var_val = var[:len(X)]
            mean_grad = mean[len(X):]
            var_grad = var[len(X):]
            i = range(0, len(X))
            j = range(len(X), 2 * len(X))
            cov_val_grad = scipy.asarray(cov[i, j]).flatten()
            
            # Get geometry from EFIT:
            t_efit = self.efit_tree.getTimeBase()
            if hasattr(self, 'times'):
                ok_idxs = self.efit_tree._getNearestIdx(self.times, t_efit)
            elif self.t_min != self.t_max:
                ok_idxs = scipy.where((t_efit >= self.t_min) & (t_efit <= self.t_max))[0]
            else:
                ok_idxs = self.efit_tree._getNearestIdx([self.t_min], t_efit)
            
            a = self.efit_tree.getAOut()[ok_idxs]
            var_a = scipy.var(a, ddof=1)
            if scipy.isnan(var_a):
                var_a = 0
            mean_a = scipy.mean(a)
            
            # Get correction factor for converting the abscissa back to Rmid:
            if self.abscissa == 'Rmid':
                mean_dX_dRmid = scipy.ones_like(X)
                var_dX_dRmid = scipy.zeros_like(X)
            elif self.abscissa == 'r/a':
                mean_dX_dRmid = scipy.mean(1.0 / a) * scipy.ones_like(X)
                var_dX_dRmid = scipy.var(1.0 / a, ddof=1) * scipy.ones_like(X)
                var_dX_dRmid[scipy.isnan(var_dX_dRmid)] = 0
            else:
                # Code taken from core.py of eqtools, modified to use
                # InterpolatedUnivariateSpline so I have direct access to derivatives:
                dX_dRmid = scipy.zeros((len(X), len(ok_idxs)))
                # Loop over time indices:
                for idx, k in zip(ok_idxs, range(0, len(ok_idxs))):
                    resample_factor = 3
                    R_grid = scipy.linspace(
                        self.efit_tree.getMagR()[idx],
                        self.efit_tree.getRGrid()[-1],
                        resample_factor * len(self.efit_tree.getRGrid())
                    )
                
                    X_on_grid = self.efit_tree.rz2rho(
                        self.abscissa,
                        R_grid,
                        self.efit_tree.getMagZ()[idx] * scipy.ones(R_grid.shape),
                        t_efit[idx]
                    )
                
                    spline = scipy.interpolate.InterpolatedUnivariateSpline(
                        X_on_grid, R_grid, k=3
                    )
                    dX_dRmid[:, k] = 1.0 / spline(X, nu=1)
                
                mean_dX_dRmid = scipy.mean(dX_dRmid, axis=1)
                var_dX_dRmid = scipy.var(dX_dRmid, ddof=1, axis=1)
                var_dX_dRmid[scipy.isnan(var_dX_dRmid)] = 0
            
            if predict_kwargs.get('full_MC', False):
                # TODO: Doesn't include uncertainty in EFIT quantities!
                # Use samples:
                val_samps = out['samp'][special_vals:len(X)+special_vals]
                grad_samps = out['samp'][len(X)+special_vals:]
                dX_dRmid = scipy.tile(mean_dX_dRmid, (val_samps.shape[1], 1)).T
                a_L_samps = -mean_a * grad_samps * dX_dRmid / val_samps
                mean_a_L = scipy.mean(a_L_samps, axis=1)
                std_a_L = scipy.std(a_L_samps, axis=1, ddof=predict_kwargs.get('ddof', 1))
            else:
                # Compute using error propagation:
                mean_a_L = -mean_a * mean_grad * mean_dX_dRmid / mean_val
                std_a_L = scipy.sqrt(
                    var_a * (mean_grad * mean_dX_dRmid / mean_val)**2 +
                    var_val * (-mean_a * mean_grad * mean_dX_dRmid / mean_val**2)**2 +
                    var_grad * (mean_a * mean_dX_dRmid / mean_val)**2 +
                    var_dX_dRmid * (mean_a * mean_grad / mean_val)**2 +
                    cov_val_grad * ((-mean_a * mean_grad * mean_dX_dRmid / mean_val**2) *
                                    (mean_a * mean_dX_dRmid / mean_val))
                )
            # Plot result:
            if plot:
                ax = plot_kwargs.pop('ax', None)
                envelopes = plot_kwargs.pop('envelopes', [1, 3])
                base_alpha = plot_kwargs.pop('base_alpha', 0.375)
                if ax is None:
                    f = plt.figure()
                    ax = f.add_subplot(1, 1, 1)
                elif ax == 'gca':
                    ax = plt.gca()
                
                l = ax.plot(X, mean_a_L, **plot_kwargs)
                color = plt.getp(l[0], 'color')
                for i in envelopes:
                    ax.fill_between(X,
                                    mean_a_L - i * std_a_L,
                                    mean_a_L + i * std_a_L,
                                    facecolor=color,
                                    alpha=base_alpha / i)
        elif self.X_dim == 2:
            raise NotImplementedError("Not there yet!")
        else:
            raise ValueError("Cannot compute gradient scale length on data with "
                             "X_dim=%d!" % (self.X_dim,))
        
        if return_prediction:
            retval = {'mean_val': mean_val,
                      'std_val': scipy.sqrt(var_val),
                      'mean_grad': mean_grad,
                      'std_grad': scipy.sqrt(var_grad),
                      'mean_a_L': mean_a_L,
                      'std_a_L': std_a_L,
                      'cov': cov,
                      'out': out,
                      'special_mean': special_mean,
                      'special_cov': special_cov
                     }
            if predict_kwargs.get('full_MC', False):
                retval['samp'] = out['samp']
            return retval
        else:
            return (mean_a_L, std_a_L)
    
    def _get_efit_times_to_average(self):
        """Get the EFIT times to average over for a profile that has already been time-averaged.
        
        If this instance has a :py:attr:`times` attribute, the nearest indices
        to those times are used. Failing that, if :py:attr:`t_min` and
        :py:attr:`t_max` are distinct, all points between them are used. Failing
        that, the nearest index to t_min is used.
        """
        t_efit = self.efit_tree.getTimeBase()
        if hasattr(self, 'times'):
            ok_idxs = self.efit_tree._getNearestIdx(self.times, t_efit)
        elif self.t_min != self.t_max:
            ok_idxs = scipy.where((t_efit >= self.t_min) & (t_efit <= self.t_max))[0]
            # Handle case where there are none:
            if not ok_idxs:
                ok_idxs = self.efit_tree._getNearestIdx([self.t_min], t_efit)
        else:
            ok_idxs = self.efit_tree._getNearestIdx([self.t_min], t_efit)
    
        return t_efit[ok_idxs]
    
    def _make_volume_averaging_matrix(self, rho_grid=None, npts=400):
        """Generate a matrix of weights to find the volume average using the trapezoid rule.
        
        At present, this only supports data that have already been time-averaged.
        
        Parameters
        ----------
        rho_grid : array-like, optional
            The points (of the instance's abscissa) to use as the quadrature
            points. If these aren't provided, a grid of points uniformly spaced
            in volnorm will be produced. Default is None (produce uniform
            volnorm grid).
        npts : int, optional
            The number of points to use when producing a uniform volnorm grid.
            Default is 400.
        
        Returns
        -------
        rho_grid : array, (`npts`,)
            The quadrature points to be used on the instance's abscissa.
        weights : array, (1, `npts`)
            The matrix of weights to multiply the values at each location in
            `rho_grid` by to get the volume average.
        """
        if self.X_dim == 1:
            times = self._get_efit_times_to_average()
            
            if rho_grid is None:
                vol_grid = scipy.linspace(0, 1, npts)
                
                if 'volnorm' not in self.abscissa:
                    rho_grid = self.efit_tree.rho2rho(
                        'volnorm',
                        self.abscissa,
                        vol_grid,
                        times,
                        each_t=True
                    )
                    rho_grid = scipy.mean(rho_grid, axis=0)
                else:
                    if self.abscissa.startswith('sqrt'):
                        rho_grid = scipy.sqrt(vol_grid)
                    else:
                        rho_grid = vol_grid
            
                N = npts - 1
                a = 0
                b = 1
            
                # Use the trapezoid rule:
                weights = 2 * scipy.ones_like(vol_grid)
                weights[0] = 1
                weights[-1] = 1
                weights *= (b - a) / (2.0 * N)
            else:
                if 'volnorm' not in self.abscissa:
                    vol_grid = self.efit_tree.rho2rho(self.abscissa, 'volnorm', rho_grid, times, each_t=True)
                else:
                    if self.abscissa.startswith('sqrt'):
                        vol_grid = scipy.asarray(rho_grid)**2
                    else:
                        vol_grid = rho_grid
                # Use nanmean in case there is a value which is teetering -- we
                # want to keep it in.
                vol_grid = scipy.stats.nanmean(vol_grid, axis=0)
                ok_mask = ~scipy.isnan(vol_grid)
                delta_vol = scipy.diff(vol_grid[ok_mask])
                weights = scipy.zeros_like(vol_grid)
                weights[ok_mask] = 0.5 * (scipy.insert(delta_vol, 0, 0) + scipy.insert(delta_vol, -1, 0))
            
            weights = scipy.atleast_2d(weights)
            return (rho_grid, weights)
        else:
            raise NotImplementedError("Volume averaging not yet supported for X_dim > 1!")
    
    def compute_volume_average(self, return_std=True, grid=None, npts=400,
                               force_update=False, gp_kwargs={}, MAP_kwargs={},
                               **predict_kwargs):
        """Compute the volume average of the profile.
        
        Right now only supports data that have already been time-averaged.
        
        Parameters
        ----------
        return_std : bool, optional
            If True, the standard deviation of the volume average is computed
            and returned. Default is True (return mean and stddev of volume average).
        grid : array-like, optional
            The quadrature points to use when finding the volume average. If
            these are not provided, a uniform grid over volnorm will be used.
            Default is None (use uniform volnorm grid).
        npts : int, optional
            The number of uniformly-spaced volnorm points to use if `grid` is
            not specified. Default is 400.
        force_update : bool, optional
            If True, a new Gaussian process will be created even if one already
            exists. Set this if you have added data or constraints since you
            created the Gaussian process. Default is False (use current Gaussian
            process if it exists).
        gp_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`create_gp` if it gets called. Default is {}.
        MAP_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`find_gp_MAP_estimate` if it gets called. Default is {}.
        **predict_kwargs : optional parameters
            All other parameters are passed to the Gaussian process'
            :py:meth:`predict` method.
        
        Returns
        -------
        mean : float
            The mean of the volume average.
        std : float
            The standard deviation of the volume average. Only returned if
            `return_std` is True. Note that this is only sufficient as an error
            estimate if you separately verify that the integration error is less
            than this!
        """
        if self.X_dim == 1:
            if force_update or self.gp is None:
                self.create_gp(**gp_kwargs)
                if not predict_kwargs.get('use_MCMC', False):
                    self.find_gp_MAP_estimate(**MAP_kwargs)
            rho_grid, weights = self._make_volume_averaging_matrix(rho_grid=grid, npts=npts)
        
            res = self.gp.predict(
                rho_grid,
                output_transform=weights,
                return_std=return_std,
                return_cov=False,
                full_output=False,
                **predict_kwargs
            )
            if return_std:
                return (res[0][0], res[1][0])
            else:
                return res[0]
        else:
            raise NotImplementedError("Volume averaging not yet supported for "
                                      "X_dim > 1!")
    
    def compute_peaking(self, return_std=True, grid=None, npts=400,
                        force_update=False, gp_kwargs={}, MAP_kwargs={},
                        **predict_kwargs):
        r"""Compute the peaking of the profile.
        
        Right now only supports data that have already been time-averaged.
        
        Uses the definition from Greenwald, et al. (2007):
        :math:`w(\psi_n=0.2)/\langle w \rangle`.
        
        Parameters
        ----------
        return_std : bool, optional
            If True, the standard deviation of the volume average is computed
            and returned. Default is True (return mean and stddev of peaking).
        grid : array-like, optional
            The quadrature points to use when finding the volume average. If
            these are not provided, a uniform grid over volnorm will be used.
            Default is None (use uniform volnorm grid).
        npts : int, optional
            The number of uniformly-spaced volnorm points to use if `grid` is
            not specified. Default is 400.
        force_update : bool, optional
            If True, a new Gaussian process will be created even if one already
            exists. Set this if you have added data or constraints since you
            created the Gaussian process. Default is False (use current Gaussian
            process if it exists).
        gp_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`create_gp` if it gets called. Default is {}.
        MAP_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`find_gp_MAP_estimate` if it gets called. Default is {}.
        **predict_kwargs : optional parameters
            All other parameters are passed to the Gaussian process'
            :py:meth:`predict` method.
        """
        if self.X_dim == 1:
            if force_update or self.gp is None:
                self.create_gp(**gp_kwargs)
                if not predict_kwargs.get('use_MCMC', False):
                    self.find_gp_MAP_estimate(**MAP_kwargs)
            rho_grid, weights = self._make_volume_averaging_matrix(rho_grid=grid, npts=npts)
            weights = scipy.append(weights, 0)
            
            # Find the relevant core location:
            if 'psinorm' in self.abscissa:
                if self.abscissa.startswith('sqrt'):
                    core_loc = scipy.sqrt(0.2)
                else:
                    core_loc = 0.2
            else:
                times = self._get_efit_times_to_average()
                
                # TODO: Verify with Martin that rho really means psinorm in his 2007 paper!
                core_loc = self.efit_tree.rho2rho('psinorm', self.abscissa, times, each_t=True)
                core_loc = scipy.mean(core_loc)
            
            rho_grid = scipy.append(rho_grid, core_loc)
            core_select = scipy.zeros_like(weights)
            core_select[-1] = 1
            weights = scipy.vstack((weights, core_select))
            res = self.gp.predict(
                rho_grid,
                output_transform=weights,
                return_std=return_std,
                return_cov=return_std,
                full_output=False,
                **predict_kwargs
            )
            if return_std:
                mean_res = res[0]
                cov_res = res[1]
                mean = mean_res[1] / mean_res[0]
                std = scipy.sqrt(cov_res[1, 1] / mean_res[0]**2 +
                                 cov_res[0, 0] * mean_res[1]**2 / mean_res[0]**4 -
                                 cov_res[0, 1] * mean_res[1] / mean_res[0]**3)
                return (mean, std)
            else:
                return res[1] / res[0]
        else:
            raise NotImplementedError("Computation of peaking factors not yet "
                                      "supported for X_dim > 1!")

def neCTS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None, remove_edge=False):
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
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
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

    if remove_edge:
        p.remove_edge_points()

    return p

def neETS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None, remove_edge=False):
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
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
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

    if remove_edge:
        p.remove_edge_points()

    return p

def ne(shot, include=['CTS', 'ETS'], **kwargs):
    """Returns a profile representing electron density from both the core and edge Thomson scattering systems.

    Parameters
    ----------
    shot : int
        The shot number to load.
    include : list of str, optional
        The data sources to include. Valid options are:

            === =======================
            CTS Core Thomson scattering
            ETS Edge Thomson scattering
            === =======================

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

def neTS(shot, **kwargs):
    """Returns a profile representing electron density from both the core and edge Thomson scattering systems.
    """
    return ne(shot, include=['CTS', 'ETS'], **kwargs)

def TeCTS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None, remove_edge=False):
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
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
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

    if remove_edge:
        p.remove_edge_points()

    return p

def TeETS(shot, abscissa='RZ', t_min=None, t_max=None, electrons=None,
          efit_tree=None, remove_edge=False):
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
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
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

    if remove_edge:
        p.remove_edge_points()

    return p

def TeFRCECE(shot, rate='s', cutoff=0.15, abscissa='Rmid', t_min=None, t_max=None,
             electrons=None, efit_tree=None, remove_edge=False):
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
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
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
        t = N.dim_of().data()[:len(Te)]
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
    
    p.add_data(X, Te, channels={1: scipy.asarray(channels)}, err_y=0.1 * scipy.absolute(Te))
    # Remove flagged points:
    # I think these are cut off channels, but I am not sure...
    p.remove_points(p.y < cutoff)
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)
    p.convert_abscissa(abscissa)

    if remove_edge:
        p.remove_edge_points()

    return p

def TeGPC2(shot, abscissa='Rmid', t_min=None, t_max=None, electrons=None,
           efit_tree=None, remove_edge=False):
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
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
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

    p.add_data(X, Te, channels={1: channels}, err_y=0.1 * scipy.absolute(Te))
    
    # Remove flagged points:
    p.remove_points(p.y == 0)
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)

    p.convert_abscissa(abscissa)

    if remove_edge:
        p.remove_edge_points()

    return p

def TeGPC(shot, cutoff=0.15, abscissa='Rmid', t_min=None, t_max=None, electrons=None,
          efit_tree=None, remove_edge=False):
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
    remove_edge : bool, optional
        If True, will remove points that are outside the LCFS. It will convert
        the abscissa to psinorm if necessary. Default is False (keep edge).
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
    
    p.add_data(X, Te, channels={1: scipy.asarray(channels)}, err_y=0.1 * scipy.absolute(Te))
    
    # Remove flagged points:
    # I think these are cut off channels, but I am not sure...
    p.remove_points(p.y < cutoff)
    if t_min is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() < t_min)
    if t_max is not None:
        p.remove_points(scipy.asarray(p.X[:, 0]).flatten() > t_max)

    p.convert_abscissa(abscissa)

    if remove_edge:
        p.remove_edge_points()

    return p

def Te(shot, include=['CTS', 'ETS', 'FRCECE', 'GPC2', 'GPC'], FRCECE_rate='s',
       FRCECE_cutoff=0.15, GPC_cutoff=0.15, remove_ECE_edge=True, **kwargs):
    """Returns a profile representing electron temperature from the Thomson scattering and ECE systems.

    Parameters
    ----------
    shot : int
        The shot number to load.
    include : list of str, optional
        The data sources to include. Valid options are:

            ====== ===============================
            CTS    Core Thomson scattering
            ETS    Edge Thomson scattering
            FRCECE FRC electron cyclotron emission
            GPC    Grating polychromator
            GPC2   Grating polychromator 2
            ====== ===============================

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
    remove_ECE_edge : bool, optional
        If True, the points outside of the LCFS for the ECE diagnostics will be
        removed. Note that this overrides remove_edge, if present, in kwargs.
        Furthermore, this may lead to abscissa being converted to psinorm if an
        incompatible option was used.
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
            p_list.append(TeFRCECE(shot, rate=FRCECE_rate, cutoff=FRCECE_cutoff,
                                   **kwargs))
            if remove_ECE_edge:
                p_list[-1].remove_edge_points()
        elif system == 'GPC2':
            p_list.append(TeGPC2(shot, **kwargs))
            if remove_ECE_edge:
                p_list[-1].remove_edge_points()
        elif system == 'GPC':
            p_list.append(TeGPC(shot, cutoff=GPC_cutoff, **kwargs))
            if remove_ECE_edge:
                p_list[-1].remove_edge_points()
        else:
            raise ValueError("Unknown profile '%s'." % (system,))
    
    p = p_list.pop()
    for p_other in p_list:
        p.add_profile(p_other)
    
    return p

def TeTS(shot, **kwargs):
    """Returns a profile representing electron temperature data from the Thomson scattering system.
    
    Includes both core and edge system.
    """
    return Te(shot, include=['CTS', 'ETS'], **kwargs)

def read_plasma_csv(*args, **kwargs):
    """Returns a profile containing the data from a CSV file.
    
    If your data are bivariate, you must ensure that time ends up being the
    first column, either by putting it first in your CSV file or by specifying
    its name first in `X_names`.
    
    The CSV file can contain metadata lines of the form "name data" or
    "name data,data,...". The following metadata are automatically parsed into
    the correct fields:
    
        ========== ======================================================
        shot       shot number
        times      comma-separated list of times included in the data
        t_min      minimum time included in the data
        t_max      maximum time included in the data
        coordinate the abscissa the data are represented as a function of
        ========== ======================================================
    
    If you don't provide `coordinate` in the metadata, the program will try to
    use the last entry in X_labels to infer the abscissa. If this fails, it will
    simply set the abscissa to the title of the last entry in X_labels. If you
    provide your data as a function of R, Z it will look for the last two
    entries in X_labels to be R and Z once surrounding dollar signs and spaces
    are removed.
    
    Parameters are the same as :py:func:`read_csv`.
    """
    p = read_csv(*args, **kwargs)
    p.__class__ = BivariatePlasmaProfile
    metadata = dict([l.split(None, 1) for l in p.metadata])
    if 'shot' in metadata:
        p.shot = int(metadata['shot'])
        p.efit_tree = eqtools.CModEFITTree(p.shot)
    if 'times' in metadata:
        p.times = [float(t) for t in metadata['times'].split(',')]
    if 't_max' in metadata:
        p.t_max = float(metadata['t_max'])
    if 't_min' in metadata:
        p.t_min = float(metadata['t_min'])
    if 'coordinate' in metadata:
        p.abscissa = metadata['coordinate']
    else:
        if (p.X_dim > 1 and p.X_labels[-2].strip('$ ') == 'R' and
                p.X_labels[-1].strip('$ ') == 'Z'):
            p.abscissa = 'RZ'
        else:
            try:
                p.abscissa = _abscissa_mapping[p.X_labels[-1]]
            except KeyError:
                p.abscissa = p.X_labels[-1].strip('$ ')
    
    return p

def read_plasma_NetCDF(*args, **kwargs):
    """Returns a profile containing the data from a NetCDF file.
    
    The file can contain metadata attributes specified in the `metadata` kwarg.
    The following metadata are automatically parsed into the correct fields:
    
        ========== ======================================================
        shot       shot number
        times      comma-separated list of times included in the data
        t_min      minimum time included in the data
        t_max      maximum time included in the data
        coordinate the abscissa the data are represented as a function of
        ========== ======================================================
    
    If you don't provide `coordinate` in the metadata, the program will try to
    use the last entry in X_labels to infer the abscissa. If this fails, it will
    simply set the abscissa to the title of the last entry in X_labels. If you
    provide your data as a function of R, Z it will look for the last two
    entries in X_labels to be R and Z once surrounding dollar signs and spaces
    are removed.
    
    Parameters are the same as :py:func:`read_NetCDF`.
    """
    metadata = kwargs.pop('metadata', [])
    metadata = set(metadata + ['shot', 'times', 't_max', 't_min', 'coordinate'])
    p = read_NetCDF(*args, metadata=metadata, **kwargs)
    p.__class__ = BivariatePlasmaProfile
    if hasattr(p, 'shot'):
        p.efit_tree = eqtools.CModEFITTree(p.shot)
    if hasattr(p, 'coordinate'):
        p.abscissa = p.coordinate
    else:
        if (p.X_dim > 1 and p.X_labels[-2].strip('$ ') == 'R' and
                p.X_labels[-1].strip('$ ') == 'Z'):
            p.abscissa = 'RZ'
        else:
            try:
                p.abscissa = _abscissa_mapping[p.X_labels[-1]]
            except KeyError:
                p.abscissa = p.X_labels[-1].strip('$ ')
    
    return p
