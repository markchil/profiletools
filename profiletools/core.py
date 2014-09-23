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

"""Provides the base :py:class:`Profile` class and other utilities.
"""

from __future__ import division

import scipy
import scipy.stats
import scipy.io
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import gptools
import sys
import os.path
import csv
import warnings
import re

def average_points(X, y, err_X, err_y, T=None, ddof=1, robust=False,
                   y_method='sample', X_method='sample', weighted=False):
    """Find the average of the points with the given uncertainties using a variety of techniques.
    
    Parameters
    ----------
    X : array, (`M`, `D`) or (`M`, `N`, `D`)
        Abscissa values to average.
    y : array, (`M`)
        Data values to average.
    err_X : array, same shape as `X`
        Uncertainty in `X`.
    err_y : array, same shape as `y`
        Uncertainty in `y`.
    T : array, (`M`, `N`), optional
        Transform for `y`. Default is None (`y` is not transformed).
    ddof : int, optional
        The degree of freedom correction used in computing the standard
        deviation. The default is 1, the standard Bessel correction to
        give an unbiased estimate of the variance.
    robust : bool, optional
        Set this flag to use robust estimators (median, IQR). Default is False.
    y_method : {'sample', 'RMS', 'total', 'of mean', 'of mean sample'}, optional
        The method to use in computing the uncertainty in the averaged `y`.
        
        * 'sample' computes the sample standard deviation.
        * 'RMS' computes the root-mean-square of the individual error bars.
        * 'total' computes the square root of the sum of the sample variance and
          the mean variance. This is only statistically reasonable if the points
          represent sample means/variances already.
        * 'of mean' computes the uncertainty in the mean using error propagation
          with the given uncertainties.
        * 'of mean sample' computes the uncertainty in the mean using error
          propagation with the sample variance.
        
        Default is 'sample' (use sample variance).
    X_method : {'sample', 'RMS', 'total', 'of mean', 'of mean sample'}, optional
        The method to use in computing the uncertainty in the averaged `X`.
        Options are the same as `y_method`. Default is 'sample' (use sample
        variance).
    weighted : bool, optional
        Set this flag to use weighted estimators. The weights are 1/err_y^2.
        Default is False (use unweighted estimators).
    
    Returns
    -------
    mean_X : array, (`D`,) or (`N`, `D`)
        Mean of abscissa values.
    mean_y : float
        Mean of data values.
    err_X : array, same shape as `mean_X`
        Uncertainty in abscissa values.
    err_y : float
        Uncertainty in data values.
    T : array, (`N`,) or None
        Mean of transformation.
    """
    allowed_methods = ['sample', 'RMS', 'total', 'of mean', 'of mean sample']
    if y_method not in allowed_methods:
        raise ValueError("Unsupported y_method '%s'!" % (y_method,))
    if X_method not in allowed_methods:
        raise ValueError("Unsupported X_method '%s'!" % (X_method,))
    
    if weighted:
        weights = 1.0 / err_y**2
        if scipy.isinf(weights).any() or scipy.isnan(weights).any():
            weights = None
            warnings.warn("Invalid weight, setting weights equal!")
    else:
        weights = None
    
    if not robust:
        # Process y:
        mean_y = meanw(y, weights=weights)
        # If there is only one member, just carry its uncertainty forward:
        if len(y) == 1:
            err_y = err_y[0]
        elif y_method == 'sample':
            err_y = stdw(y, weights=weights, ddof=ddof)
        elif y_method == 'RMS':
            err_y = scipy.sqrt(meanw(err_y**2, weights=weights))
        elif y_method == 'total':
            err_y = scipy.sqrt(
                varw(y, weights=weights, ddof=ddof) +
                meanw(err_y**2, weights=weights)
            )
        elif y_method == 'of mean':
            if weighted:
                err_y = scipy.sqrt((weights**2 * err_y**2).sum()) / weights.sum()
            else:
                err_y = scipy.sqrt((err_y**2).sum()) / len(y)
        elif y_method == 'of mean sample':
            if weighted:
                err_y = scipy.sqrt((weights**2).sum()) * stdw(y, weights=weights, ddof=ddof) / weights.sum()
            else:
                err_y = stdw(y, weights=weights, ddof=ddof) / scipy.sqrt(len(y))
        
        # Similar picture for X:
        if weights is not None:
            weights = scipy.atleast_2d(weights).T
        mean_X = meanw(X, weights=weights, axis=0)
        if len(y) == 1:
            err_X = err_X[0]
        elif X_method == 'sample':
            err_X = stdw(X, weights=weights, ddof=ddof, axis=0)
        elif X_method == 'RMS':
            err_X = scipy.sqrt(meanw(err_X**2, weights=weights, axis=0))
        elif X_method == 'total':
            err_X = scipy.sqrt(
                varw(X, weights=weights, ddof=ddof, axis=0) +
                meanw(err_X**2, weights=weights, axis=0)
            )
        elif X_method == 'of mean':
            if weighted:
                err_X = scipy.sqrt((weights**2 * err_X**2).sum(axis=0)) / weights.sum()
            else:
                err_X = scipy.sqrt((err_X**2).sum(axis=0)) / len(y)
        elif X_method == 'of mean sample':
            if weighted:
                err_X = scipy.sqrt((weights**2).sum()) * stdw(X, weights=weights, ddof=ddof, axis=0) / weights.sum()
            else:
                err_X = stdw(X, weights=weights, ddof=ddof, axis=0) / scipy.sqrt(len(y))
        
        # And again for T:
        if T is not None:
            T = meanw(T, weights=weights, axis=0)
    else:
        mean_y = medianw(y, weights=weights)
        if len(y) == 1:
            err_y = err_y[0]
        elif y_method == 'sample':
            err_y = robust_stdw(y, weights=weights)
        elif y_method == 'RMS':
            err_y = scipy.sqrt(medianw(err_y**2, weights=weights))
        elif y_method == 'total':
            err_y = scipy.sqrt((robust_stdw(y, weights=weights))**2 + medianw(err_y**2, weights=weights))
        elif y_method == 'of mean':
            # TODO: This is a very sketchy approximation!
            if weighted:
                err_y = scipy.sqrt((weights**2 * err_y**2).sum()) / weights.sum()
            else:
                err_y = scipy.sqrt((err_y**2).sum()) / len(y)
        elif y_method == 'of mean sample':
            if weighted:
                err_y = scipy.sqrt((weights**2).sum()) * robust_stdw(y, weights=weights) / weights.sum()
            else:
                err_y = robust_std(y) / scipy.sqrt(len(y))
        
        mean_X = scipy.median(X, axis=0)
        if len(y) == 1:
            err_X = err_X[0]
        elif X_method == 'sample':
            err_X = robust_stdw(X, weights=weights, axis=0)
        elif X_method == 'RMS':
            err_X = scipy.sqrt(medianw(err_X**2, weights=weights, axis=0))
        elif X_method == 'total':
            err_X = scipy.sqrt(
                (robust_stdw(X, weights=weights, axis=0))**2 +
                scipy.median(err_X**2, axis=0)
            )
        elif X_method == 'of mean':
            if weighted:
                err_X = scipy.sqrt((weights**2 * err_X**2).sum(axis=0)) / weights.sum()
            else:
                err_X = scipy.sqrt((err_X**2).sum(axis=0)) / len(y)
        elif X_method == 'of mean sample':
            if weighted:
                err_X = scipy.sqrt((weights**2).sum()) * robust_stdw(X, weights=weights, ddof=ddof, axis=0) / weights.sum()
            else:
                err_X = robust_stdw(X, weights=weights, axis=0) / scipy.sqrt(len(y))
        
        if T is not None:
            T = medianw(T, weights=weights, axis=0)
    
    return (mean_X, mean_y, err_X, err_y, T)

class Channel(object):
    """Class to store data from a single channel.
    
    This is particularly useful for storing linearly transformed data, but
    should work for general data just as well.
    
    Parameters
    ----------
    X : array, (`M`, `N`, `D`)
        Abscissa values to use.
    y : array, (`M`,)
        Data values.
    err_X : array, same shape as `X`
        Uncertainty in `X`.
    err_y : array, (`M`,)
        Uncertainty in data.
    T : array, (`M`, `N`), optional
        Linear transform to get from latent variables to data in `y`. Default is
        that `y` represents untransformed data.
    y_label : str, optional
        Label for the `y` data. Default is empty string.
    y_units : str, optional
        Units of the `y` data. Default is empty string.
    """
    def __init__(self, X, y, err_X=0, err_y=0, T=None, y_label='', y_units=''):
        self.y_label = y_label
        self.y_units = y_units
        # Verify y has only one non-trivial dimension:
        y = scipy.atleast_1d(scipy.asarray(y, dtype=float))
        if y.ndim != 1:
            raise ValueError(
                "Dependent variables y must have only one dimension! Shape of y "
                "given is %s" % (y.shape,)
            )
        
        # Handle scalar error or verify shape of array error matches shape of y:
        try:
            iter(err_y)
        except TypeError:
            err_y = err_y * scipy.ones_like(y, dtype=float)
        else:
            err_y = scipy.asarray(err_y, dtype=float)
            if err_y.shape != y.shape:
                raise ValueError(
                    "When using array-like err_y, shape must match shape of y! "
                    "Shape of err_y given is %s, shape of y given is %s."
                    % (err_y.shape, y.shape)
                )
        if (err_y < 0).any():
            raise ValueError("All elements of err_y must be non-negative!")
        
        # Handle scalar independent variable or convert array input into matrix.
        X = scipy.atleast_3d(scipy.asarray(X, dtype=float))
        if T is None and X.shape[0] != len(y):
            raise ValueError(
                "Shape of independent variables must be (len(y), D)! "
                "X given has shape %s, shape of y is %s."
                % (X.shape, y.shape,)
            )
        
        if T is not None:
            # Promote T if it is a single observation:
            T = scipy.atleast_2d(scipy.asarray(T, dtype=float))
            if T.ndim != 2:
                raise ValueError("T must have exactly 2 dimensions!")
            if T.shape[0] != len(y):
                raise ValueError("Length of first dimension of T must match length of y!")
            if T.shape[1] != X.shape[1]:
                raise ValueError("Second dimension of T must match second dimension of X!")
        else:
            T = scipy.eye(len(y))
        
        # Process uncertainty in X:
        try:
            iter(err_X)
        except TypeError:
            err_X = err_X * scipy.ones_like(X, dtype=float)
        else:
            err_X = scipy.asarray(err_X, dtype=float)
            if err_X.ndim == 1 and X.shape[2] != 1:
                err_X = scipy.tile(err_X, (X.shape[0], 1))
        err_X = scipy.atleast_2d(scipy.asarray(err_X, dtype=float))
        if err_X.shape != X.shape:
            raise ValueError(
                "Shape of uncertainties on independent variables must be "
                "(len(y), self.X_dim)! X given has shape %s, shape of y is %s."
                % (X.shape, y.shape,)
            )
        
        if (err_X < 0).any():
            raise ValueError("All elements of err_X must be non-negative!")
        
        self.X = X
        self.y = y
        self.err_X = err_X
        self.err_y = err_y
        self.T = T
    
    def keep_slices(self, axis, vals, keep_mixed=False):
        """Only keep the indices closest to given `vals`.
        
        Parameters
        ----------
        axis : int
            The column in `X` to check values on.
        vals : float or 1-d array
            The value(s) to keep the points that are nearest to.
        keep_mixed : bool, optional
            Set this flag to keep transformed quantities that depend on multiple
            values of `X[:, :, axis]`. Default is False (drop mixed quantities).
        
        Returns
        -------
        still_good : bool
            Returns True if there are still any points left in the channel,
            False otherwise.
        """
        unique_vals = []
        num_unique = []
        for pt in self.X:
            unique_vals += [scipy.unique(pt[:, axis])]
            num_unique += [len(unique_vals[-1])]
        if max(num_unique) > 1:
            if keep_mixed:
                return True
            else:
                return False
        else:
            unique_vals = scipy.asarray(unique_vals)
            keep_idxs = scipy.unique(get_nearest_idx(vals, unique_vals))
            
            self.X = self.X[keep_idxs, :, :]
            self.y = self.y[keep_idxs]
            self.err_X = self.err_X[keep_idxs, :, :]
            self.err_y = self.err_y[keep_idxs]
            self.T = self.T[keep_idxs, :]
            
            return True
        
    def average_data(self, axis=0, **kwargs):
        """Average the data along the given `axis`.
        
        Parameters
        ----------
        axis : int, optional
            Axis to average along. Default is 0.
        **kwargs : optional keyword arguments
            All additional kwargs are passed to :py:func:`average_points`.
        """
        reduced_X = scipy.delete(self.X, axis, axis=2)
        reduced_err_X = scipy.delete(self.err_X, axis, axis=2)
        self.X, self.y, self.err_X, self.err_y, self.T = average_points(
            reduced_X,
            self.y,
            reduced_err_X,
            self.err_y,
            T=self.T,
            **kwargs
        )
        self.X = scipy.expand_dims(self.X, axis=0)
        self.y = scipy.expand_dims(self.y, axis=0)
        self.err_X = scipy.expand_dims(self.err_X, axis=0)
        self.err_y = scipy.expand_dims(self.err_y, axis=0)
        self.T = scipy.expand_dims(self.T, axis=0)
    
    def remove_points(self, conditional):
        """Remove points satisfying `conditional`.
        
        Parameters
        ----------
        conditional : array, same shape as `self.y`
            Boolean array with True wherever a point should be removed.
        
        Returns
        -------
        bad_X : array
            The removed `X` values.
        bad_err_X : array
            The uncertainty in the removed `X` values.
        bad_y : array
            The removed `y` values.
        bad_err_y : array
            The uncertainty in the removed `y` values.
        bad_T : array
            The transformation matrix of the removed `y` values.
        """
        keep_idxs = ~conditional
        
        bad_X = self.X[conditional, :, :]
        bad_y = self.y[conditional]
        bad_err_X = self.err_X[conditional, :, :]
        bad_err_y = self.err_y[conditional]
        bad_T = self.T[conditional, :]
        
        self.X = self.X[keep_idxs, :, :]
        self.y = self.y[keep_idxs]
        self.err_X = self.err_X[keep_idxs, :, :]
        self.err_y = self.err_y[keep_idxs]
        self.T = self.T[keep_idxs, :]
        
        return (bad_X, bad_err_X, bad_y, bad_err_y, bad_T)

class Profile(object):
    """Object to abstractly represent a profile.
    
    Parameters
    ----------
    X_dim : positive int, optional
        Number of dimensions of the independent variable. Default value is 1.
    X_units : str, list of str or None, optional
        Units for each of the independent variables. If `X_dim`=1, this should
        given as a single string, if `X_dim`>1, this should be given as a list
        of strings of length `X_dim`. Default value is `None`, meaning a list
        of empty strings will be used.
    y_units : str, optional
        Units for the dependent variable. Default is an empty string.
    X_labels : str, list of str or None, optional
        Descriptive label for each of the independent variables. If `X_dim`=1,
        this should be given as a single string, if `X_dim`>1, this should be
        given as a list of strings of length `X_dim`. Default value is `None`,
        meaning a list of empty strings will be used.
    y_label : str, optional
        Descriptive label for the dependent variable. Default is an empty string.
    weightable : bool, optional
        Whether or not it is valid to use weighted estimators on the data, or if
        the error bars are too suspect for this to be valid. Default is True
        (allow use of weighted estimators).
    
    Attributes
    ----------
    X_dim : positive int
        The number of dimensions of the independent variable.
    X_units : list of str, (X_dim,)
        The units for each of the independent variables.
    y_units : str
        The units for the dependent variable.
    X_labels : list of str, (X_dim,)
        Descriptive labels for each of the independent variables.
    y_label : str
        Descriptive label for the dependent variable.
    y : :py:class:`Array`, (`M`,)
        The `M` dependent variables.
    X : :py:class:`Matrix`, (`M`, `X_dim`)
        The `M` independent variables.
    err_y : :py:class:`Array`, (`M`,)
        The uncertainty in the `M` dependent variables.
    err_X : :py:class:`Matrix`, (`M`, `X_dim`)
        The uncertainties in each dimension of the `M` independent variables.
    channels : :py:class:`Matrix`, (`M`, `X_dim`)
        The logical groups of points into channels along each of the independent
        variables.
    """
    def __init__(self, X_dim=1, X_units=None, y_units='', X_labels=None, y_label='',
                 weightable=True):
        self.X_dim = X_dim
        self.weightable = weightable
        if X_units is None:
            X_units = [''] * X_dim
        elif X_dim == 1:
            X_units = [X_units]
        elif len(X_units) != X_dim:
            raise ValueError("The length of X_units must be equal to X_dim!")
        
        if X_labels is None:
            X_labels = [''] * X_dim
        elif X_dim == 1:
            X_labels = [X_labels]
        elif len(X_labels) != X_dim:
            raise ValueError("The length of X_labels must be equal to X_dim!")
        
        self.X_units = X_units
        self.y_units = y_units
        
        self.X_labels = X_labels
        self.y_label = y_label
        
        self.y = scipy.array([], dtype=float)
        self.X = None
        self.err_y = scipy.array([], dtype=float)
        self.err_X = None
        self.channels = None
        
        self.transformed = scipy.array([], dtype=Channel)
        
        self.gp = None
    
    def add_data(self, X, y, err_X=0, err_y=0, channels=None):
        """Add data to the training data set of the :py:class:`Profile` instance.
        
        Will also update the Profile's Gaussian process instance (if it exists).
        
        Parameters
        ----------
        X : array-like, (`M`, `N`)
            `M` independent variables of dimension `N`.
        y : array-like, (`M`,)
            `M` dependent variables.
        err_X : array-like, (`M`, `N`), or scalar float, or single array-like (`N`,), optional
            Non-negative values only. Error given as standard deviation for
            each of the `N` dimensions in the `M` independent variables. If a
            scalar is given, it is used for all of the values. If a single
            array of length `N` is given, it is used for each point. The
            default is to assign zero error to each point.
        err_y : array-like (`M`,) or scalar float, optional
            Non-negative values only. Error given as standard deviation in the
            `M` dependent variables. If `err_y` is a scalar, the data set is
            taken to be homoscedastic (constant error). Otherwise, the length
            of `err_y` must equal the length of `y`. Default value is 0
            (noiseless observations).
        channels : dict or array-like (`M`, `N`)
            Keys to logically group points into "channels" along each dimension
            of `X`. If not passed, channels are based simply on which points
            have equal values in `X`. If only certain dimensions have groupings
            other than the simple default equality conditions, then you can
            pass a dict with integer keys in the interval [0, `X_dim`-1] whose
            values are the arrays of length `M` indicating the channels.
            Otherwise, you can pass in a full (`M`, `N`) array.
        
        Raises
        ------
        ValueError
            Bad shapes for any of the inputs, negative values for `err_y` or `n`.
        """
        # Verify y has only one non-trivial dimension:
        y = scipy.atleast_1d(scipy.asarray(y, dtype=float))
        if y.ndim != 1:
            raise ValueError(
                "Dependent variables y must have only one dimension! Shape of y "
                "given is %s" % (y.shape,)
            )
        
        # Handle scalar error or verify shape of array error matches shape of y:
        try:
            iter(err_y)
        except TypeError:
            err_y = err_y * scipy.ones_like(y, dtype=float)
        else:
            err_y = scipy.asarray(err_y, dtype=float)
            if err_y.shape != y.shape:
                raise ValueError(
                    "When using array-like err_y, shape must match shape of y! "
                    "Shape of err_y given is %s, shape of y given is %s."
                    % (err_y.shape, y.shape)
                )
        if (err_y < 0).any():
            raise ValueError("All elements of err_y must be non-negative!")
        
        # Handle scalar independent variable or convert array input into matrix.
        X = scipy.atleast_2d(scipy.asarray(X, dtype=float))
        # Correct single-dimension inputs:
        if self.X_dim == 1 and X.shape[0] == 1:
            X = X.T
        if X.shape != (len(y), self.X_dim):
            raise ValueError(
                "Shape of independent variables must be (len(y), self.X_dim)! "
                "X given has shape %s, shape of y is %s and X_dim=%d."
                % (X.shape, y.shape, self.X_dim)
            )
        
        # Process uncertainty in X:
        try:
            iter(err_X)
        except TypeError:
            err_X = err_X * scipy.ones_like(X, dtype=float)
        else:
            err_X = scipy.asarray(err_X, dtype=float)
            # TODO: Steal this idiom for handling n in gptools!
            if err_X.ndim == 1 and self.X_dim != 1:
                err_X = scipy.tile(err_X, (X.shape[0], 1))
        err_X = scipy.atleast_2d(scipy.asarray(err_X, dtype=float))
        if self.X_dim == 1 and err_X.shape[0] == 1:
            err_X = err_X.T
        if err_X.shape != X.shape:
            raise ValueError(
                "Shape of uncertainties on independent variables must be "
                "(len(y), self.X_dim)! X given has shape %s, shape of y is %s "
                "and X_dim=%d." % (X.shape, y.shape, self.X_dim)
            )
        
        if (err_X < 0).any():
            raise ValueError("All elements of err_X must be non-negative!")
        
        # Process channel flags:
        if channels is None:
            channels = scipy.tile(scipy.arange(0, len(y)), (X.shape[1], 1)).T
        else:
            if isinstance(channels, dict):
                d_channels = channels
                channels = scipy.tile(scipy.arange(0, len(y)), (X.shape[1], 1)).T
                for idx in d_channels:
                    channels[:, idx] = d_channels[idx]
            else:
                channels = scipy.asarray(channels)
                if channels.shape != (len(y), X.shape[1]):
                    raise ValueError("Shape of channels and X must be the same!")
        
        if self.X is None:
            self.X = X
        else:
            self.X = scipy.vstack((self.X, X))
        if self.channels is None:
            self.channels = channels
        else:
            self.channels = scipy.vstack((self.channels, channels))
        if self.err_X is None:
            self.err_X = err_X
        else:
            self.err_X = scipy.vstack((self.err_X, err_X))
        self.y = scipy.append(self.y, y)
        self.err_y = scipy.append(self.err_y, err_y)
        
        if self.gp is not None:
            self.gp.add_data(X, y, err_y=err_y)
        
    def add_profile(self, other):
        """Absorbs the data from one profile object.
        
        Parameters
        ----------
        other : :py:class:`Profile`
            :py:class:`Profile` to absorb.
        """
        if self.X_dim != other.X_dim:
            raise ValueError(
                "When merging profiles, X_dim must be equal between the two "
                "profiles!"
            )
        if self.y_units != other.y_units:
            raise ValueError("When merging profiles, the y_units must agree!")
        if self.X_units != other.X_units:
            raise ValueError("When merging profiles, the X_units must agree!")
        if len(other.y) > 0:
            # Modify the channels of self.channels to avoid clashes:
            if other.channels is not None and self.channels is not None:
                self.channels = (
                    self.channels - self.channels.min(axis=0) +
                    other.channels.max(axis=0) + 1
                )
            self.add_data(other.X, other.y, err_X=other.err_X, err_y=other.err_y,
                          channels=other.channels)
        
        if len(other.transformed) > 0:
            self.transformed = scipy.append(self.transformed, other.transformed)
    
    def drop_axis(self, axis):
        """Drops a selected axis from `X`.
        
        Parameters
        ----------
        axis : int
            The index of the axis to drop.
        """
        if self.X_dim == 1:
            raise ValueError("Can't drop axis from a univariate profile!")
        self.X_dim -= 1
        if self.X is not None:
            self.channels = scipy.delete(self.channels, axis, axis=1)
            self.X = scipy.delete(self.X, axis, axis=1)
            self.err_X = scipy.delete(self.err_X, axis, axis=1)
        self.X_labels.pop(axis)
        self.X_units.pop(axis)
        
        for p in self.transformed:
            p.X = scipy.delete(p.X, axis, axis=2)
            p.err_X = scipy.delete(p.err_X, axis, axis=2)
    
    def keep_slices(self, axis, vals, **kwargs):
        """Keeps only the nearest points to vals along the given axis for each channel.
        
        Parameters
        ----------
        axis : int
            The axis to take the slice(s) of.
        vals : array of float
            The values the axis should be close to.
        **kwargs : optional kwargs
            All additional kwargs are passed to :py:meth:`~gptools.core.Channel.keep_slices`.
        """
        try:
            iter(vals)
        except TypeError:
            vals = [vals]
        
        # Only handle single points if they are present...
        if self.X is not None:
            new_X = []
            new_y = []
            new_err_X = []
            new_err_y = []
            new_channels = []
            
            reduced_channels = scipy.delete(self.channels, axis, axis=1)
            channels = unique_rows(reduced_channels)
            
            for ch in channels:
                channel_idxs = (reduced_channels == ch.flatten()).all(axis=1)
                ch_axis_X = self.X[channel_idxs, axis].flatten()
                keep_idxs = scipy.unique(get_nearest_idx(vals, ch_axis_X))
                
                new_X.extend(self.X[channel_idxs, :][keep_idxs, :])
                new_y.extend(self.y[channel_idxs][keep_idxs])
                new_err_X.extend(self.err_X[channel_idxs, :][keep_idxs, :])
                new_err_y.extend(self.err_y[channel_idxs][keep_idxs])
                new_channels.extend(self.channels[channel_idxs, :][keep_idxs, :])
            self.X = scipy.vstack(new_X)
            self.y = scipy.asarray(new_y)
            self.err_X = scipy.vstack(new_err_X)
            self.err_y = scipy.asarray(new_err_y)
            self.channels = scipy.vstack(new_channels)
        
        mask = [p.keep_slices(axis, vals, **kwargs) for p in self.transformed]
        self.transformed = self.transformed[scipy.asarray(mask, dtype=bool)]
    
    def average_data(self, axis=0, **kwargs):
        """Computes the average of the profile over the desired axis.
        
        If `X_dim` is already 1, this returns the average of the quantity.
        Otherwise, the :py:class:`Profile` is mutated to contain the
        desired averaged data. `err_X` and `err_y` are populated with the
        standard deviations of the respective quantities. The averaging is
        carried out within the groupings defined by the `channels` attribute.
        
        Parameters
        ----------
        axis : int, optional
            The index of the dimension to average over. Default is 0.
        **kwargs : optional kwargs
            All additional kwargs are passed to :py:func:`average_points`.
        """
        kwargs['weighted'] = self.weightable and kwargs.get('weighted', False)
        # TODO: Add support for custom bins!
        if self.X is not None:
            reduced_channels = scipy.delete(self.channels, axis, axis=1)
            reduced_X = scipy.delete(self.X, axis, axis=1)
            reduced_err_X = scipy.delete(self.err_X, axis, axis=1)
            channels = unique_rows(reduced_channels)
            X = scipy.zeros((len(channels), self.X_dim - 1))
            y = scipy.zeros(len(channels))
            err_X = scipy.zeros_like(X)
            err_y = scipy.zeros_like(y)
            for i, chan in zip(range(0, len(channels)), channels):
                chan_mask = (
                    reduced_channels == chan.flatten()
                ).all(axis=1)
            
                X[i, :], y[i], err_X[i, :], err_y[i], dum = average_points(
                    reduced_X[chan_mask, :],
                    self.y[chan_mask],
                    reduced_err_X[chan_mask, :],
                    self.err_y[chan_mask],
                    **kwargs
                )
            
            self.X = X
            self.y = y
            self.err_X = err_X
            self.err_y = err_y
            self.channels = channels
        
        self.X_dim -= 1
        self.X_units.pop(axis)
        self.X_labels.pop(axis)
        
        for p in self.transformed:
            p.average_data(axis=axis, **kwargs)
    
    def plot_data(self, ax=None, label_axes=True, **kwargs):
        """Plot the data stored in this Profile. Only works for X_dim = 1 or 2.
        
        Parameters
        ----------
        ax : axis instance, optional
            Axis to plot the result on. If no axis is passed, one is created.
            If the string 'gca' is passed, the current axis (from plt.gca())
            is used. If X_dim = 2, the axis must be 3d.
        label_axes : bool, optional
            If True, the axes will be labelled with strings constructed from
            the labels and units set when creating the Profile instance.
            Default is True (label axes).
        **kwargs : extra plotting arguments, optional
            Extra arguments that are passed to errorbar/errorbar3d.
        
        Returns
        -------
        The axis instance used.
        """
        if self.X is not None:
            if self.X_dim > 2:
                raise ValueError("Plotting is not supported for X_dim > 2!")
            if ax is None:
                f = plt.figure()
                if self.X_dim == 1:
                    ax = f.add_subplot(1, 1, 1)
                elif self.X_dim == 2:
                    ax = f.add_subplot(111, projection='3d')
            elif ax == 'gca':
                ax = plt.gca()
        
            if 'label' not in kwargs:
                kwargs['label'] = self.y_label
        
            if 'fmt' not in kwargs and 'marker' not in kwargs:
                kwargs['fmt'] = 'o'
        
            if self.X_dim == 1:
                ax.errorbar(self.X.ravel(), self.y,
                            yerr=self.err_y, xerr=self.err_X.flatten(),
                            **kwargs)
                if label_axes:
                    ax.set_xlabel(
                        "%s [%s]" % (self.X_labels[0], self.X_units[0],) if self.X_units[0]
                        else self.X_labels[0]
                    )
                    ax.set_ylabel(
                        "%s [%s]" % (self.y_label, self.y_units,) if self.y_units
                        else self.y_label
                    )
            elif self.X_dim == 2:
                errorbar3d(ax, self.X[:, 0], self.X[:, 1], self.y,
                           xerr=self.err_X[:, 0], yerr=self.err_X[:, 1], zerr=self.err_y,
                           **kwargs)
                if label_axes:
                    ax.set_xlabel(
                        "%s [%s]" % (self.X_labels[0], self.X_units[0],) if self.X_units[0]
                        else self.X_labels[0]
                    )
                    ax.set_ylabel(
                        "%s [%s]" % (self.X_labels[1], self.X_units[1],) if self.X_units[1]
                        else self.X_labels[1]
                    )
                    ax.set_zlabel(
                        "%s [%s]" % (self.y_label, self.y_units,) if self.y_units
                        else self.y_label
                    )
        
            return ax
    
    def remove_points(self, conditional):
        """Remove points where conditional is True.
        
        Note that this does NOT remove anything from the GP -- you either need
        to call :py:meth:`create_gp` again or act manually on the :py:attr:`gp`
        attribute.
        
        Also note that this does not include any provision for removing points
        that represent linearly-transformed quantities -- you will need to
        operate directly on :py:attr:`transformed` to remove such points.
        
        Parameters
        ----------
        conditional : array-like of bool, (`M`,)
            Array of booleans corresponding to each entry in `y`. Where an
            entry is True, that value will be removed.
        
        Returns
        -------
        X_bad : matrix
            Input values of the bad points.
        y_bad : array
            Bad values.
        err_X_bad : array
            Uncertainties on the abcissa of the bad values.
        err_y_bad : array
            Uncertainties on the bad values.
        """
        idxs = ~conditional
        
        y_bad = self.y[conditional]
        X_bad = self.X[conditional, :]
        err_y_bad = self.err_y[conditional]
        err_X_bad = self.err_X[conditional, :]
        
        self.y = self.y[idxs]
        self.X = self.X[idxs, :]
        self.err_y = self.err_y[idxs]
        self.err_X = self.err_X[idxs, :]
        self.channels = self.channels[idxs, :]
        
        return (X_bad, y_bad, err_X_bad, err_y_bad)
    
    def remove_outliers(self, thresh=3, check_transformed=False,
                        force_update=False, mask_only=False, gp_kwargs={}, MAP_kwargs={},
                        **predict_kwargs):
        """Remove outliers from the Gaussian process.
        
        The Gaussian process is created if it does not already exist. The
        chopping of values assumes that any artificial constraints that have
        been added to the GP are at the END of the GP's data arrays.
        
        The values removed are returned.
        
        Parameters
        ----------
        thresh : float, optional
            The threshold as a multiplier times `err_y`. Default is 3 (i.e.,
            throw away all 3-sigma points).
        check_transformed : bool, optional
            Set this flag to check if transformed quantities are outliers.
            Default is False (don't check transformed quantities).
        force_update : bool, optional
            If True, a new Gaussian process will be created even if one already
            exists. Set this if you have added data or constraints since you
            created the Gaussian process. Default is False (use current Gaussian
            process if it exists).
        mask_only : bool, optional
            Set this flag to return only a mask of the non-transformed points
            that are flagged. Default is False (completely remove bad points).
            In either case, the bad transformed points will ALWAYS be removed if
            `check_transformed` is True.
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
        X_bad : matrix
            Input values of the bad points.
        y_bad : array
            Bad values.
        err_X_bad : array
            Uncertainties on the abcissa of the bad values.
        err_y_bad : array
            Uncertainties on the bad values.
        transformed_bad : array of :py:class:`Channel`
            Transformed points that were removed.
        """
        if force_update or self.gp is None:
            self.create_gp(**gp_kwargs)
            if not remove_kwargs.get('use_MCMC', False):
                self.find_gp_MAP_estimate(**MAP_kwargs)
        
        # Handle single points:
        mean = self.gp.predict(
            self.X,
            return_std=False,
            **predict_kwargs
        )
        deltas = scipy.absolute(mean - self.y) / self.err_y
        deltas[self.err_y == 0] = 0
        bad_idxs = (deltas >= thresh)
        
        if not mask_only:
            # Delete offending single points:
            X_bad, y_bad, err_X_bad, err_y_bad = self.remove_points(bad_idxs)
        
        # Handle transformed points:
        if check_transformed:
            bad_transformed = scipy.zeros_like(self.transformed, dtype=Channel)
            for k, pt in zip(range(0, len(self.transformed)), self.transformed):
                mean = self.gp.predict(
                    scipy.vstack(pt.X),
                    return_std=False,
                    output_transform=scipy.linalg.block_diag(*pt.T),
                    **predict_kwargs
                )
                deltas = scipy.absolute(mean - pt.y) / pt.err_y
                deltas[pt.err_y == 0] = 0
                bad_idxs = (deltas >= thresh)
            
                bad_X, bad_err_X, bad_y, bad_err_y, bad_T = pt.remove_points(bad_idxs)
                bad_transformed[k] = Channel(
                    bad_X, bad_y, err_X=bad_err_X, err_y=bad_err_y, T=bad_T,
                    y_label=pt.y_label, y_units=pt.y_units
                )
            
                # TODO: Need to do something to return/re-merge the removed points!
            
                # TODO: Need to flag points that no longer have contents!
            
                # TODO: Finish this!
        
        
        # Re-create the GP now that the points have been removed:
        # if 'k' not in gp_kwargs:
        #     gp_kwargs['k'] = self.gp.k
        # if 'noise_k' not in gp_kwargs:
        #     gp_kwargs['noise_k'] = self.gp.noise_k
        # if 'diag_factor' not in gp_kwargs:
        #     gp_kwargs['diag_factor'] = self.gp.diag_factor
        
        # self.create_gp(**gp_kwargs)
        # TODO: This will screw up edge constraints!
        
        if check_transformed:
            if mask_only:
                return (bad_idxs, bad_transformed)
            else:
                return (X_bad, y_bad, err_X_bad, err_y_bad, bad_transformed)
        else:
            if mask_only:
                return bad_idxs
            else:
                return (X_bad, y_bad, err_X_bad, err_y_bad)
        
        # TODO: Re-run MAP estimate and see what to put back in!
    
    def remove_extreme_changes(self, thresh=10, logic='and', mask_only=False):
        """Removes points at which there is an extreme change.
        
        Only for univariate data!
        
        This operation is performed by looking for points who differ by more
        than `thresh` * `err_y` from each of their neighbors. This operation
        will typically only be useful with large values of thresh. This is
        useful for eliminating bad channels.
        
        Note that this will NOT update the Gaussian process.
        
        Parameters
        ----------
        thresh : float, optional
            The threshold as a multiplier times `err_y`. Default is 10 (i.e.,
            throw away all 10-sigma changes).
        logic : {'and', 'or'}, optional
            Whether the logical operation performed should be an and or an or
            when looking at left-hand and right-hand differences. 'and' is more
            conservative, but 'or' will help if you have multiple bad channels
            in a row. Default is 'and' (point must have a drastic change in both 
            directions to be rejected).
        mask_only : bool, optional
            If True, only the boolean mask indicated where the bad points are
            will be removed, and it is up to the user to remove them. Default is
            False (actually remove the bad points).
        """
        if self.X_dim != 1:
            raise NotImplementedError("Extreme change removal is not supported "
                                      "for X_dim = %d" % (self.X_dim,))
        sort_idx = self.X.ravel().argsort()
        y_sort = self.y[sort_idx]
        err_y_sort = self.err_y[sort_idx]
        forward_diff = y_sort[:-1] - y_sort[1:]
        backward_diff = -forward_diff
        forward_diff = scipy.absolute(scipy.append(forward_diff, 0) / err_y_sort)
        backward_diff = scipy.absolute(scipy.insert(backward_diff, 0, 0) / err_y_sort)
        if logic == 'and':
            extreme_changes = (forward_diff >= thresh) & (backward_diff >= thresh)
        elif logic == 'or':
            extreme_changes = (forward_diff >= thresh) | (backward_diff >= thresh)
        else:
            raise ValueError("Unsupported logic '%s'." % (logic,))
        if mask_only:
            return extreme_changes[sort_idx.argsort()]
        else:
            return self.remove_points(extreme_changes[sort_idx.argsort()])
    
    def create_gp(self, k=None, noise_k=None, upper_factor=5, lower_factor=5,
                  x0_bounds=None, mask=None, k_kwargs={}, **kwargs):
        """Create a Gaussian process to handle the data.
        
        Parameters
        ----------
        k : :py:class:`Kernel` instance, optional
            Covariance kernel (from :py:mod:`gptools`) with the appropriate
            number of dimensions, or None. If None, a squared exponential kernel
            is used. Can also be a string from the following table:
            
                ========= ==============================
                SE        Squared exponential
                gibbstanh Gibbs kernel with tanh warping
                RQ        Rational quadratic
                SEsym1d   1d SE with symmetry constraint
                ========= ==============================
            
            The bounds for each hyperparameter are selected as follows (lower
            part is for the Gibbs kernel only):
            
                ============== =============================================
                sigma_f        [1/lower_factor, upper_factor]*range(y)
                l1             [1/lower_factor, upper_factor]*range(X[:, 1])
                ...            And so on for each length scale
                -------------- ---------------------------------------------
                l2 (gibbstanh) [10*eps, upper_factor*range(X[:, 1])]
                lw             [10*eps, upper_factor*range(X[:, 0]) / 50.0]
                x0             range(X[:, 0])
                ============== =============================================
            
            Here, eps is sys.float_info.epsilon. The initial guesses for each
            parameter are set to be halfway between the upper and lower bounds.
            Default is None (use SE kernel).
        noise_k : :py:class:`Kernel` instance, optional
            The noise covariance kernel. Default is None (use the default zero
            noise kernel, with all noise being specified by `err_y`).
        upper_factor : float, optional
            Factor by which the range of the data is multiplied for the upper
            bounds on both length scales and signal variances. Default is 5,
            which seems to work pretty well for C-Mod data.
        lower_factor : float, optional
            Factor by which the range of the data is divided for the lower
            bounds on both length scales and signal variances. Default is 5,
            which seems to work pretty well for C-Mod data.
        x0_bounds : 2-tuple, optional
            Bounds to use on the x0 (transition location) hyperparameter of the
            Gibbs covariance function with tanh warping. This is the
            hyperparameter that tends to need the most tuning on C-Mod data.
            Default is None (use range of X).
        mask : array of bool, optional
            Boolean mask of values to actually include in the GP. Default is to
            include all values.
        k_kwargs : dict, optional
            All entries are passed as kwargs to the constructor for the kernel
            if a kernel instance is not provided.
        **kwargs : optional kwargs
            All additional kwargs are passed to the constructor of
            :py:class:`gptools.GaussianProcess`.
        """
        # TODO: Create more powerful way of specifying kernels!
        # TODO: Set ranges intelligently when using all transformed data!
        # Save some time by only building these arrays once:
        # Note that using this form only gets the non-transformed values.
        y = self.y
        X = self.X
        err_y = self.err_y
        if mask is not None and X is not None:
            y = y[mask]
            X = X[mask, :]
            err_y = err_y[mask]
        if isinstance(k, gptools.Kernel):
            # Skip to the end for pure kernel instances, no need to do all the
            # testing...
            pass
        elif k is None or k == 'SE':
            y_range = y.max() - y.min()
            bounds = [(0.0, upper_factor * y_range)]
            for i in xrange(0, self.X_dim):
                X_range = X[:, i].max() - X[:, i].min()
                bounds.append((0.0, upper_factor * X_range))
            initial = [(b[1] - b[0]) / 2.0 for b in bounds]
            k = gptools.SquaredExponentialKernel(
                num_dim=self.X_dim,
                initial_params=initial,
                param_bounds=bounds,
                **k_kwargs
            )
        elif k == 'gibbstanh':
            # TODO: This is a very hackish way of supporting transformed data. Fix it!
            if self.X_dim != 1:
                raise ValueError('Gibbs kernel is only supported for univariate data!')
            try:
                y_range = y.max() - y.min()
            except (TypeError, ValueError):
                y_range = 10
            sigma_f_bounds = (0, upper_factor * y_range)
            try:
                X_range = X[:, 0].max() - X[:, 0].min()
            except TypeError:
                X_range = 1.2
            l1_bounds = (0.0, upper_factor * X_range)
            l2_bounds = (0.0, l1_bounds[1])
            lw_bounds = (l2_bounds[0], l1_bounds[1] / 50.0)
            if x0_bounds is None:
                x0_bounds = (X[:, 0].min(), X[:, 0].max())
            bounds = [sigma_f_bounds, l1_bounds, l2_bounds, lw_bounds, x0_bounds]
            initial = [(b[1] - b[0]) / 2.0 for b in bounds]
            initial[2] = initial[2] / 2
            k = gptools.GibbsKernel1dTanh(
                initial_params=initial,
                hyperprior=gptools.CoreEdgeJointPrior(bounds),
                **k_kwargs
            )
        elif k == 'gibbsdoubletanh':
            if self.X_dim != 1:
                raise ValueError('Gibbs kernel is only supported for univariate data!')
            y_range = y.max() - y.min()
            sigma_f_bounds = (0.0, upper_factor * y_range)
            X_range = X[:, 0].max() - X[:, 0].min()
            lcore_bounds = (0.0, upper_factor * X_range)
            la_bounds = (0.0, lcore_bounds[1] / 50.0)
            if x0_bounds is None:
                x0_bounds = (X[:, 0].min(), X[:, 0].max())
            bounds = [
                sigma_f_bounds,
                lcore_bounds,
                lcore_bounds,
                lcore_bounds,
                la_bounds,
                la_bounds,
                x0_bounds,
                x0_bounds
            ]
            initial = [(b[1] - b[0]) / 2.0 for b in bounds]
            k = gptools.GibbsKernel1dDoubleTanh(
                initial_params=initial,
                hyperprior=gptools.CoreMidEdgeJointPrior(bounds),
                **k_kwargs
            )
        elif k == 'RQ':
            y_range = y.max() - y.min()
            bounds = [(0.0, upper_factor * y_range), (0.0, 1e2)]
            for i in xrange(0, self.X_dim):
                X_range = X[:, i].max() - X[:, i].min()
                bounds.append((0, upper_factor * X_range))
            initial = [(b[1] - b[0]) / 2.0 for b in bounds]
            k = gptools.RationalQuadraticKernel(
                num_dim=self.X_dim,
                initial_params=initial,
                param_bounds=bounds,
                **k_kwargs
            )
            # Try to avoid some issues that were coming up during MCMC sampling:
            if 'diag_factor' not in kwargs:
                kwargs['diag_factor'] = 1e3
        elif k == 'SEsym1d':
            if self.X_dim != 1:
                raise ValueError("Symmetric SE kernel only supported for univariate data!")
            y_range = y.max() - y.min()
            bounds = [(0.0, upper_factor * y_range)]
            for i in xrange(0, self.X_dim):
                X_range = X[:, i].max() - X[:, i].min()
                bounds.append((0.0, upper_factor * X_range))
            initial = [(b[1] - b[0]) / 2.0 for b in bounds]
            k_base = gptools.SquaredExponentialKernel(
                num_dim=self.X_dim,
                initial_params=initial,
                param_bounds=bounds,
                **k_kwargs
            )
            kM1 = gptools.MaskedKernel(k_base, mask=[0], total_dim=1, scale=[1, 1])
            kM2 = gptools.MaskedKernel(k_base, mask=[0], total_dim=1, scale=[-1, 1])
            k = kM1 + kM2
        elif k == 'SEbeta':
            # TODO: Add support for k_kwargs on warp steps!
            y_range = y.max() - y.min()
            bounds = [(0.0, upper_factor * y_range)]
            for i in xrange(0, self.X_dim):
                X_range = X[:, i].max() - X[:, i].min()
                bounds.append((0.0, upper_factor * X_range))
            initial = [(b[1] - b[0]) / 2.0 for b in bounds]
            k_SE = gptools.SquaredExponentialKernel(
                num_dim=self.X_dim,
                param_bounds=bounds,
                initial_params=initial,
                **k_kwargs
            )
            # TODO: Put in hooks to vary the hyperhyperparameters/hyperprior!
            lognormal_prior = gptools.LogNormalJointPrior([0, 1], [0.25, 1])
            k_SE_beta = gptools.BetaWarpedKernel(k_SE, hyperprior=lognormal_prior)
            # TODO: Make this more intelligent!
            k = gptools.LinearWarpedKernel(k_SE_beta, -1e-3, 1.5)
        elif k == 'matern':
            y_range = y.max() - y.min()
            bounds = [(0.0, upper_factor * y_range), (1.0, 50)]
            for i in xrange(0, self.X_dim):
                X_range = X[:, i].max() - X[:, i].min()
                bounds.append((0.0, upper_factor * X_range))
            initial = [(b[1] - b[0]) / 2.0 for b in bounds]
            k = gptools.MaternKernel1d(
                # num_dim=self.X_dim,
                initial_params=initial,
                param_bounds=bounds,
                **k_kwargs
            )
        elif k == 'matern52':
            y_range = y.max() - y.min()
            bounds = [(0.0, upper_factor * y_range)]
            for i in xrange(0, self.X_dim):
                X_range = X[:, i].max() - X[:, i].min()
                bounds.append((0.0, upper_factor * X_range))
            initial = [(b[1] - b[0]) / 2.0 for b in bounds]
            k = gptools.Matern52Kernel(
                num_dim=self.X_dim,
                initial_params=initial,
                param_bounds=bounds,
                **k_kwargs
            )
        elif k == 'matern52beta':
            y_range = y.max() - y.min()
            bounds = [(0.0, upper_factor * y_range)]
            for i in xrange(0, self.X_dim):
                X_range = X[:, i].max() - X[:, i].min()
                bounds.append((0.0, upper_factor * X_range))
            initial = [(b[1] - b[0]) / 2.0 for b in bounds]
            k_M = gptools.Matern52Kernel(
                num_dim=self.X_dim,
                initial_params=initial,
                param_bounds=bounds,
                **k_kwargs
            )
            # TODO: Put in hooks to vary the hyperhyperparameters!
            lognormal_prior = gptools.LogNormalJointPrior([0.0, 1.0], [0.25, 1.0])
            k_M_beta = gptools.BetaWarpedKernel(k_M, hyperprior=lognormal_prior)
            # TODO: Make this more intelligent!
            k = gptools.LinearWarpedKernel(k_M_beta, -1e-3, 1.5)
        # TODO: I can probably just handle all of the beta-warps at once...
        elif isinstance(k, str):
            raise NotImplementedError("That kernel specification is not supported!")
        self.gp = gptools.GaussianProcess(k, noise_k=noise_k, **kwargs)
        if self.X is not None:
            self.gp.add_data(X, y, err_y=err_y)
        for p in self.transformed:
            if len(p.y) > 0:
                self.gp.add_data(
                    scipy.vstack(p.X),
                    p.y,
                    err_y=p.err_y,
                    T=scipy.linalg.block_diag(*p.T)
                )
    
    def find_gp_MAP_estimate(self, force_update=False, gp_kwargs={}, **kwargs):
        """Find the MAP estimate for the hyperparameters of the Profile's Gaussian process.
        
        If this :py:class:`Profile` instance does not already have a Gaussian
        process, it will be created. Note that the user is responsible for
        manually updating the Gaussian process if more data are added or the
        :py:class:`Profile` is otherwise mutated. This can be accomplished
        directly using the `force_update` keyword.
        
        Parameters
        ----------
        force_update : bool, optional
            If True, a new Gaussian process will be created even if one already
            exists. Set this if you have added data or constraints since you
            created the Gaussian process. Default is False (use current Gaussian
            process if it exists).
        gp_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`create_gp` if it gets called. Default is {}.
        **kwargs : optional parameters
            All other parameters are passed to the Gaussian process'
            :py:meth:`optimize_hyperparameters` method.
        """
        if force_update or self.gp is None:
            self.create_gp(**gp_kwargs)
        return self.gp.optimize_hyperparameters(**kwargs)
    
    def plot_gp(self, force_update=False, gp_kwargs={}, MAP_kwargs={}, **kwargs):
        """Plot the current state of the Profile's Gaussian process.
        
        If this :py:class:`Profile` instance does not already have a Gaussian
        process, it will be created. Note that the user is responsible for
        manually updating the Gaussian process if more data are added or the
        :py:class:`Profile` is otherwise mutated. This can be accomplished
        directly using the `force_update` keyword.
        
        Parameters
        ----------
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
        **kwargs : optional parameters
            All other parameters are passed to the Gaussian process'
            :py:meth:`plot` method.
        """
        if force_update or self.gp is None:
            self.create_gp(**gp_kwargs)
            if not kwargs.get('use_MCMC', False):
                self.find_gp_MAP_estimate(**MAP_kwargs)
        return self.gp.plot(**kwargs)
    
    def smooth(self, X, n=0, force_update=False, plot=False, gp_kwargs={},
               MAP_kwargs={}, **kwargs):
        """Evaluate the underlying smooth curve at a given set of points using Gaussian process regression.
        
        If this :py:class:`Profile` instance does not already have a Gaussian
        process, it will be created. Note that the user is responsible for
        manually updating the Gaussian process if more data are added or the
        :py:class:`Profile` is otherwise mutated. This can be accomplished
        directly using the `force_update` keyword.
        
        Parameters
        ----------
        X : array-like (`N`, `X_dim`)
            Points to evaluate smooth curve at.
        n : non-negative int, optional
            The order of derivative to evaluate at. Default is 0 (return value).
            See the documentation on :py:meth:`gptools.GaussianProcess.predict`.
        force_update : bool, optional
            If True, a new Gaussian process will be created even if one already
            exists. Set this if you have added data or constraints since you
            created the Gaussian process. Default is False (use current Gaussian
            process if it exists).
        plot : bool, optional
            If True, :py:meth:`gptools.GaussianProcess.plot` is called to
            produce a plot of the smoothed curve. Otherwise,
            :py:meth:`gptools.GaussianProcess.predict` is called directly.
        gp_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`create_gp` if it gets called. Default is {}.
        MAP_kwargs : dict, optional
            The entries of this dictionary are passed as kwargs to
            :py:meth:`find_gp_MAP_estimate` if it gets called. Default is {}.
        **kwargs : optional parameters
            All other parameters are passed to the Gaussian process'
            :py:meth:`plot` or :py:meth:`predict` method according to the
            state of the `plot` keyword.
        
        Returns
        -------
        ax : axis instance
            The axis instance used. This is only returned if the `plot`
            keyword is True.
        mean : :py:class:`Array`, (`M`,)
            Predicted GP mean. Only returned if `full_output` is False.
        std : :py:class:`Array`, (`M`,)
            Predicted standard deviation, only returned if `return_std` is True and `full_output` is False.
        full_output : dict
            Dictionary with fields for mean, std, cov and possibly random samples. Only returned if `full_output` is True.
        """
        if force_update or self.gp is None:
            self.create_gp(**gp_kwargs)
            if not kwargs.get('use_MCMC', False):
                self.find_gp_MAP_estimate(**MAP_kwargs)
        if plot:
            plot_kwargs.pop('return_prediction', True)
            return self.gp.plot(X=X, n=n, return_prediction=True, **kwargs)
        else:
            return self.gp.predict(X, n=n, **kwargs)
    
    def write_csv(self, filename):
        """Writes this profile to a CSV file.
        
        Parameters
        ----------
        filename : str
            Path of the file to write. If the file exists, it will be
            overwritten without warning.
        """
        # TODO: Add support for transformed quantities!
        # TODO: Add metadata (probably in CMod...)!
        # Could put metadata as a kwarg...
        
        # Only build these arrays once to save a bit of time:
        # Note that this form does not write any of the transformed quantities!
        X = self.X
        err_X = self.err_X
        y = self.y
        err_y = self.err_y
        
        filename = os.path.expanduser(filename)
        with open(filename, 'wb') as outfile:
            writer = csv.writer(outfile)
            X_labels = [l + ' [' + u + ']' for l, u in zip(self.X_labels, self.X_units)]
            err_X_labels = ['err_' + l for l in X_labels]
            writer.writerow(self.X_labels + err_X_labels +
                            [self.y_label + ' [' + self.y_units + ']'] +
                            ['err_' + self.y_label])
            for k in xrange(0, len(self.y)):
                writer.writerow(
                    [x for x in X[k, :]] + [x for x in err_X[k, :]] + [y[k], err_y[k]]
                )
    
def read_csv(filename, X_names=None, y_name=None, metadata_lines=None):
    """Reads a CSV file into a :py:class:`Profile`.
    
    If names are not provided for the columns holding the `X` and `y` values and
    errors, the names are found automatically by looking at the header row, and
    are used in the order found, with the last column being `y`. Otherwise, the
    columns will be read in the order specified. The column names should be of
    the form "name [units]", which will be automatically parsed to populate the
    :py:class:`Profile`. In either case, there can be a corresponding column
    "err_name [units]" which holds the 1-sigma uncertainty in that quantity.
    There can be an arbitrary number of lines of metadata at the beginning of
    the file which are read into the :py:attr:`metadata` attribute of the
    :py:class:`Profile` created. This is most useful when using
    :py:class:`BivariatePlasmaProfile` as you can store the shot and time window.
    
    Parameters
    ----------
    X_names : list of str, optional
        Ordered list of the column names containing the independent variables.
        The default behavior is to infer the names and ordering from the header
        of the CSV file. See the discussion above. Note that if you provide
        `X_names` you must also provide `y_name`.
    y_name : str, optional
        Name of the column containing the dependent variable. The default
        behavior is to infer this name from the header of the CSV file. See the
        discussion above. Note that if you provide `y_name` you must also
        provide `X_names`.
    metadata_lines : non-negative int, optional
        Number of lines of metadata to read from the beginning of the file.
        These are read into the :py:attr:`metadata` attribute of the profile
        created.
    """
    if X_names and not y_name:
        raise ValueError("If supplying an ordered list of names for the X "
                         "columns, you must also specify the name for the y "
                         "column.")
    if y_name and not X_names:
        raise ValueError("If supplying a name for the y column you must also "
                         "supply an ordered list of names for the X columns.")
    filename = os.path.expanduser(filename)
    X = []
    y = []
    err_X = []
    err_y = []
    metadata = []
    with open(filename, 'rb') as infile:
        # Capture metadata, if present:
        if metadata_lines is None:
            first_line = infile.readline()
            if first_line.startswith("metadata"):
                try:
                    metadata_lines = int(first_line.split(None, 1)[1])
                except ValueError:
                    metadata_lines = 1
            else:
                metadata_lines = 0
            infile.seek(0)
        for k in xrange(0, metadata_lines):
            metadata.append(infile.readline())
        if not (X_names and y_name):
            X_names = infile.readline().split(',')
            X_names = [name for name in X_names if not name.startswith('err_')]
            y_name = X_names.pop(-1)
            infile.seek(0)
            # Need to skip the metadata again:
            for k in xrange(0, metadata_lines):
                infile.readline()
        rdr = csv.DictReader(infile)
        for row in rdr:
            X.append([row[l] for l in X_names])
            err_X_row = []
            for l in X_names:
                try:
                    err_X_row.append(row['err_' + l])
                except KeyError:
                    err_X_row.append(0)
            err_X.append(err_X_row)
            y.append(row[y_name])
            try:
                err_y.append(row['err_' + y_name])
            except KeyError:
                err_y.append(0)
        
        y_label, y_units = parse_column_name(y_name)
        X_labels = []
        X_units = []
        for X_name in X_names:
            n, u = parse_column_name(X_name)
            X_labels.append(n)
            X_units.append(n)
        
        X_dim = len(X_labels)
        if X_dim == 1:
            X_labels = X_labels[0]
            X_units = X_units[0]
    
    p = Profile(X_dim=X_dim, X_units=X_units, y_units=y_units,
                X_labels=X_labels, y_label=y_label)
    p.add_data(X, y, err_X=err_X, err_y=err_y)
    p.metadata = metadata
    
    return p

def read_NetCDF(filename, X_names, y_name, metadata=[]):
    """Reads a NetCDF file into a :py:class:`Profile`.
    
    The file must contain arrays of equal length for each of the independent and
    the dependent variable. The units of each variable can either be specified
    as the units attribute on the variable, or the variable name can be of the
    form "name [units]", which will be automatically parsed to populate the
    :py:class:`Profile`. For each independent and the dependent variable there
    can be a corresponding column "err_name" or "err_name [units]" which holds
    the 1-sigma uncertainty in that quantity. There can be an arbitrary number
    of metadata attributes in the file which are read into the corresponding
    attributes of the :py:class:`Profile` created. This is most useful when using
    :py:class:`BivariatePlasmaProfile` as you can store the shot and time window.
    Be careful that you do not overwrite attributes needed by the class, however!
    
    Parameters
    ----------
    X_names : list of str
        Ordered list of the column names containing the independent variables.
        See the discussion above regarding name conventions.
    y_name : str
        Name of the column containing the dependent variable. See the discussion
        above regarding name conventions.
    metadata : list of str, optional
        List of attribute names to read into the corresponding attributes of the
        :py:class:`Profile` created.
    """
    with scipy.io.netcdf.netcdf_file(os.path.expanduser(filename), mode='r') as infile:
        X = []
        err_X = []
        X_labels = []
        X_units = []
        for l in X_names:
            vXl = infile.variables[l]
            X.append(vXl[:])
            n, u = parse_column_name(l)
            X_labels.append(n)
            try:
                X_units.append(vXl.units)
            except AttributeError:
                X_units.append(u)
            try:
                err_X.append(infile.variables['err_' + l])
            except KeyError:
                err_X.append(scipy.zeros_like(X[0]))
        X = scipy.hstack(X)
        err_X = scipy.hstack(err_X)
        vy = infile.variables[y_name]
        # Explicitly convert, since I've been having strange segfaults here:
        y = scipy.array(vy[:])
        y_label, u = parse_column_name(y_name)
        try:
            y_units = vy.units
        except AttributeError:
            y_units = u
        try:
            err_y = scipy.array(infile.variables['err_' + y_name][:])
        except KeyError:
            err_y = 0
        X_dim = len(X_labels)
        if X_dim == 1:
            X_labels = X_labels[0]
            X_units = X_units[0]
        p = Profile(X_dim=X_dim, X_units=X_units, y_units=y_units,
                    X_labels=X_labels, y_label=y_label)
        p.add_data(X, y, err_X=err_X, err_y=err_y)
        for m in metadata:
            try:
                if hasattr(p, m):
                    warnings.warn("Profile class already has metadata attribute %s. "
                                  "Existing value is being overwritten. This may "
                                  "lead to undesirable behavior." % (m,),
                                  RuntimeWarning)
                setattr(p, m, infile.m)
            except AttributeError:
                warnings.warn("Could not find metadata attribute %s in NetCDF file %s." % 
                              (m, filename,), RuntimeWarning)
    return p

def parse_column_name(name):
    """Parse a column header `name` into label and units.
    """
    name_split = re.split(r'^([^ \t]*)[ \t]*\[(.*)\]$', name)
    if len(name_split) == 1:
        name = name_split[0]
        units = ''
    else:
        assert len(name_split) == 4
        name = name_split[1]
        units = name_split[2]
    return (name, units)

def errorbar3d(ax, x, y, z, xerr=None, yerr=None, zerr=None, **kwargs):
    """Draws errorbar plot of z(x, y) with errorbars on all variables.
    
    Parameters
    ----------
    ax : 3d axis instance
        The axis to draw the plot on.
    x : array, (`M`,)
        x-values of data.
    y : array, (`M`,)
        y-values of data.
    z : array, (`M`,)
        z-values of data.
    xerr : array, (`M`,), optional
        Errors in x-values. Default value is 0.
    yerr : array, (`M`,), optional
        Errors in y-values. Default value is 0.
    zerr : array, (`M`,), optional
        Errors in z-values. Default value is 0.
    **kwargs : optional
        Extra arguments are passed to the plot command used to draw the
        datapoints.
    """
    fmt = kwargs.pop('fmt', kwargs.pop('marker', 'o'))
    if xerr is None:
        no_x = True
        xerr = scipy.zeros_like(x)
    else:
        no_x = False
    if yerr is None:
        no_y = True
        yerr = scipy.zeros_like(y)
    else:
        no_y = False
    if zerr is None:
        no_z = True
        zerr = scipy.zeros_like(z)
    else:
        no_z = False
    pts = ax.plot(x, y, z, fmt, **kwargs)
    color = plt.getp(pts[0], 'color')
    # Only draw the lines if the error is nonzero:
    for X, Y, Z, Xerr, Yerr, Zerr in zip(x, y, z, xerr, yerr, zerr):
        if not no_x:
            ax.plot([X - Xerr, X + Xerr], [Y, Y], [Z, Z], color=color, marker='_')
        if not no_y:
            ax.plot([X, X], [Y - Yerr, Y + Yerr], [Z, Z], color=color, marker='_')
        if not no_z:
            ax.plot([X, X], [Y, Y], [Z - Zerr, Z + Zerr], color=color, marker='_')

def unique_rows(arr):
    """Returns a copy of arr with duplicate rows removed.
    
    From Stackoverflow "Find unique rows in numpy.array."
    
    Parameters
    ----------
    arr : :py:class:`Array`, (`m`, `n`). The array to find the unique rows of.
    
    Returns
    -------
    unique : :py:class:`Array`, (`p`, `n`) where `p` <= `m`
        The array `arr` with duplicate rows removed.
    """
    b = scipy.ascontiguousarray(arr).view(
        scipy.dtype((scipy.void, arr.dtype.itemsize * arr.shape[1]))
    )
    try:
        dum, idx = scipy.unique(b, return_index=True)
    except TypeError:
        # Handle bug in numpy 1.6.2:
        rows = [_Row(row) for row in b]
        srt_idx = sorted(range(len(rows)), key=rows.__getitem__)
        rows = scipy.asarray(rows)[srt_idx]
        row_cmp = [-1]
        for k in xrange(1, len(srt_idx)):
            row_cmp.append(rows[k-1].__cmp__(rows[k]))
        row_cmp = scipy.asarray(row_cmp)
        transition_idxs = scipy.where(row_cmp != 0)[0]
        idx = scipy.asarray(srt_idx)[transition_idxs]
    return arr[idx]

def get_nearest_idx(v, a):
    """Returns the array of indices of the nearest value in `a` corresponding to each value in `v`.
    
    Parameters
    ----------
    v : Array
        Input values to match to nearest neighbors in `a`.
    a : Array
        Given values to match against.
    
    Returns
    -------
    Indices in `a` of the nearest values to each value in `v`. Has the same shape as `v`.
    """
    # Gracefully handle single-value versus array inputs, returning in the
    # corresponding type.
    try:
        return scipy.array([(scipy.absolute(a - val)).argmin() for val in v])
    except TypeError:
        return (scipy.absolute(a - v)).argmin()

class RejectionFunc(object):
    """Rejection function for use with `full_MC` mode of :py:func:`GaussianProcess.predict`.
    
    Parameters
    ----------
    mask : array of bool
        Mask for the values to include in the test.
    positivity : bool, optional
        Set this to True to impose a positivity constraint on the sample.
        Default is True.
    monotonicity : bool, optional
        Set this to True to impose a positivity constraint on the samples.
        Default is True.
    """
    def __init__(self, mask, positivity=True, monotonicity=True):
        self.mask = mask
        self.positivity = positivity
        self.monotonicity = monotonicity
    
    def __call__(self, samp):
        """Returns True if the sample meets the constraints, False otherwise.
        """
        k = len(self.mask)
        if ((self.positivity and (samp[:k][self.mask].min() < 0)) or 
                (self.monotonicity and (samp[k:2*k][self.mask].max() > 0))):
            return False
        else:
            return True

def leading_axis_product(w, x):
    """Perform a product along the leading axis, as is needed when applying weights.
    """
    return scipy.einsum('i...,i...->i...', w, x)

def meanw(x, weights=None, axis=None):
    r"""Weighted mean of data.
    
    Defined as
    
    .. math::
        
        \mu = \frac{\sum_i w_i x_i}{\sum_i w_i}
    
    Parameters
    ----------
    x : array-like
        The vector to find the mean of.
    weights : array-like, optional
        The weights. Must be broadcastable with `x`. Default is to use the
        unweighted mean.
    axis : int, optional
        The axis to take the mean along. Default is to use the whole data set.
    """
    if weights is None:
        return scipy.mean(x, axis=axis)
    else:
        x = scipy.asarray(x)
        weights = scipy.asarray(weights)
        return leading_axis_product(weights, x).sum(axis=axis) / weights.sum(axis=axis)

def varw(x, weights=None, axis=None, ddof=1, mean=None):
    r"""Weighted variance of data.
    
    Defined (for `ddof`=1) as
    
    .. math::
        
        s^2 = \frac{\sum_i w_i}{(\sum_i w_i)^2 - \sum_i w_i^2}\sum_i w_i (x_i - \mu)^2
    
    Parameters
    ----------
    x : array-like
        The vector to find the mean of.
    weights : array-like, optional
        The weights. Must be broadcastable with `x`. Default is to use the
        unweighted mean.
    axis : int, optional
        The axis to take the mean along. Default is to use the whole data set.
    ddof : int, optional
        The degree of freedom correction to use. If no weights are given, this
        is the standard Bessel correction. If weights are given, this uses an
        approximate form based on the assumption that the weights are inverse
        variances for each data point. In this case, the value has no effect
        other than being True or False. Default is 1 (apply correction assuming
        normal noise dictated weights).
    mean : array-like, optional
        The weighted mean to use. If you have already computed the weighted mean
        with :py:func:`meanw`, you can pass the result in here to save time.
    """
    if weights is None:
        return scipy.var(x, axis=axis, ddof=1)
    else:
        x = scipy.asarray(x)
        weights = scipy.asarray(weights)
        if mean is None:
            mean = meanw(x, weights=weights, axis=axis)
        else:
            mean = scipy.asarray(mean)
        V1 = weights.sum(axis=axis)
        M = leading_axis_product(weights, (x - mean)**2).sum(axis=axis)
        if ddof:
            res = V1 / (V1**2 - (weights**2).sum(axis=axis)) * M
            # Put nan where the result blow up to be consistent with scipy:
            try:
                res[scipy.isinf(res)] = scipy.nan
            except TypeError:
                if scipy.isinf(res):
                    res = scipy.nan
            return res
        else:
            return M / V1

def stdw(*args, **kwargs):
    r"""Weighted standard deviation of data.
    
    Defined (for `ddof`=1) as
    
    .. math::
        
        s = \sqrt{\frac{\sum_i w_i}{(\sum_i w_i)^2 - \sum_i w_i^2}\sum_i w_i (x_i - \mu)^2}
    
    Parameters
    ----------
    x : array-like
        The vector to find the mean of.
    weights : array-like, optional
        The weights. Must be broadcastable with `x`. Default is to use the
        unweighted mean.
    axis : int, optional
        The axis to take the mean along. Default is to use the whole data set.
    ddof : int, optional
        The degree of freedom correction to use. If no weights are given, this
        is the standard Bessel correction. If weights are given, this uses an
        approximate form based on the assumption that the weights are inverse
        variances for each data point. In this case, the value has no effect
        other than being True or False. Default is 1 (apply correction assuming
        normal noise dictated weights).
    mean : array-like, optional
        The weighted mean to use. If you have already computed the weighted mean
        with :py:func:`meanw`, you can pass the result in here to save time.
    """
    return scipy.sqrt(varw(*args, **kwargs))

# Conversion factor to get from interquartile range to standard deviation:
IQR_TO_STD = 2.0 * scipy.stats.norm.isf(0.25)

def robust_std(y, axis=None):
    r"""Computes the robust standard deviation of the given data.
    
    This is defined as :math:`IQR/(2\Phi^{-1}(0.75))`, where :math:`IQR` is the
    interquartile range and :math:`\Phi` is the inverse CDF of the standard
    normal. This is an approximation based on the assumption that the data are
    Gaussian, and will have the effect of diminishing the effect of outliers.
    
    Parameters
    ----------
    y : array-like
        The data to find the robust standard deviation of.
    axis : int, optional
        The axis to find the standard deviation along. Default is None (find
        from whole data set).
    """
    return (scipy.stats.scoreatpercentile(y, 75.0, axis=axis) -
            scipy.stats.scoreatpercentile(y, 25.0, axis=axis)) / IQR_TO_STD

def scoreatpercentilew(x, p, weights):
    """Computes the weighted score at the given percentile.
    
    Does not work on small data sets!
    
    Parameters
    ----------
    x : array
        Array of data to apply to. Only works properly on 1d data!
    p : float or array of float
        Percentile(s) to find.
    weights : array, same shape as `x`
        The weights to apply to the values in `x`.
    """
    # TODO: Vectorize this!
    x = scipy.asarray(x)
    weights = scipy.asarray(weights)
    
    srt = x.argsort()
    x = x[srt]
    w = weights[srt]
    
    Sn = w.cumsum()
    pn = 100.0 / Sn[-1] * (Sn - w / 2.0)
    k = scipy.digitize(scipy.atleast_1d(p), pn) - 1
    return x[k] + (p - pn[k]) / (pn[k + 1] - pn[k]) * (x[k + 1] - x[k])
    # TODO: This returns an array for a scalar input!

def medianw(x, weights=None, axis=None):
    """Computes the weighted median of the given data.
    
    Does not work on small data sets!
    
    Parameters
    ----------
    x : array
        Array of data to apply to. Only works properly on 1d, 2d and 3d data.
    weights : array, optional
        Weights to apply to the values in `x`. Default is to use an unweighted
        estimator.
    axis : int, optional
        The axis to take the median along. Default is None (apply to flattened
        array).
    """
    # TODO: This could be done a whole lot better!
    if weights is None:
        return scipy.median(x, axis=axis)
    else:
        if axis is None and x.ndim == 1:
            return scoreatpercentilew(x, 50, weights)[0]
        elif axis == 0 and x.ndim == 3:
            out = scipy.zeros_like(x[0])
            for i in xrange(0, out.shape[0]):
                for j in xrange(0, out.shape[1]):
                    out[i, j] = scoreatpercentilew(x[:, i, j], 50, weights)
            return out
        elif axis == 0 and x.ndim == 2:
            out = scipy.zeros(x.shape[1])
            for i in xrange(0, len(out)):
                out[i] = scoreatpercentilew(x[:, i], 50, weights)
            return out
        else:
            raise NotImplementedError("That shape/axis is not supported!")

def robust_stdw(x, weights=None, axis=None):
    """Computes the weighted robust standard deviation from the weighted IQR.
    
    Does not work on small data sets!
    
    Parameters
    ----------
    x : array
        Array of data to apply to. Only works properly on 1d, 2d and 3d data.
    weights : array, optional
        Weights to apply to the values in `x`. Default is to use an unweighted
        estimator.
    axis : int, optional
        The axis to take the robust standard deviation along. Default is None
        (apply to flattened array).
    """
    # TODO: This could be done a whole lot better!
    if weights is None:
        return robust_std(x, axis=axis)
    else:
        if axis is None and x.ndim == 1:
            lq, uq = scoreatpercentilew(x, [25, 75], weights)
            return (uq - lq) / IQR_TO_STD
        elif axis == 0 and x.ndim == 3:
            lq = scipy.zeros_like(x[0])
            uq = scipy.zeros_like(x[0])
            for i in xrange(0, lq.shape[0]):
                for j in xrange(0, lq.shape[1]):
                    lqij, uqij = scoreatpercentilew(x[:, i, j], [25, 75], weights)
                    lq[i, j] = lqij
                    uq[i, j] = uqij
            return (uq - lq) / IQR_TO_STD
        elif axis == 0 and x.ndim == 2:
            lq = scipy.atleast_1d(scipy.zeros(x.shape[1]))
            uq = scipy.atleast_1d(scipy.zeros(x.shape[1]))
            for i in xrange(0, len(lq)):
                lqi, uqi = scoreatpercentilew(x[:, i], [25, 75], weights)
                lq[i] = lqi
                uq[i] = uqi
            return (uq - lq) / IQR_TO_STD
        else:
            raise NotImplementedError("That shape/axis is not supported!")
