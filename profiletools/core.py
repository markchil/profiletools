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

"""Provides the base :py:class:`Profile` class.
"""

from __future__ import division

import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
    def __init__(self, X_dim=1, X_units=None, y_units='', X_labels=None, y_label=''):
        self.X_dim = X_dim
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
    
    def add_data(self, X, y, err_X=0, err_y=0, channels=None):
        """Add data to the training data set of the :py:class:`Profile` instance.
        
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
        try:
            iter(y)
        except TypeError:
            y = scipy.array([y], dtype=float)
        else:
            y = scipy.asarray(y, dtype=float)
            if y.ndim != 1:
                raise ValueError("Dependent variables y must have only one "
                                 "dimension with length greater than one! Shape "
                                 "of y given is %s" % (y.shape,))
        
        # Handle scalar error or verify shape of array error matches shape of y:
        try:
            iter(err_y)
        except TypeError:
            err_y = err_y * scipy.ones_like(y, dtype=float)
        else:
            err_y = scipy.asarray(err_y, dtype=float)
            if err_y.shape != y.shape:
                raise ValueError("When using array-like err_y, shape must match "
                                 "shape of y! Shape of err_y given is %s, shape "
                                 "of y given is %s." % (err_y.shape, y.shape))
        if (err_y < 0).any():
            raise ValueError("All elements of err_y must be non-negative!")
        
        # Handle scalar independent variable or convert array input into matrix.
        X = scipy.asmatrix(X, dtype=float)
        # Correct single-dimension inputs:
        if self.X_dim == 1 and X.shape[0] == 1:
            X = X.T
        if X.shape != (len(y), self.X_dim):
            raise ValueError("Shape of independent variables must be (len(y), self.X_dim)! "
                             "X given has shape %s, shape of "
                             "y is %s and X_dim=%d." % (X.shape, y.shape, self.X_dim))
        
        # Process uncertainty in err_X:
        try:
            iter(err_X)
        except TypeError:
            err_X = err_X * scipy.ones_like(X, dtype=float)
        else:
            err_X = scipy.asarray(err_X)
            if err_X.ndim == 1 and self.X_dim != 1:
                err_X = scipy.tile(err_X, (X.shape[0], 1))
        err_X = scipy.asmatrix(err_X, dtype=float)
        if self.X_dim == 1 and err_X.shape[0] == 1:
            err_X = err_X.T
        if err_X.shape != X.shape:
            raise ValueError("Shape of uncertainties on independent variables "
                             "must be (len(y), self.X_dim)! X given has shape %s, "
                             "shape of y is %s and X_dim=%d."
                             % (X.shape, y.shape, self.X_dim))
        
        if (err_X < 0).any():
            raise ValueError("All elements of err_X must be non-negative!")
        
        # Process channel flags:
        if channels is None:
            channels = X.copy()
        else:
            if isinstance(channels, dict):
                d_channels = channels
                channels = X.copy()
                for idx in d_channels:
                    channels[:, idx] = scipy.atleast_2d(d_channels[idx]).T
            else:
                channels = scipy.asmatrix(channels)
                if channels.shape != X.shape:
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
    
    def add_profile(self, other):
        """Absorbs the data from one profile object.

        Parameters
        ----------
        other : :py:class:`Profile`
            :py:class:`Profile` to absorb.
        """
        if self.X_dim != other.X_dim:
            raise ValueError("When merging profiles, X_dim must be equal between "
                             "the two profiles!")
        if self.y_units != other.y_units:
            raise ValueError("When merging profiles, the y_units must agree!")
        if self.X_units != other.X_units:
            raise ValueError("When merging profiles, the X_units must agree!")
        # Modify the channels of other.channels to avoid clashes:
        new_other_channels = (other.channels - other.channels.min(axis=0) +
                              self.channels.max(axis=0) + 1)
        self.add_data(other.X, other.y, err_X=other.err_X, err_y=other.err_y,
                      channels=new_other_channels)

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
        self.channels = scipy.delete(self.channels, axis, axis=1)
        self.X = scipy.delete(self.X, axis, axis=1)
        self.err_X = scipy.delete(self.err_X, axis, axis=1)
        self.X_labels.pop(axis)
        self.X_units.pop(axis)

    def average_data(self, axis=0, ddof=1):
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
        ddof : int, optional
            The degree of freedom correction used in computing the standard
            deviation. The default is 1, the standard Bessel correction to
            give an unbiased estimate of the variance.
        """
        # TODO: Add support for custom bins!
        if self.X_dim == 1:
            return scipy.mean(self.y)
        reduced_channels = scipy.delete(self.channels, axis, axis=1)
        reduced_X = scipy.delete(self.X, axis, axis=1)
        channels = unique_rows(reduced_channels)
        # TODO: Add support for other estimators!
        X = scipy.zeros((len(channels), self.X_dim - 1))
        y = scipy.zeros(len(channels))
        err_X = scipy.zeros_like(X)
        err_y = scipy.zeros_like(y)
        for i, chan in zip(range(0, len(channels)), channels):
            chan_mask = (scipy.asarray(reduced_channels) ==
                         scipy.asarray(chan).flatten()).all(axis=1)
            y[i] = scipy.mean(self.y[chan_mask])
            err_y[i] = scipy.std(self.y[chan_mask], ddof=ddof)
            X[i, :] = scipy.mean(reduced_X[chan_mask, :], axis=0)
            err_X[i, :] = scipy.std(reduced_X[chan_mask, :], ddof=ddof, axis=0)
        
        self.X_dim -= 1
        self.X_units.pop(axis)
        self.X_labels.pop(axis)
        self.X = X
        self.y = y
        self.err_X = err_X
        self.err_y = err_y
        self.channels = channels
    
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
            ax.errorbar(scipy.asarray(self.X).flatten(), self.y,
                        yerr=self.err_y, xerr=scipy.asarray(self.err_X).flatten(),
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
            X_arr = scipy.asarray(self.X)
            err_X_arr = scipy.asarray(self.err_X)
            errorbar3d(ax, X_arr[:, 0], X_arr[:, 1], self.y,
                       xerr=err_X_arr[:, 0], yerr=err_X_arr[:, 1], zerr=self.err_y,
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
        
        Parameters
        ----------
        conditional : array-like of bool, (`M`,)
            Array of booleans corresponding to each entry in `y`. Where an
            entry is True, that value will be removed.
        """
        idxs = ~conditional
        self.y = self.y[idxs]
        self.X = self.X[idxs, :]
        self.err_y = self.err_y[idxs]
        self.err_X = self.err_X[idxs, :]
        self.channels = self.channels[idxs, :]

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
