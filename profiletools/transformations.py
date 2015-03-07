import scipy
import warnings
import matplotlib.pyplot as plt
import eqtools
try:
    import TRIPPy
except ImportError:
    warnings.warn("Module TRIPPy could not be loaded!", RuntimeWarning)
import profiletools

class ConversionWrapper(object):
    """Class to wrap a coordinate transform to avoid the overhead of using an anonymous (lambda) function.
    
    Parameters
    ----------
    abscissa : str
        The coordinate to convert to.
    efit_tree : :py:class:`eqtools.Equilibrium`
        The EFIT tree used to perform the mappings.
    """
    def __init__(self, abscissa, efit_tree):
        self.abscissa = abscissa
        self.efit_tree = efit_tree
    
    def __call__(self, *args, **kwargs):
        return self.efit_tree.rz2rho(self.abscissa, *args, **kwargs)

def get_transforms(abscissa, tci_chords, efit_tree, times, point_array, Z_point, theta, ds=1e-3):
    """Retrieves the weights to be used for TCI transforms.
    
    Parameters
    ----------
    abscissa : str
        Coordinate system to use, passed as first arg to :py:meth:`rz2rho`.
    tci_chords : array of float
        Radial locations of the chords.
    efit_tree : :py:class:`eqtools.Equilibrium`
        EFIT tree used to perform mappings.
    times : array of float
        Times to compute the weights transform at.
    point_array : array of float
        The quadrature points to use. Must be strictly monotonically increasing.
    Z_point : float
        Z coordinate of the starting point of the rays (should be well outside
        the tokamak). Units are meters.
    theta : float
        Angle of the chords. Units are radians.
    """
    tokamak = TRIPPy.plasma.Tokamak(efit_tree)
    rays = ray_array(tokamak, tci_chords, Z_point, theta)
    T = TRIPPy.invert.fluxFourierSens(
        rays,
        ConversionWrapper(abscissa, efit_tree),
        tokamak.center,
        times,
        point_array,
        ds=ds
    )
    
    return T

def ray_array(tokamak, tci_chords, Z_point, theta):
    """Produce an array of :py:class:`TRIPPy.beam.Ray` objects for each TCI chord.
    
    Takes the tokamak object and the radial locations of the TCI chords and
    produces the ray array geometry objects to be passed into the
    :py:func:`TRIPPy.fluxFourierSens` function.
    
    Parameters
    ----------
    tokamak : :py:class:`TRIPPy.Tokamak`
        The geometry object to compute the rays in.
    tci_chords : array of float
        Radial locations of the chords.
    Z_point : float
        Z coordinate of the starting point of the rays (should be well outside
        the tokamak). Units are meters.
    theta : float
        Angle of the chords. Units are radians.
    """
    # Create array of Vecr, one for each chord:
    vec_array = [TRIPPy.geometry.Vecr([R_val, theta, Z_point]) for R_val in tci_chords]
    
    # Create array of Point, one for each chord:
    point_array = [TRIPPy.geometry.Point(v, tokamak) for v in vec_array]
    
    # Directional vector for the chords:
    vert_vector = TRIPPy.geometry.Vecx(scipy.array([0., 0., 1.]))
    
    # Flag the points that need correction for how many times the pass through
    # the vessel:
    # TODO: THIS NEEDS TO BE CHECKED FOR OLDER/NEWER SHOTS!
    limiter_array = scipy.zeros(len(tci_chords), dtype=int)
    limiter_array[0:5] = 2
    
    # Create an array of Ray, one for each chord:
    ray_array = [TRIPPy.beam.Ray(p, vert_vector) for p in point_array]
    
    # Trace each ray through the tokamak:
    for r, l in zip(ray_array, limiter_array):
        tokamak.trace(r, limiter=l)
    
    return ray_array
