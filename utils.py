"""
Small helper functions
"""

from typing import Tuple, List
import numpy as np

try:
    import cupy as cp
    from cupyx.scipy.fft import fftn, ifftn, fftshift
except ImportError as error:
    print("No cupy, using numpy instead")
    import numpy as cp
    from scipy.fft import fftn, ifftn, fftshift


def gpu2cpu(vars: cp.ndarray | List | Tuple) -> Tuple | np.ndarray:
    # 1st Check if several or single input
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, cp.ndarray):
        return vars.get()
    else:
        return tuple(var.get() if hasattr(var, "get") else var for var in vars)

def cpu2gpu(vars: List | Tuple | np.ndarray | cp.ndarray) -> Tuple | cp.ndarray:
    if isinstance(vars, np.ndarray) or isinstance(vars, cp.ndarray):
        return cp.asarray(vars)
    else:
        return tuple(
            cp.asarray(var) if isinstance(var, np.ndarray) else var for var in vars
        )


def expand_3d(array_2d: np.ndarray, k: int) -> np.ndarray:
    array_3d = cp.tile(cp.expand_dims(array_2d, axis=2), k)
    return array_3d


def dot_prod(a, b):
    c = cp.abs(a * b.conj()).sum()
    return c


def norm_corr(obj, rec):
    a = dot_prod(obj, obj)
    b = dot_prod(rec, rec)
    c = dot_prod(obj, rec)
    return c / cp.sqrt(a * b)


def norm_delta(obj, rec):
    a = dot_prod(obj, obj)
    b = dot_prod(rec, rec)
    c = dot_prod(obj - rec, obj - rec)
    return c / cp.sqrt(a * b)


def align_phase(a, b):
    a_phase = cp.angle((b * a.conj()).sum())
    a = a * cp.exp(1.0j * a_phase)
    return a


def zero_dev(a, b):
    c = cp.nan_to_num(cp.divide(a, b))
    return c


def rxy_map(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    # rxy_map(n)
    Returns an (n x n) matrix with Radius values from center point.
    fftshift of (0,0) is always 0.
    dx = dy = 1.
    """
    if n % 2 == 0:
        x = cp.linspace(-n / 2, n / 2 - 1, n)
    else:
        x = cp.linspace(-(n - 1) / 2, (n - 1) / 2, n)
    x_grid, y_grid = cp.meshgrid(x, x)
    r_grid = cp.hypot(x_grid, y_grid)
    return r_grid, x_grid, y_grid


def circle(n: int, r_grid: int | float) -> np.ndarray:
    """
    # circle(camera_n, R)
    Draws an image with NxN dimensions,
    with a circle in its middle with a radius of R pixels.
    Circle values are 1, other values are 0.
    """
    circ = (rxy_map(n)[0] < r_grid) * 1.0
    return circ


def rectangle(n, xy):
    """
    Draws an image with NxN dimensions,
    with a (xy[0] x xy[1]) rectangle in its middle
    rectangle values are 1, other values are 0.
    """
    _, x_grid, y_grid = rxy_map(n)

    rect = ((abs(x_grid) < (xy[0] / 2)) * (abs(y_grid) < (xy[1] / 2))) * 1.0
    return rect


def lp_filt(obj: cp.ndarray, lp_type: str, filter_size: float | int) -> np.ndarray:
    """
    Apply a low-pass filter to a given array using specified filter types and sizes.

    Parameters
    ----------
    obj : cp.ndarray
        The input array to which the low-pass filter is to be applied. Expected to be a 2D array.
    lp_type : str
        The type of low-pass filter to apply. Options include:
        - "none": No filtering, returns the original array.
        - "circ": Circular low-pass filter.
        - "square": Square-shaped low-pass filter.
        - "gaussian": Gaussian low-pass filter.
    filter_size : float or int
        The size of the filter. Interpretation of this parameter depends on the filter type:
        - For "circ" and "square", it represents the radius or side length respectively.
        - For "gaussian", it represents the standard deviation of the Gaussian.

    Returns
    -------
    np.ndarray
        The filtered array after applying the specified low-pass filter.

    Raises
    ------
    TypeError
        If `lp_type` is not one of the specified options ("none", "circ", "square", "gaussian").

    Notes
    -----
    - The function makes use of several helper functions such as `circle`, `rectangle`, `fftshift`, `rxy_map`, `fftn`, and `ifftn`.
    - The function assumes that `obj` is a 2D array and that its dimensions are equal (square).
    """
    match lp_type:
        case "none":
            lp_filter = cp.ones_like(obj)
        case "circ":
            n = obj.shape[1]
            r = filter_size
            lp_filter = circle(n, r / 2)
            lp_filter = fftshift(lp_filter)
        case "square":
            n = obj.shape[1]
            xy = (filter_size, filter_size)
            lp_filter = rectangle(n, xy)
            lp_filter = fftshift(lp_filter)
        case "gaussian":
            n = obj.shape[0]  # Should be square
            radius_2d = fftshift(rxy_map(n)[0])
            gaussian = cp.exp(-(radius_2d**2) / (2 * filter_size**2))
            lp_filter = fftn(gaussian / gaussian.sum())
        case _:
            raise TypeError(
                "Second input should be a tuple with 1st entry either 'circ' or 'square' for lp_filter type "
            )
    return ifftn(lp_filter * fftn(obj))
