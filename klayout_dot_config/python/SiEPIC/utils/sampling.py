# Adapted from: https://stackoverflow.com/questions/14084634/adaptive-plotting-of-a-function-in-python
# Old license: "Adaptive sampling of 1D functions" by unknown CC0

# Modified by Thomas Ferreira de Lima @thomaslima
# License: CC BY 4.0 https://creativecommons.org/licenses/by/4.0/

import numpy as np


def sample_function(func, points, tol=0.001, min_points=8, max_level=16,
                    sample_transform=None):
    """
    Sample a 1D function to given tolerance by adaptive subdivision.

    The function itself can be multidimensional.

    The result of sampling is a set of points that, if plotted,
    produces a smooth curve with also sharp features of the function
    resolved.

    Parameters
    ----------
    func : callable
        Function func(x) of a single argument. It is assumed to be vectorized.
    points : array-like, 1D
        Initial points to sample, sorted in ascending order.
        These will determine also the bounds of sampling.
    tol : float, optional
        Tolerance to sample to. The condition is roughly that the total
        length of the curve on the (x, y) plane is computed up to this
        tolerance.
    min_point : int, optional
        Minimum number of points to sample.
    max_level : int, optional
        Maximum subdivision depth.
    sample_transform : callable, optional
        Function w = g(x, y). The x-samples are generated so that w
        is sampled.

    Returns
    -------
    x : ndarray
        X-coordinates
    y : ndarray
        Corresponding values of func(x)

    Notes
    -----
    This routine is useful in computing functions that are expensive
    to compute, and have sharp features --- it makes more sense to
    adaptively dedicate more sampling points for the sharp features
    than the smooth parts.

    Examples
    --------
    >>> def func(x):
    ...     '''Function with a sharp peak on a smooth background'''
    ...     a = 0.001
    ...     return x + a**2/(a**2 + x**2)
    ...
    >>> x, y = sample_function(func, [-1, 1], tol=1e-3)

    >>> import matplotlib.pyplot as plt
    >>> xx = np.linspace(-1, 1, 12000)
    >>> plt.plot(xx, func(xx), '-', x, y[0], '.')
    >>> plt.show()

    """
    with np.errstate(divide='ignore', invalid='ignore'):
        return _sample_function(func, points, values=None, mask=None, depth=0,
                                tol=tol, min_points=min_points, max_level=max_level,
                                sample_transform=sample_transform)


def _sample_function(func, points, values=None, mask=None, tol=0.05,
                     depth=0, min_points=16, max_level=16,
                     sample_transform=None):
    points = np.unique(points)

    if values is None:
        values = np.atleast_2d(func(points))

    if mask is None:
        mask = slice(None)

    if depth > max_level:
        # recursion limit
        return points, values

    x_a = points[..., :-1][..., mask]
    x_b = points[..., 1:][..., mask]

    x_c = .5 * (x_a + x_b)
    y_c = np.atleast_2d(func(x_c))

    x_2 = np.r_[points, x_c]
    y_2 = np.r_['-1', values, y_c]
    j = np.argsort(x_2)

    x_2 = x_2[..., j]
    y_2 = y_2[..., j]

    # -- Determine the intervals at which refinement is necessary

    if len(x_2) < min_points:
        mask = np.ones([len(x_2) - 1], dtype=bool)
    else:
        # represent the data as a path in N dimensions (scaled to unit box)
        if sample_transform is not None:
            y_2_val = sample_transform(x_2, y_2)
        else:
            y_2_val = y_2

        p = np.r_['0',
                  x_2[None, :],
                  y_2_val.real.reshape(-1, y_2_val.shape[-1]),
                  y_2_val.imag.reshape(-1, y_2_val.shape[-1])
                  ]

        sz = (p.shape[0] - 1) // 2

        xscale = x_2.ptp(axis=-1)
        yscale = np.abs(y_2_val.ptp(axis=-1)).ravel()

        p[0] /= xscale

        p[1:sz + 1] /= yscale[:, None]
        p[sz + 1:] /= yscale[:, None]

        # compute the length of each line segment in the path
        dp = np.diff(p, axis=-1)
        s = np.sqrt((dp**2).sum(axis=0))
        s_tot = s.sum()

        # compute the angle between consecutive line segments
        dp /= s
        dcos = np.arccos(np.clip((dp[:, 1:] * dp[:, :-1]).sum(axis=0), -1, 1))

        # determine where to subdivide: the condition is roughly that
        # the total length of the path (in the scaled data) is computed
        # to accuracy `tol`
        dp_piece = dcos * .5 * (s[1:] + s[:-1])
        mask = (dp_piece > tol * s_tot)

        mask = np.r_[mask, False]
        mask[1:] |= mask[:-1].copy()

    # -- Refine, if necessary

    if mask.any():
        return _sample_function(func, x_2, y_2, mask, tol=tol, depth=depth + 1,
                                min_points=min_points, max_level=max_level,
                                sample_transform=sample_transform)
    else:
        return x_2, y_2
