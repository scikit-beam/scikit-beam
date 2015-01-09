from functools import wraps
from copy import copy
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors
import numpy as np


def ensure_ax(func):
    @wraps(func)
    def inner(*args, **kwargs):
        if 'ax' in kwargs:
            ax = kwargs.pop('ax', None)
        elif len(args) > 0 and isinstance(args[0], matplotlib.axes.Axes):
            ax = args[0]
            args = args[1:]
        else:
            ax = plt.gca()
        ret = func(ax, *args, **kwargs)
        ax.figure.canvas.draw()  # possible perf hit
        return ret
    return inner


@ensure_ax
def draw_rings(ax, radii, center, circle_style=None, cross_hairs_style=None):
    """Draw circular rings and cross-hairs around a specified center point.

    Parameters
    ----------
    radii : list
        a list of radii in pixels
    center : float
    circle_style : dict, optional
        Overrides default style passed to axvline and axhline
    cross_hairs_style : dict, optional
        Overrides default style passed to Circle
    ax : matplotlib.Axes, optional

    Returns
    -------
    list, containing a patch for each circle and two lines
    """
    if circle_style is None:
        circle_style = dict()
    if cross_hairs_style is None:
        cross_hairs_style = dict()
    style = dict(facecolor='none', edgecolor='r', lw=2, linestyle='dashed')
    style.update(circle_style)
    patches = []
    for r in radii:
        c = matplotlib.patches.Circle(center[::-1], r, **style)
        patches.append(c)
        ax.add_patch(c)
    style = dict(color='r')
    style.update(cross_hairs_style)
    patches.append(ax.axhline(center[0], **style))
    patches.append(ax.axvline(center[1], **style))

    if len(radii) > 0:
        ax.set_ylim([center[0] - radii[-1], center[0] + radii[-1]])
        ax.set_xlim([center[1] - radii[-1], center[1] + radii[-1]])
    return patches


@ensure_ax
def lognorm_imshow(ax, image, image_style=None):
    """
    Plot an image with log-scaled intensity and smart scaling.

    This is just a wrapper of matplotlib.pyplot.imshow with some
    intricate defaults.

    Parameters
    ----------
    image : array
    image_style : dict, optional
        Overrides style arguments passed to imshow
    ax: matplotlib.Axes, optional

    Returns
    -------
    AxesImage
    """
    if image_style is None:
        image_style = dict()
    vmin, vmax = np.percentile(image, [1, 99])
    my_cmap = copy(matplotlib.cm.get_cmap('gray'))
    my_cmap.set_bad('k')
    style = dict(cmap=my_cmap, interpolation='none',
                 norm=matplotlib.colors.LogNorm(),
                 vmin=vmin, vmax=vmax)
    style.update(image_style)
    im = ax.imshow(image, **style)
    return im
