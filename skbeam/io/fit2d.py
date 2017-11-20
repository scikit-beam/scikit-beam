from __future__ import print_function, division
import numpy as np
from skbeam.io.save_powder_output import _create_file_path
from fabio.fit2dmaskimage import Fit2dMaskImage
import fabio


def fit2d_save(mask, filename, dir_path=None):
    """
    Compresses and wraps the mask for Fit2D use
    Parameters
    ----------
    mask: ndarray
        The mask
    filename: str
        The filename
    dir_path: str, optional
        Path to the destination file
    """
    saver = Fit2dMaskImage(data=~np.flipud(mask))
    saver.write(_create_file_path(dir_path, filename, '.msk'))


def read_fit2d_msk(filename):
    """
    Reads mask from file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    ndarray:
        The mask as boolian array
    """
    a = np.flipud(fabio.open(filename))
    return ~a.data.astype(bool)
