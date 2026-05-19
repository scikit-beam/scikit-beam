from __future__ import division, print_function
from typing import Optional

import fabio
from fabio.fit2dmaskimage import Fit2dMaskImage
import numpy as np

from skbeam.io.save_powder_output import _create_file_path


def fit2d_save(mask: np.ndarray, filename: str, dir_path: Optional[str] = None) -> None:
    """
    Compresses and wraps the mask for Fit2D use
    Parameters
    ----------
    mask: np.ndarray
        The mask
    filename: str
        The filename
    dir_path: str, optional
        Path to the destination file
    """
    saver = Fit2dMaskImage(data=~mask)
    saver.write(_create_file_path(dir_path, filename, ".msk"))


def read_fit2d_msk(filename: str) -> np.ndarray:
    """
    Reads mask from file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    np.ndarray:
        The mask as boolean array
    """
    a = fabio.open(filename)
    return ~a.data.astype(bool)
