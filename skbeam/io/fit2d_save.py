import warnings

from .fit2d import fit2d_save, read_fit2d_msk  # noqa: F401

warnings.warn("Fit2d IO functionality has been moved to fit2d", DeprecationWarning)
