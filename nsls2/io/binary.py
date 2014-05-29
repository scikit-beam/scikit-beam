import numpy as np


def read_binary(filename, nx, ny, nz, dsize, headersize):
    """
    docstring, woo!

    Parameters
    ----------
    filename: String
              The name of the file to open
    nx: integer
        The number of data elements in the x-direction
    ny: integer
        The number of data elements in the y-direction
    nz: integer
        The number of data elements in the z-direction
    dsize: numpy data type
           The size of each element in the numpy array
    headersize: integer
                The size of the file header in bytes

    Returns
    -------
    (data, header)
    data: ndarray
            data.shape = (x, y, z) if z > 1
            data.shape = (x, y) if z == 1
            data.shape = (x,) if y == 1 && z == 1
    header: String
            header = file.read(headersize)
    """

    opened_file = open(filename, "rb")

    # read the file header
    header = opened_file.read(headersize)

    data = np.fromfile(opened_file, dsize, -1)

    if nz is not 1:
        data.resize(nx, ny, nz)
    elif ny is not 1:
        data.resize(nx, ny)

    return data, header
