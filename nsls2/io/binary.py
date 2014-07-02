import numpy as np


def read_binary(filename, nx, ny, nz, dsize, headersize, **kwargs):
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
    extras: dict
            unnecessary keys that were passed to this function through
            dictionary unpacking

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

    # open the file
    opened_file = open(filename, "rb")

    # read the file header
    header = opened_file.read(headersize)

    # read the entire file in as 1D list
    data = np.fromfile(opened_file, dsize, -1)

    # reshape the array to 3D
    if nz is not 1:
        data.resize(nx, ny, nz)
    # unless the 3rd dimension is 1, in which case reshape the array to 2D
    elif ny is not 1:
        data.resize(nx, ny)
    # unless the 2nd dimension is also 1, in which case leave the array as 1D

    # return the array and the header
    return data, header
