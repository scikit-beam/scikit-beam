import numpy as np


def parabola_gen(x, center, height, width):
    return width * (x - center) ** 2 + height


def gauss_gen(x, center, height, width):
    return height * np.exp(-(((x - center) / width) ** 2))
