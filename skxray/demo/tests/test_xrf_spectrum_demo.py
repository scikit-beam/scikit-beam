from __future__ import (unicode_literals, print_function, absolute_import,
                        division)
import six
import time as ttime


def test_xrf_spectrum_demo():
    # smoketest the demo
    from ..demo_xrf_spectrum import run_demo
    import matplotlib.pyplot as plt
    plt.ion()
    run_demo()
