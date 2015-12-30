import cython
cimport cython
import numpy as np
cimport numpy as np
import itertools

@cython.boundscheck(False)
@cython.wraparound(False)
def print_memory_addresses(np.ndarray[np.float_t, ndim=3] arr,
                           long ilen, long jlen, long klen):
    print(len(arr))
    cdef long ptr, i, j, k
    cdef np.float_t val

    cdef np.float_t [:,:,:] view2 = arr
    cdef np.float_t [:] view1 = arr[0][0]

    for i, j, k in itertools.product(range(ilen), range(jlen), range(klen)):
        ptr = i*jlen*klen + j*klen + k
        val1 = view1[ptr]
        val2 = view2[i, j, k]
        print('%s %s %s %s %s %s' % (ptr, i, j, k, val1, val2))
