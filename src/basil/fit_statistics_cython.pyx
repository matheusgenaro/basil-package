# Licensed under a 3-clause BSD style license - see LICENSE.rst
# cython: language_level=3
import numpy as np
from libc.math cimport exp, log
from gammapy.stats.fit_statistics_cython import TRUNCATION_VALUE
cimport numpy as np
cimport cython

cdef double stable_logsum_binomial(
    int N,
    double logs,
    double logb,
    double[:] logcomb,
    int offset
) nogil:

    cdef:
        int j
        double val
        double m
        double s

    #
    # First pass: find max
    #
    m = (
        logcomb[offset]
        + N * logb
    )

    for j in range(1, N + 1):

        val = (
            logcomb[offset + j]
            + j * logs
            + (N - j) * logb
        )

        if val > m:
            m = val

    #
    # Second pass: exp sum
    #
    s = 0.0

    for j in range(N + 1):

        val = (
            logcomb[offset + j]
            + j * logs
            + (N - j) * logb
        )

        s += exp(val - m)

    return m + log(s)


@cython.cdivision(True)
@cython.boundscheck(False)
def basil_logsum_cython(long[:] counts,
                    double[:] npred_s,
                    double[:] npred_b,
                    double[:] logcomb):
    """Summed BASiL 3D fit statistics.

    Parameters
    ----------
    counts : `~numpy.ndarray`
        Counts array.
    npred_s : `~numpy.ndarray`
        Predicted counts array.
    npred_b : `~numpy.ndarray`
        Predicted background counts array.
    comb : `~numpy.ndarray`
        Combinatory factor array.
    """
    cdef np.float_t sum = 0
    cdef np.float_t npr, s, b
    cdef unsigned int i, ni, k, j, N
    cdef np.float_t trunc = TRUNCATION_VALUE
    cdef np.float_t logtrunc = log(TRUNCATION_VALUE)

    ni = counts.shape[0]
    k = 0
    for i in range(ni):
        npr = npred_s[i] + npred_b[i]
        N = int(counts[i])
        if npr < trunc:
            npr = trunc
        if N == 0:
            sum += npr
        else:
            s = max(npred_s[i], trunc)
            b = max(npred_b[i], trunc)
            if (s == 0) and (b != 0):
                sum += npr - (logcomb[i+k] + N*log(b))
            elif (s != 0) and (b == 0):
                sum += npr - (logcomb[i+k+N] + N*log(s))
            else:
                sum += npr - stable_logsum_binomial(N, log(s), log(b), logcomb, i+k)
                
            k += N

    return 2 * sum
