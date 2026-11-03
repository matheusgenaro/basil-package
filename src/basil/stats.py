# Licensed under a 3-clause BSD style license - see LICENSE and LICENSE-Gammapy
#
# This file contains original code and code adapted from Gammapy.
#
# Portions of this file are derived from Gammapy:
# https://github.com/gammapy/gammapy
#
# The original Gammapy code is licensed under the BSD 3-Clause License.
# Copyright notices and license conditions applicable to that code
# are preserved in accordance with the original license.
# See LICENSE-Gammapy for licensing information.
#
# Modifications and additional code:
# Copyright (c) 2026, Matheus Genaro Dantas Xavier
# See LICENSE for licensing information.

from gammapy.stats.fit_statistics_cython import TRUNCATION_VALUE
import numpy as np
import math

from .utils import log_sum_P_sN_mod

__all__ = ["BASiL_3D_logsum"]


def BASiL_3D_logsum(n_on, mu_s, mu_b, logcomb, truncation_value=TRUNCATION_VALUE):
    r"""BASiL statistics for 3D case.
    """
    n_on = np.asanyarray(n_on)
    mu_s = np.asanyarray(mu_s)
    mu_b = np.asanyarray(mu_b)

    mu_s = np.where(mu_s <= truncation_value, truncation_value, mu_s)
    mu_b = np.where(mu_b <= truncation_value, truncation_value, mu_b)

    mu_on = mu_s + mu_b

    nbins_lon = n_on.shape[-1]
    nbins_lat = n_on.shape[-2]
    nbins_en = n_on.shape[-3]

    truncation_value = np.asanyarray(truncation_value)
    if np.any(truncation_value) <= 0:
        raise ValueError("Cash statistic truncation value must be positive.")

    mu_on = np.where(mu_on <= truncation_value, truncation_value, mu_on)

    stat = np.zeros_like(mu_on, dtype=float)
    
    with np.errstate(divide="ignore"):
        log_mu_s = np.log(mu_s)
        log_mu_b = np.log(mu_b)

    mask_nonzero = n_on > 0

    mask_s0_bgt0 = mask_nonzero & (mu_s == 0) & (mu_b > 0)
    mask_sgt0_b0 = mask_nonzero & (mu_s > 0) & (mu_b == 0)
    mask_both0   = mask_nonzero & (mu_s == 0) & (mu_b == 0)
    mask_general = mask_nonzero & (mu_s > 0) & (mu_b > 0)


    # stat computation
    
    stat[mask_s0_bgt0] = 2*(mu_on[mask_s0_bgt0]	- (np.array([logcomb[i][j][k][0] for i, j, k in zip(*np.where(mask_s0_bgt0))]) + n_on[mask_s0_bgt0]*log_mu_b[mask_s0_bgt0]))

    stat[mask_sgt0_b0] = 2*(mu_on[mask_sgt0_b0] - (np.array([logcomb[i][j][k][-1] for i, j, k in zip(*np.where(mask_sgt0_b0))]) + n_on[mask_sgt0_b0]*log_mu_s[mask_sgt0_b0]))
    
    stat[mask_both0] = np.inf

    idxs = np.argwhere(mask_general)

    for i, j, k in idxs:
        non = int(n_on[i, j, k])

        ns = np.arange(non + 1)

        xn = logcomb[i][j][k] + ns*log_mu_s[i, j, k] + (non - ns)*log_mu_b[i, j, k]

        log_term = log_sum_P_sN_mod(xn)

        stat[i, j, k] = 2 * (mu_on[i, j, k] - log_term)

    # n_on == 0 case
    stat[~mask_nonzero] = 2 * mu_on[~mask_nonzero]

    return stat
    
