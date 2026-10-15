import numpy as np
import math
import copy

# 1D analysis ---------------------------------------------

def basil_like_general_v5(n_on, mu_sig, energy, dict_logprod):
    '''
       Compute the log of the marginal likelihood (posterior of mu_sig)
       Not yet fully implemented for multiple event variables.
    '''

    res = np.zeros(n_on.shape)

    # Loop in energy bins
    for i in range(len(n_on)):
        logprod = dict_logprod[str(energy[i])]
        # Loop in n_s (0 to n_on)
        mu_s = mu_sig[i][0][0]
        if mu_s > 0:  # avoid 0**0 case
            res[i][0][0] = log_sum_P_sN_mod(logprod + np.linspace(0,len(logprod)-1, len(logprod))*math.log(mu_s)) - mu_s
        elif mu_s == 0:
            res[i][0][0] = logprod[0]
        else:
            res[i][0][0] = -np.inf
    return -2 * res


def log_sum_P_sN_mod(xs):
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))

# 3D analysis ---------------------------------------------

def slice_mask_flat_comb_new(new_comb, mask, slices):
    '''
       Mask and slice the comb list for BASiL.
    '''
    idx = slices['energy'].start
    new_comb_copy = copy.deepcopy(new_comb[idx])
    mask_sliced = mask.slice_by_idx(slices=slices)
    comb_sliced_list = []
    for i in range(len(mask_sliced.data[0])):
        for j in range(len(mask_sliced.data[0][0])):
            if mask_sliced.data[0][i][j] == True:
                if np.isscalar(new_comb_copy[i][j]):
                    comb_sliced_list.append(float(new_comb_copy[i][j]))
                else:
                    for k in range(len(new_comb_copy[i][j])):
                        comb_sliced_list.append(float(new_comb_copy[i][j][k]))

    return np.asanyarray(comb_sliced_list)
