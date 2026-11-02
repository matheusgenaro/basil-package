import numpy as np
import math
import copy
import pickle

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


class basil_factors:
    "Computation of essential 1D BASiL elements."
    
    def __init__(
        self,
        n_on=None,
        n_off=None,
        alpha=None,
        energy=None,
    ):
        self.n_on=n_on
        self.n_off=n_off
        self.alpha=alpha
        self.energy=energy

    def compute_log_factor(self, n_on_, n_off_, alpha_):
        n_on = int(n_on_)
        n_off = int(n_off_)
        
        if n_on == 0:
            log_factor_bin = [0.0]
        else:
            C = 1.0 + 1.0 / float(alpha_)
            logC = math.log(C)

            N = n_on + n_off

            # log(f_0) = log((N)! / (n_off)!)
            log_f = math.lgamma(N + 1) - math.lgamma(n_off + 1)

            size = n_on + 1
            log_factor_bin = [0.0] * size
            log_factor_bin[0] = log_f

            for j in range(1, size):
                log_f += logC - math.log(N - (j - 1))
                log_factor_bin[j] = log_f

        return log_factor_bin
        
    def logsumexp(self, x, y):
        m = max(x, y)
        return m + math.log1p(math.exp(-abs(x - y)))
        
    def combinatory_log(self, like_bkg, like_sig):
        log_bkg = [math.log(x) for x in like_bkg]
        log_sig = [math.log(x) for x in like_sig]

        n = len(like_bkg)
        log_comb = [-math.inf] * (n + 1)
        log_comb[0] = 0.0  # log(1)

        for i in range(n):
            for j in range(i + 1, -1, -1):
                x = log_bkg[i] + log_comb[j]
                if j == 0:
                    log_comb[j] = x
                else:
                    y = log_sig[i] + log_comb[j - 1]
                    log_comb[j] = self.logsumexp(x, y)

        return log_comb
    
    def write_pickle(self, content, filename):
        """
           Write content to pickle file.
        """
        with open(filename, 'wb') as h:
            pickle.dump(content, h, pickle.HIGHEST_PROTOCOL)
        return

    def run(self, like_sig_list, like_bkg_list, file_path):
        """
           Does all steps to compute BASiL factors and generate the corresponding pickle files.
           
           Parameters
           ----------
           like_sig_list:
           
           like_bkg_list:
           
           file_path: where combinatory factors will be stored
          
        """
        n_on = self.n_on
        n_off = self.n_off
        alpha = self.alpha
        energy = self.energy
        
        for i in range(len(n_on)):
            logfactor = self.compute_log_factor(n_on[i][0][0], n_off[i][0][0], alpha[i][0][0])
            
            if n_on[i][0][0] > 0:
                like_sig = like_sig_list[i]
                like_bkg = like_bkg_list[i]
            else:
                like_sig = []
                like_bkg = []
            logcomb = self.combinatory_log(like_bkg, like_sig) 
            logprod = np.array(logcomb) + np.array(logfactor)
                
            self.write_pickle(logfactor, file_path+'/factor_bin'+str(energy[i])+'.pkl')
            self.write_pickle(logcomb, file_path+'/comb_bin'+str(energy[i])+'.pkl')
            self.write_pickle(logprod, file_path+'/prod_bin'+str(energy[i])+'.pkl')
                     
        return
        
    def run_stacked(self, label_factors, like_sig, like_bkg):
        n_on = self.n_on
        n_off = self.n_off
        alpha = self.alpha
        energy = self.energy
    
        for i in range(len(n_on)):
            if n_on[i][0][0] > 0:
                logfactor = self.compute_log_factor(n_on[i][0][0], n_off[i][0][0], alpha[i][0][0])
    
                logcomb = self.combinatory_log(like_bkg[i], like_sig[i]) 
                logprod = np.array(logcomb) + np.array(logfactor)

            else:
                like_sig_ = np.ones(int(n_on[i][0][0]))
                like_bkg_ = np.ones(int(n_on[i][0][0]))
                logcomb = self.combinatory_log(like_bkg_, like_sig_)
                logprod = np.array(logcomb) #+ np.array(logfactor)
            
            if n_on[i][0][0] > 0:
                self.write_pickle(logfactor, '/home/matheus/Documents/BASiL/dist/factors/'+label_factors+'/factor_bin'+str(energy[i])+'.pkl')
            self.write_pickle(logcomb, '/home/matheus/Documents/BASiL/dist/factors/'+label_factors+'/comb_bin'+str(energy[i])+'.pkl')
            self.write_pickle(logprod, '/home/matheus/Documents/BASiL/dist/factors/'+label_factors+'/prod_bin'+str(energy[i])+'.pkl')
            
        return

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
    
def reshape_comb(comb, dataset_basil):
    geom = dataset_basil.geoms['geom']
    
    # Reshaping the combinatory list -- for stat_array
    new_comb = []
    # Loop over energies
    for i in range(len(comb)):
        en_matrix = []
        # Loop over spatial bins
        k = 0
        while k < geom.data_shape[-1]*geom.data_shape[-2]:
            row = []
            row_likesig = []
            row_likebkg = []
            for j in range(geom.data_shape[-2]):
                if len(comb[i][k+j]) == 1:
                    row.append([comb[i][k+j][0]])
                else:
                    row.append(comb[i][k+j])
            k+=geom.data_shape[-2]
            en_matrix.append(row)
        new_comb.append(en_matrix)
    
    
    new_comb_copy = copy.deepcopy(new_comb)
    indx_mask = np.argwhere(dataset_basil.mask.data == False)
    for i in range(len(indx_mask)):
        indx_ = indx_mask[i]
        new_comb_copy[indx_[0]][indx_[1]][indx_[2]] = -1
    
    
    new_comb_masked = []
    for i in range(len(new_comb_copy)):
        en_matrix = []
        for j in range(len(new_comb_copy[i])):
            row = []
            for k in range(len(new_comb_copy[i][j])):
                if new_comb_copy[i][j][k] != -1:
                    row.append(new_comb_copy[i][j][k])
            if len(row) > 0:
                en_matrix.append(row)
        if len(en_matrix) > 0:
            new_comb_masked.append(en_matrix)
    
    # Flattening the combinatory list for stat_sum()
    flat_comb_masked = []
    # Loop over energies
    for i in range(len(new_comb_masked)):
        for j in range(len(new_comb_masked[i])):
            for k in range(len(new_comb_masked[i][j])):
                if np.isscalar(new_comb_masked[i][j][k]):
                    flat_comb_masked.append(float(new_comb_masked[i][j][k]))
                else:
                    for l in range(len(new_comb_masked[i][j][k])):
                        flat_comb_masked.append(float(new_comb_masked[i][j][k][l]))

    return new_comb, flat_comb_masked

