# Licensed under a 3-clause BSD style license - see LICENSE and LICENSE-Gammapy
#
# Copyright (c) 2026, Matheus Genaro Dantas Xavier
# See LICENSE for licensing information.

import numpy as np
import math
import copy
import pickle
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid, quad
from scipy.optimize import brentq, minimize
from matplotlib import pyplot as plt
from astropy import units as u
from gammapy.maps import WcsNDMap, Map
from collections import defaultdict
from astropy.coordinates import Angle
from astropy.convolution import Tophat2DKernel

# 1D analysis ---------------------------------------------

def basil_posterior(n_on, mu_sig, energy, dict_logprod):
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
            res[i][0][0] = log_sum(logprod + np.linspace(0,len(logprod)-1, len(logprod))*math.log(mu_s)) - mu_s
        elif mu_s == 0:
            res[i][0][0] = logprod[0]
        else:
            res[i][0][0] = -np.inf
    return -2 * res


def log_sum(xs):
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

    def factor_log(self, n_on_, n_off_, alpha_):
        n_on = int(n_on_)
        n_off = int(n_off_)
        
        if n_on == 0:
            log_factor_bin = [0.0]
        else:
            C = 1.0 + 1.0 / float(alpha_)
            logC = math.log(C)

            N = n_on + n_off

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
           like_sig_list: list of signal likelihoods per energy bin.
           
           like_bkg_list: list of background likelihoods per energy bin.
           
           file_path: where combinatory factors will be stored
          
        """
        n_on = self.n_on
        n_off = self.n_off
        alpha = self.alpha
        energy = self.energy
        
        for i in range(len(n_on)):
            logfactor = self.factor_log(n_on[i][0][0], n_off[i][0][0], alpha[i][0][0])
            
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
        
def significance_basil(n_on, n_off, alpha, like_sig, like_bkg, sig_max, sigma=True):
    '''
    
        sig_max (array): prior of maximum expected signal
        sigma (bool): if False, returns the log of the probability of the null hypothesis,
                    if True, do Gaussian conversion to sigma.
    '''
    basil_f = basil_factors()
    log_comb = basil_f.combinatory_log(like_bkg, like_sig)
    log_factor = basil_f.factor_log(n_on, n_off, alpha)
    log_prod = np.array(log_comb)+np.array(log_factor)
    
    xs = log_prod + np.stack([math.lgamma(n+1) for n in range(len(log_prod))])
    logBF_pre = math.lgamma(n_off+1) - math.lgamma(n_on+n_off+1) - log_comb[0] + log_sum(xs)
 
    logs_max = np.log(sig_max)
    logBF = logBF_pre - logs_max
    
    log_pH0, sigma_value = pH0_from_logBF(logBF)
    
    if sigma == False:
        return -log_pH0
    # sign of logprob moved to exponential below
    logprob = max(0, logBF) + math.log1p(math.exp(-abs(logBF)))
    
    if sigma == False:
        return -logprob
    else:
        return sigma

def pH0_from_logBF(logBF):
    log_pH0 = max(0, logBF) + math.log1p(math.exp(-abs(logBF))) # opposite sign

    # Significance in sigma
    if log_pH0 < 700:  # safe range
        sigma = norm.isf(math.exp(-log_pH0))
    else:
        t = np.sqrt(2 * log_pH0)
        sigma = t - (np.log(t) + 0.5*np.log(2*np.pi)) / t
    return log_pH0, sigma


def find_peak(x,interp):
    return interp(x)

def hpd_interval(x, y, alpha, eps=1e-6):
    '''
         Function to find the limits of the high posterior density (HPD) interval,
         with the condition that p(x_low) = p(x_high)
         
         --------
         x : array of x values of the posterior.
         y : array of values of the posterior for given x.
         alpha : probability interval to compute (between 0 and 1).
         eps : absolute tolerance for brentq method (root finding).
         
         ---------
         Returns:
            x_low : lower bound of HPD interval.
            x_high : upper bound of HPD interval.
            y_low : corrresponding posterior value  for lower bound, y_low = p(x_low).
            y_high : corrresponding posterior value  for upper bound, y_high = p(x_high).
    '''
    interp = interp1d(x, y, kind='cubic', bounds_error=False, fill_value=0)

    def mass_above(threshold):
        mask = y >= threshold
        if not np.any(mask):
            return 0
        return trapezoid(y[mask], x[mask])

    total_area = trapezoid(y, x)

    def objective(t):
        return mass_above(t) / total_area - alpha

    # Step 1: find HPD threshold
    t_star = brentq(objective, 0, max(y), xtol=eps)

    # Step 2: find where y(x) = t_star 
    def f(x_val):
        return float(interp(x_val) - t_star)

    # Find brackets from the grid
    mask = y >= t_star
    indices = np.where(mask)[0]

    i_low = indices[0]
    i_high = indices[-1]

    x_low = brentq(f, x[i_low - 1], x[i_low], xtol=eps) if i_low > 0 else x[i_low]
    x_high = brentq(f, x[i_high], x[i_high + 1], xtol=eps) if i_high < len(x) - 1 else x[i_high]

    y_low = float(interp(x_low))
    y_high = float(interp(x_high))

    return x_low, x_high, y_low, y_high

def plot_posterior(dataset, energy_bin, sig_max, credible_interval=0.6827):
    '''
        Plot the posterior probability distribution for a given energy.
        
        dataset : SpectrumDatasetOnOffBASiL.
        energy_bin : Index of corresponding energy.
        sig_max : Maximum signal to compute the posterior.
        credible_interval : Probability value to compute
                        the High Density Posterior (HPD) interval.
        ---------        
        Returns:
        
        result : Dictionary containing `fig`,`ax` of plot;
                `x`,`y` values of the posterior distribution;
                `mode` of the posterior distribution;
                HPD interval `hpd_interval` and the corresponding
                y values in the posterior (`hpd_interval_y`)
    '''
    sig_array = 10**np.linspace(np.log10(0.01),np.log10(sig_max),int(sig_max*10))
    mu_sig = dataset.npred_signal().data
    energy_axis = dataset.geoms['geom'].axes['energy']
    pos_basil = np.array([])
    for j in sig_array:    
        for i in range(len(mu_sig)):
            mu_sig[i][0][0] = j
            
        posterior = basil_posterior(dataset.counts.data, mu_sig, energy_axis.center.value, dataset.dict_logprod)
        pos_basil = np.append(pos_basil, posterior[energy_bin][0][0])


    fig, ax = plt.subplots()
    x_inf, x_sup, y_inf, y_sup = hpd_interval(sig_array, np.exp((-pos_basil+min(pos_basil))/2), credible_interval)
    y_plot = np.exp((-pos_basil+min(pos_basil))/2)
    ax.plot(sig_array, y_plot)
    ax.fill_between(x=sig_array, y1= y_plot,where= (x_inf < sig_array)&(sig_array < x_sup),color= "b", alpha= 0.5)
    ax.hlines(y_inf, min(sig_array), max(sig_array), ls='dashed', color='black', lw=1)
    ax.hlines(y_sup, min(sig_array), max(sig_array), ls='dotted', color='black', lw=1)
    ax.set_xlabel('Signal '+r'$\mu_{\rm s}$', fontsize=15)
    ax.set_ylabel('Posterior distribution', fontsize=15)
    interp = interp1d(sig_array, pos_basil, kind='cubic', bounds_error=False, fill_value=0)
    res_min = minimize(find_peak, 3, args=(interp), bounds=[(0, None)])
    ax.vlines(res_min.x[0], 0, 1, color='black')

    result = {'fig': fig,
              'ax': ax,
              'x': sig_array,
              'y': y_plot,
              'mode': res_min.x[0],
              'hpd_interval':[x_inf,x_sup],
              'hpd_interval_y':[y_inf,y_sup]}

    return result



# 3D analysis ---------------------------------------------

def create_events_en(events, en_bin_l, en_bin_u):    
    return events.table[(events.table['ENERGY'] >= en_bin_l) & (events.table['ENERGY'] < en_bin_u)]

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
    
def compute_comb_3D(dataset, list_events, en_thres_run, list_excl_mask):
    '''
        Computation of the combinatorial factor for 3D BASiL.
        Likelihoods are organized in a list per pixel structure.
        
        -------
        Returns:
            comb (list) : Combinatorial factor per pixel.
            like_sig_data (list) : Signal likelihoods per pixel.
            like_bkg_data (list) : Background likelihoods per pixel.
    '''
    geom = dataset.geoms['geom']
    en_bins = dataset.geoms['geom'].axes['energy'].edges.value
    basil_f = basil_factors()
    
    # Compute the combinatory factor for each spatial and energy bin, given the likelihood values
    comb = []
    like_sig_data = []
    like_bkg_data = []
    for k in range(len(en_bins)-1):
        # For each energy bin, identify the location of the events
        events_en_list = [
            create_events_en(list_events[l], en_bins[k], en_bins[k+1])
            for l in range(len(list_events))
        ]
        
        spatial_dict_sig = []
        spatial_dict_bkg = []
        for l in range(len(list_events)):
            spatial_dict_sig_ = defaultdict(list)
            spatial_dict_bkg_ = defaultdict(list)
            for event in events_en_list[l]:
                key = (event['INDX_DEC'], event['INDX_RA'])
                spatial_dict_sig_[key].append(event['LIKE_SIG'])
                spatial_dict_bkg_[key].append(event['LIKE_BKG'])
        
            spatial_dict_sig.append(spatial_dict_sig_)
            spatial_dict_bkg.append(spatial_dict_bkg_)
        
        # For each spatial bin in given energy, compute combinatory factor from likelihoods
        comb_en = []
        like_sig_en = []
        like_bkg_en = []
        # Loop over spatial bins
        for j in range(geom.data_shape[-2]): # lat
            for i in range(geom.data_shape[-1]): # lon
                like_sig = []
                like_bkg = []
                # Loop over runs
                for l in range(len(list_events)):
                    # Check the energy threshold of the run
                    if k >= en_thres_run[l]:
                        # Check if event is inside spatial mask
                        if list_excl_mask[l].data[k][j][i] == True:                      
                            # Get zeta-BDT values for spatial bin
                            likesig_values = spatial_dict_sig[l].get((j, i), [])
                            likebkg_values = spatial_dict_bkg[l].get((j, i), [])
                            like_sig.extend(likesig_values)
                            like_bkg.extend(likebkg_values)
                            
                if len(like_sig) == 0:
                    comb_ = [0.0]
                    like_sig = [0.0]
                    like_bkg = [0.0]
                else:
                    comb_ = basil_f.combinatory_log(np.asarray(like_bkg), np.asarray(like_sig))
                comb_en.append(comb_)
                like_sig_en.append(like_sig)
                like_bkg_en.append(like_bkg)
        comb.append(comb_en)
        like_sig_data.append(like_sig_en)
        like_bkg_data.append(like_bkg_en)

    return comb, like_sig_data, like_bkg_data
    
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

def reshape_likelihoods(like_sig_data, like_bkg_data, dataset_basil):
    geom = dataset_basil.geoms['geom']
    
    # Reshaping the combinatory list -- for stat_array
    new_like_sig_data = []
    new_like_bkg_data = []
    # Loop over energies
    for i in range(len(like_sig_data)):
        likesig_en_matrix = []
        likebkg_en_matrix = []
        # Loop over spatial bins
        k = 0
        while k < geom.data_shape[-1]*geom.data_shape[-2]:
            row_likesig = []
            row_likebkg = []
            for j in range(geom.data_shape[-2]):
                if len(like_sig_data[i][k+j]) == 1:
                    row_likesig.append([like_sig_data[i][k+j][0]])
                    row_likebkg.append([like_bkg_data[i][k+j][0]])
                else:
                    row_likesig.append(like_sig_data[i][k+j])
                    row_likebkg.append(like_bkg_data[i][k+j])
            k+=geom.data_shape[-2]
            likesig_en_matrix.append(row_likesig)
            likebkg_en_matrix.append(row_likebkg)
        new_like_sig_data.append(likesig_en_matrix)
        new_like_bkg_data.append(likebkg_en_matrix)

    return new_like_sig_data, new_like_bkg_data
    

def logBF_3D_pixel(logcomb, mu_b, s_max):
    '''
        Compute log of Bayes factor for a pixel in 3D analysis.
        
    '''
    if mu_b <= 0:
        res = -np.log(s_max)
    else:
        xs = logcomb + np.stack([math.lgamma(n+1) for n in range(len(logcomb))]) - np.stack([n for n in range(len(logcomb))])*np.log(mu_b)
        res = log_sum(xs) - logcomb[0] - np.log(s_max)

    return res

def create_significance_map(energy_bin, sig_max, dataset_basil, new_like_sig_data, new_like_bkg_data, correlation_radius=Angle("0.2 deg")):
    '''
        Create significance map for BASiL 3D in a given energy range.
        
        energy_bin : index of energy bin of energy axis.
        sig_max : prior on the signal counts.
        dataset_basil : MapDatasetBASiL
        new_like_sig_data : Signal likelihood per pixel.
        new_like_bkg_data : Background likelihood per pixel.
        correlation_radius : Size of top hat kernel for correlating events.
        --------
        Returns:
            significance_map : WcsNDMap 
    
    '''

    # Defining the kernel from correlation radius
    pixel_size = dataset_basil.counts.geom.pixel_scales[0].value
    size = correlation_radius.deg/pixel_size
    geom = dataset_basil.counts.geom.to_image()
    geom = geom.to_odd_npix(max_radius=correlation_radius)
    
    kernel = Tophat2DKernel(size)
    maps = Map.from_geom(geom, data=kernel.array)
    kernel_data = maps.data / maps.data.max()
    Nkernel = maps.data.shape[0]
    
    n_on = dataset_basil.counts * dataset_basil.mask_safe
    mu_bkg = dataset_basil.npred_background() * dataset_basil.mask_safe
    mu_bkg_conv = mu_bkg.convolve(kernel_data).data

    basil_f = basil_factors()
    
    half = Nkernel//2
    counts_map = n_on.data[energy_bin]
    likesig_map = new_like_sig_data[energy_bin]
    likebkg_map = new_like_bkg_data[energy_bin]
    mask_safe_data = dataset_basil.mask_safe.data[energy_bin]
    counts_map_conv = np.zeros(dataset_basil.counts.data[energy_bin].shape)
    significance_matrix = np.zeros(dataset_basil.counts.data[energy_bin].shape)
    for i in range(15, counts_map_conv.shape[0]-15):
        for j in range(15, counts_map_conv.shape[1]-15):
            likesig_pixel = []
            likebkg_pixel = []
    
            for plus_x in range(-half, half+1):
                for plus_y in range(-half, half+1):
                    kernel_value = kernel_data[half+plus_x, half+plus_y]
                    if kernel_value > 0:
                        counts_pixel = counts_map[i+plus_x,j+plus_y]
                        counts_map_conv[i,j] += counts_pixel#*kernel_data[half+plus_x, half+plus_y]
                        # collect likelihoods
                        if counts_pixel > 0:
                            likesig_pixel_arr = likesig_map[i+plus_x][j+plus_y]
                            likebkg_pixel_arr = likebkg_map[i+plus_x][j+plus_y]
                            for k in range(len(likesig_pixel_arr)):
                                likesig_pixel.append(likesig_pixel_arr[k])
                                likebkg_pixel.append(likebkg_pixel_arr[k])
            if len(likesig_pixel) > 0:
                # Compute logcomb
                comb_pixel = basil_f.combinatory_log(likebkg_pixel, likesig_pixel)
                # Compute significance and store in matrix
                logBF = logBF_3D_pixel(comb_pixel, mu_bkg_conv[energy_bin][i,j], sig_max)
                log_pH0, sigma = pH0_from_logBF(logBF)
                significance_matrix[i,j] = sigma

    significance_map = WcsNDMap(geom=dataset_basil.counts.sum_over_axes().geom,data=significance_matrix)

    return significance_map

def integrand_BF_term_H1(norm, sum_s, logcomb_flat, Ns_array, N_pixels,
                         indx_pixels, s_base, mu_b, log_value=False, exp_value=0):
    '''
        Integrand for Bayes factor (BF) to test significance of flux point,
        in the context of BASiL.
        
        norm: amplitude scale (if norm = 0, compute under H_0).
        sum_s: sum of base signal over the pixels.
        logcomb_flat: log of combinatorial factor in a 1D array.
        Ns_array: possible values of N_s organized in a 1D array.
        N_pixels: Number of pixels.
        indx_pixels: index location of each pixel in Ns_array.
        s_base: base npred_signal() computed.
        mu_b: npred_background()
        log_value: if True, compute the logarithm of the integrand.
        exp_value: normalization to numerically stabilize the integration.
    
    '''
    if norm < 0:
        raise ValueError("norm must be non-negative.")
    
    term = np.zeros(N_pixels)
    
    if norm == 0:
        xs = logcomb_flat
        for i, (start, end) in enumerate(zip(indx_pixels[:-1], indx_pixels[1:])):
            term[i] = xs[start]
        
        # Handling the last pixel
        i += 1
        term[i] = xs[end]
    else:
        xs = logcomb_flat + Ns_array*np.log(norm)

        # Computing the term per pixel
        for i, (start, end) in enumerate(zip(indx_pixels[:-1], indx_pixels[1:])):
            xs_ = xs[start:end]
            if (s_base[i] > 0) & (mu_b[i] > 0):
                xs_final = xs_ + Ns_array[start:end]*(np.log(s_base[i]) - np.log(mu_b[i]))
                term[i] = log_sum(xs_final)
            elif (s_base[i] == 0) & (mu_b[i] > 0):
                term[i] = xs_[0]
            else:
                raise ValueError("Negative value in s_base={:} or mu_b={:}, for pixel {:}".format(s_base[i],mu_b[i],i))

        # Handling the last pixel
        i += 1
        xs_ = xs[end]
        if (s_base[i] > 0) & (mu_b[i] > 0):
            xs_final = xs_ + Ns_array[end]*(np.log(s_base[i]) - np.log(mu_b[i]))
            term[i] = log_sum(xs_final)
        elif (s_base[i] == 0) & (mu_b[i] > 0):
            term[i] = xs_
        else:
            raise ValueError("Negative value in s_base={:} or mu_b={:}, for pixel {:}".format(s_base[i],mu_b[i],i))

    log_integrand = -norm*sum_s + np.sum(term)
            
    if log_value:
        res = log_integrand
    else:
        res = np.exp(log_integrand - exp_value)
        
    return res


def significance_3D_fluxpoint(dataset_basil, en_bin, amp_base, amp_max):
    '''
     Significance computation for 3D BASiL based on the Bayes factor (BF)
             
        BF = p(N,X|H_1)/p(N,X|H_0)
        
    The test is defined for the amplitude A at a given energy bin.
    Alternative hypothesis H_1: A > 0.
    Null hypothesis H_0: A = 0.
    
     dataset_basil: MapDatasetBASiL.
     en_bin: index of energy axis bin to estimate the significance.
     amp_base: base amplitude to compute predicted signal.
     amp_max: prior on the amplitude norm (scaling factor).
     
    '''
    
    # Counts, background, predicted signal
    flat_masked_bkg = dataset_basil.npred_background().data[en_bin].flatten()[np.argwhere(dataset_basil.mask_safe.data[en_bin].flatten()==True)].flatten()
    flat_masked_counts = dataset_basil.counts.data[en_bin].flatten()[np.argwhere(dataset_basil.mask_safe.data[en_bin].flatten()==True)].flatten().astype(int)

    dataset_basil_copy = dataset_basil.copy()
    dataset_basil_copy.models = dataset_basil.models.copy()
    dataset_basil_copy.models.parameters['amplitude'].value = amp_base
    flat_masked_s = dataset_basil_copy.npred_signal().data[en_bin].flatten()[np.argwhere(dataset_basil.mask_safe.data[en_bin].flatten()==True)].flatten()
    flat_masked_s[flat_masked_s < 0] = 0

    # Get flat comb list
    indx_safemask_bin = np.argwhere(dataset_basil.mask_safe.data[en_bin] == True)
    comb_bin = dataset_basil.comb[en_bin]
    flat_comb_bin = []
    for i in range(len(indx_safemask_bin)):
        elem = comb_bin[indx_safemask_bin[i][0]][indx_safemask_bin[i][1]]
        for j in range(len(elem)):
            flat_comb_bin.append(float(elem[j]))

    # Relevant indices for the array manipulation
    Ns_array = np.zeros(len(flat_comb_bin))
    indx = 0
    indx_pixels = np.zeros(len(flat_masked_counts), dtype=int)
    i = 0
    for count in flat_masked_counts:
        indx_pixels[i] = indx
        if count > 0:
            Ns_array[indx:indx+count+1] += np.stack([n for n in range(count+1)]) 
        indx += count + 1
        i += 1

    # BF term H0 (set norm = 0)
    logP0 = integrand_BF_term_H1(0, np.sum(flat_masked_s), flat_comb_bin, Ns_array, len(flat_masked_counts),
                                 indx_pixels, flat_masked_s, flat_masked_bkg, True)
    
    # Compute integral
    int_res = quad(integrand_BF_term_H1, 0, amp_max,
                   args=(np.sum(flat_masked_s), flat_comb_bin, Ns_array, len(flat_masked_counts),indx_pixels, flat_masked_s, flat_masked_bkg, False, logP0))
    
    # To log space
    logBF = np.log(int_res[0]/amp_max)    
    log_pH0, sigma = pH0_from_logBF(logBF)

    return log_pH0, sigma
    
