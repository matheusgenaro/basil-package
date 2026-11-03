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

from gammapy.datasets import SpectrumDatasetOnOff
from gammapy.utils.scripts import make_name
from .utils import basil_posterior
import numpy as np


__all__ = ["SpectrumDatasetOnOffBASiL"]


class SpectrumDatasetOnOffBASiL(SpectrumDatasetOnOff):
    """
    New dataset type that includes the BASiL approach for
    computing the posterior of the signal in On/Off analysis, through
    the modified `stat_array()`
    
    	Class structure and slice_by_idx() method extracted from Gammapy.
    	
    """
    def __init__(
            self,
            models=None,
            counts=None,
            counts_off=None,
            acceptance=None,
            acceptance_off=None,
            exposure=None,
            mask_fit=None,
            psf=None,
            edisp=None,
            name=None,
            mask_safe=None,
            gti=None,
            meta_table=None,
            dict_logprod=None
    ):
        self._name = make_name(name)
        self._evaluators = {}

        self.counts = counts
        self.counts_off = counts_off
        self.exposure = exposure
        self.acceptance = acceptance
        self.acceptance_off = acceptance_off
        self.gti = gti
        self.mask_fit = mask_fit
        self.psf = psf
        self.edisp = edisp
        self.models = models
        self.mask_safe = mask_safe
        self.meta_table = meta_table
        self.dict_logprod = dict_logprod
        
    def stat_array(self):
        """
        Likelihood per bin in BASiL approach given the current model parameters
        """
        mu_sig = self.npred_signal().data
        on_stat_ = basil_posterior(
            n_on=self.counts.data,
            mu_sig=mu_sig,
            energy=self.counts.geom.axes["energy"].center.value,
            dict_logprod=self.dict_logprod,
        )
        return np.nan_to_num(on_stat_)

    def slice_by_idx(self, slices, name=None):
        """Slice sub dataset. 
        	Same code and documentation as Gammapy. 
        	Addition of "dict_logprod" variable.
        	Check MapDataset in original Gammapy code for details.
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name, "meta_table": self.meta_table}

        if self.counts is not None:
            kwargs["counts"] = self.counts.slice_by_idx(slices=slices)

        if self.counts_off is not None:
            kwargs["counts_off"] = self.counts_off.slice_by_idx(slices=slices)

        if self.acceptance is not None:
            kwargs["acceptance"] = self.acceptance.slice_by_idx(slices=slices)

        if self.acceptance_off is not None:
            kwargs["acceptance_off"] = self.acceptance.slice_by_idx(slices=slices)

        if self.exposure is not None:
            kwargs["exposure"] = self.exposure.slice_by_idx(slices=slices)

        if self.background is not None and self.stat_type == "cash":
            kwargs["background"] = self.background.slice_by_idx(slices=slices)

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.slice_by_idx(slices=slices)

        if self.psf is not None:
            kwargs["psf"] = self.psf.slice_by_idx(slices=slices)

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.slice_by_idx(slices=slices)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.slice_by_idx(slices=slices)
            
        if self.dict_logprod is not None:
            kwargs["dict_logprod"] = self.dict_logprod

        return self.__class__(**kwargs)

