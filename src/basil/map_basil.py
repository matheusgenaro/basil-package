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

from gammapy.datasets import MapDataset 
from gammapy.irf import EDispKernelMap, EDispMap, PSFMap
from gammapy.utils.fits import HDULocation
from gammapy.utils.scripts import make_name

from .utils import slice_mask_flat_comb_new
from .stats import BASiL_3D_logsum
from .fit_statistics_cython import basil_logsum_cython
import numpy as np

__all__ = ["MapDatasetBASiL"]



class MapDatasetBASiL(MapDataset):
    """
    MapDataset class with BASiL statistics adapted for 3D analysis.
    """
    def __init__(
        self,
        models=None,
        counts=None,
        exposure=None,
        background=None,
        psf=None,
        edisp=None,
        mask_safe=None,
        mask_fit=None,
        gti=None,
        meta_table=None,
        name=None,
        comb_flat=None,
        comb=None,
    ):
        self._name = make_name(name)
        self._evaluators = {}

        self.counts = counts
        self.exposure = exposure
        self.background = background
        self._background_cached = None
        self._background_parameters_cached = None
        self.mask_fit = mask_fit
        self.comb_flat = comb_flat
        self.comb = comb

        if psf and not isinstance(psf, (PSFMap, HDULocation)):
            raise ValueError(
                f"'psf' must be a 'PSFMap' or `HDULocation` object, got {type(psf)}"
            )

        self.psf = psf

        if edisp and not isinstance(edisp, (EDispMap, EDispKernelMap, HDULocation)):
            raise ValueError(
                "'edisp' must be a 'EDispMap', `EDispKernelMap` or 'HDULocation' "
                f"object, got `{type(edisp)}` instead."
            )

        self.edisp = edisp
        self.mask_safe = mask_safe
        self.gti = gti
        self.models = models
        self.meta_table = meta_table


    def stat_array(self):
        """Statistic function value per bin given the current model parameters."""
        print('It is using BASiL')
        return BASiL_3D_logsum(n_on=self.counts.data, mu_s=self.npred_signal().data, mu_b=self.npred_background().data, logcomb=self.comb)

    def stat_sum(self):
        # Original method has some prior conditions that may be adapted
        prior_stat_sum = 0.0
        if self.models is not None:
            prior_stat_sum = self.models.parameters.prior_stat_sum()
            
  
        counts, npred_s, npred_b = self.counts.data.astype(float), self.npred_signal().data.astype(float), self.npred_background().data.astype(float)
        
        comb_f = self.comb_flat
        comb_f = np.ascontiguousarray(comb_f, dtype=np.float64)
        
        if self.mask is not None:
            mask = ~(self.mask.data == False)  # noqa
            counts = counts[mask]
            npred_s = npred_s[mask]
            # Replace negative values
            npred_s[npred_s < 0] = 0
            npred_b = npred_b[mask]            
            if self.mask.data.dtype == bool or self.stat_type == "basil":
                npred_s = np.ascontiguousarray(npred_s, dtype=np.float64)
                npred_b = np.ascontiguousarray(npred_b, dtype=np.float64)
                counts = np.ascontiguousarray(counts, dtype=np.long)
                basil_sum = basil_logsum_cython(counts, npred_s, npred_b, comb_f)
            elif self.stat_type == "cash_weighted":
                weight = self.mask.data[mask]
                cash_sum = weighted_cash_sum_cython(counts, npred, weight)
            else:
                raise ValueError(
                    f"'stat_type' must be a 'basil' or `cash_weighted`."
                    f", got `{self.stat_type}` instead."
                )
        else:
             basil_sum = basil_logsum_cython(counts.ravel(), npred_s.ravel(), npred_b.ravel(), comb_f)
             print('It is using BASiL')
        return basil_sum
          
    def slice_by_idx(self, slices, name=None):
        """Slice sub dataset.

        The slicing only applies to the maps that define the corresponding axes.
        Modification so that it is compatible to stat_type == 'basil'
        
        Parameters
        ----------
        slices : dict
            Dictionary of axes names and integers or `slice` object pairs. Contains one
            element for each non-spatial dimension. For integer indexing the
            corresponding axes is dropped from the map. Axes not specified in the
            dict are kept unchanged.
        name : str, optional
            Name of the sliced dataset. Default is None.

        Returns
        -------
        dataset : `MapDataset` or `SpectrumDataset`
            Sliced dataset.

        Examples
        --------
        >>> from gammapy.datasets import MapDataset
        >>> dataset = MapDataset.read("$GAMMAPY_DATA/cta-1dc-gc/cta-1dc-gc.fits.gz")
        >>> slices = {"energy": slice(0, 3)} #to get the first 3 energy slices
        >>> sliced = dataset.slice_by_idx(slices)
        >>> print(sliced.geoms["geom"])
        WcsGeom
        <BLANKLINE>
            axes       : ['lon', 'lat', 'energy']
            shape      : (np.int64(320), np.int64(240), 3)
            ndim       : 3
            frame      : galactic
            projection : CAR
            center     : 0.0 deg, 0.0 deg
            width      : 8.0 deg x 6.0 deg
            wcs ref    : 0.0 deg, 0.0 deg
        <BLANKLINE>
        """
        name = make_name(name)
        kwargs = {"gti": self.gti, "name": name, "meta_table": self.meta_table}

        if self.counts is not None:
            kwargs["counts"] = self.counts.slice_by_idx(slices=slices)

        if self.exposure is not None:
            kwargs["exposure"] = self.exposure.slice_by_idx(slices=slices)

        if self.background is not None and self.stat_type == "basil":
            kwargs["background"] = self.background.slice_by_idx(slices=slices)

        if self.edisp is not None:
            kwargs["edisp"] = self.edisp.slice_by_idx(slices=slices)

        if self.psf is not None:
            kwargs["psf"] = self.psf.slice_by_idx(slices=slices)

        if self.mask_safe is not None:
            kwargs["mask_safe"] = self.mask_safe.slice_by_idx(slices=slices)

        if self.mask_fit is not None:
            kwargs["mask_fit"] = self.mask_fit.slice_by_idx(slices=slices)
            
        if self.comb is not None:
            kwargs["comb_flat"] = slice_mask_flat_comb_new(self.comb, self.mask, slices)


        return self.__class__(**kwargs)     
