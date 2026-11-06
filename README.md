# basil_for_gammapy

This is an extension package for Gammapy that implements
the Bayesian Analysis including Single-event Likelihoods (BASiL) --
a method initially proposed by D'Amico et al. (2021).

Details, performance evaluation and examples for this implementation
are described in Genaro et al. (in prep.).

This is not part of the official Gammapy distribution. This package
is distributed under the BSD 3-Clause License (see LICENSE). Original
Gammapy license is also preserved (see LICENSE-Gammapy).

## Methodology

BASiL performs a Bayesian estimation of the gamma-ray signal by
including gamma-hadron separation variables in the posterior
probability density. This allows removing selection cuts, which
increases the amount of signal events and improves the sensitivity 
and statistical constraints for source modeling.

The package creates new dataset classes that inherit from
Gammapy classes and change the statistics computation associated
with them (`stat_array()` and `stat_sum()` methods). It can be used
for 1D and 3D analyses.

The user must provide the lower level information, such as producing
event lists and instrument response functions without cuts, as well as
attributing likelihood values to the events based the variable distributions
associated with gamma rays and background. The code provides a tentative
suggestion for computing the likelihoods, given the distributions, but it
must also be adapted by the user.

## Features

- New dataset classes: SpectrumDatasetOnOffBASiL and MapDatasetBASiL
for 1D and 3D analyses with BASiL.
- Example notebooks guiding the preparation of the datasets.
- Example data: simulated sources based on the first public test data release
of H.E.S.S. (H.E.S.S. Collaboration 2018). Generated event lists contain values
for a gamma-hadron separation variable `GAMMANESS` that does not correspond to a
real analysis. Likelihood values for the events also accompany the data.

## Installation

First, install Gammapy: reference


Then install this package:

git clone https://github.com/username/repository.git
cd repository
pip install .

## Compatibility

- Python: >= 3.10 ??
- Gammapy: = 1.3

Compatibility with Gammapy >= 2.0 have not been implemented yet.
Compatibility with Gammapy < 1.3 have not been tested.

## Tutorial

Use Jupyter notebooks `example_notebook.ipynb` and `example_notebook_3D.ipynb`
to learn how to incorporate BASiL in a Gammapy analysis.

## Acknowledgements

This package builds upon the work of the
[Gammapy project](https://gammapy.org/), an open-source
Python package for gamma-ray astronomy.

Some parts of this package are derived from or extend
Gammapy functionality.

Gammapy is distributed under the BSD 3-Clause License.
Copyright notices and license conditions for derived
code are retained accordingly.

Please refer to the original
[Gammapy repository](https://github.com/gammapy/gammapy)
for further information.


[![gammapy](https://img.shields.io/badge/powered%20by-gammapy-orange.svg?style=flat)](https://www.gammapy.org/)
