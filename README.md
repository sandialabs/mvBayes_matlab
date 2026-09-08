[![Pipeline Status](https://github.com/sandialabs/mvBayes_matlab/actions/workflows/matlab.yml/badge.svg)](https://github.com/sandialabs/mvBayes_matlab/actions/workflows/matlab.yml)

# mvBayes

![](logo.png)

A MATLAB implementation of the multivariate Bayesian regression (mvBayes) framework. Decomposes a multivariate/functional response using a user-specified orthogonal basis decomposition, and then models each basis component independently using an arbitrary user-specified (univariate) Bayesian regression model. Includes prediction and plotting methods.


## Examples
* [Friedman Example](example.m) - An extension of the "Friedman function" to functional response. The Bayesian regression model here is BASS (Bayesian Adaptive Smoothing Splines)


### Installation
------------------------------------------------------------------------------
1. Download zip or tar.gz of package or clone repository
------------------------------------------------------------------------------

### Dependencies

Core functionality requires the Statistics and Machine Learning Toolbox
(`sobolset`, `scramble` for Sobol' sampling).

Two optional features require external packages on the MATLAB path:

* `basisType="pns"` requires [fdasrvf_MATLAB](https://github.com/jdtuck/fdasrvf_MATLAB)
  (`fastpns`, `fastPNSe2s`, `PNSs2e`).
* Closed-form Sobol' indices for BASS models require
  [bass_matlab](https://github.com/sandialabs/bass_matlab) (`BassBasis`,
  `sobolBasis`). Without it, `mvSobol` falls back to Monte Carlo.

## References


************

Author: Gavin Q. Collins and J. Derek Tucker
Sandia National Laboratories

