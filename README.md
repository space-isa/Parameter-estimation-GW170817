# Parameter-estimation-GW170817

Project files for a Bayesian parameter estimation (BPE) study of the outflow and structured jet properties of the short gamma-ray burst (SGRB) resutling from the binary neutron star merger event GW170817.

---
## Table of Contents 
1. [How to use?](#how-to-use) 
2. [Requirements](#requirements)
3. [Ideas for future development](#ideas-for-future-development)
4. [Credits](#credits)
5. [Author](#author)
---

## About the code

Refer to the manuscript PDF located in the `doc/thesis-manuscript` directory to learn more about the project backgorund (Ch. 1-3), analysis methodology (Ch. 4), and contextualized results (Ch. 5-6).

## How to use

- In the `data` directory:
  - The folder `afterglow` houses data and code to reproduce our strucutred jet SGRB afterglow model results. Contents:
    1) a set of backend (.h5) files containing analysis data (e.g., `emcee-afterglow-backend-01.h5`) 
    2) a script `compile_backend.py` to compile and plot data 
    3) a directory `example-plots` with examples of what the plots resulting from (2) should look like 
  - The folder `outflow-to-jet` houses data and code to reproduce our outflow to structured jet model results. Contents:
    1) a file `download-data.txt` with instructions on how to download the dataset
    2) a script `run_backend.py` to compile and plot data
    3) a directory `example-plots` with examples of what the plots resulting from (2) should look like 
  
- In the `doc` directory:
  - `thesis-manuscript` holds both the manuscript PDF and LaTeX project folder (containing .tex and .bib files as well as all figures used) 
  - `defense-slides` contains a PDF version of my defense presentation

- In the `src`directory:
  - Both `afterglow` and `outflow-to-jet` contain samples of code that could be integrated into an existing project
    -  [NOTE: Models are currently proprietary and are not provided.]
  - `afterglow` contains the raw observational dataset `observations_Mooley.dat` and a cleaning script `cleanData170817.py`

## Requirements
This code was developed using Python 3.8.6.

Besides standard Python library, this project utilzed:

- Non-standard Python libraries
  - `numpy`
  - `scipy`
  - `matplotlib`
- `emcee` imports (see Credits to learn more)
  - `emcee`
  - `corner`
  - `h5py`

## Ideas for future development
 
- Expand the functionality of `exception_handler.py`.
- Developing a robust test to probe model sensitivity to priors.
- Utilizing machine learning to speed up `emcee` runtime at the model-data comparison stage.

## Credits 
`emcee`: The MCMC Hammer package: https://emcee.readthedocs.io/en/stable/

## Author
Isabel J. Rodriguez, Oregon State University

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
