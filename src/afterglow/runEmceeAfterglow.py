#!/usr/bin/python3
# Tested with Python 3.8.6
#------------------------------------------------------------------------------
#    runEmceeAfterglow.py
#------------------------------------------------------------------------------
# Authors: Isabel J. Rodriguez, Davide Lazzati
# Oregon State University
#------------------------------------------------------------------------------
""" 
Use emcee module to generate parameter fits based on DL afterglow code. 

Imported files
--------------
init_params.py
plots.py
multibandDataMooley.py
exceptionHandler.py
<afterglowModel.py>

Functions 
--------------

cleanTempFolder(None)
    Remove files from /temp folder.
    Used by: main()

createParamFiles(*args)
    Store emcee parameter values in .dat file. 
    Used by: main()

runAfterglow(*args)
    Call afterglow script to calculate lightcurves.
    Used by: logLikelihood()

logPrior(*agrs)
    Create parameter lables using math text.
    Used by: logProbability()

logLikelihood(*args)
    Used by: logProbability()

logProbability(*args)
    Used by: main()

main(None)
    Run emcee package, save and plot results. 
"""

#  Standard Python library imports 
import numpy as np
from math import log10
import time
import shutil #  for cleaning temp folder 
import os 
os.environ["OMP_NUM_THREADS"] = "1"
import sys
print("Python version {}".format(sys.version))
import multiprocessing
from multiprocessing import Pool

#  Emcee imports 
import emcee
print("emcee version", emcee.__version__)
import tqdm #  for progress bar 

#  Importing from companion scripts 
from cleanDataGW170817 import time_obs, flux_obs, flux_uncert
from init_params import params_list
import runAfterglowDL as run_ag
from exceptionHandler import exception_handler


def cleanTempFolder():
    """Remove files from temp folder."""

    folder = "./temp/"
    for filename in os.listdir(folder):
        file_path = folder + filename
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

def createParamFiles(params_list):
    """
    Determine whether a duplicate file exists 
    using pid and timestamp as identifiers.
    If file exists, create a new one with a different timestamp.
    If not, create a temp file containing emcee parameter samples.

    Parameters
    ----------
    params_list: list
        List of emcee parameter values. 

    Returns
    ----------
    params_datafile: .dat file 
        Contains emcee parameter values for use by runAfterglow.
    """

    folder = "./temp/"
    filename = "params-pid-{}-time-{}.dat".format(
            str(os.getpid()), str(time.time()))
    #  Ensure filenames are unique
    if os.path.isfile(folder + filename):
        new_filename = 'params-pid-{}-time-{}.dat'.format(
            str(os.getpid()), str(time.time()))
        params_datafile = folder + new_filename
    else:
        params_datafile = folder + filename

    dataout = []
    for i, item in enumerate(params_list):
        dataout.append(item[0])
    np.savetxt(params_datafile, dataout, fmt='%s')
    return params_datafile

def runAfterglow(eps_E, eps_B, p_e, 
                 n_ISM, E_j, E_c, theta_j, 
                 theta_c, theta_obs, Gamma_0):
    """
    Convert emcee parameter values from log space to linear space.  
    Call afterglow script runAfterglowDL.py to calculate lightcurves.

    Parameters
    ----------
    theta = {eps_E,...,Gamma_0}

    Returns
    ----------
    lightcurve: float 
    """ 

    params = zip((eps_E, eps_B, p_e, 
                  n_ISM, E_j, E_c, theta_j, 
                  theta_c, theta_obs, Gamma_0))
    params_list = (list(params))
    params_datafile = createParamFiles(params_list)
    lightcurve = run_ag.main(params_datafile=params_datafile)
    return lightcurve

def logPrior(theta):
    """
    Define flat ("uninformative") prior distributions for 
    a set of parameters.

    Parameters
    ----------
    theta: set of parameters 

    Returns
    ----------
    0.0 if sample drawn within the bounds, -infinity otherwise. 
    """ 
    eps_E, eps_B, p_e, \
    n_ISM, E_j, E_c, theta_j, \
    theta_c, theta_obs, Gamma_0 = theta
    #  NOTE: eps_E, eps_B, n_ISM, E_j, E_c, and Gamma_0 are flat priors
    #  in log space

    if (-4 <  eps_E < -0.3 and 
        -4 < eps_B < -0.3 and 
        2 < p_e < 2.5 and
        -4 < n_ISM < -0.3 and
        -4 < E_j < 50 and
        -4 < E_c < 49 and
        0 < theta_j < 10 and 
        theta_j + 0.6 < theta_c < 20 and
        0 < theta_obs < 90 and
        -4 < Gamma_0 < 2.7):
        return 0.0
    return -np.inf

def logLikelihood(theta, x, y, yerr):
    """
    Define log-likelihood function assuming 
    a Gaussian distribution.
    
    Parameters
    ----------
    theta: set of parameters
    y: array, float 
        Observed flux 
    yerr: array, float 
        Observed flux uncertainty 

    Returns
    ----------
    -0.5 * np.sum(((y-model)/yerr)**2): float
        Likelihood function 
    """ 
    eps_E, eps_B, p_e, \
    n_ISM, E_j, E_c, theta_j, \
    theta_c, theta_obs, Gamma_0 = theta  

    lightcurve = runAfterglow(eps_E, eps_B, p_e, 
                              n_ISM, E_j, E_c, theta_j, 
                              theta_c, theta_obs, Gamma_0)
    model = lightcurve
    return -0.5 * np.sum(((y-model)/yerr)**2) 

def logProbability(theta, x, y, yerr):
    """Define full log-probabilty function."""
    if not np.isfinite(logPrior(theta)):
        return -np.inf
    return logPrior(theta) + logLikelihood(theta, x, y, yerr)

def emceeSampler(params_list):
    """" 
    Run emcee sampler and check for convergence every n steps.  

    Parameters
    ----------
    params_list: list, float
        NOTE: This is a global variable, 
        imported from init_params.py (see imports list, line 63). 

    Returns
    ----------
    None  
    """
    def _prepEmcee(params_list, Gaussian_ball=False):
        """
        Iniitalize walkers around initial guess. 
        
        If 'Gaussian_ball' is set to True, initialize walkers in a small 
        Gaussian ball around the inital guess.
        """
        num_params = len(params_list)
        print("# of parameters emcee is fitting: {}".format(num_params))
        print("Initial parameter guesses:{}".format(params_list))
        params_list = np.reshape(params_list, (1, num_params))
        if Gaussian_ball == True:
            pos = params_list + 1e-4 * np.random.randn(n_walkers, num_params)
        else:
            pos = params_list * np.random.randn(n_walkers, num_params)
            print(pos)
        print("Initial walkers set.")
        nwalkers, ndim = pos.shape
        return nwalkers, ndim, pos

    def _createBackendFile():
        """Generate a .h5 backend file to save and monitor progress.""" 
        print(os.getcwd())
        folder = "./backend"
        datestamp = time.strftime("%Y%m%d-%H%M")
        filename = "backend-file-{}.h5".format(datestamp)
        backend = emcee.backends.HDFBackend(folder+filename)

        return backend

    def _saveResults(backend, samples):
        """Rename backend file to match when the emcee run completed."""
        datestamp = time.strftime("%Y%m%d-%H%M")
        backend_folder = './backend/'
        filename = "backend-file-{}.h5".format(datestamp)
        os.rename(backend.filename,
                  backend_folder + filename) 

    def _runEmcee(backend, nwalkers, ndim, pos):
        """            
        Set up a pool process to run emcee in parallel. 
        Run emcee sampler and check for convergence very n steps,
        where n is user-defined. 
        """
        backend.reset(nwalkers, ndim)
        index = 0
        autocorr = np.empty(max_iter)
        old_tau = np.inf
        
        #  Set up parallel processing 
        with Pool(processes = n_processes) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, 
                                            ndim, 
                                            logProbability,
                                            args = (x,y,yerr), 
                                            backend=backend, 
                                            pool=pool)
            #  Run emcee 
            for sample in sampler.sample(
                pos, iterations=max_iter, progress=True):

                #print("log_prob = {} ".format(sampler.get_log_prob()))
                #print("tau = {}".format(sampler.get_autocorr_time()))
                #print("acceptance fraction = {} ".format(sampler.acceptance_fraction))  

                #  Check for convergence very "check_iter" steps
                if sampler.iteration % check_iter:
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[index] = np.mean(tau)
                index += 1
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau

                #  Get samples 
                samples = sampler.chain[:, :, :].reshape((-1,ndim))
                print(samples.shape, samples)
        return samples

    backend = _createBackendFile()
    nwalkers, ndim, pos = _prepEmcee(params_list)
    samples = _runEmcee(backend, nwalkers, ndim, pos)
    _saveResults(backend, samples)
    print("Emcee run complete. Access backend file to plot.")

@exception_handler
def main():
    """Clean temp folder and run emcee sampler."""
    cleanTempFolder()
    #  Run emcee sampler code 
    emceeSampler(params_list)


if __name__ == "__main__":
    # Global variables from imports  
    x = time_obs
    y = flux_obs
    yerr = flux_uncert 
    num_params = len(params_list)
    params = np.reshape(params_list, (1, num_params))

    # User-defined global variables 
    n_walkers = 20  
    n_processes = 1
    max_iter = 1

    #  Check for convergence every n iterations
    #  NOTE: max_iter must be divisible by n
    check_iter = 1 

    #  Run script 
    main() 
