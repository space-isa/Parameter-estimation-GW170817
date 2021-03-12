#!/usr/bin/python3
# Tested with Python 3.8.6
#------------------------------------------------------------------------------
#    runEmceeOutflow.py
#------------------------------------------------------------------------------
# Authors: Isabel J. Rodriguez, Davide Lazzati
# Oregon State University
#------------------------------------------------------------------------------
""" 
Use emcee module for Bayesian parameter estimation (BPE) analysis using outflow 
code developed by Lazzati and Perna (2019). Outflow code is proprietary and is 
thus not provided, however the general analysis process can be followed using 
one's own code.  

FIT PARAMETERS
--------------
	lmdot: log(mass loss) [M_sol/s]
	lLj: log(jet luminosity at injection) [erg/s]
	thinj: jet opening angle at injection [deg]
	dt: time delat to jet launch [s]
	vW: wind veloicty [c]
	Teng: engine activity duration [s]
	lG0: log(Lorentz factor at injection)

IMPORTED FILES
--------------
<outflowModel.py>

FUNCTIONS
--------------

afterglowFitData(None)
	Define best value and uncertainties generated from afterglow BPE analysis.
	Used by: main()

priorBounds(None)
	Define prior bounds for parameters of interest. We use a flat prior 
	distribution, with some parameters in log space.
	Used by: main()

logLikelihood(*args)
    Used by: logProbability()

logPrior(*agrs)
    Create parameter lables using math text.
    Used by: logProbability()

logProbability(*args)
    Used by: main()

runEmcee(*args)
    Call afterglow script to calculate lightcurves.
    Used by: logLikelihood()

main(None)
    Run emcee package and save results in backend file.

"""

# Standard Python imports
import os
import time 

#  Non-starndard Python imports
import numpy as np

#  Emcee module
try: 
	import emcee
except:
	print("Import the emcee module to continute.")

#  Companion scripts
#from <Outflow model> import cocoon


def afterglowFitData():
	# These are the best fit values (w/uncertainties) 
	# from the afterglow analysis as of 03052021.
	  # NOTE: Thetas are given in degrees.
	log_E_jet = [49.828, 0.456, 0.366]
	log_E_cocoon = [48.514, 0.795, 0.798]
	theta_jet = [2.154, 0.799, 0.576]
	theata_cocoon = [14.570, 7.242, 7.040]
	log_Gamma_0 = [3.544, 0.363, 0.890]

	bestValue=np.array([log_E_jet[0], log_E_cocoon[0], 
						theta_jet[0], theata_cocoon[0], 
						log_Gamma_0[0]
						)

	upperSigma=np.array([log_E_jet[1], log_E_cocoon[1], 
						 theta_jet[1], theata_cocoon[1], 
						 log_Gamma_0[1]
						 )
	
	lowerSigma=np.array([log_E_jet[2], log_E_cocoon[2], 
						 theta_jet[2], theata_cocoon[2], 
						 log_Gamma_0[2]
						 )
	return bestValue, upperSigma, lowerSigma

def priorBounds():
	"""Define priors."""

	priorBnds=[-3, 2,  # Log(mdot_w) [solar masses / s]
			   48, 53,  # Log(jet injection luminosity) [erg/s]
			   1, 30,  # theta_injection [deg]
			   0.01, 2,  # time delay [s]
			   0.01, 0.75,  # wind velocity [c]
			   0.1, 2,  # T_engine [s]
			   1, 4]  # Log(Gamma injection) [--]
	return priorBnds

def logLikelihood(fitParms,bestValue,lowerSigma,upperSigma):
    """
    Define log-likelihood function assuming 
    a Gaussian distribution.
    
    PARAMETERS
    ----------
    fitParms: set of parameters
    bestValue: array, float 
    lowerSigma: array, float 
	upperSigma: array, float

    Returns
    ----------
    chisqr: float
		Chi squared
     
    """ 
	lmdot, lLj, thinj, dt, vW, Teng, lG0 = fitParms
	
	#  Call outflow model
	theta_j, theta_c, e_c = cocoon(mdot_w=10**lmdot*2e33,L_j=10**lLj,
		                           theta_0_deg=thinj,delta_t=dt,
								   v_w=vW*3e10,t90=Teng,Gamma_0=10**lG0)
	e_j = 10 ** lLj * Teng - e_c

	if e_j<0: return -np.inf
	model = np.array([np.log10(e_j),np.log10(e_c),theta_j,theta_c,lG0])

	#  Calculate chi squared
	chisqr = ((model - bestValue) ** 2 / upperSigma ** 2)
	jj = np.where(model < bestValue)
	chisqr[jj] = ((model[jj] - bestValue[jj]) ** 2 / lowerSigma[jj] ** 2)
	chisqr = -0.5 * chisqr.sum()
	# print(chisqr)
	return chisqr

def logPrior(fitParms,priorBnds):
    """
    Define priors for a set of parameters, theta.
    
    PARAMETERS
    ----------
    fitParms: set of parameters 

    RETURNS
    ----------
    0.0 if sample is drawn within the bounds, -infinity otherwise. 
    """ 
	lmdot, lLj, thinj, dt, vW, Teng, lG0 = fitParms

	if (priorBnds[0] <  lmdot < priorBnds[1] and
		priorBnds[2] < lLj < priorBnds[3] and 
		priorBnds[4] < thinj < priorBnds[5] and 
		priorBnds[6] < dt < priorBnds[7] and 
		priorBnds[8] < vW < priorBnds[9] and 
		priorBnds[10] < Teng < priorBnds[11] and  
		priorBnds[12] < lG0 < priorBnds[13]): 
		return 0.0
	return -np.inf

def logProbability(fitParms,bestValue,lowerSigma,upperSigma,priorBnds):
	"""Define full log-probabilty function."""

	lp = logPrior(fitParms, priorBnds)
	if not np.isfinite(lp):
		return -np.inf
	return lp + logLikelihood(fitParms, bestValue,
							  lowerSigma, upperSigma)


def runEmcee(priorBnds, bestValue, lowerSigma, upperSigma):
	"""Run emcee sampler and save progress to backend file."""

	nWalkers = 100
	nSteps = 10000
	nDim = len(priorBnds) // 2

	#  Initialize walkers at random positions
	pos = np.zeros([nWalkers,nDim]) 
	for i in range(nDim):
		r_num = np.random.rand(nWalkers)
		pos[:,i] = r_num * (priorBnds[2*i+1] - priorBnds[2*i]) + priorBnds[2*i]

	#  Initialize backend file
	timestamp = time.strftime("%Y%m%d-%H%M")
	filename = "emcee-outflow-backend-{}.h5".format(timestamp)
	backend = emcee.backends.HDFBackend(filename)
	backend.reset(nWalkers, nDim)

	#  Initialize and run emcee sampler
	sampler = emcee.EnsembleSampler(nWalkers, nDim, 
									log_probability,
									args=(bestValue, lowerSigma,
									upperSigma, priorBnds),
									backend=backend)
	
	sampler.run_mcmc(pos, nSteps, progress=True);

	#  Retrieve samples
	burn_in = nSteps // 5
	samples = sampler.get_chain(discard=burn_in, flat=True)
	print("Emcee run complete. Check backend file for results.")

def main():
	priorBnds = priorBounds()
	bestValue, upperSigma, lowerSigma = afterglowFitData()
	runEmcee(priorBnds, bestValue, lowerSigma, upperSigma)

if __name__ == "__main__":
	main()

