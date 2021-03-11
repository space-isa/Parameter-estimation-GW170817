#!/usr/bin/python3
# Tested with Python 3.8.6
#------------------------------------------------------------------------------
#    run_backend.py
#------------------------------------------------------------------------------
# Author: Isabel J. Rodriguez
# Oregon State University
#------------------------------------------------------------------------------
"""
Search for most recent emcee backend file (ending in .h5) in the current directory. 
Pull samples from file and generate 1) a plot of the parameter  chains, 2) a corner 
plot of the posterior distributions, and 3) display best fit values and uncertainties. 
Save plots as png files.  

INPUTS
------
    NONE

OUTPUTS
-------
    NONE
"""

#  Imported libraries
try:
    import emcee
except: 
    print("Install the emcee module to continue.")
try: 
    import corner
except: 
    print("Install the corner module to continue.")

# Standard Python library imports
import os
import glob
import time

# Non-standard libraries
from matplotlib import pyplot as plt
import numpy as np
import pylab as plb

def retriveRecentFile():
    """Return the most recent backend file in directory."""
    try:
        files = glob.glob('./*.h5')
        print(max(files))
        return max(files)
    except:
        print("There are no files in this folder")

def createLabels():
    """Create parameter lables using math text."""

    params = [r'log($\dot{m}_w$)',
              r'log(L$_0$)',
              r'$θ_{0}$',
              r'$Δ(t)$',
              r'v$_w$',
              r'T$_{engine}$',
              r'log($\Gamma_{0}$)']

    labels = params[:]
    return labels

def pullSamples(backend_filename):
    """Pull samples from backend file."""

    reader = emcee.backends.HDFBackend(backend_filename)
    samples = reader.get_chain(discard=burn_in, flat=True, thin=thin) 
    return samples 

def plotChains(samples, labels, timestamp):
    fig, axes = plt.subplots(samples.shape[1], figsize=(10, 7), sharex=True)
    for i in range(samples.shape[1]):
        ax = axes[i]
        ax.plot(samples[:, i], "k", alpha=0.7)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("step number")
    
    fig.savefig("chain-burnin-{}-steps-{}.png".format(burn_in, timestamp))

def cornerPlot(samples, labels, timestamp):
    """Generate main plot."""

    xkcd_color = 'xkcd:' + 'black'
    corner_plot = corner.corner(samples,
                                labels=labels,
                                label_kwargs={"fontsize":20},
                                quantiles=[0.16, 0.5, 0.84],
                                color=xkcd_color,
                                histtype='bar',
                                show_titles=True,
                                title_kwargs={"fontsize":14},
                                # title_fmt='.1e',
                                smooth=True,
                                fill_contours=True,
                                plot_density=True,
                                use_math_text=False,
                                hist_kwargs={
                                "color": 'grey',
                                "fill": True,
                                "edgecolor": 'k',
                                "linewidth": 1.2
                                },
                                top_ticks=False,
                                figsize=((12, 12)))
    for ax in corner_plot.get_axes():
        ax.tick_params(axis='both', 
                        labelsize=16, 
                        direction='in', 
                        pad=8.0)
    filename = "corner-plot-outflow-burnin-{}-steps-{}.png".format(burn_in, timestamp)
    corner_plot.savefig(filename)
    print("Corner plot generated. " \
        "Check /plots folder to view results.")

def showFinalNumbers(labels, samples):
    """
    Use the 50th, 16th and 84th quantiles as best value 
    and lower/upper uncertanties respectively.
    """

    # Number of dimensions = number of parameters
    num_dim = np.shape(samples)[1]
    mcmc_quantiles=np.zeros([num_dim,3])
    best_val = 50
    lower_sigma = 16
    upper_sigma = 84

    for i in range(num_dim):
        mcmc_quantiles[i,:] = np.percentile(samples[:, i], 
                                           [lower_sigma, 
                                            best_val, 
                                            upper_sigma])
        print("{} = {:.3f}".format(labels[i], mcmc_quantiles[i,1]),
              " +{0:.3f}".format(mcmc_quantiles[i,2]-mcmc_quantiles[i,1]),
              " -{0:.3f}".format(mcmc_quantiles[i,1]-mcmc_quantiles[i,0]))

def main():
    """
    Contains a pipeline that generates a list of compiled samples, 
    creates plotting lables, and displays final best fit values.
    """

    timestamp = time.strftime("%Y%m%d-%H%M")
    backend_filename = retriveRecentFile()
    samples = pullSamples(backend_filename)
    labels = createLabels()

    #  Generate plots
    plotChains(samples, labels, timestamp)
    cornerPlot(samples, labels, timestamp)


if __name__ == "__main__":
    """Run code."""

    burn_in = 0
    thin = 20
    main()
