#!/usr/bin/python3
# Tested with Python 3.8.6
#------------------------------------------------------------------------------
#    compile_backend.py
#------------------------------------------------------------------------------
# Author: Isabel J. Rodriguez
# Oregon State University
#------------------------------------------------------------------------------
"""
Search for all emcee backend files (ending in .h5) in the current directory. 
Pull and compile samples from each file and generate 1) a plot of the parameter 
chains, and 2) a corner plot of the posterior distributions. Save plots as png 
files and display the best fit values.  

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
import numpy as np
import pylab as plb
from matplotlib import pyplot as plt


def retriveBackendFiles():
    """Return all backend files current directory."""

    try:
        files = glob.glob('./*.h5')
        files = np.sort(files)
        print(files)
        return files
    except:
        print("There are no files in this folder")

def createLabels():
    """Create parameter lables using math text."""
    params  = [r"$log(ɛ_{\mathrm{e}})$",
               r"$log(ɛ_{\mathrm{b}})$",
               r"$p_{\mathrm{index}}$",
               r"$log(n_{\mathrm{ISM}})$",
               r"$log(E_{\mathrm{j}})$",
               r"$log(E_{\mathrm{c}})$",
               r"$θ_{\mathrm{j}}$",
               r"$θ_{\mathrm{c}}$",
               r"$θ_{\mathrm{obs}}$",
               r"$log(Γ_{\mathrm{0}})$"]
    
    labels = params[:]
    return labels

def compileBackendSamples():
    """Take list of backend files and compile samples."""

    files = retriveBackendFiles()
    compiled_samples = []

    for backend in files:
        reader = emcee.backends.HDFBackend(backend)
        #thin = 15
        # tau = reader.get_autocorr_time()
        #burn_in = int(2 * np.max(tau))
        #thin = int(0.5 * np.min(tau))
        samples = reader.get_chain(flat=True, discard=burn_in)
        compiled_samples.extend(samples)

    # print(np.shape(compiled_samples))
    compiled_samples = np.asarray(compiled_samples)
    return compiled_samples 

def plotChains(labels, samples):
    """Plot chains for each parameter."""

    fig, axes = plt.subplots(np.shape(samples)[1], figsize=(10, 7), sharex=True)
    for i in range(np.shape(samples)[1]):
        ax = axes[i]
        ax.plot(samples[:, i], "k", alpha=0.7)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("step number")
    fig.savefig("chain-burnin-{}-steps-{}.png".format(burn_in, 
               time.strftime("%Y%m%d-%H%M")))

def cornerPlot(labels, samples):
    """Generate main plot."""

    xkcd_color = 'xkcd:' + 'black'
    corner_plot = corner.corner(samples,
                                labels=labels,
                                label_kwargs={'fontsize': 20},
                                quantiles=[0.16, 0.5, 0.84],
                                # color=xkcd_color,
                                histtype='bar',
                                show_titles=False,
                                title_kwargs={"fontsize": 18},
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

    filename = "corner-plot-DL-" \
               "burnin-{}-steps-{}.png".format(burn_in, 
               time.strftime("%Y%m%d-%H%M"))

    corner_plot.savefig(filename)
    print("Corner plot generated.")

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
    creates plotting lables, and 
      """

    compiled_samples = compileBackendSamples()
    labels = createLabels()
    showFinalNumbers(labels, samples=compiled_samples)

    #  Generate plots 
    plotChains(labels, samples=compiled_samples)
    cornerPlot(labels, samples=compiled_samples)


if __name__ == "__main__":
    """Run code."""

    burn_in = 0
    main()