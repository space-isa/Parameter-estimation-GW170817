"""
Takes a list of pre-defined frequencies found in 'observations_Mooley.dat', 
processes and creates output arrays. Not all frequencies have the same nubmer 
of data points, so to create 2D arrays missing data points are filled with 'NaN'
('Not a Number') and are masked (i.e., 'NaN' --> '--') so that future calculations 
can be performed without error. 

To use, import into a Python script as shown:

    Example: 
        >> from multibandDataMooley import time_obs, flux_obs, flux_uncert

INPUTS
------
    None 
        NOTE: User can define frequencies of interest based on how they are 
              presented in the fourth column in 'observations_Mooley.dat' by 
              updating the variable 'wanted_frequencies' (line ). 
RETURNS
-------
    time_obs  : masked array 
    flux_obs  : masked array 
    flux_uncert : masked array   
    output_freq : list 
"""

#  Standard Python library imports 
import numpy as np
import csv 
from itertools import chain

#  Import to create masked arrays 
import numpy.ma as ma

def pull_observation_data(datafile=None):
    """
    Pull data from all frequency bands. Convert to the correct dtypes. 
    Convert fluxes and uncertainties from mJy to Jy.
    """
    obs_time = []
    freq_labels = []
    obs_flux = []
    flux_uncert = []
    obs_freq = []

    with open(datafile, 'r') as data:
        read_data = csv.reader(data, delimiter="\t")
        for line in read_data:
            if line:
                obs_time.append(int(line[0]))
                freq_labels.append(line[1])
                obs_flux.append(float(line[2]) * 1e-3)
                flux_uncert.append(float(line[3]) * 1e-3)
                obs_freq.append(line[4])

    #  Create a set containing unique frequencies and times (i.e., no duplicates)
    unique_frequencies = list(set(obs_freq))
    # unique_times = list(set(obs_time))
    data = list(zip(obs_time, obs_flux, flux_uncert, obs_freq, freq_labels))
    return data, unique_frequencies


def create_storage_dictionary(data, unique_frequencies):
    """Create a nested storage dictionary."""

    stored_frequencies = {band:
                            {'Freq band': '',
                             'Time observed (s)' : [],
                             'Flux (Jy)': [],
                             'Flux uncertainty (Jy)' : []}
                         for band in unique_frequencies}
    return stored_frequencies

def update_dictionary(data, stored_frequencies):
    """Fill in nested dictionary with data points."""

    data = list(data) 
    for row in data: 
        obs_time, obs_flux, flux_uncert, obs_freq, freq_labels = row

        #  Open and update appropriate sub dictionary 
        nested_dict = stored_frequencies[obs_freq] 
        nested_dict['Freq band'] = freq_labels
        nested_dict['Time observed (s)'].append(obs_time)
        nested_dict['Flux (Jy)'].append(obs_flux)
        nested_dict['Flux uncertainty (Jy)'].append(flux_uncert)

    for key, val in stored_frequencies.items():
        val['Num datapoints'] = len(val['Time observed (s)'])

def pull_data_of_interest(stored_frequencies, wanted_frequencies):
    """
    Pull data from storage dictionary, using frequencies defined by user 
    in 'wanted frequencies' as look-up keys.
    """

    flux = []
    flux_uncert = []
    time = []
    output_freq = []

    for key, val in stored_frequencies.items():
        if key in wanted_frequencies: 
            flux.append(val['Flux (Jy)'])
            flux_uncert.append(val['Flux uncertainty (Jy)'])
            time.append(val['Time observed (s)'])
            output_freq.append(key)

    # Transform from row-wise to column-wise
    output_freq = np.asarray(output_freq).T
    flux = np.asarray(flux).T
    flux_uncert = np.asarray(flux_uncert).T
    time = np.asarray(time).T

    return flux, flux_uncert, time, output_freq

def find_unique_times(time):
    flatten_time = list(chain.from_iterable(time)) 
    unique_times = list(set(flatten_time))
    unique_times = np.sort(unique_times)
    return unique_times

def convert_freq_to_ev(output_freq):
    output_freq = np.array(output_freq, dtype='float64')

    for i in range(len(output_freq)):
        # X-ray
        if output_freq[i] == -1: 
            output_freq[i] = 1000
        # Optical
        elif output_freq[i] == 4.95:
            output_freq[i] = 2.066404
        # Radio in GHz
        else: 
            output_freq[i] = (np.float(output_freq[i]) * 4.13567E-06) # *1e9*6.63e-27/1.6e-12) 
    
    return output_freq


def create_mask(unique_times, time_array):
    return np.in1d(unique_times, time_array, invert=True)


def create_filler(unique_times):
    return np.ones(len(unique_times)) * np.nan


def find_indicies(mask):
    return np.where(mask==False)[0]


def fill_array(filler_array, indicies, array):
    filler_array[indicies] = array
    return filler_array


def mask_array(array, mask):
    return ma.masked_array(array, mask)


def cleanData():
    """
     Contains a pipeline that processes observation data 
     based on the frequency bands the user is interested in.    

    ARGUMENTS
    --------- 
        NONE
        
    RETURNS
    -------
        time_obs : list 
        flux_obs : list
        flux_uncert : list 
        output_freq : list
        unique_times : list  
    """

    data, unique_frequencies = pull_observation_data(datafile='./observations_Mooley.dat')
    stored_frequencies = create_storage_dictionary(data, unique_frequencies)
    update_dictionary(data, stored_frequencies)
    flux, flux_uncert, time, output_freq = pull_data_of_interest(stored_frequencies, wanted_frequencies)
    unique_times = find_unique_times(time)
    output_freq = convert_freq_to_ev(output_freq)
    return time, flux, flux_uncert, output_freq, unique_times

def maskData(time, flux, flux_uncert, unique_times):
    """
     Contains a pipeline that processes cleaned data 
     based on the frequency bands the user is interested in.    

    ARGUMENTS
    --------- 
        NONE
        
    RETURNS
    -------
        time_obs : masked array (object) 
        flux_obs : masked array (object) 
        flux_uncert : masked array (object)  
    """

    time_mask = []
    indicies = []
    flux_filler = []
    flux_uncert_filler = []
    time_filler = []

    #  Create arrays that are the same size as unique times
    filler = create_filler(unique_times)
    filler_array_flux = np.tile(filler, (num_frequencies,1))
    filler_array_flux_uncert = np.copy(filler_array_flux)
    filler_array_time = np.copy(filler_array_flux)

    for i in range(len(wanted_frequencies)):
        mask = create_mask(unique_times, time[i])
        time_mask.append(mask)
        where_false = find_indicies(mask)
        indicies.append(where_false)
        flux_filler.append(fill_array(filler_array_flux[i], 
                                      where_false, 
                                      flux[i]))
        flux_uncert_filler.append(fill_array(filler_array_flux_uncert[i], 
                                             where_false, 
                                             flux_uncert[i]))
        time_filler.append(fill_array(filler_array_time[i], 
                                             where_false, 
                                             time[i]))

    #  Transform row-wise --> column-wise
    time_mask = np.asarray(time_mask).T
    flux_filler = np.asarray(flux_filler).T
    flux_uncert_filler = np.asarray(flux_uncert_filler).T
    time_filler = np.asarray(time_filler).T

    #  Final outputs
    flux_obs = ma.masked_array(flux_filler, time_mask)
    flux_uncert = ma.masked_array(flux_uncert_filler, time_mask)
    time_obs = ma.masked_array(time_filler, time_mask)
    return time_obs, flux_obs, flux_uncert

#  Run code

#  Frequencies of interest
    #  [Note: -1 = xray, 4.95 = optical]
wanted_frequencies = ['3.00E+00', '6.00E+00', '1.30E+00', 
                        '7.25E+00', '4.50E+00', '-1', '4.95'] 
num_frequencies = len(wanted_frequencies)

time_obs, flux_obs, flux_uncert, output_freq, unique_times = cleanData()
time_obs, flux_obs, flux_uncert = maskData(time_obs, flux_obs, flux_uncert, unique_times)

sort_index=np.argsort(output_freq)
output_freq=output_freq[sort_index]

time_copy = time_obs.copy()
flux_copy = flux_obs.copy()
uncert_copy = flux_uncert.copy()

#  Sort data to match frequency order
for i in range(unique_times.size):
    time_copy[i,:]=time_obs[i,sort_index]
    flux_copy[i,:]=flux_obs[i,sort_index]
    uncert_copy[i,:]=flux_uncert[i,sort_index]

# Final outputs
time_obs = time_copy
flux_obs = flux_copy
flux_uncert = uncert_copy
# freqObsGHz = output_freq / (4.135667662e-15 * 1e9)
print("Data cleaned.")


