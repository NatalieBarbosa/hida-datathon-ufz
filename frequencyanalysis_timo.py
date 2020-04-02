# -*- coding: utf-8 -*-
import os.path as osp
import netCDF4
from netcdf_helpers.reader import get_time_series_from_location
from plot.plot import plot_time_series_for_locations
from frequencyanalysis.power_spectrum import power_spectrum
from frequencyanalysis.plot import plot_spectrum
import numpy as np
import matplotlib.pyplot as plt

# path for output plot
path = "/Users/houben/phd/hackathons/hida_datathon/repos/hida-datathon-ufz/frequencyanalysis_output"
# time step size
time_step_size = 365*86400
# set a path to the directory containing the data
directory = "/Users/houben/phd/hackathons/hida_datathon/data/MyChallengePaleo"
# set the file names
filename_temp_data_r1 = "T2m_R1_ym_1stMill.nc"
filename_temp_data_r2 = "T2m_R2_ym_1stMill.nc"
# Load input files
temp_data_r1 = netCDF4.Dataset(osp.join(directory, filename_temp_data_r1), "r")
temp_data_r2 = netCDF4.Dataset(osp.join(directory, filename_temp_data_r2), "r")
# set coordinate from desired location
lat_target = 0
lon_target = 0
# returns a time series at closest location
time_series, closest = get_time_series_from_location(
    temp_data_r1, "T2m", lat_target, lon_target
)

# # plot example
# names = ["a", "b", "c"]
# time_series_list = [time_series, time_series]
# time = temp_data_r1.variables['time'][:]
# plot_time_series_for_locations(time_series_list, time, names, linewidth=.1)


# function for power spectral density
def power_spec_plot(tempslice, label):
    # plot a power spectrum
    frequency, spectrum = power_spectrum(tempslice, tempslice, time_step_size, method="scipyperio", o_i="i")
    plot_spectrum(spectrum, frequency, path=path, linestyle="-", marker="", labels=label, lims=[(1e-11, 1e-7),(1,1e10)])

# -----------------------------
# aggregate data for different latitudes
# -----------------------------
# in 10 degree units
temp_r1 = temp_data_r1.variables["T2m"]
temp_r2 = temp_data_r2.variables["T2m"]

# set other variables, still they are classes but in contrast to temp_r1 only 1D
time_r1 = temp_data_r1.variables["time"]
time_r2 = temp_data_r2.variables["time"]
lat_r1 = temp_data_r1.variables["lat"]
lat_r2 = temp_data_r2.variables["lat"]
lon_r1 = temp_data_r1.variables["lon"]
lon_r2 = temp_data_r2.variables["lon"]

# make arrays from these classes
time = time_r1[:]
lat = lat_r1[:]
lon = lon_r1[:]

# make a temperature slice from the class temp_r1
temp_r1[:]
tempslice_r1 = temp_r1[:, (lat > 0) & (lat < 10) , :]
np.shape(tempslice_r1)
# take mean for that slice
tempslice_r1_mean = np.mean(tempslice_r1, axis=(1,2))
np.shape(tempslice_r1_mean)


#power_spec_plot(tempslice_r1_mean, "test")

frequency, spectrum = power_spectrum(tempslice_r1_mean, tempslice_r1_mean, time_step_size, method="scipyperio", o_i="i")

# define a list of markevery cases to plot
cases = ["-90 - -70", "-70 - -50", "-50 - -30", "-30 - -10", "-10 - 10", "10 - 30", "30 - 50", "50 - 70", "70 - 90"]
cases_u = [-70, -50, -30, -10, 10, 30, 50, 70, 90]
cases_l = [-90, -70, -50, -30, -10, 10, 30, 50, 70]
# define the figure size and grid layout properties
figsize = (10, 9)
cols = 2
rows = len(cases) // cols + 1
# define the data for cartesian plots
delta = 0.11


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


axs = plt.figure(figsize=figsize, constrained_layout=True).subplots(rows, cols)
axs = trim_axs(axs, len(cases))
for ax, case, case_l, case_u in zip(axs, cases, cases_l, cases_u):
    ax.set_title('latitude band ' + str(case) + ' degree')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel("Spectral Power")
    ax.set_xlabel("Frequency [Hz]")
    tempslice_r1 = temp_r1[:, (lat > case_l) & (lat < case_u), :]
    tempslice_r1_mean = np.mean(tempslice_r1, axis=(1,2))
    tempslice_r2 = temp_r2[:, (lat > case_l) & (lat < case_u), :]
    tempslice_r2_mean = np.mean(tempslice_r2, axis=(1,2))
    frequency, spectrum = power_spectrum(tempslice_r2_mean, tempslice_r1_mean, time_step_size, method="scipyperio", o_i="oi")
    ax.plot(frequency, spectrum, ls='-', ms=4)
    ax.set_ylim(1e-3, 1e3)
#ax.set_title("Power Spectral Density of T2m for R1 model. \n Aggregated for different latitudes")
plt.savefig(path + "/R2_R1_temp_spectral_density.png", dpi=300)
