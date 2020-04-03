# -*- coding: utf-8 -*-
import os.path as osp
import netCDF4
from netcdf_helpers.reader import get_time_series_from_location
from plot.plot import plot_time_series_for_locations
from frequencyanalysis.power_spectrum import power_spectrum
from frequencyanalysis.plot import plot_spectrum
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
import scipy.signal as signal
from frequencyanalysis.filter import butter_bandstop_filter

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

# Make power spectra for different zones
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
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



# filter out the solar activity: or a frequency of x years from anomaly temp
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
global_mean_ts_r1 = np.mean(temp_r1, axis=(1,2))
global_mean_ts_r1_anom = global_mean_ts_r1 - np.mean(global_mean_ts_r1)

spectrum = fftpack.fft(global_mean_ts_r1_anom)
len_input = len(global_mean_ts_r1_anom)
frequency = fftpack.fftfreq(len_input, time_step_size)

# define min and max frequency in years
min_freq = 1 / (365*86400*9)
max_freq = 1 / (365*86400*12)
filtered_9_12 = butter_bandstop_filter(global_mean_ts_r1_anom, max_freq, min_freq, 1/time_step_size, order=1)
min_freq = 1 / (365*86400*5)
max_freq = 1 / (365*86400*1000)
filtered_5_1000 = butter_bandstop_filter(global_mean_ts_r1_anom, max_freq, min_freq, 1/time_step_size, order=1)
plt.figure(figsize=(20,10))
plt.plot(global_mean_ts_r1_anom[:1000], label="signal")
plt.plot(filtered_9_12[:1000], label="Filter 9-12 years", linestyle="-", linewidth=0.5)
plt.plot(filtered_5_1000[:1000], label="Filter 5-1000 years", linestyle="-", linewidth=0.5)
plt.legend()
plt.savefig("frequencyanalysis_output/t2m_r1_filtered.png", dpi=300)
# filter out the solar activity: or a frequency of x years from temp
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
global_mean_ts_r1 = np.mean(temp_r1, axis=(1,2))

spectrum = fftpack.fft(global_mean_ts_r1)
len_input = len(global_mean_ts_r1)
frequency = fftpack.fftfreq(len_input, time_step_size)

# define min and max frequency in years
min_freq = 1 / (365*86400*9)
max_freq = 1 / (365*86400*12)
filtered_9_12 = butter_bandstop_filter(global_mean_ts_r1, max_freq, min_freq, 1/time_step_size, order=1)
min_freq = 1 / (365*86400*50)
max_freq = 1 / (365*86400*100)
filtered_50_100 = butter_bandstop_filter(global_mean_ts_r1, max_freq, min_freq, 1/time_step_size, order=1)
plt.plot(global_mean_ts_r1[:300], label="signal")
plt.plot(filtered_9_12[:300], label="Filter 9-12 years", linestyle="-", linewidth=0.5)
plt.plot(filtered_50_100[:300], label="Filter 50-100 years", linestyle="-", linewidth=0.5)
plt.legend()


# Make power spectra for different zones of filtered temp anomaly
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# define min and max frequency in years
min_freq = 1 / (365*86400*9)
max_freq = 1 / (365*86400*12)
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
    # filter the signal:
    tempslice_filtered = butter_bandstop_filter(tempslice_r2_mean, max_freq, min_freq, 1/time_step_size, order=1)
                         #butter_bandstop_filter(global_mean_ts_r1_anom, max_freq, min_freq, 1/time_step_size, order=1)
    frequency, spectrum = power_spectrum(tempslice_filtered, tempslice_filtered, time_step_size, method="scipyperio", o_i="i")
    ax.plot(frequency, spectrum, ls='-', ms=4)
    #ax.set_ylim(1e-3, 1e3)
#ax.set_title("Power Spectral Density of T2m for R1 model. \n Aggregated for different latitudes")
plt.savefig(path + "/R2_temp_spectral_density_filtered_9_12.png", dpi=300)

plt.plot(tempslice_filtered, label="filtered")
plt.plot(tempslice_r2_mean, label="signal")















# global_mean_ts_r1_anom_filtered = butter_bandstop_filter(global_mean_ts_r1_anom, max_freq, min_freq, 1/time_step_size, order=5)
# plt.plot(global_mean_ts_r1_anom)
# plt.plot(global_mean_ts_r1_anom_filtered)






#
# # Bandstop filter - Does not work
# def butter_bandstop(lowcut, highcut, fs, order=5):
#     nyq = 0.5 * fs
#     low = lowcut / nyq
#     high = highcut / nyq
#     b, a = butter(order, [low, high], btype='highpass')
#     return b, a
#
# def butter_bandstop_filter(data, lowcut, highcut, fs, order=5):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y


# pop first value of frequency since it's zero and all negative frequencies
# frequency_half = frequency[1:int((len(frequency)+1)/2)]
#
# frequency_cut = (frequency_half < min_freq) & (frequency_half > max_freq)
# fre ...
# np.fft.ifft(spectrum)
# plt.plot(np.fft.ifft(spectrum))


#
# def run():
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.signal import freqz
#
#     # Sample rate and desired cutoff frequencies (in Hz).
#     fs = time_step_size
#     lowcut = min_freq
#     highcut = max_freq
#
#     # Plot the frequency response for a few different orders.
#     plt.figure(1)
#     plt.clf()
#     for order in [3, 6, 9]:
#         b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#         w, h = freqz(b, a, worN=2000)
#         plt.plot((fs * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
#
#     plt.plot([0, 0.5 * fs], [np.sqrt(0.5), np.sqrt(0.5)],
#              '--', label='sqrt(0.5)')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Gain')
#     plt.grid(True)
#     plt.legend(loc='best')
#
#     # Filter a noisy signal.
#     T = 0.05
#     nsamples = T * fs
#     t = np.linspace(0, T, nsamples, endpoint=False)
#     a = 0.02
#     f0 = 600.0
#     x = 0.1 * np.sin(2 * np.pi * 1.2 * np.sqrt(t))
#     x += 0.01 * np.cos(2 * np.pi * 312 * t + 0.1)
#     x += a * np.cos(2 * np.pi * f0 * t + .11)
#     x += 0.03 * np.cos(2 * np.pi * 2000 * t)
#     plt.figure(2)
#     plt.clf()
#     plt.plot(t, x, label='Noisy signal')
#
#     print(np.shape(x))
#
#     y = butter_bandpass_filter(x, lowcut, highcut, fs, order=6)
#     plt.plot(t, y, label='Filtered signal (%g Hz)' % f0)
#     plt.xlabel('time (seconds)')
#     plt.hlines([-a, a], 0, T, linestyles='--')
#     plt.grid(True)
#     plt.axis('tight')
#     plt.legend(loc='upper left')
#
#     plt.show()
#
# run()



#
# n = 61
# a = signal.firwin(n, cutoff = 0.3, window = "hamming")
# #Frequency and phase response
# mfreqz(a)
# show()
# #Impulse and step response
# figure(2)
# impz(a)
# show()
#
#
# spectrum = abs(spectrum[: int(round(len(spectrum) / 2))]) ** 2
# power_spectrum_input = spectrum[1:]
# spectrum = fftpack.fft(output)
# spectrum = abs(spectrum[: int(round(len(spectrum) / 2))]) ** 2
# power_spectrum_output = spectrum[1:]
# if len_input == len_output:
#     power_spectrum_result = power_spectrum_output / power_spectrum_input
# frequency_input = (
#     abs(fftpack.fftfreq(len_input, time_step_size))[
#         : int(round(len_input / 2))
#     ]
# )[1:]
# frequency_output = (
#     abs(fftpack.fftfreq(len_output, time_step_size))[
#         : int(round(len_output / 2))
#     ]
# )[1:]
