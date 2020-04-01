# -*- coding: utf-8 -*-
import os.path as osp
import netCDF4
from netcdf_helpers.reader import say_hello, get_time_series_from_location
from plot.plot import plot_time_series_for_locations
# example script to load own modules

say_hello()

# set a path to the directory containing the data
directory = "/Users/houben/phd/hackathons/hida_datathon/data/MyChallengePaleo"
# set the file names
filename_temp_data_r1 = "T2m_R1_ym_1stMill.nc"
# load netCDF
temp_data_r1 = netCDF4.Dataset(osp.join(directory, filename_temp_data_r1), "r")
# set coordinate from desired location
lat_target = 65
lon_target = 44
# returns a time series at closest location
time_series, closest = get_time_series_from_location(
    temp_data_r1, "T2m", lat_target, lon_target
)

# plot example
names = ["a", "b", "c"]
time_series_list = [time_series, time_series]
time = temp_data_r1.variables['time'][:]
plot_time_series_for_locations(time_series_list, time, names)
