# -*- coding: utf-8 -*-
import netCDF4
import numpy as np
import os.path as osp


# Examples how to work with netCDF4 (derived from the official netCDF4 GitRepo)
# https://github.com/Unidata/netcdf4-python/blob/master/examples/reading_netCDF.ipynb
# -----------------------------------------------------------------------------

# set a path to the directory containing the data
directory = "/Users/houben/phd/hackathons/hida_datathon/data/MyChallengePaleo"
# set the file names
filename_temp_data_r1 = "T2m_R1_ym_1stMill.nc"
filename_temp_data_r2 = "T2m_R1_ym_1stMill.nc"

# Load input files
temp_data_r1 = netCDF4.Dataset(osp.join(directory, filename_temp_data_r1), "r")
temp_data_r2 = netCDF4.Dataset(osp.join(directory, filename_temp_data_r2), "r")

# get all variable names
temp_data_r1.variables.keys()
temp_data_r2.variables.keys()

# assign temp to a variable
temp_r1 = temp_data_r1.variables["T2m"]
temp_r2 = temp_data_r2.variables["T2m"]

# List the dimensions
for dim in temp_data_r1.dimensions.items():
    print(dim)

temp_r1.dimensions
temp_r1.shape

# Each dimension typically has a variable associated with it (called a coordinate variable).
# Coordinate variables are 1D variables that have the same name as dimensions.
# Coordinate variables and auxiliary coordinate variables (named by the coordinates attribute) locate values in time and space.

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
tempslice = temp_r1[time < 20716, lat > 86.72253095, lon < 1.875]


# What is the temperature at a certain (arbitrary) location?
target = (85, 17)
# We need to find the closest point in the dataset:
# function to find index to nearest point
def near(array, value):
    diff_array = np.abs(array - value)
    id = np.argmin(diff_array)
    return id


# get the index for target point
lat_indx_target = near(lat, target[0])
lon_indx_target = near(lon, target[1])

# get the value of that index
lat_val_target = lat[lat_indx_target]
lon_val_target = lon[lon_indx_target]

# get the corresponding temp time series at target location
temp_target = temp_r1[:, lat_val_target, lon_val_target]
