# -*- coding: utf-8 -*-
import netCDF4
import numpy as np
import os.path as osp


def say_hello():
    print("hello")


def get_time_series_from_location(data, var_name, lat_target, lon_target):
    """
    Get a time series for a variable at a specific location with the
    corresponding time.

    Parameters
    ----------
    data : <class 'netCDF4._netCDF4.Dataset'>
        netCDF4-file
    var_name : string
        Name of the variable to look for.
    lat_target : Float
        Latitude  value of desired location.
    lon_target : Float
        Longitude value of desired location.

    Returns
    -------

    time_series : 1D-array
        Time series of variable at given location.
    closest :
    """

    var = data.variables[var_name]
    lat = data.variables["lat"][:]
    lon = data.variables["lon"][:]

    def near(array, value):
        diff_array = np.abs(array - value)
        id = np.argmin(diff_array)
        return id

    # get the index for target point
    lat_indx_target = near(lat, lat_target)
    lon_indx_target = near(lon, lon_target)

    # get the value of that index
    lat_val_target = lat[lat_indx_target]
    lon_val_target = lon[lon_indx_target]
    closest = (lat_val_target, lon_val_target)

    # get the corresponding temp time series at target location
    time_series = var[:, lat_val_target, lon_val_target]
    return time_series, closest
