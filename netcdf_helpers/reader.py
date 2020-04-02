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

    def getclosest_ij(latarray,longarray,lat_target,long_target):
        # find squared distance of every point on grid
        leastdist = 1000000 #initial random value
        latindx = np.arange(0,len(latarray),1)
        lonindx = np.arange(0,len(longarray),1)
        for y,x in zip(latindx, lonindx):
            dist_sq = (latarray[y]-lat_target)**2 + (longarray[x]-long_target)**2 #calculate square of distance
#            print(dist_sq)
            dist_sq_old = dist_sq #move to another variable to enable comparison wiht next calculation
            if dist_sq < leastdist or dist_sq < dist_sq_old: #if the new value is less than old calculation or the least distance calculated so far
                leastdist = dist_sq
                leastlatidx = y
                leastlongidx = x
                leastdist = dist_sq #then save the indices associated with this least value of distance
#                print(leastdist, latarray[y], longarray[x])  
        print ("The closest coordinates are: "+ str(leastdist)+ " meters away")
        return leastlatidx, leastlongidx
    # get the index for target point
#    lat_indx_target = near(lat, lat_target)
#    lon_indx_target = near(lon, lon_target)
    lat_indx_target, lon_indx_target = getclosest_ij(lat, lon, lat_target, lon_target)

    # get the value of that index
    lat_val_target = lat[lat_indx_target]
    lon_val_target = lon[lon_indx_target]
    closest = (lat_val_target, lon_val_target)

    # get the corresponding temp time series at target location
    time_series = var[:, lat_val_target, lon_val_target]
    return time_series, closest
