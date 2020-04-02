# -*- coding: utf-8 -*-
import os.path as osp
import netCDF4
from netcdf_helpers.reader import say_hello, get_time_series_from_location
from plot.plot import plot_time_series_for_locations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import xarray as xr
import pandas as pd
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt

say_hello()

# set a path to the directory containing the data
directory = "/Users/houben/phd/hackathons/hida_datathon/data/MyChallengePaleo"
# set the file names
filename_temp_data_r1 = "T2m_R1_ym_1stMill.nc"
filename_temp_data_r2 = "T2m_R2_ym_1stMill.nc"
filename_solar_data = "Solar_forcing_1st_mill.nc"
filename_volc_data = "Volc_Forc_AOD_1st_mill.nc"
# load netCDF
#temp_data_r1 = netCDF4.Dataset(osp.join(directory, filename_temp_data_r1), "r")
#temp_data_r2 = netCDF4.Dataset(osp.join(directory, filename_temp_data_r2), "r")
temp_data_r1 = xr.open_dataset(osp.join(directory, filename_temp_data_r1))
temp_data_r2 = xr.open_dataset(osp.join(directory, filename_temp_data_r2))

df_r1 = temp_data_r1.to_dataframe()["T2m"]
df_r1.index.names
print(df_r1.index.get_level_values('time'))
timelist = df_r1.index.get_level_values('time')

Globalmeantemp = df_r1.groupby('time').mean()
mean = np.mean(Globalmeantemp)
Var_frommean = Globalmeantemp - mean
plt.plot(Globalmeantemp)
plt.plot(Var_frommean)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 5)
X = Var_frommean.to_numpy().reshape(-1,1)
kmeans.fit(X)
kmeans.cluster_centers_
print("Cluster memberships:\n{}".format(kmeans.labels_))
classes = kmeans.predict(X)
dip = np.argwhere(classes==4)
dipinyear = list(int(timelist[i][0]/10000) for i in dip)
len(dipinyear)
print(kmeans.predict(X))

import mglearn
mglearn.discrete_scatter(X, X, kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_, kmeans.cluster_centers_, [0,1,2], markers='^', markeredgewidth=2)

X = df_r1.loc[df_r1.time < 30]
kmeans.fit(X["T2m"].to_numpy().reshape(-1,1))
mglearn.discrete_scatter(X["lon"], X["lat"], kmeans.labels_, markers='o')
mglearn.discrete_scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], [0,1,2], markers='^', markeredgewidth=2)

print("Cluster memberships:\n{}".format(kmeans.labels_))
classes = kmeans.predict(X)
dip = np.argwhere(classes==2)
dipinyear = list(timelist[i] for i in dip)
print(kmeans.predict(X))
