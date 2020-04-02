import os.path as osp
#import netCDF4
#from netcdf_helpers.reader import say_hello, get_time_series_from_location
#from plot.plot import plot_time_series_for_locations
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import MinMaxScaler
import numpy as np
import xarray as xr
#import pandas as pd
#import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import mglearn

say_hello()

# set a path to the directory containing the data
directory = "D:\HIDA-Datathon-UFZ\hida-datathon-ufz\data"
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

#Understand the data more and see the levels of each column
df = temp_data_r1.to_dataframe()["T2m"]
print(df.index.get_level_values('time'))
timelist = df.index.get_level_values('time')
latlist = df.index.get_level_values('lat')
lonlist = df.index.get_level_values('lon')

#Reset the indices (I find it easier to work this way)
df_r1 = temp_data_r1.to_dataframe().reset_index(level=['lat', 'lon', 'time'])#["T2m"]
#Calculate a global annual mean temperature time series
Globalmeantemp = df_r1.groupby('time').mean()
#Calculate the mean of the time series to focus on the variation from the mean
mean = np.mean(Globalmeantemp["T2m"])
Var_frommean = Globalmeantemp["T2m"] - mean
plt.plot(Var_frommean)

from sklearn.cluster import KMeans
#Initialize the algorithm and fit it with the data
kmeans = KMeans(n_clusters = 5)
X = Var_frommean.to_numpy().reshape(-1,1)
kmeans.fit(X)
print("Cluster memberships:\n{}".format(kmeans.labels_))
#Assign classes to each data point based on the model
classes = kmeans.predict(X)
#Inspect the centroids of the clusters
print(kmeans.cluster_centers_)
#Shortcut to see/visualize the datapoints and range of each cluster
mglearn.discrete_scatter(X, X, kmeans.labels_, markers='o')
#Volcanic activity is expected to have the maximum impact out of all forcings so look for the time points which are in the cluster associated with the lowest centroid
dip = np.argwhere(classes==np.argmin(kmeans.cluster_centers_))
#look for the years which have the biggest dips
dipinyear = list(int(timelist[i][0]/10000) for i in dip)
len(dipinyear)

shortlistedtimeseries = list(timelist[i][0] for i in dip)

#Filter the dataset and look at only the years which have the biggest dips for the data analysis/image analysis
#Also divide it into 6 zones as previously discussed: tropical, temperate and polar in northern and southern hemispheres
df_r1_time = df_r1[df_r1.time.isin(shortlistedtimeseries)]
df_North = df_r1[df_r1.lat>0]
df_North_trop = df_North[df_North.lat<30]
df_North_nottrop = df_North[df_North.lat>30]
df_North_temp = df_North_nottrop[df_North_nottrop.lat<60]
df_North_polar = df_North_nottrop[df_North_nottrop.lat>60]
df_South = df_r1[df_r1.lat<0]
df_South_trop = df_South[df_South.lat>-30]
df_South_nottrop = df_South[df_South.lat<-30]
df_South_temp = df_South_nottrop[df_South_nottrop.lat>-60]
df_South_polar = df_South_nottrop[df_South_nottrop.lat<-60]
