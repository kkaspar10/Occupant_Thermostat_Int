#!/usr/bin/env python
# coding: utf-8

# Modified by Kathryn Kaspar on 06-17-2023
# Original code by Saptak Dutta from https://github.com/saptakdutta

#%% Libraries
import pandas as pd, numpy as np, requests as req, json
from pathlib import Path
import csv


# Must repeat all processes below if multiple years are required.
#Adjust lines 38, 39, 57 & 60

# Available time formats: LST/UTC
timeformat = 'LST'

# check https://power.larc.nasa.gov/#resources for full list of parameters and modify as needed

# ALLSKY_SFC_SW_DNI - direct normal irradiance (W/m2)
# ALLSKY_SFC_SW_DIFF - diffuse horizontal irradiance (W/m2)
# ALLSKY_SFC_SW_DWN - global horizontal irradiance (W/m2)
# T2M - temperature 2 m above ground (degC)
# RH2M - relative humidity 2 m above ground level (%)
# WS2M - wind speed 2 m above ground level (m/s)

params = 'T2M,RH2M,ALLSKY_SFC_SW_DNI,ALLSKY_SFC_SW_DIFF'
#Always use RE (renewable energy) for this purpose
community = 'RE' 
#Obtain LAT/LON from google maps, Austin TX
location = {
    'latitude':'45.4656851',
    'longitude':'-73.7480617'
    }
# Start/end time in format: 'YYYYMMDD'
##NOTE: If using predictions (e.g. 24 h ahead, query another day or two (as needed) beyond the end date of simulation)
sTime = '20220101'
eTime = '20220402'


#%% API call for given lat/long
cwd = Path.cwd()
path = cwd.__str__()
url = 'https://power.larc.nasa.gov/api/temporal/hourly/point?Time='+timeformat+'&parameters='+params+'&community='+community+'&longitude='+location['longitude']+'&latitude='+location['latitude']+'&start='+sTime+'&end='+eTime+'&format=JSON'
data = req.get(url)

data = data.json()
data = pd.DataFrame((data['properties']['parameter']))
data = data.set_index(pd.to_datetime(data.index, format='%Y%m%d%H'))


# Convert the index to a datetime format
data.index = pd.to_datetime(data.index)

# Filter for relevant dates
filtered_df = data.loc['2022-01-01 00:00:00':'2022-04-01 23:00:00']


filtered_df.to_parquet(path+'/weather_2022.parquet')