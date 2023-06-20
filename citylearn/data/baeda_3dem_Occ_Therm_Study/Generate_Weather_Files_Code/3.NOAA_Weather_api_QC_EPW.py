#!/usr/bin/env python
# coding: utf-8

# Modified by Kathryn Kaspar on 06-17-2023
# Original code by Saptak Dutta from https://github.com/saptakdutta

#%% Libraries
import pandas as pd, numpy as np, requests as req, json
from pathlib import Path
import csv
import os

# Must repeat all processes below if multiple years are required.
#Adjust lines 38, 39, 53 & 60
#This time, output format specified in Line 48 as "EPW"

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
sTime = '20220101'
eTime = '20220401'


# In[35]:


#%% API call for given lat/long
cwd = Path.cwd()
path = cwd.__str__()
url = 'https://power.larc.nasa.gov/api/temporal/hourly/point?Time='+timeformat+'&parameters='+params+'&community='+community+'&longitude='+location['longitude']+'&latitude='+location['latitude']+'&start='+sTime+'&end='+eTime+'&format=EPW'
data = req.get(url)

#Write data to output file & specify output path
path = os.getcwd()
output_file = path + '/WeatherFile_EPW_2022.epw'  #Change file name as desired

# Save the content of the response to a file
with open(output_file, 'w') as file:
    file.write(data.text)