#!/usr/bin/env python
# coding: utf-8

# Created by Kathryn Kaspar on 06-17-2023
#virtual environment to run this code: ecobee1

import pandas as pd
import numpy as np
import openpyxl
import pyarrow as pa
import pyarrow.parquet as pq
import glob
import os
from pandas.io.formats.excel import ExcelFormatter
from pandas.io import gbq
from openpyxl.workbook import Workbook
from shutil import copyfile
from pathlib import Path


# use glob to get all the parquet files in the folder
# one parquet file should be created for each period/year from 1.NOAA_Weather_api_QC.py

path = os.getcwd()
parquet_files = glob.glob(os.path.join(path, "*.parquet"))


# Read each parquet file into DataFrame
##Note: 2020 is a leap year, so the lengths of the dataframe will be different
#I separated 2020 first and performed the below code and then did the same code for 2021, 2022 subsequently
# Dataframe lengths: 2184 for 2020, 2160 for 2021/2022
#Should have been written into the script, but as it stands now lines 44-49, 52, and 95 need to be edited depending on if leap year or not
#Edit lines 98,99 depending on file names
for file in parquet_files:
    df = (pd.read_parquet(file))
    df = df.reset_index().rename(columns={'index': 'datetime'})
    df = df.rename(columns={'T2M': 'Outdoor Drybulb Temperature [C]', 'RH2M': 'Outdoor Relative Humidity [%]', 'ALLSKY_SFC_SW_DNI': 'Direct Solar Radiation [W/m2]', 'ALLSKY_SFC_SW_DIFF': 'Diffuse Solar Radiation [W/m2]'})
    
    column_names = df.columns.tolist()  # Get the list of column names
    column_names[4], column_names[3] = column_names[3], column_names[4]  # Switch the positions of Column 3 and Column 4 just to match CityLearn sample weather files
    df = df.reindex(columns=column_names)
    df_2 = df.copy()
    
    #generate noise for each year file
    noise_T_6H = np.random.uniform(low=-0.3, high=0.3, size=(2184,))
    noise_T_12H = np.random.uniform(low=-0.65, high=0.65, size=(2184,))
    noise_T_24H = np.random.uniform(low=-1.35, high=1.35, size=(2184,))
    noise_RH_6H = np.random.uniform(low=-2.5, high=2.5, size=(2184,))
    noise_RH_12H = np.random.uniform(low=-5.0, high=5.0, size=(2184,))
    noise_RH_24H = np.random.uniform(low=-10.0, high=10.0, size=(2184,))

    for i in range(0,2184):
        df_2.loc[i, '6h Prediction Outdoor Drybulb Temperature [C]'] = df_2.loc[i+6, 'Outdoor Drybulb Temperature [C]'] + noise_T_6H[i]
        df_2.loc[i, '12h Prediction Outdoor Drybulb Temperature [C]'] = df_2.loc[i+12, 'Outdoor Drybulb Temperature [C]'] + noise_T_12H[i]
        df_2.loc[i, '24h Prediction Outdoor Drybulb Temperature [C]'] = df_2.loc[i+24, 'Outdoor Drybulb Temperature [C]'] + noise_T_24H[i]
    
        df_2.loc[i, '6h Prediction Outdoor Relative Humidity [%]'] = df_2.loc[i+6, 'Outdoor Relative Humidity [%]'] + noise_RH_6H[i]
        df_2.loc[i, '12h Prediction Outdoor Relative Humidity [%]'] = df_2.loc[i+12, 'Outdoor Relative Humidity [%]'] + noise_RH_12H[i]
        df_2.loc[i, '24h Prediction Outdoor Relative Humidity [%]'] = df_2.loc[i+24, 'Outdoor Relative Humidity [%]'] + noise_RH_24H[i]
        
        ###generate noise for diffuse/direct, which is a percentage of what the future reading is.
        value_twop = df_2.loc[i+6, 'Diffuse Solar Radiation [W/m2]']
        two_p = value_twop*0.025
        noise_DIF_6H = np.random.uniform(low=-two_p, high=two_p, size=(1,))
        
        value_fivep = df_2.loc[i+12, 'Diffuse Solar Radiation [W/m2]']
        five_p = value_fivep*0.05
        noise_DIF_12H = np.random.uniform(low=-five_p, high=five_p, size=(1,))
        
        value_tenp = df_2.loc[i+24, 'Diffuse Solar Radiation [W/m2]']
        ten_p = value_tenp*0.1
        noise_DIF_24H = np.random.uniform(low=-ten_p, high=ten_p, size=(1,))
        
        df_2.loc[i, '6h Prediction Diffuse Solar Radiation [W/m2]'] = df_2.loc[i+6, 'Diffuse Solar Radiation [W/m2]'] + noise_DIF_6H[0]
        df_2.loc[i, '12h Prediction Diffuse Solar Radiation [W/m2]'] = df_2.loc[i+12, 'Diffuse Solar Radiation [W/m2]'] + noise_DIF_12H[0]
        df_2.loc[i, '24h Prediction Diffuse Solar Radiation [W/m2]'] = df_2.loc[i+24, 'Diffuse Solar Radiation [W/m2]'] + noise_DIF_24H[0]
        
        value_twop_dir = df_2.loc[i+6, 'Direct Solar Radiation [W/m2]']
        two_p_dir = value_twop_dir*0.025
        noise_DIR_6H = np.random.uniform(low=-two_p_dir, high=two_p_dir, size=(1,))
        
        value_fivep_dir = df_2.loc[i+12, 'Direct Solar Radiation [W/m2]']
        five_p_dir = value_fivep_dir*0.05
        noise_DIR_12H = np.random.uniform(low=-five_p_dir, high=five_p_dir, size=(1,))
        
        value_tenp_dir = df_2.loc[i+24, 'Direct Solar Radiation [W/m2]']
        ten_p_dir = value_tenp_dir*0.1
        noise_DIR_24H = np.random.uniform(low=-ten_p_dir, high=ten_p_dir, size=(1,))
        
        df_2.loc[i, '6h Prediction Direct Solar Radiation [W/m2]'] = df_2.loc[i+6, 'Direct Solar Radiation [W/m2]'] + noise_DIR_6H[0]
        df_2.loc[i, '12h Prediction Direct Solar Radiation [W/m2]'] = df_2.loc[i+12, 'Direct Solar Radiation [W/m2]'] + noise_DIR_12H[0]
        df_2.loc[i, '24h Prediction Direct Solar Radiation [W/m2]'] = df_2.loc[i+24, 'Direct Solar Radiation [W/m2]'] + noise_DIR_24H[0]
        
    end = df_2.shape[0]  #total number of rows to subtract last rows that are not needed
    df_2.drop(df_2.index[2184:end], inplace=True)
    
    file_name_new = os.path.basename(file)
    file_name_new_2 = file_name_new[:-8] + '_Final.parquet'
    folder_path = '/root/ecobee_OccupancyProfiles/Weather_Files/'
    file_path = os.path.join(folder_path, file_name_new_2)
    df_2.to_parquet(file_path)

