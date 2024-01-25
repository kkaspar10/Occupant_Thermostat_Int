import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

csv_path = os.getenv("OPTUNA_CSV_RESULTS_PATH")

# for all files .csv in the folder csv_path, extract a row where there is the min of column "RMSE CL" and a row where there is the min "RMSE Val"
# save them in a new dataframe in iterative way and save it in a new csv file
df_list = []
for file in os.listdir(csv_path):
    if file.endswith(".csv") and "bld" in file:
        df_temp = pd.read_csv(csv_path + "/" + file)
        # From this dataframe, extract the rows were Weight_decay is not nan
        df_temp = df_temp.loc[df_temp['Weight_decay'].notna()]
        df_temp_cl_min = df_temp.loc[df_temp['RMSE CL'] == df_temp['RMSE CL'].min()]
        df_temp_val_min = df_temp.loc[df_temp['RMSE Val'] == df_temp['RMSE Val'].min()]
        df_list.extend([df_temp_cl_min, df_temp_val_min])

df = pd.concat(df_list, ignore_index=True)

# Find rows where 'RMSE CL' is minimized
df_cl = df.loc[df.groupby('Building Key')['RMSE CL'].idxmin()]
df_cl = df_cl[['Building Key', 'Num_hidden', 'Num_layers', 'Dropout', 'Learning rate', 'Weight_decay']]

# Rename columns to match the JSON format
df_cl.rename(columns={'Num_hidden': 'n_hidden', 'Num_layers': 'n_layers', 'Dropout': 'dropout', 'Learning rate': 'learning_rate', 'Weight_decay': 'Weight_decay'}, inplace=True)
df_cl.set_index('Building Key', inplace=True)

# Find rows where 'RMSE Val' is minimized
df_val = df.loc[df.groupby('Building Key')['RMSE Val'].idxmin()]
df_val = df_val[['Building Key', 'Num_hidden', 'Num_layers', 'Dropout', 'Learning rate', 'Weight_decay']]

# Rename columns to match the JSON format
df_val.rename(columns={'Num_hidden': 'n_hidden', 'Num_layers': 'n_layers', 'Dropout': 'dropout', 'Learning rate': 'learning_rate', 'Weight_decay': 'Weight_decay'}, inplace=True)
df_val.set_index('Building Key', inplace=True)

# Create dictionaries directly from DataFrames
best_cl = df_cl.to_dict(orient='index')
best_val = df_val.to_dict(orient='index')

# Save dictionaries in JSON format
import json
with open(csv_path + '/best_cl.json', 'w') as fp:
    json.dump(best_cl, fp)

with open(csv_path + '/best_val.json', 'w') as fp:
    json.dump(best_val, fp)




