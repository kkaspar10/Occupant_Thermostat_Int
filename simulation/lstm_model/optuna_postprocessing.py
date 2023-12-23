import  pandas as pd
import os
import json

from dotenv import load_dotenv

load_dotenv()

csv_results_path = os.getenv('OPTUNA_CSV_RESULTS_PATH')

if __name__ == "__main__":

    # read the csv file in summary folder
    df = pd.read_csv(os.path.join(csv_results_path, 'summary', 'optuna_best_trial.csv'),delimiter=';', decimal=',')

    # transform the df dataframe into a json file
    # using the Building Key column as the primary key,
    # and all other columns as values, maintaining the name of the column as a key
    optuna_json = {}
    for row in df.to_dict('records'):
        building_key = row.pop('building_key')
        optuna_json[building_key] = row

    print(optuna_json)

    # save the json file
    with open(('simulation\\data\\lstm_pth\\' + 'best_config.json'),
              'w') as fp:
        json.dump(optuna_json, fp)


    # #read all file from csv_dir and merge them into one
    # df = pd.DataFrame()
    # for file in os.listdir(csv_dir):
    #     if file.endswith(".csv"):
    #         df = df.append(pd.read_csv(os.path.join(csv_dir, file)))
    #
    # # save this file as csv
    # df.to_csv(os.path.join(csv_dir, 'optuna_lstm_results_global.csv'), index=False)
    #
    # # for each Building Key from df, plot hte RMSE CL and RMSE val in a scatter plot
    #
    # # For each group, select the row with the minimum RMSE CL value
    # min_rmse_cl = df.groupby('Building Key').apply(lambda x: x.iloc[x['RMSE CL'].idxmin()])
    #
    # # For each group, select the row with the minimum RMSE val value
    # min_rmse_val = df.groupby('Building Key').apply(lambda x: x.iloc[x['RMSE Val'].idxmin()])
    #
    # # For each Building Key, create a json with the optimal trial
    # optuna_json = {199613: 15, 20199: 2, 247942: 20, 4421: 15, 481052: 13,
    #                498771: 16, 508889: 20, 546814: 2, 75252: 8, 79194: 12}
    #
    # # Create a json file for each building selecting from df:
    # # Num_layers, Num_hidden, Dropout, Learning rate
    # # For each building, select the row with the optimal trial from optuna_json
    # optuna_df = pd.DataFrame()
    # for key, value in optuna_json.items():
    #     optuna_df = optuna_df.append(df[(df['Building Key'] == key) & (df['Trial'] == value)])
    #
    # # Create a json file with the optimal configuration for each building
    # opt_config = {}
    # for key, value in optuna_json.items():
    #     opt_config[key] = {'num_layer': optuna_df[optuna_df['Building Key'] == key]['Num_layers'].values[0],
    #                        'hidden_size': optuna_df[optuna_df['Building Key'] == key]['Num_hidden'].values[0],
    #                        # 'dropout': optuna_df[optuna_df['Building Key'] == key]['Dropout'].values[0],
    #                        'learning_rate': optuna_df[optuna_df['Building Key'] == key]['Learning rate'].values[0]}

