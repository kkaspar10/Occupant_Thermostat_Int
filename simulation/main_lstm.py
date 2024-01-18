# One configuration for all my lstm
# I have to set only the configuration setup at each run
# I want run models that have data in input with different timestamp
# I want to choose to include wandb and optuna
# I want to choose to save the model

import random
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import wandb
import pickle
import os
import json

from torch.utils.data import TensorDataset, DataLoader


# IMPORT MY FUNCTIONS AND CLASSES
from lstm_model.classes import *
from lstm_model.config import config
from lstm_model.deployment import closed_loop, CL_log
from lstm_model.testing import testing, test_log
from lstm_model.training import training, train_log, log_metrics
from lstm_model.preprocessing import *
from lstm_model.media_log import *
# from preprocess.import_data import *
# from visualization.media_log_wandb import *
# from pipeline.lstm_pipeline import *

config = AttributeDict(config)

# ENSURE DETERMINISTIC BEHAVIOR
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % config.seed)
np.random.seed(hash("improves reproducibility") % config.seed)
torch.manual_seed(hash("by removing stochasticity") % config.seed)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % config.seed)

# MAPE function
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    absolute_error = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(absolute_error) * 100
    return mape

import seaborn as sns


def plot_average_indoor_temperature_colored(df):
    # Plot the average indoor temperature colored by the simulation_reference
    unique_years = df.year.unique()
    num_years = len(unique_years)
    num_cols = 2  # Number of columns for subplots

    num_rows = (num_years + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))
    axes = axes.flatten()

    for i, year in enumerate(unique_years):
        year_data = df[df['year'] == year]
        unique_sim_refs = year_data.simulation_reference.unique()
        colors = sns.color_palette('tab10', len(unique_sim_refs))

        ax = axes[i]

        for j, sim_ref in enumerate(unique_sim_refs):
            sim_data = year_data[year_data['simulation_reference'] == sim_ref]
            sns.lineplot(data=sim_data, x=sim_data.index, y='average_indoor_air_temperature', color=colors[j], ax=ax,
                         label=str(sim_ref))

        ax.set_xlabel('Index')
        ax.set_ylabel('Average Indoor Temperature')
        ax.set_title(f'Average Indoor Temperature for each Simulation Reference - Year {year}')

    # Set legend outside the subplots
    fig.legend(title='Simulation Reference', loc='center right')

    plt.tight_layout()
    plt.show()

def plot_average_indoor_temperature(df, train_x, val_x, test_x):
    # plot average indoor temperature of df, highlithing train, val and test
    plt.figure(figsize=(20, 10))
    plt.plot(df.average_indoor_air_temperature, label='df')
    plt.plot(np.arange(train_x.shape[0], train_x.shape[0] + val_x.shape[0]),
             df.average_indoor_air_temperature[train_x.shape[0]:train_x.shape[0] + val_x.shape[0]], label='val')
    plt.plot(np.arange(train_x.shape[0] + val_x.shape[0], train_x.shape[0] + val_x.shape[0] + test_x.shape[0]),
             df.average_indoor_air_temperature[
             train_x.shape[0] + val_x.shape[0]:train_x.shape[0] + val_x.shape[0] + test_x.shape[0]], label='test')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # ---------------------------------- IMPORT DATA ---------------------------------------------------------------#
    dict_of_df_path = "simulation\\lstm_model\\dict_building_v2.pickle"

    # if dict_of_df_path exists
    if os.path.exists(dict_of_df_path):
        # open dict_building as pickle
        with open(dict_of_df_path, 'rb') as handle:
            dict_building = pickle.load(handle)
    else:
        # import data
        dir_path = 'simulation\\data\\lstm_train_data\\'

        dict_building = import_data(dir_path)

        # save dict_building as pickle
        with open(dict_of_df_path, 'wb') as handle:
            pickle.dump(dict_building, handle)

    # Open json file with hyperparameters configuration of the lstm
    with open('simulation\\data\\lstm_pth\\' + 'best_config.json') as json_file:
        lstm_config = json.load(json_file)

    # Multiple deployment
    # ------------------------------- ENTER IN MODEL PIPELINE -----------------------------------------------------#
    for key in dict_building:
        config = AttributeDict(config)

        # key = 508889

        df = dict_building[key]

        # How many simulation_reference unique values there are in the df
        print(df.simulation_reference.unique())
        # Return me the number of rows for each simulation_reference and for each year
        print(df.groupby(['simulation_reference', 'year']).size())


        # NA check in df
        # print(df.isnull().sum())

        if config.wandb_on:
            print('This run will be uploaded to WandB servers! \n')
            run_name = str(df.resstock_county_id.unique()[0]) + '_' + str(df.resstock_building_id.unique()[0])
            print(run_name)

            # # tell wandb to get started
            wandb.init(project=config.prj_name,
                       entity=config.entity,
                       config={
                           'batch_size': config.batch_size,
                           'device': config.device,
                           'num_layer': lstm_config[str(key)]['n_layers'],
                           'hidden_size': lstm_config[str(key)]['n_hidden'],
                           'learning_rate': lstm_config[str(key)]['learning_rate'],
                           'dropout': lstm_config[str(key)]['dropout'],
                           'optimizer': config.optimizer_name,
                           'epochs': config.epochs,
                           'run_id': config.run_id
                       },
                       name=run_name,  # If prj_name is CityLearn2.0 use hyperparameters.run_id
                       reinit=True)  # Allow to load multiple running
            # access all HPs through wandb.config, so logging matches execution!
            # config = wandb.config
        else:
            print('This run will be not uploaded to WandB servers! \n'
                  'CHECK if torch.save is turn on if you want save your model. \n')

        # ------------------------------------------------------- DATA PREPARATION ------------------------------------------------------------ #
        # df.columns
        if (df.resstock_county_id == 'TX, Travis County').all():
            lstm_columns = ['direct_solar_radiation',
                            'diffuse_solar_radiation',
                            'outdoor_air_temperature',
                            # 'setpoint',
                            'occupant_count',
                            'cooling_load',
                            # 'ideal_cooling_load_proportion', 'ideal_heating_load_proportion', # there are a lot of nans
                            'sin_month', 'cos_month', 'sin_hour', 'cos_hour', 'sin_DoW', 'cos_DoW',
                            'average_indoor_air_temperature']
        else:
            lstm_columns = ['direct_solar_radiation',
                            'diffuse_solar_radiation',
                            'outdoor_air_temperature',
                            # 'setpoint',
                            'occupant_count',
                            'heating_load',
                            # 'ideal_cooling_load_proportion', 'ideal_heating_load_proportion', # there are a lot of nans
                            'sin_month', 'cos_month', 'sin_hour', 'cos_hour', 'sin_DoW', 'cos_DoW',
                            'average_indoor_air_temperature']

        df_lstm = df[lstm_columns]

        df_lstm = df_lstm.to_numpy().astype(np.float32)

        # STEP 1: PORTIONING - define train, validation and test dataset
        # Plot the average indoor temperature colored by the simulation_reference
        # plot_average_indoor_temperature_colored(df)

        # df has 2 year of data. Use all simulation reference of 2020 for training,
        # simulation reference 2, 3 and 4 of 2021 for validation and simulation reference 5 and 6 of 2021 for testing
        train_indices = ((df['year'] == 2020) & (df['simulation_reference'].isin(['2', '3', '4', '5', '6', '7'])))
        val_indices = ((df['year'] == 2021) & (df['simulation_reference'].isin(['2', '3', '4', '5'])))
        test_indices = ((df['year'] == 2021) & (df['simulation_reference'].isin(['6', '7'])))

        train_df = df[train_indices]

        val_df = df[val_indices]

        test_df = df[test_indices]

        # Select onlu lstm_columns
        train_df = train_df[lstm_columns]
        val_df = val_df[lstm_columns]
        test_df = test_df[lstm_columns]

        # convert to numpy array
        train_data = train_df.to_numpy().astype(np.float32)
        val_data = val_df.to_numpy().astype(np.float32)
        test_data = test_df.to_numpy().astype(np.float32)

        # STEP 2: SLIDING WINDOW - define inputs and outputs matrix for the LSTM
        train_x, train_y = sliding_windows(train_data, seq_length=config.lb, output_len=config.output_pred)
        val_x, val_y = sliding_windows(val_data, seq_length=config.lb, output_len=config.output_pred)
        test_x, test_y = sliding_windows(test_data, seq_length=config.lb, output_len=config.output_pred)

        # ------------------------------------------------------------------------------------------------------------------------------------#
        # Plot the Heating or Cooling Power related to indoor Temperature
        t_out = df_lstm[-(test_x.shape[0]):, lstm_columns.index("outdoor_air_temperature")]

        if 'cooling_load' in lstm_columns:
            p_th = df_lstm[-(test_x.shape[0]):, lstm_columns.index("cooling_load")]
            # fig_db_ax = pth_temp_plot(p_th, test_y, 'Cooling Load [kW]', 'Mean Air Temperature [°C]', title='Thermal Load and Temperature')
            fig_db_ax = pth_temp_in_temp_out_plot(p_th, test_y, t_out,
                                                  'Cooling Load [kW]', 'Indoor Air Temperature [°C]',
                                                  'Outdoor Air Temperature [°C]',
                                                  title='Thermal Load and Temperature')
        else:
            p_th = df_lstm[-(test_x.shape[0]):, lstm_columns.index("heating_load")]
            # fig_db_ax = pth_temp_plot(p_th, test_y, 'Heating Load [kW]', 'Mean Air Temperature [°C]', title='Thermal Load and Temperature')
            fig_db_ax = pth_temp_in_temp_out_plot(p_th, test_y, t_out,
                                                  'Heating Load [kW]', 'Indoor Air Temperature [°C]',
                                                  'Outdoor Air Temperature [°C]',
                                                  title='Thermal Load and Temperature')

        if config.wandb_on:
            # bokeh_log(fig_db_ax, title='Thermal Load and Temperature')
            graph_log(fig_db_ax, title='Thermal Load and Temperature')
            # html_log(fig_db_ax, title='Thermal Load and Temperature')
        # else:
            # fig_db_ax.show()

        # -----------------------------------------------------------------------------------------------------------------------------------#

        # STEP 3: NORMALIZATION - normalize the input and output variable
        maxT = np.max(np.concatenate((train_y, val_y), axis=0)).astype(np.float32)
        minT = np.min(np.concatenate((train_y, val_y), axis=0)).astype(np.float32)

        tr_val = np.concatenate((train_x, val_x), axis=0)

        max_df = np.max(tr_val.reshape(-1, tr_val.shape[-1]), axis=0).astype(np.float32)
        min_df = np.min(tr_val.reshape(-1, tr_val.shape[-1]), axis=0).astype(np.float32)

        # inputs = max_min_norm(inputs, max_df, min_df)
        # labels = max_min_norm(labels, maxT, minT)
        train_x = max_min_norm(train_x, max_df, min_df)
        train_y = max_min_norm(train_y, maxT, minT)
        val_x = max_min_norm(val_x, max_df, min_df)
        val_y = max_min_norm(val_y, maxT, minT)
        test_x = max_min_norm(test_x, max_df, min_df)
        test_y = max_min_norm(test_y, maxT, minT)

        # create a dataframe where columns are the lstm_columns and rows are the max_df array
        max_df = pd.DataFrame(max_df.reshape(1, max_df.size), index=df.resstock_building_id.unique(), columns=lstm_columns)
        min_df = pd.DataFrame(min_df.reshape(1, min_df.size), index=df.resstock_building_id.unique(), columns=lstm_columns)

        # create a csv file for max_df and one for min_df. At each step of the loop,
        # concatenate the new max_df and min_df to the previous ones
        # Write the header to the CSV file only if the file doesn't exist
        if not os.path.exists('simulation\\data\\max.csv'):
            max_df.to_csv('simulation\\data\\max.csv', index=True, header=True)
        else:
            max_df.to_csv('simulation\\data\\max.csv', index=True, header=False, mode='a')

        # Write the header to the CSV file only if the file doesn't exist
        if not os.path.exists('simulation\\data\\min.csv'):
            min_df.to_csv('simulation\\data\\min.csv', index=True, header=True)
        else:
            min_df.to_csv('simulation\\data\\min.csv', index=True, header=False, mode='a')

        # STEP 4: DATALOADER - define train and test dataset as a dataloader
        # df_dataset, df_loader = dataset_dataloader(inputs, labels, config.batch_size, shuffle=False)
        train_dataset, train_loader = dataset_dataloader(train_x, train_y, config.batch_size, shuffle=True)
        validation_dataset, validation_loader = dataset_dataloader(val_x, val_y, config.batch_size, shuffle=False)
        test_data, test_loader = dataset_dataloader(test_x, test_y, config.batch_size, shuffle=False)

        # STEP 5: MODEL - define the model
        # INPUT & OUTPUT SIZE
        input_size = train_x.shape[-1]
        output_size = train_y.shape[-1]

        # define the lstm model with the hyperparameters from lstm_config
        # lstm = LSTM_model_optuna(n_features=input_size,
        #                         n_output=output_size,
        #                         seq_len=config.lb,
        #                         num_layers=lstm_config[str(key)]['n_layers'],
        #                         num_hidden=lstm_config[str(key)]['n_hidden'],
        #                         drop_prob=lstm_config[str(key)]['dropout']).to(config.device)
        #
        lstm = LSTM_attention(n_features=input_size,
                                n_output=output_size,
                                seq_len=config.lb,
                                num_layers=lstm_config[str(key)]['n_layers'],
                                num_hidden=lstm_config[str(key)]['n_hidden'],
                                drop_prob=lstm_config[str(key)]['dropout']).to(config.device)


         # lstm = LSTM_model_wandb(n_features=input_size,
        #                         n_output=output_size,
        #                         seq_len=config.lb).to(config.device)

        # Define the loss and optimizer
        criterion = torch.nn.MSELoss()  # mean-squared error for regression
        optimizer = getattr(torch.optim, config.optimizer_name)(lstm.parameters(), lr=lstm_config[str(key)]['learning_rate'])

        if config.wandb_on:
            # Tell wandb to watch what the model gets up to: gradients, weights, and more!
            wandb.watch(lstm, criterion, log="all", log_freq=10)

        # ---------------------------------------------------------- TRAINING ------------------------------------------------------------------- #
        LOSS_TRAIN = []
        LOSS_VAL = []
        for epoch in range(config.epochs):
            loss_train, loss_val, \
                ylab_train, ypred_train, \
                ylab_val, ypred_val = training(model=lstm, train_loader=train_loader, val_loader=validation_loader,
                                               optimizer=optimizer, criterion=criterion, config=config,
                                               maxT=maxT, minT=minT)

            LOSS_TRAIN.append(loss_train)
            LOSS_VAL.append(loss_val)

            if config.wandb_on:
                train_log(train_loss=LOSS_TRAIN, val_loss=LOSS_VAL, epoch=epoch)
            else:
                if epoch % config.log_interval == 0:
                    print("Epoch: %d, Train loss: %1.5f, Val loss: %1.5f" % (epoch, LOSS_TRAIN[epoch], LOSS_VAL[epoch]))

        # METRICS
        MAPE_train = mean_absolute_percentage_error(ylab_train, ypred_train)
        RMSE_train = mean_squared_error(ylab_train, ypred_train, squared=False)

        MAPE_val = mean_absolute_percentage_error(ylab_val, ypred_val)
        RMSE_val = mean_squared_error(ylab_val, ypred_val, squared=False)

        if config.wandb_on:
            log_metrics(MAPE=MAPE_train, RMSE=RMSE_train, set='Train')
            log_metrics(MAPE=MAPE_val, RMSE=RMSE_val, set='Validation')

        # --------------------------------------------------------- SAVE MODEL ----------------------------------------------------------------- #
        run_name = df['resstock_county_id'].unique()[0] + '_' + str(df['resstock_building_id'].unique()[0])

        print('Il nome del file è', run_name)

        if config.save_local:
            checkpoint = {
                'model_state_dict': lstm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'train_loss': LOSS_TRAIN[-1],
                'val_loss': LOSS_VAL[-1],
            }
            save_folder = 'simulation\\data\\lstm_pth'

            save_path = save_folder + '\\' + 'model_pth_' + run_name + '.pth'

            torch.save(checkpoint, save_path)
            print('File saved correctly!')

        # ----------------------------------------------------------- LOAD MODEL ------------------------------------------------------------ #
        if config.load_model:
            model_path = save_folder + '\\' + 'model_pth_' + 'VT, Chittenden County_75252.pth'
        else:
            model_path = None

        # ----------------------------------------------------------- TESTING ------------------------------------------------------------------ #
        LOSS_TEST, ylab_test, ypred_test = testing(model=lstm, model_path=model_path, test_loader=test_loader,
                                                   optimizer=optimizer,
                                                   criterion=criterion,
                                                   config=config, maxT=maxT, minT=minT, ylab=[], ypred=[])


        # METRICS
        MAPE_test = mean_absolute_percentage_error(ylab_test, ypred_test)
        RMSE_test = mean_squared_error(ylab_test, ypred_test, squared=False)
        R2_test = r2_score(ylab_test, ypred_test)

        if config.wandb_on:
            test_log(test_loss=LOSS_TEST, MAPE=MAPE_test, RMSE=RMSE_test)

        # LOSS_DF, ylab_df, ypred_df = testing(model=lstm, model_path=model_path, test_loader=df_loader,
        #                                      optimizer=optimizer,
        #                                      criterion=criterion,
        #                                      config=config, maxT=maxT, minT=minT, ylab=[], ypred=[])
        #
        # # METRICS
        # MAPE_DF = mean_absolute_percentage_error(ylab_test, ypred_test)
        # RMSE_DF = mean_squared_error(ylab_test, ypred_test, squared=False)

        print('Final Loss Train: {:.5f}'.format(LOSS_TRAIN[-1]))
        print('Final Loss Validation: {:.5f}'.format(LOSS_VAL[-1]))
        print('Final Loss Test: {:.5f}'.format(LOSS_TEST))
        # print('Global Loss DF: {:.5f}'.format(LOSS_DF))

        print('MAPE Test: %0.5f%%' % MAPE_test)
        print('RMSE Test: {:.5f}'.format(RMSE_test.item()))
        print('R2 Test: {:.5f}'.format(R2_test.item()))
        # print('MAPE Global: %0.5f%%' % MAPE_DF)
        # print('RMSE Global: {:.5f}'.format(RMSE_DF.item()))

        # --------------------------------------------------------- CLOSED LOOP ----------------------------------------------------------------------#
        if config.load_model:
            model_path = save_folder + '\\' + 'model_pth_' + 'VT, Chittenden County_100753.pth'
        else:
            model_path = None

        Tpred, Treal = closed_loop(model=lstm, optimizer=optimizer, model_path=model_path,
                                   test_x=test_x, test_y=test_y, config=config, maxT=maxT, minT=minT)

        # METRICS
        MAPE_sim = mean_absolute_percentage_error(Treal, Tpred)
        RMSE_sim = mean_squared_error(Treal, Tpred) ** 0.5
        R2_sim = r2_score(Treal, Tpred)
        print('MAPE_sim:%0.5f%%' % MAPE_sim)
        print('RMSE_sim: {:.5f}'.format(RMSE_sim.item()))
        print('R2_sim: {:.5f}'.format(R2_sim.item()))

        if config.wandb_on:
            CL_log(MAPE=MAPE_sim, RMSE=RMSE_sim, R2=R2_sim)


        # ------------------------------------------- PLOTTING --------------------------------------------------------#

        fig_test = plot_graph(ypred=ypred_test, ylab=ylab_test, config=config,
                              title='Temperatue prediction TESTING SET- %d T lag for next %d T' % (
                                  config.lb, config.output_pred))

        # fig_all_df = plot_graph(ypred=ypred_df, ylab=ylab_df, config=config,
        #                         title='Temperatue prediction ALL DATASET - %d T lag for next %d T' % (
        #                             config.lb, config.output_pred))

        error_dist = error_distribution(ypred=ypred_test, yreal=ylab_test)

        plot_test_CL = plot_graph(Tpred, Treal, config=config,
                                  title='Simulation Closed Loop LSTM in Test')

        scatter_CL = plot_scatter(Tpred, Treal)

        if config.wandb_on:
            graph_log(fig_test, title='Indoor Temperature Test Simulation')
            # graph_log(fig_all_df, title='Indoor Temperature all Simulation')
            image_log(error_dist, title='Error distribution')
            graph_log(plot_test_CL, title='Indoor Temperature in Closed Loop')
            image_log(scatter_CL, title='CL Predicted vs Actual Indoor air temperature')
        else:
            fig_test.show()
            # fig_all_df.show()
            plot_test_CL.show()
            scatter_CL.show()





