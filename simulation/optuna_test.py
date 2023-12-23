import optuna
import torch
from optuna.trial import TrialState
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

from lstm_model.classes import *
from lstm_model.config import config
from lstm_model.deployment import closed_loop, CL_log
from lstm_model.testing import testing, test_log
from lstm_model.training import training, train_log, log_metrics
from lstm_model.preprocessing import *
from lstm_model.media_log import *

import os
from dotenv import load_dotenv

load_dotenv()

csv_results_path = os.getenv('OPTUNA_CSV_RESULTS_PATH')

config = AttributeDict(config)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    absolute_error = np.abs((y_true - y_pred) / y_true)
    mape = np.mean(absolute_error) * 100
    return mape

# DEFINE MODEL FUNCTON
def define_model(trial, config, key, dict_building):

    # ------------------------------- ENTER IN MODEL PIPELINE -----------------------------------------------------#

    # bld_key = 508889
    bld_key = key

    df = dict_building[bld_key]

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
                   config=config,
                   name=run_name,  # If prj_name is CityLearn2.0 use hyperparameters.run_id
                   reinit=True)  # Allow to load multiple running
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
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

    # STEP 4: DATALOADER - define train and test dataset as a dataloader
    # df_dataset, df_loader = dataset_dataloader(inputs, labels, config.batch_size, shuffle=False)
    train_dataset, train_loader = dataset_dataloader(train_x, train_y, config.batch_size, shuffle=True)
    validation_dataset, validation_loader = dataset_dataloader(val_x, val_y, config.batch_size, shuffle=False)
    test_data, test_loader = dataset_dataloader(test_x, test_y, config.batch_size, shuffle=False)

    # STEP 5: MODEL - define the model
    # INPUT & OUTPUT SIZE
    input_size = train_x.shape[-1]
    output_size = train_y.shape[-1]


    # 1. DEFINE THE MODEL
    # define the hyperparameters
    num_hidden = trial.suggest_int('n_units', 8, 64, step=8)
    num_layers = trial.suggest_int('n_layers', 1, 3, step=1)
    dropout_prob = trial.suggest_float('drop_prob', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('lr', 1e-6, 1e-2)
    # optimizer = trial.suggest_categorical('optimizer', ["Adam", "RMSprop", "L-BFGS"])

    # define the model
    lstm = LSTM_model_optuna(n_features=input_size,
                             n_output=output_size,
                             drop_prob=dropout_prob,
                             seq_len=config.lb,
                             num_hidden=num_hidden,
                             num_layers=num_layers).to(config.device)

    # 2. DEFINE THE LOSS FUNCTION AND THE OPTIMIZER
    # Make the loss and optimizer
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = getattr(torch.optim, config.optimizer_name)(lstm.parameters(), lr=learning_rate)

    return lstm, train_loader, validation_loader, test_loader, criterion, optimizer, maxT, minT, test_x, test_y, bld_key


# Create an Optuna study object to find the optimal hyperparameters
def objective(trial, config, key, dict_building):
    # 1. DEFINE THE MODEL
    lstm, train_loader, validation_loader, test_loader, criterion, optimizer, maxT, minT, test_x, test_y, bld_key = define_model(
        trial, config, key, dict_building)

    n_trial = trial.number + 1

    # 2. TRAINING
    for epoch in range(config.epochs):
        LOSS_TRAIN, LOSS_VAL, \
            ylab_train, ypred_train, \
            ylab_val, ypred_val = training(model=lstm, train_loader=train_loader, val_loader=validation_loader,
                                           optimizer=optimizer, criterion=criterion, config=config,
                                           maxT=maxT, minT=minT)

        if epoch % config.log_interval == 0:
            print("Epoch: %d, Train loss: %1.5f, Val loss: %1.5f" % (epoch, LOSS_TRAIN, LOSS_VAL))
    # METRICS
    MAPE_train = mean_absolute_percentage_error(ylab_train, ypred_train)
    RMSE_train = mean_squared_error(ylab_train, ypred_train, squared=False)

    MAPE_val = mean_absolute_percentage_error(ylab_val, ypred_val)
    RMSE_val = mean_squared_error(ylab_val, ypred_val, squared=False)

    if config.wandb_on:
        log_metrics(MAPE=MAPE_train, RMSE=RMSE_train, set='Train')
        log_metrics(MAPE=MAPE_val, RMSE=RMSE_val, set='Validation')

    # 3. TESTING

    model_path = None

    LOSS_TEST, ylab_test, ypred_test = testing(model=lstm, model_path=model_path, test_loader=test_loader,
                                               optimizer=optimizer,
                                               criterion=criterion,
                                               config=config, maxT=maxT, minT=minT, ylab=[], ypred=[])

    MAPE_test = mean_absolute_percentage_error(ylab_test, ypred_test)
    RMSE_test = mean_squared_error(ylab_test, ypred_test, squared=False)

    # 4. CLOSED LOOP
    Tpred, Treal = closed_loop(model=lstm, optimizer=optimizer, model_path=model_path,
                               test_x=test_x, test_y=test_y, config=config, maxT=maxT, minT=minT)

    MAPE_sim = mean_absolute_percentage_error(Treal, Tpred)
    RMSE_sim = mean_squared_error(Treal, Tpred) ** 0.5
    R2_sim = r2_score(Treal, Tpred)


    # 7. SAVING
    name = f'LSTM_optuna_bld_{bld_key}_trial_{n_trial}.pth'
    basedir = 'simulation\\lstm_model\\optuna_model\\'

    checkpoint = {
        'model_state_dict': lstm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': LOSS_TRAIN,
        'val_loss': LOSS_VAL
    }

    torch.save(checkpoint,  os.path.join(basedir, name))
    print('File saved correctly!')

    # 8. HYPERPARAMETERS DATAFRAME
    # MAKING DF OH HYPERPARAMETERS
    hyperP = pd.DataFrame([[bld_key, n_trial, lstm.n_layers, lstm.n_hidden, config.batch_size,
                            optimizer.defaults['lr'], config.epochs, lstm.dropout.p,
                            LOSS_VAL, LOSS_TEST,
                            MAPE_train, RMSE_train,
                            MAPE_val, RMSE_val,
                            MAPE_test, RMSE_test,
                            MAPE_sim, RMSE_sim, R2_sim]],
                          columns=['Building Key', 'Trial', 'Num_layers', 'Num_hidden', 'Batch size',
                                   'Learning rate', 'Epochs', 'Dropout',
                                   'Validation Loss', 'Test Loss',
                                   'MAPE Train', 'RMSE Train',
                                   'MAPE Val', 'RMSE Val',
                                   'MAPE Test', 'RMSE Test',
                                   'MAPE CL', 'RMSE CL', 'R2 CL'])

    runsLogName = csv_results_path + f'\\optuna_lstm_results_bld_{bld_key}.csv'

    if not os.path.exists(runsLogName):
        hyperP.to_csv(path_or_buf=runsLogName,
                      sep=',', decimal='.', index=False)
    else:
        runsLog = pd.read_csv(runsLogName, sep=',', decimal='.')
        runsLog = runsLog.append(hyperP)
        runsLog.to_csv(path_or_buf=runsLogName,
                       sep=',', decimal='.', index=False)

    return RMSE_val, RMSE_sim

if __name__ == "__main__":

    # import configuration file
    config = AttributeDict(config)

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

    # Create a study object and optimize the objective function.
    for building_key in dict_building.keys():
        study = optuna.create_study(directions=["minimize", "minimize"])
        study.optimize(lambda trial: objective(trial, config, building_key, dict_building), n_trials=20, timeout=None)

        # optuna.visualization.plot_pareto_front(study, target_names=["RMSE_train", "RMSE_val"])

        pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trials


        print("  Params: ")
        for key, value in trial[0].params.items():
            print("    {}: {}".format(key, value))

            # create a dataframe with the best trial for each building_key
            best_trial = pd.DataFrame([[trial[0].params['n_units'], trial[0].params['n_layers'], trial[0].params['drop_prob'], trial[0].params['lr']]],
                                      columns=['Num_hidden', 'Num_layers', 'Dropout', 'Learning rate'])
            # add building_key column
            best_trial['Building Key'] = building_key

            # save csv in dynamic mode
            best_trial.to_csv(os.path.join(csv_results_path, 'optuna_lstm_results_global.csv'), index=False, mode='a', header=False)





