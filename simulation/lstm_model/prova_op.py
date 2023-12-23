import optuna
import torch
from optuna.trial import TrialState
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import TensorDataset, DataLoader

from LSTM.classes import LSTM_model_optuna, AttributeDict
from LSTM.deployment import closed_loop
from LSTM.metrics import mean_absolute_percentage_error
from LSTM.testing import testing
from LSTM.training import training
from pipeline.define_tensor import set_train_test
from preprocess.import_data import *


def define_model(trial, config):
    # --------------------------------- DEFINE VARIABLE RANGE VALUE ----------------------------------------------- #
    num_hidden = trial.suggest_int('n_units', 8, 64, step=8)
    num_layers = trial.suggest_int('n_layers', 1, 3, step=1)
    dropout_prob = trial.suggest_float('drop_prob', 0.0, 0.5, step=0.1)
    learning_rate = trial.suggest_float('lr', 1e-6, 1e-2)
    # optimizer = trial.suggest_categorical('optimizer', ["Adam", "RMSprop", "L-BFGS"])

    # ---------------------------------- IMPORT DATA ---------------------------------------------------------------#
    dict_of_df_path = "data\\clim_time_dict_of_df.pickle"

    dict_building = building_dict(dict_of_df_path)
    # ------------------------------- ENTER IN MODEL PIPELINE -----------------------------------------------------#
    # Choose one randnom building for each County

    # bld_key = 100753  # VT
    # bld_key = 100781  # CA
    bld_key = 100862  # TX

    # df_test = next(iter(dict_building.values()))
    df_test = dict_building[bld_key]

    # Remove simulation reference 3 with 0% of ideal load and evaluate the LSTM performance
    # df_test = df_test.loc[df_test['simulation_reference'] != 3]

    # ------------------------------- ENTER IN MODEL PIPELINE -----------------------------------------------------#
    # df_test = next(iter(dict_building.values()))

    # create the model, data, and optimization problem
    def dataset_dataloader(x, y, BATCH_SIZE, shuffle=True):
        TENSOR = TensorDataset(torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32)))
        LOADER = DataLoader(TENSOR, shuffle=shuffle, batch_size=BATCH_SIZE, drop_last=True)
        return TENSOR, LOADER

    # TRAIN VALIDATION & TEST PORTIONING
    maxT, minT, max_df, min_df, inputs, labels, train_x, train_y, val_x, val_y, test_x, test_y = set_train_test(
        df=df_test, config=config)

    # DATASET & DATALOADER
    df_tensor, df_loader = dataset_dataloader(inputs, labels, config.batch_size, shuffle=False)
    train_dataset, train_loader = dataset_dataloader(train_x, train_y, config.batch_size, shuffle=True)
    validation_dataset, validation_loader = dataset_dataloader(val_x, val_y, config.batch_size, shuffle=False)
    test_data, test_loader = dataset_dataloader(test_x, test_y, config.batch_size, shuffle=False)

    input_size = train_x.shape[-1]
    output_size = train_y.shape[-1]

    lstm = LSTM_model_optuna(n_features=input_size,
                             n_output=output_size,
                             drop_prob=dropout_prob,
                             seq_len=config.lb,
                             num_hidden=num_hidden,
                             num_layers=num_layers).to(config.device)

    # Make the loss and optimizer
    criterion = torch.nn.MSELoss()  # mean-squared error for regression
    optimizer = getattr(torch.optim, config.optimizer_name)(lstm.parameters(), lr=learning_rate)

    return lstm, train_loader, validation_loader, test_loader, df_loader, criterion, optimizer, maxT, minT, test_x, test_y, bld_key


def objective(trial):
    from LSTM.configuration import config
    config = AttributeDict(config)

    n_trial = trial.number + 1

    # initialize the network,criterion and optimizer
    lstm, train_loader, validation_loader, test_loader, df_loader, criterion, optimizer, maxT, minT, test_x, test_y, bld_key = define_model(
        trial, config=config)

    # --------------------------------------------- TRAINING --------------------------------------------------------- #
    for epoch in range(config.epochs):
        LOSS_TRAIN, LOSS_VAL, \
        ylab_train, ypred_train, \
        ylab_val, ypred_val = training(model=lstm, train_loader=train_loader, val_loader=validation_loader,
                                       optimizer=optimizer, criterion=criterion, config=config,
                                       maxT=maxT, minT=minT)
        if epoch % 10 == 0:
            print("Epoch: %d, Train loss: %1.5f, Val loss: %1.5f" % (epoch, LOSS_TRAIN[epoch], LOSS_VAL[epoch]))

    # METRICS
    MAPE_train = mean_absolute_percentage_error(ylab_train, ypred_train)
    RMSE_train = mean_squared_error(ylab_train, ypred_train, squared=False)

    MAPE_val = mean_absolute_percentage_error(ylab_val, ypred_val)
    RMSE_val = mean_squared_error(ylab_val, ypred_val, squared=False)

    # ------------------------------------------ TESTING OPEN LOOP --------------------------------------------------- #
    model_path = None

    LOSS_TEST, ylab_test, ypred_test = testing(model=lstm, model_path=model_path, test_loader=test_loader,
                                                 optimizer=optimizer,
                                                 criterion=criterion,
                                                 config=config, maxT=maxT, minT=minT, ylab=[], ypred=[])

    # METRICS
    MAPE_test = mean_absolute_percentage_error(ylab_test, ypred_test)
    RMSE_test = mean_squared_error(ylab_test, ypred_test, squared=False)
    R2_test = r2_score(ylab_test, ypred_test)

    # --------------------------------------------- CLOSED LOOP ------------------------------------------------------ #
    Tpred, Treal = closed_loop(model=lstm, optimizer=optimizer, model_path=model_path,
                               test_x=test_x, test_y=test_y, config=config, maxT=maxT, minT=minT)

    # METRICS
    MAPE_sim = mean_absolute_percentage_error(Treal, Tpred)
    RMSE_sim = mean_squared_error(Treal, Tpred) ** 0.5
    R2_sim = r2_score(Treal, Tpred)

    # START TRAINING PROCESS
    # LSTM SONO O STATELESS O STATEFULL:
    # STATEFULL: HIDDEN DEL BATCH PRECEDENTE VIENE PASSATA ALLA BATCH SUCCESSIVA --> NON HA SENSO FARLO PERO' SE FACCIO SHUFFLE QUINDI
    # SI USA STATELESS: NON PASSO HIDDEN DEL BATCH PRECEDENTE AL SUCCESSIVO PERCHE' NON C'E CONTINUITA' TEMPORALE TRA BATCH

    # ---------------------------------------------- SAVING THE MODEL ---------------------------------------------- #
    name = f'LSTM_optuna_bld_{bld_key}_trial_{n_trial}.pth'
    basedir = 'models'
    csv_dir = 'C:\\Users\\BAEDA\\OneDrive - Politecnico di Torino\\BAEDA\\Ricerca ASO\\Adaptive and predictive control strategies in buildings\\Projects\\2022_12 CityLearn2_LSTM\\Results_Buscemi'
    # csv_dir = 'C:\\Users\\pc\\OneDrive - Politecnico di Torino\\BAEDA\\Ricerca ASO\\Adaptive and predictive control strategies in buildings\\Projects\\2022_12 CityLearn2_LSTM\\Results_Buscemi'

    checkpoint = {
        'model_state_dict': lstm.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_loss': LOSS_TRAIN[-1],
        'val_loss': LOSS_VAL[-1]
    }

    torch.save(checkpoint,  os.path.join(basedir, name))
    print('File saved correctly!')

    # torch.save(lstm.state_dict(), os.path.join(basedir, name))

    # MAKING DF OH HYPERPARAMETERS
    hyperP = pd.DataFrame([[bld_key, n_trial, lstm.n_layers, lstm.n_hidden, config.batch_size,
                            optimizer.defaults['lr'], config.epochs,
                            LOSS_VAL[-1], LOSS_TEST,
                            MAPE_train, RMSE_train,
                            MAPE_val, RMSE_val,
                            MAPE_test, RMSE_test, R2_test,
                            MAPE_sim, RMSE_sim, R2_sim]],
                          columns=['Building Key', 'Trial', 'Num_layers', 'Num_hidden', 'Batch size',
                                   'Learning rate', 'Epochs',
                                   'Validation Loss', 'Test Loss',
                                   'MAPE Train', 'RMSE Train',
                                   'MAPE Val', 'RMSE Val',
                                   'MAPE Test', 'RMSE Test', 'R2 Test'
                                   'MAPE CL', 'RMSE CL', 'R2 CL'])

    runsLogName = csv_dir + f'\\optuna_lstm_results_bld_{bld_key}.csv'
    if not os.path.exists(runsLogName):
        hyperP.to_csv(path_or_buf=runsLogName,
                      sep=',', decimal='.', index=False)
    else:
        runsLog = pd.read_csv(runsLogName, sep=',', decimal='.')
        runsLog = runsLog.append(hyperP)
        runsLog.to_csv(path_or_buf=runsLogName,
                       sep=',', decimal='.', index=False)

    return LOSS_VAL[-1]


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
