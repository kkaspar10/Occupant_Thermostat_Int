import numpy as np
import torch
import wandb

def closed_loop(model, optimizer, model_path, test_x, test_y, config, maxT, minT, k=0, CL_batch_size=1):
    if config.load_model:
        print('Model Loading...')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # optimizer = getattr(torch.optim, config.optimizer_name)(model.parameters(), lr=config.learning_rate)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.to(config.device)
        print('Model successfully loaded!')
        print(model)
        print(optimizer)
    else:
        print('Trying Closed loop approach to the model trained in this run!')

    model.eval()

    h = model.init_hidden(CL_batch_size, config.device)

    Tout = np.zeros(len(test_y))

    for b in range(0, len(test_y)):
        Current_variables = test_x[b, :, :-1]  # all variables are used t timestep t except for target variable (internal temperature)
        T_input = test_x[b, :, -1]  # internal temperature selected at timestep  t-1 t-lookback
        if k < config.lb:
            if k == 0:
                T_lag = T_input
            else:
                T_lag = np.concatenate([T_input[0:config.lb - k], Tout[0:k]])
        else:
            T_lag = Tout[k - config.lb:k]

        inputs = np.column_stack(
            [Current_variables, T_lag])  # stack input at timestep t with internal temperature at timestep t-1
        inputs = torch.tensor(inputs, dtype=torch.float32)
        inputs = inputs[np.newaxis, :, :]
        inputs = inputs.to(config.device)
        h = tuple([each.data for each in h])
        test_output, h = model(inputs.float(), h)

        # RESCALE OUTPUT
        # if config.device.type == 'cuda':
        test_output = test_output.to('cpu')
        test_output = test_output.detach().numpy()

        Tout[k] = test_output[:, 0]
        k = k + 1
        if b % 100 == 0: print('Step of the loop: ', b)
    Tpred = Tout * (maxT - minT) + minT
    Treal = test_y[:, 0] * (maxT - minT) + minT
    # print('Step of the loop: ', b)

    return Tpred, Treal

# WANDB LOG
def CL_log(MAPE, RMSE, R2):
    # Where the magic happens
    wandb.log({'MAPE CL': MAPE, 'RMSE CL': RMSE, 'R2 CL': R2})
    print("MAPE CL: %1.5f, RMSE CL: %1.5f, R2 CL: %1.5f" % (MAPE, RMSE,R2))
