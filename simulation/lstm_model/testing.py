import torch
import wandb


# TESTING STEP
def testing(model, model_path, test_loader, optimizer, criterion, config, maxT, minT, ypred=[], ylab=[]):
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
        print('Testing the model in the same run of training!')

    model.eval()

    h = model.init_hidden(config.batch_size, config.device)

    for batch in test_loader:
        input_test, target_test = batch
        input_test = input_test.to(config.device)
        target_test = target_test.to(config.device)

        h = tuple([each.data for each in h])

        # FORWARD PASS
        output_test, h = model(input_test.float(), h)
        # target_test = target_test.unsqueeze(1)
        # obtain loss function
        optimizer.zero_grad()
        loss_test = criterion(output_test, target_test.float())

        # if config.device.type == 'cuda':
        output_test = output_test.to('cpu')
        output_test = output_test.detach().numpy()
        # RESCALE OUTPUT
        output_test = output_test[:, 0]
        # output_test = np.reshape(output_test, (-1, 1)).shape
        output_test = output_test * (maxT - minT) + minT

        # labels = labels.item()
        # if config.device.type == 'cuda':
        target_test = target_test.to('cpu')
        target_test = target_test.detach().numpy()
        target_test = target_test[:, 0]
        # target_test = np.reshape(target_test, (-1, 1))
        # RESCALE LABELS
        target_test = target_test * (maxT - minT) + minT
        ypred.append(output_test)
        ylab.append(target_test)
        # print('Step in the loop: ', step)
        # step = step + 1

    flatten = lambda l: [item for sublist in l for item in sublist]
    ypred = flatten(ypred)
    ylab = flatten(ylab)

    LOSS_TEST = loss_test.item()

    return LOSS_TEST, ylab, ypred


def testing_lstm_mlp(model, model_path, test_loader, optimizer, criterion, config, maxT, minT):
    ypred = []
    ylab = []

    if config.load_model:
        print('Model Loading...')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])

        # optimizer = getattr(torch.optim, config.optimizer_name)(model.parameters(), lr=config.learning_rate)

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        model.to(config.device)
        model.eval()
        print('Model successfully loaded!')
    else:
        print('Testing the model in the same run of training!')
        model.eval()

    model.eval()

    h = model.init_hidden(config.batch_size, config.device)

    for batch in test_loader:
        input_lstm_test, input_mlp_test, target_test = batch
        input_lstm_test = input_lstm_test.to(config.device)
        input_mlp_test = input_mlp_test.to(config.device)
        target_test = target_test.to(config.device)

        h = tuple([each.data for each in h])

        # FORWARD PASS
        output_test, h = model(input_lstm_test.float(), input_mlp_test.float(), h)
        # target_test = target_test.unsqueeze(1)
        # obtain loss function
        optimizer.zero_grad()
        loss_test = criterion(output_test, target_test.float())

        # if config.device.type == 'cuda':
        output_test = output_test.to('cpu')
        output_test = output_test.detach().numpy()
        # RESCALE OUTPUT
        output_test = output_test[:, 0]
        # output_test = np.reshape(output_test, (-1, 1)).shape
        output_test = output_test * (maxT - minT) + minT

        # labels = labels.item()
        # if config.device.type == 'cuda':
        target_test = target_test.to('cpu')
        target_test = target_test.detach().numpy()
        target_test = target_test[:, 0]
        # target_test = np.reshape(target_test, (-1, 1))
        # RESCALE LABELS
        target_test = target_test * (maxT - minT) + minT
        ypred.append(output_test)
        ylab.append(target_test)
        # print('Step in the loop: ', step)
        # step = step + 1

    flatten = lambda l: [item for sublist in l for item in sublist]
    ypred = flatten(ypred)
    ylab = flatten(ylab)

    LOSS_TEST = loss_test.item()

    return LOSS_TEST, ylab, ypred


# WANDB LOG
def test_log(test_loss, MAPE, RMSE):
    # Where the magic happens
    wandb.log({"Loss Test": test_loss, 'MAPE Test': MAPE, 'RMSE Test': RMSE})
    print("Test loss: %1.5f, MAPE Test: %1.5f, RMSE Test: %1.5f" % (test_loss, MAPE, RMSE))
