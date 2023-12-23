import wandb
import torch

# TRAINING STEP

def training(model, train_loader, val_loader, optimizer, criterion, config,
             maxT, minT, train_loss_list=[], val_loss_list=[]):
    ypred_train = []
    ylab_train = []
    ypred_val = []
    ylab_val = []
    """
    config: Parameter dictionary;
    """
    model.train()
    # Hidden state is initialized at each epochs
    h = model.init_hidden(config.batch_size, config.device)
    for batch in train_loader:
        input_train, target_train = batch
        input_train = input_train.to(config.device)
        target_train = target_train.to(config.device)

        # since the batch is big enough, a stateless mode is used
        # (also considering the possibility to shuffle the training examples,
        # which increase the generalization ability of the network)
        h = model.init_hidden(config.batch_size, config.device)
        h = tuple([each.data for each in h])

        # FORWARD PASS
        output_train, h = model(input_train.float(), h)
        loss_train = criterion(output_train, target_train.float())

        # BACKWARD PASS
        optimizer.zero_grad()
        # obtain the loss function
        loss_train.backward()

        # STEP WITH OPTIMIZER
        optimizer.step()

        # EVALUATE METRICS FOR TRAIN
        # if config.device.type == 'cuda':
        output_train = output_train.to('cpu')
        output_train = output_train.detach().numpy()
        # RESCALE OUTPUT
        output_train = output_train[:, 0]
        # output_test = np.reshape(output_test, (-1, 1)).shape
        output_train = output_train * (maxT - minT) + minT

        # labels = labels.item()
        # if config.device.type == 'cuda':
        target_train = target_train.to('cpu')
        target_train = target_train.detach().numpy()
        target_train = target_train[:, 0]
        # target_test = np.reshape(target_test, (-1, 1))
        # RESCALE LABELS
        target_train = target_train * (maxT - minT) + minT
        ypred_train.append(output_train)
        ylab_train.append(target_train)

    # train_loss_list.append(loss_train.item())
    train_loss_list = loss_train.item()

    flatten = lambda l: [item for sublist in l for item in sublist]
    ypred_train = flatten(ypred_train)
    ylab_train = flatten(ylab_train)

    h = model.init_hidden(config.batch_size, config.device)
    for batch in val_loader:
        input_val, target_val = batch
        input_val = input_val.to(config.device)
        target_val = target_val.to(config.device)

        h = tuple([each.data for each in h])
        output_val, h = model(input_val.float(), h)
        # target_val = target_val.unsqueeze(1)

        optimizer.zero_grad()

        # obtain loss function
        loss_val = criterion(output_val, target_val.float())

        # EVALUATE METRICS FOR TRAIN
        # if config.device.type == 'cuda':
        output_val = output_val.to('cpu')
        output_val = output_val.detach().numpy()
        # RESCALE OUTPUT
        output_val = output_val[:, 0]
        # output_test = np.reshape(output_test, (-1, 1)).shape
        output_val = output_val * (maxT - minT) + minT

        # labels = labels.item()
        # if config.device.type == 'cuda':
        target_val = target_val.to('cpu')
        target_val = target_val.detach().numpy()
        target_val = target_val[:, 0]
        # target_test = np.reshape(target_test, (-1, 1))
        # RESCALE LABELS
        target_val = target_val * (maxT - minT) + minT
        ypred_val.append(output_val)
        ylab_val.append(target_val)

    # val_loss_list.append(loss_val.item())
    val_loss_list = loss_val.item()

    # if len(train_loss_list) >= (config.epochs - 1):
    #     train_loss_list = []
    # if len(val_loss_list) >= (config.epochs - 1):
    #     val_loss_list = []

    return train_loss_list, val_loss_list, ylab_train, ypred_train, ylab_val, ypred_val

# def training_GPT(model, train_loader, val_loader, optimizer, criterion, config, train_loss_list=[], val_loss_list=[]):
#     model.train()
#     # Hidden state is initialized at each epochs
#     h = model.init_hidden(config.batch_size, config.device)
#     for batch in train_loader:
#         lstm_input, mlp_input, target = batch
#         lstm_input = lstm_input.to(config.device)
#         mlp_input = mlp_input.to(config.device)
#         target = target.to(config.device)
#         h = model.init_hidden(config.batch_size, config.device)
#         h = tuple([each.data for each in h])
#
#         # FORWARD PASS
#         output, h = model(lstm_input.float(), mlp_input.float(), h)
#         loss = criterion(output, target.float())
#
#         # BACKWARD PASS
#         optimizer.zero_grad()
#         # obtain the loss function
#         loss.backward()
#
#         # STEP WITH OPTIMIZER
#         optimizer.step()
#
#         # EVALUATE METRICS FOR TRAIN
#         output = output.to('cpu')
#         output = output.detach().numpy()
#         target = target.to('cpu')
#         target = target.detach().numpy()
#         train_loss_list.append(loss.item())
#     # Perform the same steps for the validation set
#     model.eval()
#     h = model.init_hidden(config.batch_size, config.device)
#     with torch.no_grad():
#         for batch in val_loader:
#             lstm_input, mlp_input, target = batch
#             lstm_input = lstm_input.to(config.device)
#             mlp_input = mlp_input.to(config.device)
#             target = target.to(config.device)
#             h = model.init_hidden(config.batch_size, config.device)
#             h = tuple([each.data for each in h])
#             output, h = model(lstm_input.float(), mlp_input.float(), h)
#             loss = criterion(output, target.float())
#             output = output.to('cpu')
#             output = output.detach().numpy()
#             target = target.to('cpu')
#             target = target.detach().numpy()
#             val_loss_list

# WANDB LOG
def train_log(train_loss, val_loss, epoch):
    # Where the magic happens
    wandb.log({"epoch": epoch, "Loss Train": train_loss[epoch], 'Loss Validation': val_loss[epoch]})
    print("Epoch: %d, Train loss: %1.5f, Val loss: %1.5f" % (epoch, train_loss[epoch], val_loss[epoch]))

def log_metrics(MAPE, RMSE, set):
    # Where the magic happens
    wandb.log({f'MAPE {set}': MAPE, f'RMSE {set}': RMSE})
    print(f"MAPE {set}: %1.5f, RMSE {set}: %1.5f" % (MAPE, RMSE))
