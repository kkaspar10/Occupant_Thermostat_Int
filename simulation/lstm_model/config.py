import torch

# TRACK METADATA AND HYPERPARAMETERS
config = dict(
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    # aggregate_df=True, # CHECK PROJECT NAME
    # train_model=True,
    save_local=True,
    load_model=False,
    # For wandb
    wandb_on=True,
    # prj_name="pytorch-lstm",
    # prj_name='CityLearn2.0',
    prj_name='OCC_10VT_buildings_V2',
    entity='gim07',
    run_id='RUN_NO_SP',

    epochs=100,
    batch_size=int(24 * 7),  # CHECK DF TIMESTAMP
    lb=12,  # CHECK DF TIMESTAMP
    output_pred=1,
    learning_rate=0.008,
    optimizer_name='Adam',
    num_layer=2,
    hidden_size=16,
    dropout=0.0,
    log_interval=5,
    architecture="LSTM",
    seed=1234)
