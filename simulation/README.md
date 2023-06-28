# Occupant Thermostat Int: Simulation

## EnergyPlus Simulation

Execute the following if buildings have not already been set in [settings.yaml](settings.yaml) otherwise, skip this step. Will need to have a `DOE_XStock` database:

```console
python energyplus.py select_buildings
```

Execute the following to run EnergyPlus simulations on selected buildings and set the LSTM training data:

```console
python energyplus.py simulate
```

The training data is written to `data/lstm_train_data` directory.