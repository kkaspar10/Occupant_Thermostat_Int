# Effects of occupant thermostat preferences and override behavior on residential demand response in CityLearn
The source code in this directory is used to reproduce the results found in our work titled: `Effects of occupant thermostat preferences and override behavior on residential demand response in CityLearn`.

## Installation
First, clone this repository as:

```bash 
git clone https://github.com/kkaspar10/Occupant_Thermostat_Int/
```

Install the dependencies in requirements.txt:
```bash
pip install -r requirements.txt
```

It is important that the specified stable-baseline3 version or earlier is used to avoid issues with stable-baseline3 stopping support for gym environments, which the CityLearn version used in this work is.

## Running simulations

Navigate to this directory in the repository:
```bash
cd simulation
```

The `simulate.py` file contains information about the data stored for each timestep and the subparsers to specify in the CLI when running the simulation.

**Positional Arguments:**
- `library` (str): Path to the library to be used for the simulation.
- `agent` (str): Identifier for the agent.

**Optional Arguments:**
- `-a`, `--schema_filename` (str): Path to the schema file (optional).
- `-x`, `--simulation_id_suffix` (str): Suffix to append to the simulation ID (optional).
- `-w`, `--reward_function` (str): Reward function to be used (optional).
- `-k`, `--reward_function_kwargs` (json): JSON-formatted keyword arguments for the reward function (optional).
- `-r`, `--rbc` (str): Resource-based control strategy (optional).
- `-e`, `--episodes` (int): Number of episodes to run (default: 1).
- `-st`, `--train_episode_time_steps` (int): Time steps for training episodes (can specify multiple values, optional).
- `-et`, `--evaluation_episode_time_steps` (int): Time steps for evaluation episodes (can specify multiple values, optional).
- `-s`, `--random_seed` (int): Random seed for reproducibility (default: 0).
- `-c`, `--central_agent`: Flag to specify if a central agent should be used (optional).
- `-b`, `--buildings` (int): List of building indices to include in the simulation (optional).
- `-io`, `--inactive_observations` (str): List of inactive observations (optional).

**Example Usage:**

To simulate Building 1 using LoD 3, the line appearing in final.sh would appear as:
```bash
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "Building_1-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4 , \"higher_exponent\": 1.4 }" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 0
```

## Results
The results from the previous simulation should be then found under `simulation/data/citylearn_simulation/` as a `.json` file for each building.

## Documentation
Refer to the [docs](https://intelligent-environments-lab.github.io/CityLearn/) for further documentation of the CityLearn API.
