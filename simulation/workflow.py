import os
import shutil
import pandas as pd
from src.utilities import FileHandler

timesteps = {
    2020: (0, 2159),
    2021: (2160, 4319),
    2022: (4320, 6479),
}
episodes = 10
rbc_agent = 'citylearn RBC'
rl_agent = 'stable-baselines3 SAC'
reward = 'ComfortReward'

# # Reward tuning
# cmd_list = []

# for i in range(20):
#     for j, e in enumerate([1.0, 1.2, 1.4, 1.6, 1.8, 2.0]):
#         reward_kwargs = "{\\\"lower_exponent\\\": " + str(e) + " , \\\"higher_exponent\\\": " + str(e) + " }"
#         cmd = f'python -m src.simulate general 2 simulate-citylearn {rl_agent} -x "Building_{i + 1}-reward_tuning-{j + 1}-2022" -w {reward} -k "{reward_kwargs}" -e {episodes} -st {timesteps[2020][0]} {timesteps[2020][1]} -st {timesteps[2021][0]} {timesteps[2021][1]} -et {timesteps[2022][0]} {timesteps[2022][1]} -c -b {i}'
#         cmd_list.append(cmd)

# with open(os.path.join(FileHandler.WORKFLOW_DIRECTORY, 'reward_tuning.sh'), 'w') as f:
#     cmd_list.append('')
#     f.write('\n'.join(cmd_list))

# LoDs
cmd_list = []
reward_tuning_results = pd.read_csv(os.path.join(FileHandler.METADATA_DIRECTORY, 'comfort_reward_tuning_best_result.csv'))

for l in range(1, 4):
    cmd_list.append(f'# LoD-{l}')
    if l == 1:
        for y, t in timesteps.items():
            cmd = f'python -m src.simulate general {l} simulate-citylearn {rbc_agent} -x "building_all-final-{y}" -e 1 -st {t[0]} {t[1]} -et {t[0]} {t[1]} -c'
            cmd_list.append(cmd)

    else:
        for b, e1, e2, d in reward_tuning_results.sort_values(['building_index'])[['building_index', 'lower_exponent', 'higher_exponent', 'id']].to_records(index=False):
            reward_kwargs = "{\\\"lower_exponent\\\": " + str(e1) + " , \\\"higher_exponent\\\": " + str(e2) + " }"
            cmd = f'python -m src.simulate general {l} simulate-citylearn {rl_agent} -x "Building_{b + 1}-final-2022" -w {reward} -k "{reward_kwargs}" -e {episodes} -st {timesteps[2020][0]} {timesteps[2020][1]} -st {timesteps[2021][0]} {timesteps[2021][1]} -et {timesteps[2022][0]} {timesteps[2022][1]} -c -b {b}'
            cmd_list.append(cmd)

            if l == 2:
                source_filepath = os.path.join(FileHandler.DEFAULT_OUTPUT_DIRECTORY, f'{d}.json')
                
                if os.path.isfile(source_filepath):
                    destination_filepath = os.path.join(FileHandler.DEFAULT_OUTPUT_DIRECTORY, f'lod_{l}-stable_baselines3-sac-norbc-centralized-comfortreward-building_{b + 1}-final-2022.json')
                    shutil.copy(source_filepath, destination_filepath)

                else:
                    pass
            
            else:
                pass

with open(os.path.join(FileHandler.WORKFLOW_DIRECTORY, 'final.sh'), 'w') as f:
    cmd_list.append('')
    f.write('\n'.join(cmd_list))