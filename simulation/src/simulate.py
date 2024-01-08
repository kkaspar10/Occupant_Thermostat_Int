import argparse
import concurrent.futures
from datetime import datetime
import getpass
import inspect
from multiprocessing import cpu_count
import os
from pathlib import Path
import socket
import subprocess
import sys
from typing import Any, List, Mapping, Tuple, Union
from citylearn.agents.base import Agent
from citylearn.agents.marlisa import MARLISA, MARLISARBC
from citylearn.agents.rbc import BasicBatteryRBC, BasicRBC, OptimizedRBC, RBC
from citylearn.agents.sac import SACRBC
from citylearn.building import LSTMDynamicsBuilding
from citylearn.reward_function import ComfortReward, IndependentSACReward, MARL, RewardFunction, SolarPenaltyReward, SolarPenaltyAndComfortReward
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from citylearn.utilities import read_json, write_json
import pandas as pd
import simplejson as json
from stable_baselines3 import A2C, DDPG, PPO, SAC, TD3
from src.occ_citylearn import OCCCityLearnEnv
from src.occ_agent import FullPowerHeatPumpRBC, ZeroPowerHeatPumpRBC
from src.occ_reward import AverageComfortReward, CostPenalty, DiscomfortPenalty, DiscomfortAndSetpointReward, DiscomfortPenaltyAndConsumptionPenalty, DiscomfortPenaltyAndCostPenalty, MinimumComfortReward
from src.utilities import FileHandler

def run_work_order(work_order_filepath, max_workers=None, start_index=None, end_index=None, virtual_environment_path=None, windows_system=None):
    work_order_filepath = Path(work_order_filepath)

    if virtual_environment_path is not None:    
        if windows_system:
            virtual_environment_command = f'"{os.path.join(virtual_environment_path, "Scripts", "Activate.ps1")}"'
        
        else:
            virtual_environment_command = f'source "{os.path.join(virtual_environment_path, "bin", "activate")}"'
    
    else:
        virtual_environment_command = 'echo "No virtual environment"'

    with open(work_order_filepath,mode='r') as f:
        args = f.read()
    
    args = args.strip('\n').split('\n')
    start_index = 0 if start_index is None else start_index
    end_index = len(args) - 1 if end_index is None else end_index
    assert start_index <= end_index, 'start_index must be <= end_index'
    assert start_index < len(args), 'start_index must be < number of jobs'
    args = args[start_index:end_index + 1]
    args = [a for a in args if not a.startswith('#')]
    args = [f'{virtual_environment_command} && {a}' for a in args]
    max_workers = cpu_count() if max_workers is None else max_workers
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        print(f'Will use {max_workers} workers for job.')
        print(f'Pooling {len(args)} jobs to run in parallel...')
        results = [executor.submit(subprocess.run,**{'args':a, 'shell':True}) for a in args]
            
        for future in concurrent.futures.as_completed(results):
            try:
                print(future.result())
            
            except Exception as e:
                print(e)

class CityLearnSimulation:
    @staticmethod
    def simulate(
        level_of_detail: int, library: str, agent: str, schema_filename: str = None, simulation_id_suffix: str = None, central_agent: bool = None, buildings: List[int] = None, inactive_observations: List[str] = None, random_seed: int = None, reward_function: str = None, reward_function_kwargs: Mapping[str, Any] = None, rbc: str = None, 
        episodes: int = None, output_directory: Union[str, Path] = None, train_episode_time_steps: List[List[int]] = None, evaluation_episode_time_steps: List[int] = None,
    ):  
        train_start_timestamp = datetime.utcnow()
        env, agent = CityLearnSimulation.learn(
            level_of_detail, library, agent, schema_filename=schema_filename, central_agent=central_agent, buildings=buildings, inactive_observations=inactive_observations, random_seed=random_seed, reward_function=reward_function, 
            reward_function_kwargs=reward_function_kwargs, rbc=rbc, episodes=episodes, episode_time_steps=train_episode_time_steps
        )
        train_end_timestamp = datetime.utcnow()
        evaluation_start_timestamp = datetime.utcnow()
        env, agent, actions = CityLearnSimulation.evaluate(library, env, agent, episode_time_steps=evaluation_episode_time_steps)
        evaluation_end_timestamp = datetime.utcnow()
        evaluation_summary = CityLearnSimulation.get_evaluation_summary(env)
        agent_name, rbc_name, central_agent, reward_function_name = CityLearnSimulation.get_simulation_id(env, agent)
        evaluation_summary = {
            'hostname': socket.gethostname(),
            'username': getpass.getuser(),
            'level_of_detail': level_of_detail,
            'library': library,
            'agent': agent_name,
            'rbc_name': rbc_name,
            'central_agent': central_agent,
            'random_seed': random_seed,
            'reward_function_name': reward_function_name,
            'reward_function_kwargs': reward_function_kwargs,
            'episode_time_steps': [env.episode_tracker.episode_start_time_step, env.episode_tracker.episode_end_time_step],
            'simulation_timestamps': {
                'train_start_timestamp': train_start_timestamp,
                'train_end_timestamp': train_end_timestamp,
                'evaluation_start_timestamp': evaluation_start_timestamp,
                'evaluation_end_timestamp': evaluation_end_timestamp,
            },
            **evaluation_summary,
            'actions': actions,
        }
        os.makedirs(output_directory, exist_ok=True)
        simulation_id = f'lod_{int(level_of_detail)}-{library.replace("-", "_")}-{agent_name}-{rbc_name}-{central_agent}-{reward_function_name}'
        simulation_id += '.json' if simulation_id_suffix is None else f'-{simulation_id_suffix}.json'
        simulation_id = simulation_id.lower()
        filepath = os.path.join(output_directory, simulation_id)
        write_json(filepath, evaluation_summary)

    @staticmethod
    def get_simulation_id(env: OCCCityLearnEnv, agent: Any) -> tuple:
        central_agent = 'Centralized' if env.central_agent else 'Decentralized'
        reward_function_name = env.reward_function.__class__.__name__
        rbc_name = 'NoRBC'
        agent_name = agent.__class__.__name__

        if isinstance(agent, RBC) or agent.__class__ == Agent:
            reward_function_name = 'NoReward'
        else:
            if reward_function_name == 'RewardFunction':
                reward_function_name = 'ConsumptionReward'
            else:
                pass

        if isinstance(agent, (MARLISARBC, SACRBC)) and agent.rbc.__class__ != RBC:
            rbc_name = agent.rbc.__class__.__name__
        else:
            pass

        if isinstance(agent, (MARLISA, MARLISARBC)):
            agent_name = 'MARLISA'
        elif isinstance(agent, SACRBC):
            agent_name = 'SAC'
        elif agent.__class__ == Agent:
            agent_name = 'RandomAgent'
        else:
            pass

        return agent_name, rbc_name, central_agent, reward_function_name

    @staticmethod
    def get_evaluation_summary(env: OCCCityLearnEnv) -> dict:
        comfort_band = FileHandler.get_settings()['reward_function']['attributes']['band']
        evaluation = env.evaluate(comfort_band=comfort_band).pivot(index='name', columns='cost_function', values='value')
        data = {
            'rewards': env.episode_rewards,
            'occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_summary': {
                b.name: b.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_summary for b in env.buildings},
            'evaluation': evaluation.to_dict('index'),
            'time_series': CityLearnSimulation.get_time_series(env).to_dict('list'),
        }

        return data

    @staticmethod
    def get_time_series(env: OCCCityLearnEnv):
        data_list = []

        for b in env.buildings:
            b: LSTMDynamicsBuilding
            data = pd.DataFrame({
                'bldg_name': b.name,
                'net_electricity_consumption': b.net_electricity_consumption,
                'net_electricity_consumption_without_storage': b.net_electricity_consumption_without_storage,
                'net_electricity_consumption_without_storage_and_partial_load': b.net_electricity_consumption_without_storage_and_partial_load,
                'net_electricity_consumption_without_storage_and_partial_load_and_pv': b.net_electricity_consumption_without_storage_and_partial_load_and_pv,
                'indoor_dry_bulb_temperature': b.indoor_dry_bulb_temperature,
                'indoor_dry_bulb_temperature_without_partial_load': b.indoor_dry_bulb_temperature_without_partial_load,
                'indoor_dry_bulb_temperature_set_point': b.energy_simulation.indoor_dry_bulb_temperature_set_point,
                'indoor_dry_bulb_temperature_set_point_without_control': b.energy_simulation.indoor_dry_bulb_temperature_set_point_without_control,
                'occupant_count': b.energy_simulation.occupant_count,
                'heating_electricity_consumption': b.heating_electricity_consumption,
                'heating_demand': b.heating_demand,
                'heating_demand_without_partial_load': b.heating_demand_without_partial_load,
                'dhw_electricity_consumption': b.dhw_electricity_consumption,
                'dhw_demand': b.dhw_demand,
                'non_shiftable_load': b.non_shiftable_load,
                'non_shiftable_load_electricity_consumption': b.non_shiftable_load_electricity_consumption,
                'energy_to_non_shiftable_load': b.energy_to_non_shiftable_load,
                'energy_from_dhw_device': b.energy_from_dhw_device,
                'energy_from_heating_device': b.energy_from_heating_device,
                'occupant_a_increase': b.occupant.parameters.a_increase,
                'occupant_b_increase': b.occupant.parameters.b_increase,
                'occupant_a_decrease': b.occupant.parameters.a_decrease,
                'occupant_b_decrease': b.occupant.parameters.b_decrease,
                'occupant_increase_setpoint_probability': b.occupant.probabilities['increase_setpoint'],
                'occupant_decrease_setpoint_probability': b.occupant.probabilities['decrease_setpoint'],
                'occupant_random_probability': b.occupant.probabilities['random'],
                'occupant_interaction_indoor_dry_bulb_temperature_set_point_delta': b.energy_simulation.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta,
                'net_electricity_consumption_cost': b.net_electricity_consumption_cost,
            })

            data['time_step'] = data.index
            data_list.append(data)

        return pd.concat(data_list, ignore_index=True)

    @staticmethod
    def evaluate(library: str, env: OCCCityLearnEnv, agent: Any, episode_time_steps: List[int] = None) -> Tuple[OCCCityLearnEnv, Any, List[List[Mapping[str, float]]]]:
        if episode_time_steps is not None:
            env.unwrapped.episode_time_steps = [episode_time_steps]
        
        else:
            pass

        observations = env.reset()
        actions_list = []

        while not env.done:
            if library in ['citylearn']:
                actions = agent.predict(observations, deterministic=True)
                actions_list.append(env.unwrapped._parse_actions(actions))
            
            elif library in ['stable-baselines3']:
                actions, _ = agent.predict(observations, deterministic=True)
                actions_list.append(env.unwrapped._parse_actions([actions]))

            else:
                raise Exception(f'Unknown library: {library}')

            observations, _, _, _ = env.step(actions)

        return env, agent, actions_list

    @staticmethod
    def learn(
        level_of_detail: int, library: str, agent: str, schema_filename: str = None, central_agent: bool = None, buildings: List[int] = None, random_seed: int = None, 
        inactive_observations: List[str] = None, reward_function: str = None, reward_function_kwargs: Mapping[str, Any] = None, rbc: str = None, episodes: int = None, episode_time_steps: List[List[int]] = None
    ) -> Tuple[OCCCityLearnEnv, Any]:
        env, agent = CityLearnSimulation.get_agent(
            level_of_detail, library, agent, schema_filename=schema_filename, central_agent=central_agent, buildings=buildings, random_seed=random_seed, inactive_observations=inactive_observations,
            reward_function=reward_function, reward_function_kwargs=reward_function_kwargs, rbc=rbc, episode_time_steps=episode_time_steps
        )
        kwargs = {}

        if library in ['citylearn']:
            kwargs = {**kwargs, 'episodes': episodes}
        
        elif library in ['stable-baselines3']:
            kwargs = {**kwargs, 'total_timesteps': episodes*env.time_steps}

        else:
            raise Exception(f'Unknown library: {library}')
        
        agent.learn(**kwargs)

        return env, agent


    @staticmethod
    def get_agent(
        level_of_detail: int, library: str, agent: str, schema_filename: str = None, central_agent: bool = None, buildings: List[int] = None, random_seed: int = None, inactive_observations: List[str] = None,
        reward_function: str = None, reward_function_kwargs: Mapping[str, Any] = None, rbc: str = None, episode_time_steps: List[List[int]] = None
    ) -> Tuple[OCCCityLearnEnv, Any]:
        schema_filename = 'schema.json' if schema_filename is None else schema_filename
        schema = read_json(os.path.join(FileHandler.SCHEMA_DIRECTORY, schema_filename))
        schema['root_directory'] = FileHandler.SCHEMA_DIRECTORY

        if episode_time_steps is not None:
            schema['episode_time_steps'] = episode_time_steps
        else:
            pass
        
        if level_of_detail == 1:
            for b in schema['buildings']:
                inactive_actions = schema['buildings'][b].get('inactive_actions', []) + ['cooling_device', 'heating_device']
                schema['buildings'][b]['inactive_actions'] = list(set(inactive_actions))
        
        else:
            pass

        # set inactive observations and action
        if inactive_observations is not None:
            for k in schema['observations']:
                if k in inactive_observations:
                    schema['observations'][k]['active'] = False
                
                else:
                    pass
        
        reward_function = reward_function if reward_function is None else CityLearnSimulation.get_reward_function(reward_function)
        env_kwargs = {
            'schema': schema,
            'central_agent': central_agent,
            'reward_function': reward_function,
            'buildings': buildings,
            'random_seed': random_seed
        }
        agent_kwargs = {}
        agent_classes = {
            'citylearn': {
                'Agent': Agent,
                'BasicBatteryRBC': BasicBatteryRBC,
                'BasicRBC': BasicRBC,
                'MARLISA': MARLISA,
                'OptimizedRBC': OptimizedRBC,
                'RBC': RBC,
                'SAC': SACRBC,
                'FullPowerHeatPumpRBC': FullPowerHeatPumpRBC,
                'ZeroPowerHeatPumpRBC': ZeroPowerHeatPumpRBC,
            },
            'stable-baselines3': {
                'A2C': (A2C, 'MlpPolicy'),
                'DDPG': (DDPG, 'MlpPolicy'),
                'PPO': (PPO, 'MlpPolicy'),
                'SAC': (SAC, 'MlpPolicy'),
                'TD3': (TD3, 'MlpPolicy'),
            }
        }
        agent_class = None
        
        if library in ['citylearn']:
            agent_kwargs = {**agent_kwargs, 'random_seed': random_seed}
            agent_kwargs = {**agent_kwargs, 'rbc': agent_classes[library][rbc]} if rbc is not None else agent_kwargs
            agent_class = agent_classes[library][agent]
        
        elif library in ['stable-baselines3']:
            agent_class, policy = agent_classes[library][agent]
            agent_kwargs = {**agent_kwargs, 'policy': policy, 'seed': random_seed}
            env_kwargs = {**env_kwargs, 'wrappers': [NormalizedObservationWrapper, StableBaselines3Wrapper]}

        else:
            raise Exception(f'Unknown library: {library}')

        env = CityLearnSimulation.env_creator(**env_kwargs)

        for b in env.unwrapped.buildings:
            if level_of_detail < 3:
                b.ignore_occupant = True
    
            else:
                b.ignore_occupant = False

        if reward_function is not None and reward_function_kwargs is not None:
            env.unwrapped.reward_function = reward_function(env_metadata=env.get_metadata(), **reward_function_kwargs)
        
        else:
            pass

        agent = agent_class(env=env, **agent_kwargs)

        return env, agent

    @staticmethod
    def env_creator(schema: Union[dict, Path, str], wrappers: list = None, **kwargs) -> OCCCityLearnEnv:
        env = OCCCityLearnEnv(schema, **kwargs)
        wrappers = [] if wrappers is None else wrappers

        for wrapper in wrappers:
            env = wrapper(env)

        return env
    
    @staticmethod
    def get_reward_function(reward_function: str) -> RewardFunction:
        reward_functions = {
            'ComfortReward': ComfortReward,
            'AverageComfortReward': AverageComfortReward, 
            'MinimumComfortReward': MinimumComfortReward,
            'IndependentSACReward': IndependentSACReward,
            'MARL': MARL,
            'RewardFunction': RewardFunction,
            'SolarPenaltyReward': SolarPenaltyReward,
            'SolarPenaltyAndComfortReward': SolarPenaltyAndComfortReward,
            'DiscomfortPenalty': DiscomfortPenalty,
            'CostPenalty': CostPenalty,
            'DiscomfortPenaltyAndCostPenalty': DiscomfortPenaltyAndCostPenalty,
            'DiscomfortPenaltyAndConsumptionPenalty': DiscomfortPenaltyAndConsumptionPenalty,
            'DiscomfortAndSetpointReward': DiscomfortAndSetpointReward,
        }

        return reward_functions[reward_function]

def main():
    parser = argparse.ArgumentParser(prog='citylearn-challenge-2023', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')

    # run work order
    subparser_run_work_order = subparsers.add_parser('run_work_order')
    subparser_run_work_order.add_argument('work_order_filepath', type=Path)
    subparser_run_work_order.add_argument('-m', '--max_workers', dest='max_workers', type=int)
    subparser_run_work_order.add_argument('-s', '--start_index', default=0, dest='start_index', type=int)
    subparser_run_work_order.add_argument('-e', '--end_index', default=None, dest='end_index', type=int)
    subparser_run_work_order.set_defaults(func=run_work_order)

    subparser_general = subparsers.add_parser('general')  
    subparser_general.add_argument('level_of_detail', choices=[1, 2, 3], type=int)
    subparser_general.add_argument('-d', '--output_directory', dest='output_directory', type=str, default=FileHandler.DEFAULT_OUTPUT_DIRECTORY)
    general_subparsers = subparser_general.add_subparsers(title='subcommands', required=True, dest='subcommands')
    
    # CityLearn simulation
    subparser_simulate_citylearn = general_subparsers.add_parser('simulate-citylearn')
    subparser_simulate_citylearn.add_argument('library', type=str)
    subparser_simulate_citylearn.add_argument('agent', type=str)
    subparser_simulate_citylearn.add_argument('-a', '--schema_filename', dest='schema_filename', type=str)
    subparser_simulate_citylearn.add_argument('-x', '--simulation_id_suffix', dest='simulation_id_suffix', type=str)
    subparser_simulate_citylearn.add_argument('-w', '--reward_function', dest='reward_function', type=str, default=None)
    subparser_simulate_citylearn.add_argument('-k', '--reward_function_kwargs', dest='reward_function_kwargs', type=json.loads, default=None)
    subparser_simulate_citylearn.add_argument('-r', '--rbc', dest='rbc', type=str, default=None)
    subparser_simulate_citylearn.add_argument('-e', '--episodes', dest='episodes', type=int, default=1)
    subparser_simulate_citylearn.add_argument('-st', '--train_episode_time_steps', dest='train_episode_time_steps', type=int, nargs='+', action='append')
    subparser_simulate_citylearn.add_argument('-et', '--evaluation_episode_time_steps', dest='evaluation_episode_time_steps', type=int, nargs='+')
    subparser_simulate_citylearn.add_argument('-s', '--random_seed', dest='random_seed', type=int, default=0)
    subparser_simulate_citylearn.add_argument('-c', '--central_agent', dest='central_agent', action='store_true')
    subparser_simulate_citylearn.add_argument('-b', '--buildings', dest='buildings', type=int, nargs='+')
    subparser_simulate_citylearn.add_argument('-io', '--inactive_observations', dest='inactive_observations', type=str, nargs='+')
    subparser_simulate_citylearn.set_defaults(func=CityLearnSimulation.simulate)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }
    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())