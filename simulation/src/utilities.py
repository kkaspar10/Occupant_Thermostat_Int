import os
from pathlib import Path
import pickle
import shutil
from typing import Any, List, Union
from citylearn.utilities import read_json
import pandas as pd
import simplejson as json
import yaml

class FileHandler:
    ROOT_DIRECTORY = os.path.join(*Path(os.path.dirname(os.path.abspath(__file__))).parts[0:-2])
    INTERACTION_MODEL_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'citylearn', 'Interaction_Models')
    SIMULATION_ROOT_DIRECTORY = os.path.join(ROOT_DIRECTORY, 'simulation')
    SETTINGS_FILEPATH = os.path.join(SIMULATION_ROOT_DIRECTORY, 'settings.yaml')
    DATA_DIRECTORY = os.path.join(SIMULATION_ROOT_DIRECTORY, 'data')
    FIGURES_DIRECTORY = os.path.join(SIMULATION_ROOT_DIRECTORY, 'figures')
    WORKFLOW_DIRECTORY = os.path.join(SIMULATION_ROOT_DIRECTORY, 'workflow')
    JOURNAL_PAPER_FIGURES_DIRECTORY = os.path.join(FIGURES_DIRECTORY, 'journal_paper')
    DEFAULT_OUTPUT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'citylearn_simulation')
    OSM_DIRECTORY = os.path.join(DATA_DIRECTORY, 'osm')
    SCHEDULES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'schedules')
    ENERGYPLUS_SIMULATION_OUTPUT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'energyplus_simulation')
    LSTM_TRAIN_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'lstm_train_data')
    EPW_DIRECTORY = os.path.join(DATA_DIRECTORY, 'EPW_Files')
    CITYLEARN_WEATHER_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'CityLearn_Weather_Files')
    LSTM_MODEL_DIRECTORY = os.path.join(DATA_DIRECTORY, 'lstm_pth')
    LSTM_MODEL_CONFIG_FILEPATH = os.path.join(LSTM_MODEL_DIRECTORY, 'best_config.json')
    SCHEMA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'schema')
    METADATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'metadata')

    @staticmethod
    def read_yaml(filepath: Union[str, Path]) -> dict:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        return data
    
    @staticmethod
    def get_settings() -> dict:
        return FileHandler.read_yaml(FileHandler.SETTINGS_FILEPATH)
    
    @staticmethod
    def delete_directory(directory, remake: bool = None):
        remake = True if remake is None else remake
        
        if os.path.isdir(directory):
            shutil.rmtree(directory)
        else:
            pass

        if remake:
            os.makedirs(directory, exist_ok=True)
        else:
            pass

    @staticmethod
    def read_pickle(filepath: str) -> Any:
        """Return pickle object.
        
        Parameters
        ----------
        filepath : str
        pathname of pickle file.
        
        Returns
        -------
        obj: Any
            JSON document converted to dictionary.
        """

        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        return data

class DataHandler:
    KPI_LABELS = {
        'electricity_consumption_total': 'Consumption',
        'cost_total': 'Cost',
        'carbon_emissions_total': 'Emissions',
        'discomfort_too_cold_proportion': 'Unmet cold',
        'discomfort_too_hot_proportion': 'Unmet hot',
        'discomfort_proportion': 'Unmet', 
        'ramping_average': 'Ramping',
        'daily_peak_average': 'Peak daily',  
        'annual_peak_average': 'Peak all', 
        'daily_one_minus_load_factor_average': 'Load factor',
        'monthly_one_minus_load_factor_average': 'Monthly load factor',
        'one_minus_thermal_resilience_proportion': 'Thermal resilience',
        'power_outage_normalized_unserved_energy_total': 'Unserved energy',
        'average': 'Score',
    }
                
    @staticmethod
    def get_weighted_evaluation(simulation_id_key: str, output_directory: Union[Path, str] = None):
        kpi_data = DataHandler.get_concat_data(simulation_id_key, 'evaluation', output_directory=output_directory)
        id_vars = ['id', 'phase', 'library', 'agent', 'rbc_name', 'central_agent', 'reward_function_name', 'environment']
        value_vars = [c for c in kpi_data.columns if c not in id_vars]
        kpi_data = kpi_data.melt(id_vars=id_vars, value_vars=value_vars, var_name='kpi')
        kpi_data = kpi_data.dropna()
        kpi_labels = pd.DataFrame({'kpi': list(DataHandler.KPI_LABELS.keys()), 'kpi_label': list(DataHandler.KPI_LABELS.values())})
        kpi_data = kpi_data.merge(kpi_labels, on='kpi', how='left')

        return kpi_data
    
    @staticmethod
    def get_concat_data(simulation_id_key: str, key: str, output_directory: Union[Path, str] = None):
        evaluation_summary = DataHandler.get_evaluation_summary(simulation_id_key, output_directory=output_directory)
        data_list = []

        for k, v in evaluation_summary.items():
            if key == 'evaluation':
                data = pd.DataFrame.from_dict(v[key], orient='index')
                data.index.name = 'environment'
                data = data.reset_index()

            elif key == 'time_series':
                data = pd.DataFrame(v[key])
                data['environment'] = data['bldg_name']

            elif key == 'rewards':
                data = v[key]
                edata_list = []

                for i, r in enumerate(data):
                    edata = pd.DataFrame.from_dict(r, orient='columns')
                    edata['episode'] = i

                    if v['central_agent'] == 'Centralized':
                        edata['environment'] = 'District'
                    else:
                        edata['environment'] = list(v['evaluation'].keys())[:-1]
                    
                    edata_list.append(edata)

                data = pd.concat(edata_list, ignore_index=True)

            else:
                raise Exception(f'Unknown key: {key}')
            
            data['id'] = k
            data['lod'] = data['id'].str.split('-', expand=True)[0].str.split('_', expand=True)[1].astype(int)
            data['year'] = data['id'].str.split('-', expand=True)[8].astype(int)
            data['library'] = v['library']
            data['agent'] = v['agent']
            data['rbc_name'] = v['rbc_name']
            data['random_seed'] = v.get('random_seed', None)
            data['central_agent'] = v['central_agent']
            data['reward_function_name'] = v['reward_function_name']
            data['reward_function_kwargs'] = json.dumps(v['reward_function_kwargs'])
            data['reward_function_kwargs'] = data['reward_function_kwargs'].map(lambda x: json.loads(x))
            data_list.append(data)

        data = pd.concat(data_list, ignore_index=True)

        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        else:
            pass

        return data

    @staticmethod
    def get_evaluation_summary(simulation_id_key: Union[str, List[str]], output_directory: Union[Path, str] = None):
        output_directory = FileHandler.DEFAULT_OUTPUT_DIRECTORY if output_directory is None else output_directory
        data = {}

        for f in os.listdir(output_directory):
            if f.endswith('json') and (
                (isinstance(simulation_id_key, str) and simulation_id_key in f) 
                    or (isinstance(simulation_id_key, list) and len([k for k in simulation_id_key if k in f]) == len(simulation_id_key))
                        ):
                data[f.split('.')[0]] = read_json(os.path.join(output_directory, f))
            else:
                pass

        return data


    