import argparse
import inspect
import os
import shutil
import sys
from doe_xstock.database import SQLiteDatabase
from doe_xstock.data import CityLearnData
from doe_xstock.lstm import TrainData
from doe_xstock.utilities import get_data_from_path, read_json, write_data, write_json
import numpy as np
import pandas as pd
import yaml
from citylearn.data import DataSet
from src.occ_citylearn import OCCCityLearnEnv
from src.utilities import FileHandler

def build_schema():
    dynamics_normalization_minimum = pd.read_csv(os.path.join(FileHandler.LSTM_MODEL_DIRECTORY, 'min.csv'), index_col=0)
    dynamics_normalization_maximum = pd.read_csv(os.path.join(FileHandler.LSTM_MODEL_DIRECTORY, 'max.csv'), index_col=0)
    lstm_config = read_json(FileHandler.LSTM_MODEL_CONFIG_FILEPATH)
    settings = FileHandler.get_settings()
    years = settings['years']['train'] + settings['years']['test']
    episode_time_steps = []
    data_list = []

    # if os.path.isdir(FileHandler.SCHEMA_DIRECTORY):
    #     shutil.rmtree(FileHandler.SCHEMA_DIRECTORY)
    # else:
    #     pass

    # os.makedirs(FileHandler.SCHEMA_DIRECTORY)
    
    # set weather data
    for i, y in enumerate(years):
        filepath = os.path.join(FileHandler.CITYLEARN_WEATHER_DATA_DIRECTORY, f'weather_{y}_Final.parquet')
        data = pd.read_parquet(filepath)
        
        # remove 29th of Feb to be consistent
        data = data[(data['datetime'].dt.strftime('%d-%m')!='29-02')].copy()
        data = data.drop(columns=['datetime'], errors='ignore')
        data_list.append(data)

        # set training episode splits
        if y in years:
            if i == 0:
                episode_time_steps.append((0, data.shape[0] - 1))
            else:
                start = episode_time_steps[-1][1] + 1
                end = start + data.shape[0] - 1
                episode_time_steps.append((start, end))
        else:
            pass
    
    weather_data = pd.concat(data_list, ignore_index=True)
    weather_data.to_csv(os.path.join(FileHandler.SCHEMA_DIRECTORY, 'weather.csv'), index=False)
    
    # write schema
    # general
    schema = DataSet.get_schema(settings['schema_template'])
    schema['root_directory'] = None
    schema['central_agent'] = True
    schema['simulation_start_time_step'] = 0
    schema['simulation_end_time_step'] = episode_time_steps[-2][1]
    schema['episode_time_steps'] = episode_time_steps
    schema['rolling_episode_split'] = False
    schema['random_episode_split'] = False

    # set observations and action
    for t in ['observations', 'actions']:
        for k in schema[t]:
            if k in settings[f'active_{t}']:
                schema[t][k]['active'] = True
            
            else:
                schema[t][k]['active'] = False

    key = 'occupant_interaction_indoor_dry_bulb_temperature_set_point_delta'
    schema['observations'][key] = {'active': True if key in  settings[f'active_observations'] else False, 'shared_in_central_agent': False}

    # set reward function
    schema['reward_function'] = settings['reward_function']

    # set buildings
    schema['buildings'] = {}

    for bldg_id, setpoint_id in zip(settings['building_selection']['bldg_ids'], settings['building_selection']['setpoint_ids']):
        bldg_name = str(bldg_id)

        # get simulation data for building
        data_list = []

        for y in years:
            data = get_citylearn_building_data(bldg_id, f'{bldg_id}-{y}')
            data_list.append(data)

        simulation_data = pd.concat(data_list, ignore_index=True)
        simulation_data.to_csv(os.path.join(FileHandler.SCHEMA_DIRECTORY, f'{bldg_name}.csv'), index=False)

        # set random seed
        np.random.seed(int(bldg_id))
        
        # general building info
        schema['buildings'][bldg_name] = {
            'type': settings['building_type'],
            'include': True,
            'energy_simulation': f'{bldg_name}.csv',
            'weather': 'weather.csv',
            'carbon_intensity': None,
            'pricing': 'pricing.csv',
            'inactive_observations': [],
            'inactive_actions': [],
            'set_point_hold_time_steps': settings['set_point_hold_time_steps'],
        }

        # heat pump for heating
        schema['buildings'][bldg_name]['heating_device'] = {
            'type': settings['heating_device_type'],
            'autosize': True,
            'autosize_attributes': {'safety_factor': settings['heating_device_autosize_safety_factor']},
            'attributes': {
                'nominal_power': None,
                'efficiency': np.random.uniform(
                    settings['heating_device_efficiency']['minimum'],
                    settings['heating_device_efficiency']['maximum'],
                ),
                'target_cooling_temperature': 8.0,
                'target_heating_temperature': np.random.uniform(
                    settings['heating_device_target_heating_temperature']['minimum'],
                    settings['heating_device_target_heating_temperature']['maximum'],
                )
            }
        }

        # electric heater for DHW heating
        schema['buildings'][bldg_name]['dhw_device'] = {
            'type': settings['dhw_device_type'],
            'autosize': True,
            'autosize_attributes': {'safety_factor': settings['dhw_device_autosize_safety_factor']},
            'attributes': {
                'nominal_power': None,
                'efficiency': np.random.uniform(
                    settings['dhw_device_efficiency']['minimum'],
                    settings['dhw_device_efficiency']['maximum'],
                ),
            }
        }
        
        # temperature dynamics
        dynamics_bldg_name = bldg_name
        source_filepath = os.path.join(FileHandler.LSTM_MODEL_DIRECTORY, f'model_pth_VT, Chittenden County_{bldg_name}.pth')

        destination_filepath = os.path.join(FileHandler.SCHEMA_DIRECTORY, f'{dynamics_bldg_name}.pth')
        shutil.copy(source_filepath, destination_filepath)
        schema['buildings'][bldg_name]['dynamics'] = {}
        
        for m in ['cooling', 'heating']:
            schema['buildings'][bldg_name]['dynamics'][m] = {
                'type': settings['dynamics']['type'],
                'attributes': {
                    'input_size': dynamics_normalization_minimum.shape[1],
                    'hidden_size': lstm_config[bldg_name]['n_hidden'],
                    'num_layers': lstm_config[bldg_name]['n_layers'],
                    'dropout': lstm_config[bldg_name]['dropout'],
                    'lookback': settings['dynamics']['attributes']['lookback'],
                    'filename': f'{dynamics_bldg_name}.pth',
                    'input_normalization_minimum': dynamics_normalization_minimum.loc[int(dynamics_bldg_name)].values.tolist(),
                    'input_normalization_maximum': dynamics_normalization_maximum.loc[int(dynamics_bldg_name)].values.tolist(),
                    'input_observation_names': settings['dynamics']['attributes']['input_observation_names'][m]
                }
            }

        # occupant
        occupant_type = 'Tolerant' if setpoint_id == 'Cluster_0_SPs' else 'Average'
        increase_source_filepath = os.path.join(FileHandler.INTERACTION_MODEL_DIRECTORY, f'{occupant_type}_Amount_SP_Increase_v2.pkl')
        decrease_source_filepath = os.path.join(FileHandler.INTERACTION_MODEL_DIRECTORY, f'{occupant_type}_Amount_SP_Decrease_v2.pkl')
        increase_destination_filename = f'{bldg_id}_setpoint_increase.pkl'
        increase_destination_filepath = os.path.join(FileHandler.SCHEMA_DIRECTORY, increase_destination_filename)
        decrease_destination_filename = f'{bldg_id}_setpoint_decrease.pkl'
        decrease_destination_filepath = os.path.join(FileHandler.SCHEMA_DIRECTORY, decrease_destination_filename)
        shutil.copy(increase_source_filepath, increase_destination_filepath)
        shutil.copy(decrease_source_filepath, decrease_destination_filepath)
        parameters_filename = f'{bldg_id}_occupant_parameters.csv'
        parameters_filepath = os.path.join(FileHandler.SCHEMA_DIRECTORY, parameters_filename)
        parameters = get_occupant_parameters(simulation_data, occupant_type)
        parameters.to_csv(parameters_filepath, index=False)
        schema['buildings'][bldg_name]['occupant'] = {
            'type': settings['occupant']['type'],
            'parameters_filename': parameters_filename,
            'attributes': {
                'setpoint_increase_model_filename': increase_destination_filename,
                'setpoint_decrease_model_filename': decrease_destination_filename,
                'delta_output_map': settings['occupant']['attributes']['delta_output_map'],
            }
        }

    # env initialization check and sizing
    try:
        env = OCCCityLearnEnv(schema, root_directory=FileHandler.SCHEMA_DIRECTORY)
        print('Passed env initialization test')
    
    except Exception as e:
        print('Failed env initialization test')
        raise e
    
    for b in env.buildings:
        schema['buildings'][b.name]['heating_device']['autosize'] = False
        schema['buildings'][b.name]['dhw_device']['autosize'] = False
        schema['buildings'][b.name]['heating_device']['attributes']['nominal_power'] = b.heating_device.nominal_power
        schema['buildings'][b.name]['dhw_device']['attributes']['nominal_power'] = b.dhw_device.nominal_power

    schema['simulation_end_time_step'] = episode_time_steps[-1][1]
    write_json(os.path.join(FileHandler.SCHEMA_DIRECTORY, 'schema.json'), schema)

def get_occupant_parameters(data, occupant_type):
    """Get LogisticRegressionOccupant a and b parameters time series."""

    settings = FileHandler.get_settings()
    parameters = settings['occupant']['parameters'][occupant_type]

    for interaction, interaction_params in parameters.items():
        for _, status_parameters in interaction_params.items():
            for h in status_parameters['hours']:
                data.loc[
                    (data['Hour']>=h[0] + 1) 
                    & (data['Hour']<=h[1] + 1),
                    (f'a_{interaction}', f'b_{interaction}')
                ] = (status_parameters['a'], status_parameters['b'])

    columns = ['a_increase', 'b_increase', 'a_decrease', 'b_decrease']
    assert data[columns].isnull().sum().sum() == 0, 'Null parameters found'

    data = data[columns].copy()

    return data

def get_citylearn_building_data(bldg_id, simulation_id):
    """Get CityLearn input data from E+ simulation database."""
    
    settings = FileHandler.get_settings()
    kwargs = {
        'simulation_output_directory': FileHandler.ENERGYPLUS_SIMULATION_OUTPUT_DIRECTORY,
        'simulation_id': simulation_id,
        'bldg_id': bldg_id,
        **settings['resstock_dataset'],
    }
    data = CityLearnData.get_building_data(**kwargs)
    setpoint_data = CityLearnData.get_building_data(reference='1-ideal', **kwargs)
    data['Temperature Set Point (C)'] = setpoint_data['Temperature Set Point (C)'].tolist()
    data['Solar Generation (W/kW)'] = 0.0
    data['Cooling Load (kWh)'] = 0.0
    data = data[data['Month'].isin(settings['months'])].copy()

    return data

def simulate():
    """Runs EnergyPlus simulations for selected buildings and sets LSTM train and CityLearn input data."""
    
    settings = FileHandler.get_settings()
    setpoint_data = pd.read_csv(os.path.join(FileHandler.DATA_DIRECTORY, 'SP_Averages_by_Cluster.csv'))

    for d in [
        FileHandler.LSTM_TRAIN_DATA_DIRECTORY, 
        # FileHandler.SCHEMA_DIRECTORY
    ]:
        if os.path.isdir(d):
            shutil.rmtree(d)
        else:
            pass

        os.makedirs(d, exist_ok=True)

    for bldg_id, setpoint_id in zip(settings['building_selection']['bldg_ids'], settings['building_selection']['setpoint_ids']):
        data_list = []

        for year in settings['years']['train'] + settings['years']['test']:
            epw = get_data_from_path(os.path.join(FileHandler.EPW_DIRECTORY, f'WeatherFile_EPW_Full_{year}.epw'))
            osm = get_data_from_path(os.path.join(FileHandler.OSM_DIRECTORY, f'{bldg_id}.osm'))
            schedules = read_json(os.path.join(FileHandler.SCHEDULES_DIRECTORY, f'{bldg_id}.json'))
            setpoints = (setpoint_data[setpoint_id] - 32.0)*5.0/9.0 # F -> C
            setpoints = {'setpoint': setpoints.tolist()*365} # set time series for 1 year

            # simulation ID is built from dataset reference and bldg_id
            simulation_id = f'{bldg_id}-{year}'
            output_directory = os.path.join(FileHandler.ENERGYPLUS_SIMULATION_OUTPUT_DIRECTORY, f'{simulation_id}')

            if os.path.isdir(output_directory):
                shutil.rmtree(output_directory)
            else:
                pass

            # initialize lstm train data class
            ltd = TrainData(
                idd_filepath=settings['idd_filepath'],
                osm=osm,
                epw=epw,
                schedules=schedules,
                setpoints=setpoints,
                seed=settings['seed'],
                iterations=settings['partial_load']['iterations'],
                max_workers=settings['max_workers'],
                simulation_id=simulation_id,
                output_directory=output_directory,
            )
            _, partial_loads_data = ltd.run(
                partial_load_multiplier_minimum_value=settings['partial_load']['multiplier_minimum'],
                partial_load_multiplier_maximum_value=settings['partial_load']['multiplier_maximum'],
                partial_load_multiplier_probability=settings['partial_load']['multiplier_probability'],
            )
            
            # collect lstm train data for current year
            if year in settings['years']['train']:
                for k, v in partial_loads_data.items():
                    data = pd.DataFrame(v)
                    data['simulation_reference'] = k
                    data['year'] = year
                    data_list.append(data)

            else:
                pass

        # write lstm train data
        data = pd.concat(data_list, ignore_index=True, sort=False)
        data = data[data['month'].isin(settings['months'])].copy()
        data['resstock_county_id'] = settings['resstock_county_id']
        data['resstock_building_id'] = bldg_id
        data['ecobee_building_id'] = None
        data[[
            'resstock_county_id',
            'resstock_building_id',
            'ecobee_building_id',
            'simulation_reference',
            'timestep',
            'year',
            'month',
            'day',
            'day_of_week',
            'hour',
            'minute',
            'direct_solar_radiation',
            'diffuse_solar_radiation',
            'outdoor_air_temperature',
            'average_indoor_air_temperature',
            'occupant_count',
            'cooling_load',
            'heating_load',
            'setpoint'
        ]].sort_values(['simulation_reference', 'year', 'timestep']).to_csv(os.path.join(FileHandler.LSTM_TRAIN_DATA_DIRECTORY, f'{bldg_id}.csv'), index=False)

def select_buildings():
    data = get_valid_buildings()
    select_n_buildings(data)
    set_simulation_input()

def get_valid_buildings():
    """select valid buildings."""

    settings = FileHandler.get_settings()
    data = get_database().query_table(f"""
    SELECT
        p.bldg_id,
        r.label
    FROM (
        SELECT 
            bldg_id,
            MAX(point_five_degree_partial_unmet/730.0) AS unmet_proportion
        FROM dynamic_lstm_train_data_thermal_comfort_summary 
        WHERE 
            timestep_resolution = 'monthly'
            AND in_resstock_county_id = '{settings['resstock_county_id']}'
        GROUP BY
            bldg_id
    ) p
    LEFT JOIN metadata m ON m.bldg_id = p.bldg_id
    LEFT JOIN metadata_clustering_label r ON r.metadata_id = m.id
    INNER JOIN optimal_metadata_clustering o ON o.clustering_id = r.clustering_id
    CROSS JOIN constant c
    WHERE p.unmet_proportion <= c.ashrae_maximum_unmet_hour_proportion
    ORDER BY p.bldg_id, r.label
    ;
    """)

    return data

def select_n_buildings(data):
    """select desired number of buildings from valid buildings."""  
    
    settings = FileHandler.get_settings()
    settings['building_selection']['bldg_ids'] = []

    while len(settings['building_selection']['bldg_ids']) < settings['building_selection']['count']:
        for _, v in data[~data['bldg_id'].isin(settings['building_selection']['bldg_ids'])].groupby('label'):
            settings['building_selection']['bldg_ids'] += v['bldg_id'].sample(1, random_state=settings['seed']).tolist()

    settings['building_selection']['bldg_ids'] = settings['building_selection']['bldg_ids'][:settings['building_selection']['count']]

    update_settings(settings)

def set_simulation_input():
    """retrieve and save OSM and schedule files for selectd buildings."""

    settings = FileHandler.get_settings()

    for d in [
        FileHandler.OSM_DIRECTORY, 
        FileHandler.SCHEDULES_DIRECTORY, 
        FileHandler.LSTM_TRAIN_DATA_DIRECTORY, 
        # FileHandler.SCHEMA_DIRECTORY
    ]:
        if os.path.isdir(d):
            shutil.rmtree(d)
        else:
            pass

        os.makedirs(d, exist_ok=True)

    for bldg_id in settings['building_selection']['bldg_ids']:
            metadata_id, osm = get_database().query_table(f"""
            SELECT
                i.metadata_id,
                i.bldg_osm AS osm
            FROM energyplus_simulation_input i
            WHERE 
                i.dataset_type = '{settings['resstock_dataset']['dataset_type']}'
                AND i.dataset_weather_data = '{settings['resstock_dataset']['weather_data']}'
                AND i.dataset_year_of_publication = {settings['resstock_dataset']['year_of_publication']}
                AND i.dataset_release = {settings['resstock_dataset']['release']}
                AND i.bldg_id = {bldg_id}
            """).iloc[0].tolist()
            schedules = get_database().query_table(f"""
            SELECT 
                *
            FROM schedule 
            WHERE metadata_id = {metadata_id}
            """)
            schedules = schedules.drop(columns=['metadata_id','timestep',])
            schedules = schedules.to_dict(orient='list')

            write_data(osm, os.path.join(FileHandler.OSM_DIRECTORY, f'{bldg_id}.osm'))
            write_json(os.path.join(FileHandler.SCHEDULES_DIRECTORY, f'{bldg_id}.json'), schedules)

def update_settings(settings):
    with open(FileHandler.SETTINGS_FILEPATH, 'w') as f:
        yaml.safe_dump(settings, f, sort_keys=False, indent=4)

def get_database():
    return SQLiteDatabase(FileHandler.get_settings()['database_filepath'])

def main():
    parser = argparse.ArgumentParser(prog='occupant-thermostat-int-energyplus', description='Run EnergyPlus simulations to get LSTM train data.')
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')
    
    # select buildings
    subparser_select_buildings = subparsers.add_parser('select_buildings', description='Select n buildings from DOE_XStock database if exists.')
    subparser_select_buildings.set_defaults(func=select_buildings)

    # simulate
    subparser_simulate = subparsers.add_parser('simulate', description='EnergyPlus simulations on selected buildings.')
    subparser_simulate.set_defaults(func=simulate)

    # build schema
    subparser_build_schema = subparsers.add_parser('build_schema', description='Set CityLearn schemas.')
    subparser_build_schema.set_defaults(func=build_schema)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {
        key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }

    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())