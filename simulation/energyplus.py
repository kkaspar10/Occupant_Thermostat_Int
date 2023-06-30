import argparse
import inspect
import os
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from doe_xstock.database import SQLiteDatabase
from doe_xstock.data import CityLearnData
from doe_xstock.lstm import TrainData
from doe_xstock.utilities import get_data_from_path, read_json, write_data, write_json
import pandas as pd
import yaml

DATA_DIRECTORY = os.path.join(Path(os.path.dirname(__file__)).absolute(), 'data')
SETTINGS_FILEPATH = 'settings.yaml'
OSM_DIRECTORY = os.path.join(DATA_DIRECTORY, 'osm')
SCHEDULES_DIRECTORY = os.path.join(DATA_DIRECTORY, 'schedules')
ENERGYPLUS_SIMULATION_OUTPUT_DIRECTORY = os.path.join(DATA_DIRECTORY, 'energyplus_simulation')
LSTM_TRAIN_DATA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'lstm_train_data')
SCHEMA_DIRECTORY = os.path.join(DATA_DIRECTORY, 'schema')

def simulate():
    """Runs EnergyPlus simulations for selected buildings and sets LSTM train and CityLearn input data."""
    
    settings = get_settings()
    setpoint_data = pd.read_csv(os.path.join('data', 'SP_Averages_by_Cluster.csv'))

    for d in [LSTM_TRAIN_DATA_DIRECTORY, SCHEMA_DIRECTORY]:
        if os.path.isdir(d):
            shutil.rmtree(d)
        else:
            pass

        os.makedirs(d, exist_ok=True)

    for bldg_id, setpoint_id in zip(settings['building_selection']['bldg_ids'], settings['building_selection']['setpoint_ids']):
        data_list = []

        for year in settings['years']['train'] + settings['years']['test']:
            epw = get_data_from_path(os.path.join('data', 'EPW_Files', f'WeatherFile_EPW_Full_{year}.epw'))
            osm = get_data_from_path(os.path.join(OSM_DIRECTORY, f'{bldg_id}.osm'))
            schedules = read_json(os.path.join(SCHEDULES_DIRECTORY, f'{bldg_id}.json'))
            setpoints = (setpoint_data[setpoint_id] - 32.0)*5.0/9.0 # F -> C
            setpoints = {'setpoint': setpoints.tolist()*365} # set time series for 1 year

            # simulation ID is built from dataset reference and bldg_id
            simulation_id = f'{bldg_id}-{year}'
            output_directory = os.path.join(ENERGYPLUS_SIMULATION_OUTPUT_DIRECTORY, f'{simulation_id}')

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
                run_period_begin_month=settings['energyplus_run_period']['begin_month'],
                run_period_begin_day_of_month=settings['energyplus_run_period']['begin_day_of_month'],
                run_period_begin_year=year,
                run_period_end_month=settings['energyplus_run_period']['end_month'],
                run_period_end_day_of_month=settings['energyplus_run_period']['end_day_of_month'],
                run_period_end_year=year,
                seed=settings['seed'],
                iterations=settings['partial_load_iterations'],
                max_workers=settings['max_workers'],
                simulation_id=simulation_id,
                output_directory=output_directory,
            )
            results = ltd.simulate_partial_loads()
            
            # collect lstm train data for current year
            if year in settings['years']['train']:
                for reference, data in results.items():
                    data = pd.DataFrame(data)
                    data['resstock_building_id'] = bldg_id
                    data['simulation_reference'] = int(reference.split('-')[-2])
                    data['year'] = year
                    data_list.append(data)
            else:
                pass

            # write CityLearn input data
            year_schema_directory = os.path.join(SCHEMA_DIRECTORY, str(year))
            os.makedirs(year_schema_directory, exist_ok=True)
            data = CityLearnData.get_building_data(**{
                'energyplus_output_directory': ENERGYPLUS_SIMULATION_OUTPUT_DIRECTORY,
                'simulation_id': simulation_id,
                'bldg_id': bldg_id,
                **settings['resstock_dataset']
            })
            data['Solar Generation (W/kW)'] = 0.0
            data.to_csv(os.path.join(year_schema_directory, f'{bldg_id}.csv'), index=False)

            weather_data_filepath = os.path.join(year_schema_directory, 'weather.csv')

            if not os.path.isfile(weather_data_filepath):
                weather_data = pd.read_parquet(os.path.join(DATA_DIRECTORY, 'CityLearn_Weather_Files', f'weather_{year}_Final.parquet'))
                weather_data.to_csv(weather_data_filepath, index=False)
            else:
                pass

        # write lstm train data
        data = pd.concat(data_list, ignore_index=True, sort=False)
        data['location'] = settings['location']
        data['ecobee_building_id'] = None
        data[[
            'location',
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
        ]].to_csv(os.path.join(LSTM_TRAIN_DATA_DIRECTORY, f'{bldg_id}.csv'), index=False)

def select_buildings():
    metadata = get_valid_buildings()
    select_n_buildings(metadata)
    set_simulation_input()

def get_valid_buildings():
    """select valid buildings."""

    settings = get_settings()
    location_query = f"(t.in_resstock_county_id = '{settings['location']}' AND t.month IN {str(tuple(settings['months']))})"
    con = sqlite3.connect(settings['database_filepath'])
    con.create_function('REGEXP', 2, regexp)
    metadata = pd.read_sql(f"""
    WITH t AS (
        SELECT
            t.in_resstock_county_id,
            t.metadata_id,
            t.cluster_label,
            ROW_NUMBER() OVER (
                PARTITION BY t.in_resstock_county_id, t.cluster_label 
                ORDER BY t.proportion_unmet_hour) AS rank_unmet_hour,
            t.proportion_unmet_hour,
            t.count_unmet_hour,
            t.min_delta,
            t.max_delta
        FROM (
            SELECT
                t.in_resstock_county_id,
                t.metadata_id,
                t.cluster_label,
                SUM(t.count_unmet_hour)/SUM(t.count_timestep) proportion_unmet_hour,
                SUM(t.count_unmet_hour) AS count_unmet_hour,
                MIN(t.min_delta) AS min_delta,
                MAX(t.max_delta) AS max_delta
            FROM energyplus_simulation_monthly_unmet_hour_summary t
            WHERE {location_query}
            AND t.metadata_id NOT IN (SELECT metadata_id FROM energyplus_simulation_error)
            AND t.metadata_id IN (SELECT metadata_id FROM model WHERE osm REGEXP 'OS:AirLoopHVAC:UnitarySystem')
            GROUP BY
                t.in_resstock_county_id,
                t.metadata_id,
                t.cluster_label
        ) t
    )

    SELECT
        t.cluster_label,
        t.proportion_unmet_hour,
        t.count_unmet_hour,
        t.min_delta,
        t.max_delta,
        t.metadata_id,
        m.bldg_id,
        t.in_resstock_county_id,
        m.in_sqft,
        m.in_geometry_floor_area,
        m.in_occupants,
        m.in_orientation,
        m.in_pv_system_size,
        m.in_vintage,
        m.in_roof_area_ft_2,
        m.in_window_area_ft_2
    FROM t
    LEFT JOIN metadata m ON m.id = t.metadata_id
    WHERE 
        t.proportion_unmet_hour <= {settings['building_selection']['unmet_hour_proportion_limit']}
        AND (m.in_pv_system_size IS NULL OR m.in_pv_system_size = '' OR m.in_pv_system_size = 'None')
    ORDER BY m.bldg_id
    """, con)

    con.close()

    return metadata

def select_n_buildings(metadata):
    """select desired number of buildings from valid buildings."""  
    
    settings = get_settings()
    settings['building_selection']['bldg_ids'] = []

    while len(settings['building_selection']['bldg_ids']) < settings['building_selection']['count']:
        for _, v in metadata[~metadata['bldg_id'].isin(settings['building_selection']['bldg_ids'])].groupby('cluster_label'):
            settings['building_selection']['bldg_ids'] += v['bldg_id'].sample(1, random_state=settings['seed']).tolist()

    settings['building_selection']['bldg_ids'] = settings['building_selection']['bldg_ids'][:settings['building_selection']['count']]

    update_settings(settings)

def set_simulation_input():
    """retrieve and save OSM and schedule files for selectd buildings."""

    settings = get_settings()

    for d in [OSM_DIRECTORY, SCHEDULES_DIRECTORY, LSTM_TRAIN_DATA_DIRECTORY, SCHEMA_DIRECTORY]:
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

            write_data(osm, os.path.join(OSM_DIRECTORY, f'{bldg_id}.osm'))
            write_json(os.path.join(SCHEDULES_DIRECTORY, f'{bldg_id}.json'), schedules)

def update_settings(settings):
    with open(SETTINGS_FILEPATH, 'w') as f:
        yaml.safe_dump(settings, f, sort_keys=False, indent=4)

def get_database():
    return SQLiteDatabase(get_settings()['database_filepath'])

def get_settings():
    with open(SETTINGS_FILEPATH, 'r') as f:
        settings = yaml.safe_load(f)

    return settings

def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

def main():
    parser = argparse.ArgumentParser(prog='occupant-thermostat-int-energyplus', description='Run EnergyPlus simulations to get LSTM train data.')
    subparsers = parser.add_subparsers(title='subcommands', required=True, dest='subcommands')
    
    # select buildings
    subparser_simulate = subparsers.add_parser('select_buildings', description='Select n buildings from DOE_XStock database if exists.')
    subparser_simulate.set_defaults(func=select_buildings)

    # simulate
    subparser_simulate = subparsers.add_parser('simulate', description='EnergyPlus simulations on selected buildings.')
    subparser_simulate.set_defaults(func=simulate)

    args = parser.parse_args()
    arg_spec = inspect.getfullargspec(args.func)
    kwargs = {
        key:value for (key, value) in args._get_kwargs() 
        if (key in arg_spec.args or (arg_spec.varkw is not None and key not in ['func','subcommands']))
    }

    args.func(**kwargs)

if __name__ == '__main__':
    sys.exit(main())