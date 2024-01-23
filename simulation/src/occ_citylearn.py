import importlib
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Tuple, Union
from citylearn.base import Environment, EpisodeTracker
from citylearn.building import Building, LSTMDynamicsBuilding
from citylearn.citylearn import CityLearnEnv
from citylearn.data import EnergySimulation, TimeSeriesData
from citylearn.reward_function import RewardFunction
import numpy as np
import pandas as pd
from src.utilities import FileHandler

class OccupantInteractionBuildingEnergySimulation(EnergySimulation):
    # hacky way for now to make the occ interaction delta available in the citylearn.building.Building.observations function.
    # this way, it is part of the citylearn.data.EnergySimulation time series data that is included in the observations by default

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta = np.zeros(len(self.solar_generation), dtype='float32')
        self.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_without_control = np.zeros(len(self.solar_generation), dtype='float32')

class LogisticRegressionOccupantParameters(TimeSeriesData):
    def __init__(self, a_increase: Iterable[float], b_increase: Iterable[float], a_decrease: Iterable[float], b_decrease: Iterable[float], presence: Iterable[float] = None, start_time_step: int = None, end_time_step: int = None):
        super().__init__(start_time_step=start_time_step, end_time_step=end_time_step)
        
        # setting dynamic parameters
        # (can we give a and b another name that is clearer about what they represent?)
        self.a_increase = np.array(a_increase, dtype='float32')
        self.b_increase = np.array(b_increase, dtype='float32')
        self.a_decrease = np.array(a_decrease, dtype='float32')
        self.b_decrease = np.array(b_decrease, dtype='float32')
        self.presence = np.zeros(len(self.a_increase), dtype='float32') if presence is None else np.array(presence, dtype='float32')

class Occupant(Environment):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def predict(self) -> float:
        delta = 0.0

        return delta

class LogisticRegressionOccupant(Occupant):
    def __init__(
            self, setpoint_increase_model_filepath: Union[Path, str], setpoint_decrease_model_filepath: Union[Path, str], 
            delta_output_map: Mapping[int, float], parameters: LogisticRegressionOccupantParameters, **kwargs
        ):
        super().__init__(**kwargs)
        self.__setpoint_increase_model = None
        self.__setpoint_decrease_model = None
        self.__probabilities = None
        self.setpoint_increase_model_filepath = setpoint_increase_model_filepath
        self.setpoint_decrease_model_filepath = setpoint_decrease_model_filepath
        self.delta_output_map = delta_output_map
        self.parameters = parameters

    @property
    def probabilities(self) -> Mapping[str, float]:
        return self.__probabilities

    @property
    def setpoint_increase_model_filepath(self) -> Union[Path, str]:
        return self.__setpoint_increase_model_filepath
    
    @property
    def setpoint_decrease_model_filepath(self) -> Union[Path, str]:
        return self.__setpoint_decrease_model_filepath
    
    @property
    def delta_output_map(self) -> Mapping[int, float]:
        return self.__delta_output_map
    
    @setpoint_increase_model_filepath.setter
    def setpoint_increase_model_filepath(self, value: Union[Path, str]):
        self.__setpoint_increase_model_filepath = value
        self.__setpoint_increase_model = FileHandler.read_pickle(self.setpoint_increase_model_filepath)

    @setpoint_decrease_model_filepath.setter
    def setpoint_decrease_model_filepath(self, value: Union[Path, str]):
        self.__setpoint_decrease_model_filepath = value
        self.__setpoint_decrease_model = FileHandler.read_pickle(self.setpoint_decrease_model_filepath)

    @delta_output_map.setter
    def delta_output_map(self, value: Mapping[Union[str, int], float]):
        self.__delta_output_map = {int(k): v for k, v in value.items()}

    def predict(self, x: Tuple[float, List[List[float]]]) -> float:
        delta = super().predict()
        response = None
        interaction_input, delta_input = x
        interaction_probability = lambda  a, b, x_ : 1/(1 + np.exp(-(a + b*x_)))
        increase_setpoint_probability = interaction_probability(self.parameters.a_increase[self.time_step], self.parameters.b_increase[self.time_step], interaction_input)
        decrease_setpoint_probability = interaction_probability(self.parameters.a_decrease[self.time_step], self.parameters.b_decrease[self.time_step], interaction_input)
        random_seed = max(self.random_seed, 1)*self.time_step
        nprs = np.random.RandomState(random_seed)
        random_probability = nprs.uniform()
        self.__probabilities['increase_setpoint'][self.time_step] = increase_setpoint_probability
        self.__probabilities['decrease_setpoint'][self.time_step] = decrease_setpoint_probability
        self.__probabilities['random'][self.time_step] = random_probability
        
        if (increase_setpoint_probability < random_probability and decrease_setpoint_probability < random_probability) \
                or (increase_setpoint_probability >= random_probability and decrease_setpoint_probability >= random_probability):
            pass

        elif increase_setpoint_probability >= random_probability:
            response = self.__setpoint_increase_model.predict(delta_input)
            delta = self.delta_output_map[response[0]]

        elif decrease_setpoint_probability >= random_probability:
            response = self.__setpoint_decrease_model.predict(delta_input)
            delta = -self.delta_output_map[response[0]]

        else:
            pass

        return delta
    
    def reset(self):
        super().reset()
        self.__probabilities = {
            'increase_setpoint': np.zeros(self.episode_tracker.episode_time_steps, dtype='float32'),
            'decrease_setpoint': np.zeros(self.episode_tracker.episode_time_steps, dtype='float32'),
            'random': np.zeros(self.episode_tracker.episode_time_steps, dtype='float32'),
        }

class OccupantInteractionBuilding(LSTMDynamicsBuilding):
    def __init__(self, *args, occupant: Occupant = None, ignore_occupant: bool = None, **kwargs):
        # occupant is an optional parameter for now.
        # When the CityLearnEnv._load function eventually gets updated in 
        # CityLearn release to support occupant model, will make occupant 
        # parameter compulsory for this building type
        
        self.occupant = Occupant() if occupant is None else occupant
        self.ignore_occupant = False if ignore_occupant is None else ignore_occupant
        super().__init__(*args, **kwargs)
        
    @LSTMDynamicsBuilding.episode_tracker.setter
    def episode_tracker(self, episode_tracker: EpisodeTracker):
        LSTMDynamicsBuilding.episode_tracker.fset(self, episode_tracker)
        self.occupant.episode_tracker = episode_tracker

    @LSTMDynamicsBuilding.random_seed.setter
    def random_seed(self, seed: int):
        LSTMDynamicsBuilding.random_seed.fset(self, seed)
        self.occupant.random_seed = self.random_seed

    def update_setpoints(self):
        """Update building indoor temperature dry-bulb temperature, humidity, etc setpoint using occupant interaction model."""

        raise NotImplementedError
    
    def apply_actions(self, **kwargs):
        super().apply_actions(**kwargs)

        if self.simulate_dynamics:
            self.update_setpoints()
        else:
            pass

    def next_time_step(self):
        super().next_time_step()
        self.occupant.next_time_step()

    def reset(self):
        super().reset()
        self.occupant.reset()

    def reset_dynamic_variables(self):
        super().reset_dynamic_variables()
        start_ix = 0
        end_ix = self.episode_tracker.episode_time_steps
        self.energy_simulation.indoor_dry_bulb_temperature_set_point[start_ix:end_ix] = self.energy_simulation.indoor_dry_bulb_temperature_set_point_without_control.copy()[start_ix:end_ix]

class LogisticRegressionOccupantInteractionBuilding(OccupantInteractionBuilding):
    def __init__(self, *args, occupant: LogisticRegressionOccupant = None, set_point_hold_time_steps: int = None, **kwargs):
        super().__init__(*args, occupant=occupant, **kwargs)
        self.occupant: LogisticRegressionOccupant
        self.energy_simulation: OccupantInteractionBuildingEnergySimulation
        self.__set_point_hold_time_step_counter = None
        self.set_point_hold_time_steps = set_point_hold_time_steps
        self.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_summary = []
        
    @property
    def set_point_hold_time_steps(self) -> int:
        return self.__set_point_hold_time_steps
    
    @set_point_hold_time_steps.setter
    def set_point_hold_time_steps(self, value: int):
        assert value is None or value >= 0, 'set_point_hold_time_steps must be >= 0'
        self.__set_point_hold_time_steps = np.inf if value is None else int(value)

    def update_setpoints(self):
        current_setpoint = self.energy_simulation.indoor_dry_bulb_temperature_set_point[self.time_step]
        previous_setpoint = self.energy_simulation.indoor_dry_bulb_temperature_set_point[self.time_step - 1]
        current_temperature = self.energy_simulation.indoor_dry_bulb_temperature[self.time_step]
        previous_temperature = self.energy_simulation.indoor_dry_bulb_temperature[self.time_step - 1]
        interaction_input = current_temperature
        delta_input = [[current_setpoint, previous_setpoint, previous_temperature - previous_setpoint]]
        model_input = (interaction_input, delta_input)   
        setpoint_delta = self.occupant.predict(x=model_input)
        self.energy_simulation.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta[self.time_step] = setpoint_delta

        if abs(setpoint_delta) > 0.0 and not self.ignore_occupant:
            self.energy_simulation.indoor_dry_bulb_temperature_set_point[self.time_step:] = current_setpoint + setpoint_delta
            self.__set_point_hold_time_step_counter = self.set_point_hold_time_steps

        elif self.__set_point_hold_time_step_counter is None:
            pass

        else:
            self.__set_point_hold_time_step_counter -= 1
        
        # revert back to default setpoint schedule if no occupant interaction in defined window
        if self.__set_point_hold_time_step_counter is not None and self.__set_point_hold_time_step_counter == 0:
            self.energy_simulation.indoor_dry_bulb_temperature_set_point[self.time_step + 1:] = self.energy_simulation.indoor_dry_bulb_temperature_set_point_without_control[self.time_step + 1:]
            self.__set_point_hold_time_step_counter = None

        else:
            pass

    def estimate_observation_space_limits(self, include_all: bool = None, periodic_normalization: bool = None) -> Tuple[Mapping[str, float], Mapping[str, float]]:
        # include the occ setpoint delta in the observation space limits so that it shows up when setting the observation space
        limit = {k: 0.0 for k, v in self.action_metadata.items() if v or include_all}
        low_limit, high_limit = limit, limit
        observation_name = 'occupant_interaction_indoor_dry_bulb_temperature_set_point_delta'
        observation_limit = 1.5

        try:
            low_limit, high_limit = super().estimate_observation_space_limits(include_all, periodic_normalization)

            if observation_name in low_limit.keys():
                low_limit[observation_name] = -observation_limit
                high_limit[observation_name] = observation_limit
                print(f'successfully set observation limits.')
            
            else:
                pass

        except KeyError as e:
            if e.args[0] == observation_name:
                print(f'unable to set observations limits because no data source for {observation_name}; will try again later.')

            else:
                raise e
            
        finally:
            return low_limit, high_limit

    def reset_data_sets(self):
        super().reset_data_sets()
        start_time_step = self.episode_tracker.episode_start_time_step
        end_time_step = self.episode_tracker.episode_end_time_step
        self.occupant.parameters.start_time_step = start_time_step
        self.occupant.parameters.end_time_step = end_time_step

    def reset_dynamic_variables(self):
        super().reset_dynamic_variables()
        start_ix = 0
        end_ix = self.episode_tracker.episode_time_steps
        delta_summary = np.unique(self.energy_simulation.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta[start_ix:end_ix], return_counts=True)
        self.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_summary.append([delta_summary[0].tolist(), delta_summary[1].tolist()])
        self.energy_simulation.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta[start_ix:end_ix] =\
            self.energy_simulation.occupant_interaction_indoor_dry_bulb_temperature_set_point_delta_without_control.copy()[start_ix:end_ix]

    def reset(self):
        super().reset()
        self.__set_point_hold_time_step_counter = None

class OCCCityLearnEnv(CityLearnEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # delete after CityLearn>2.1b12 release
        for b in self.buildings:
            # temporary fix for setting random seed in buildings and occupants
            b.random_seed = self.random_seed

            # fix set_point_hold_time_steps
            b.set_point_hold_time_steps = self.schema['buildings'][b.name]['set_point_hold_time_steps']

    def _load(self, **kwargs) -> Tuple[Union[Path, str], List[Building], Union[int, List[Tuple[int, int]]], bool, bool, float, RewardFunction, bool, List[str], EpisodeTracker]:
        args = super()._load(**kwargs)
        root_directory = args[0]
        buildings_ix = 1
        episode_tracker = args[-1]

        for i, b in enumerate(args[buildings_ix]):
            # set occupant
            b: LogisticRegressionOccupantInteractionBuilding
            building_occupant = self.schema['buildings'][b.name]['occupant']
            occupant_type = building_occupant['type']
            occupant_module = '.'.join(occupant_type.split('.')[0:-1])
            occupant_name = occupant_type.split('.')[-1]
            occupant_constructor = getattr(importlib.import_module(occupant_module), occupant_name)
            attributes: dict = building_occupant.get('attributes', {})
            parameters_filepath = os.path.join(root_directory, building_occupant['parameters_filename'])
            parameters = pd.read_csv(parameters_filepath)
            presence = b.energy_simulation.__getattr__(
                'occupant_count', 
                start_time_step=b.episode_tracker.simulation_start_time_step, 
                end_time_step=b.episode_tracker.simulation_end_time_step
            ).copy()
            attributes['parameters'] = LogisticRegressionOccupantParameters(*parameters.values.T, presence=presence)
            attributes['episode_tracker'] = episode_tracker

            for k in ['increase', 'decrease']:
                attributes[f'setpoint_{k}_model_filepath'] = os.path.join(root_directory, attributes[f'setpoint_{k}_model_filename'])
                _ = attributes.pop(f'setpoint_{k}_model_filename')

            b.occupant = occupant_constructor(random_seed=b.random_seed, **attributes)
            args[buildings_ix][i] = b

            # update citylearn.building.Building.energy_simulation so that the
            # occupant_interaction_indoor_dry_bulb_temperature_set_point_delta observation is available to set observation limits and space
            energy_simulation = pd.read_csv(os.path.join(root_directory, self.schema['buildings'][b.name]['energy_simulation']))
            b.energy_simulation = OccupantInteractionBuildingEnergySimulation(
                *energy_simulation.values.T, 
                start_time_step=b.energy_simulation.start_time_step,
                end_time_step=b.energy_simulation.end_time_step,
            )
            b.observation_space = b.estimate_observation_space(include_all=False, normalize=False)

            b.reset_data_sets()

        return args