import importlib
import os
from pathlib import Path
from typing import Iterable, List, Mapping, Tuple, Union
from citylearn.base import Environment, EpisodeTracker
from citylearn.building import Building, LSTMDynamicsBuilding
from citylearn.citylearn import CityLearnEnv
from citylearn.data import TimeSeriesData
from citylearn.reward_function import RewardFunction
import numpy as np
import pandas as pd
from src.utilities import FileHandler

class LogisticRegressionOccupantParameters(TimeSeriesData):
    def __init__(self, a_increase: Iterable[float], b_increase: Iterable[float], a_decrease: Iterable[float], b_decrease: Iterable[float], start_time_step: int = None, end_time_step: int = None):
        super().__init__(start_time_step=start_time_step, end_time_step=end_time_step)
        
        # setting dynamic parameters
        # (can we give a and b another name that is clearer about what they represent?)
        self.a_increase = np.array(a_increase, dtype=float)
        self.b_increase = np.array(b_increase, dtype=float)
        self.a_decrease = np.array(a_decrease, dtype=float)
        self.b_decrease = np.array(b_decrease, dtype=float)

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
        random_probability = np.random.uniform()
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
            delta = self.delta_output_map[response[0]]

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

    def update_setpoints(self):
        """Update building indoor temperature dry-bulb temperature, humidity, etc setpoint using occupant interaction model."""

        raise NotImplementedError
    
    def apply_actions(self, **kwargs):
        super().apply_actions(**kwargs)

        if self.simulate_dynamics and not self.ignore_occupant:
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
    def __init__(self, *args, occupant: LogisticRegressionOccupant = None, **kwargs):
        super().__init__(*args, occupant=occupant, **kwargs)
        self.occupant: LogisticRegressionOccupant

    def update_setpoints(self):
        current_setpoint = self.energy_simulation.indoor_dry_bulb_temperature_set_point[self.time_step]
        previous_setpoint = self.energy_simulation.indoor_dry_bulb_temperature_set_point[self.time_step - 1]
        current_temperature = self.energy_simulation.indoor_dry_bulb_temperature[self.time_step]
        previous_temperature = self.energy_simulation.indoor_dry_bulb_temperature[self.time_step - 1]
        interaction_input = current_temperature
        delta_input = [[current_setpoint, previous_setpoint, previous_temperature - previous_setpoint]]
        model_input = (interaction_input, delta_input)      
        setpoint_delta = self.occupant.predict(x=model_input)

        if abs(setpoint_delta) > 0.0:
            self.energy_simulation.indoor_dry_bulb_temperature_set_point[self.time_step:] = current_setpoint + setpoint_delta
        else:
            pass

    def reset_data_sets(self):
        super().reset_data_sets()
        start_time_step = self.episode_tracker.episode_start_time_step
        end_time_step = self.episode_tracker.episode_end_time_step
        self.occupant.parameters.start_time_step = start_time_step
        self.occupant.parameters.end_time_step = end_time_step

class OCCCityLearnEnv(CityLearnEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _load(self, **kwargs) -> Tuple[Union[Path, str], List[Building], Union[int, List[Tuple[int, int]]], bool, bool, float, RewardFunction, bool, List[str], EpisodeTracker]:
        args = super()._load(**kwargs)
        root_directory = args[0]
        buildings_ix = 1
        episode_tracker = args[-1]

        for i, b in enumerate(args[buildings_ix]):
            b: LogisticRegressionOccupantInteractionBuilding
            building_occupant = self.schema['buildings'][b.name]['occupant']
            occupant_type = building_occupant['type']
            occupant_module = '.'.join(occupant_type.split('.')[0:-1])
            occupant_name = occupant_type.split('.')[-1]
            occupant_constructor = getattr(importlib.import_module(occupant_module), occupant_name)
            attributes: dict = building_occupant.get('attributes', {})
            parameters_filepath = os.path.join(root_directory, building_occupant['parameters_filename'])
            parameters = pd.read_csv(parameters_filepath)
            attributes['parameters'] = LogisticRegressionOccupantParameters(*parameters.values.T)
            attributes['episode_tracker'] = episode_tracker

            for k in ['increase', 'decrease']:
                attributes[f'setpoint_{k}_model_filepath'] = os.path.join(root_directory, attributes[f'setpoint_{k}_model_filename'])
                _ = attributes.pop(f'setpoint_{k}_model_filename')

            b.occupant = occupant_constructor(**attributes)
            b.reset_data_sets()
            args[buildings_ix][i] = b

        return args