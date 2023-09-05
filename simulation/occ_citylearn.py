import importlib
import os
from pathlib import Path
import pickle
from typing import Any, Iterable, List, Mapping, Tuple, Union
from citylearn.base import Environment, EpisodeTracker
from citylearn.building import Building, LSTMDynamicsBuilding
from citylearn.citylearn import CityLearnEnv
from citylearn.data import TimeSeriesData
from citylearn.reward_function import RewardFunction
import numpy as np
import pandas as pd

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
        self.setpoint_increase_model_filepath = setpoint_increase_model_filepath
        self.setpoint_decrease_model_filepath = setpoint_decrease_model_filepath
        self.delta_output_map = delta_output_map
        self.parameters = parameters

    @property
    def setpoint_increase_model_filepath(self) -> Union[Path, str]:
        return self.__setpoint_increase_model_filepath
    
    @property
    def setpoint_decrease_model_filepath(self) -> Union[Path, str]:
        return self.__setpoint_decrease_model_filepath
    
    @setpoint_increase_model_filepath.setter
    def setpoint_increase_model_filepath(self, value: Union[Path, str]):
        self.__setpoint_increase_model_filepath = value
        self.__setpoint_increase_model = read_pickle(self.setpoint_increase_model_filepath)

    @setpoint_decrease_model_filepath.setter
    def setpoint_decrease_model_filepath(self, value: Union[Path, str]):
        self.__setpoint_decrease_model_filepath = value
        self.__setpoint_decrease_model = read_pickle(self.setpoint_decrease_model_filepath)

    def predict(self, x: Tuple[float, List[float]]) -> float:
        delta = super().predict()
        interaction_input, delta_input = x
        interaction_probability = lambda  a, b: 1/(1 + np.exp(-(a[self.time_step] + b[self.time_step]*interaction_input)))
        increase_interaction_probability = interaction_probability(self.parameters.a_increase, self.parameters.b_increase)
        decrease_interaction_probability = interaction_probability(self.parameters.a_decrease, self.parameters.b_decrease)
        
        if increase_interaction_probability > 0.0 and decrease_interaction_probability > 0.0:
            pass

        elif increase_interaction_probability > 0.0 and np.random.uniform() <= increase_interaction_probability:
            response = self.__setpoint_increase_model.predict(delta_input)
            delta = self.delta_output_map[response]

        elif decrease_interaction_probability > 0.0 and np.random.uniform() <= decrease_interaction_probability:
            response = self.__setpoint_decrease_model.predict(delta_input)
            delta = self.delta_output_map[response]

        else:
            pass

        return delta

class OccupantInteractionBuilding(LSTMDynamicsBuilding):
    def __init__(self, *args, occupant: Occupant = None, **kwargs):
        # occupant is an optional parameter for now.
        # When the CityLearnEnv._load function eventually gets updated in 
        # CityLearn release to support occupant model, will make occupant 
        # parameter compulsory for this building type
        self.occupant = Occupant() if occupant is None else occupant
        super().__init__(*args, **kwargs)

    
        
    @LSTMDynamicsBuilding.episode_tracker.setter
    def episode_tracker(self, episode_tracker: EpisodeTracker):
        LSTMDynamicsBuilding.episode_tracker.fset(self, episode_tracker)
        self.occupant.episode_tracker = episode_tracker

    def update_setpoints(self):
        """Update building indoor temperature dry-bulb temperature, humidity, etc setpoint using occupant interaction model."""

        raise NotImplementedError

    def next_time_step(self):
        super().next_time_step()
        self.occupant.next_time_step()

        if self.simulate_dynamics:
            self.update_setpoints()
        else:
            pass

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

    def update_setpoints(self):
        current_setpoint = self.energy_simulation.indoor_dry_bulb_temperature_set_point[self.time_step]
        previous_setpoint = self.energy_simulation.indoor_dry_bulb_temperature_set_point[self.time_step - 1]
        current_temperature = self.energy_simulation.indoor_dry_bulb_temperature[self.time_step]
        previous_temperature = self.energy_simulation.indoor_dry_bulb_temperature[self.time_step - 1]
        interaction_input = current_temperature
        delta_input = [current_setpoint, previous_setpoint, previous_temperature - previous_setpoint]
        model_input = (interaction_input, delta_input)       
        setpoint_delta = self.occupant.predict(model_input)
        self.energy_simulation.indoor_dry_bulb_temperature[self.time_step] = current_setpoint + setpoint_delta

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
            args[buildings_ix][i] = b

        return args
    
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