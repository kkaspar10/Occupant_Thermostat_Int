from pathlib import Path
from typing import List, Mapping, Tuple, Union
import numpy as np
from citylearn.base import Environment
from citylearn.data import LogisticRegressionOccupantParameters
from citylearn.utilities import read_pickle

class Occupant(Environment):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def predict(self) -> float:
        delta = 0.0

        return delta

class LogisticRegressionOccupant(Occupant):
    def __init__(
            self, setpoint_increase_model_filepath: Union[Path, str], setpoint_decrease_model_filepath: Union[Path, str], 
            model_output_map: Mapping[int, float], parameters: LogisticRegressionOccupantParameters, lookback: int = None, **kwargs):
        super().__init__(**kwargs)
        self.__setpoint_increase_model = None
        self.__setpoint_decrease_model = None
        self.setpoint_increase_model_filepath = setpoint_increase_model_filepath
        self.setpoint_decrease_model_filepath = setpoint_decrease_model_filepath
        self.model_output_map = model_output_map
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
        self.__setpoint_increase_model = read_pickle(self.__setpoint_increase_model_filepath)

    @setpoint_decrease_model_filepath.setter
    def setpoint_decrease_model_filepath(self, value: Union[Path, str]):
        self.__setpoint_decrease_model_filepath = value
        self.__setpoint_decrease_model = read_pickle(self.__setpoint_decrease_model_filepath)

    def predict(self, x: Tuple[float, List[float]]) -> float:
        # make predictions on Y/N to update setpoint and amount of temperature delta here
        # THE setpoint update is made for the current time step?
        # Only return the setpoint delta?
        # default is to set delta to 0.0 C
        delta = super().predict()
        interaction_input, delta_input = x

        interaction_probability = 1/(1 + np.exp(-(
            self.parameters.a[self.time_step] 
            + self.parameters.b[self.time_step]*interaction_input
        )))
        
        # if there is interaction, how does one decide between increase or decrease?
        if np.random.uniform() <= interaction_probability:
            # how do we choose betweeen increase and decrease?
            increase_response = self.__setpoint_increase_model.predict(delta_input)
            decrease_response = self.__setpoint_decrease_model.predict(delta_input)
            increase_delta = self.model_output_map[increase_response]
            decrease_delta = self.model_output_map[decrease_response]

        else:
            pass

        return delta

        