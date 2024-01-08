from typing import List, Any, Mapping, Tuple, Union
from citylearn.reward_function import ComfortReward, RewardFunction
import numpy as np

class DiscomfortAndSetpointReward(RewardFunction):
    def __init__(self, env_metadata: Mapping[str, Any], exponents: Tuple[float, float] = None):
        super().__init__(env_metadata)
        self.exponents = exponents

    @property
    def exponents(self) -> Tuple[float, float]:
        return self.__exponents
    
    @exponents.setter
    def exponents(self, value: Tuple[float, float]):
        self.__exponents = (2.0, 2.0) if value is None else value

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        a = [abs(o['occupant_interaction_indoor_dry_bulb_temperature_set_point_delta']) for o in observations]
        b = [abs(o['indoor_dry_bulb_temperature'] - o['indoor_dry_bulb_temperature_set_point']) for o in observations]
        reward_list = [-(a_**self.exponents[0] + b_**self.exponents[1]) for a_, b_ in zip(a, b)]

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward

class DiscomfortPenalty(RewardFunction):
    """Penalty for occupant thermal discomfort.

    The reward is the calculated as the change in setpoint during the time of a thermostat override (absolute value) raised to an exponent (override_exponent).
    If there is no change to the setpoint, this reward should be zero.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    override_exponent: float, default = 2.0
        Penalty exponent
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], override_exponent: float = None):
        super().__init__(env_metadata)
        self.override_exponent = override_exponent
    
    @property
    def override_exponent(self) -> float:
        return self.__override_exponent
    
    @override_exponent.setter
    def override_exponent(self, override_exponent: float):
        self.__override_exponent = 2.0 if override_exponent is None else override_exponent


    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = [-(abs(o['occupant_interaction_indoor_dry_bulb_temperature_set_point_delta'])**self.override_exponent) for o in observations]

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list

        return reward

class CostPenalty(RewardFunction):
    """Penalty for high energy costs.

    The reward is calculated as the sum of the energy costs at each timestep.

    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    coefficient_cost: float, default = 1.0
        Coefficient to multiply energy cost
    """

    def __init__(self, env_metadata: Mapping[str, Any], coefficient_cost: float = None):
        super().__init__(env_metadata)
        self.coefficient_cost = coefficient_cost

    @property
    def coefficient_cost(self) -> float:
        return self.__coefficient_cost
    
    @coefficient_cost.setter
    def coefficient_cost(self, coefficient_cost: float):
        self.__coefficient_cost = 1.0 if coefficient_cost is None else coefficient_cost


    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward_list = [o['net_electricity_consumption']*o['electricity_pricing']*self.coefficient_cost*-1.0 for o in observations]

        if self.central_agent:
            reward = [sum(reward_list)]

        else:
            reward = reward_list
        
        return reward

class DiscomfortPenaltyAndCostPenalty(RewardFunction):
    """Addition of :py:class:`citylearn.reward_function.DiscomfortPenalty` and :py:class:`citylearn.reward_function.CostPenalty`.
    
    Parameters
    ----------
    env_metadata: Mapping[str, Any]:
        General static information about the environment.
    override_exponent: float, default = 2.0
        Penalty exponent
    coefficient_cost: float, default = 1.0
        Coefficient to multiply energy cost
    """
    
    def __init__(self, env_metadata: Mapping[str, Any], override_exponent: float = None, coefficient_cost: float = None, coefficients: Tuple = None):
        self.__functions: List[RewardFunction] = [
            DiscomfortPenalty(env_metadata, override_exponent=override_exponent),
            CostPenalty(env_metadata, coefficient_cost=coefficient_cost)
        ]
        super().__init__(env_metadata)
        self.coefficients = coefficients

    @property
    def coefficients(self) -> Tuple:
        return self.__coefficients
    
    @coefficients.setter
    def coefficients(self, coefficients: Tuple):
        coefficients = [1.0]*len(self.__functions) if coefficients is None else coefficients
        assert len(coefficients) == len(self.__functions), f'{type(self).__name__} needs {len(self.__functions)} coefficients.' 
        self.__coefficients = coefficients

    @RewardFunction.env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]):
        RewardFunction.env_metadata.fset(self, env_metadata)
        
        for f in self.__functions:
            f.env_metadata = self.env_metadata

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward = np.array([f.calculate(observations) for f in self.__functions], dtype='float32')
        reward = reward*np.reshape(self.coefficients, (len(self.coefficients), 1))
        reward = reward.sum(axis=0).tolist()

        return reward
    
class DiscomfortPenaltyAndConsumptionPenalty(RewardFunction):
    def __init__(self, env_metadata: Mapping[str, Any], override_exponent: float = None, coefficients: Tuple = None):
        self.__functions: List[RewardFunction] = [
            DiscomfortPenalty(env_metadata, override_exponent=override_exponent),
            RewardFunction(env_metadata)
        ]
        super().__init__(env_metadata)
        self.coefficients = coefficients

    @property
    def coefficients(self) -> Tuple:
        return self.__coefficients
    
    @coefficients.setter
    def coefficients(self, coefficients: Tuple):
        coefficients = [1.0]*len(self.__functions) if coefficients is None else coefficients
        assert len(coefficients) == len(self.__functions), f'{type(self).__name__} needs {len(self.__functions)} coefficients.' 
        self.__coefficients = coefficients

    @RewardFunction.env_metadata.setter
    def env_metadata(self, env_metadata: Mapping[str, Any]):
        RewardFunction.env_metadata.fset(self, env_metadata)
        
        for f in self.__functions:
            f.env_metadata = self.env_metadata

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward = np.array([f.calculate(observations) for f in self.__functions], dtype='float32')
        reward = reward*np.reshape(self.coefficients, (len(self.coefficients), 1))
        reward = reward.sum(axis=0).tolist()

        return reward

class AverageComfortReward(ComfortReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward = super().calculate(observations)

        if self.central_agent:
            reward = [reward[0]/len(observations)]
        
        else:
            pass

        return reward
    
class MinimumComfortReward(ComfortReward):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @ComfortReward.central_agent.getter
    def central_agent(self) -> bool:
        return False

    def calculate(self, observations: List[Mapping[str, Union[int, float]]]) -> List[float]:
        reward = super().calculate(observations)
        reward = [min(reward)]

        return reward