from typing import Iterable, List, Mapping, Union
from citylearn.reward_function import ComfortReward

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