from typing import Any, List, Mapping, Union
from citylearn.agents.rbc import BasicBatteryRBC
from citylearn.citylearn import CityLearnEnv

class ZeroPowerHeatPumpRBC(BasicBatteryRBC):
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)

    @BasicBatteryRBC.action_map.setter
    def action_map(self, action_map: Union[List[Mapping[str, Mapping[int, float]]], Mapping[str, Mapping[int, float]], Mapping[int, float]]):
        BasicBatteryRBC.action_map.fset(self, action_map)
        action_map = self.action_map

        for i in range(len(action_map)):
            for k, v in action_map[i].items():
                if 'device' in k:
                    for h in v:
                        action_map[i][k][h] = 0.0
                else:
                    pass

        BasicBatteryRBC.action_map.fset(self, action_map)

class FullPowerHeatPumpRBC(BasicBatteryRBC):
    def __init__(self, env: CityLearnEnv, **kwargs: Any):
        super().__init__(env, **kwargs)

    @BasicBatteryRBC.action_map.setter
    def action_map(self, action_map: Union[List[Mapping[str, Mapping[int, float]]], Mapping[str, Mapping[int, float]], Mapping[int, float]]):
        BasicBatteryRBC.action_map.fset(self, action_map)
        action_map = self.action_map

        for i in range(len(action_map)):
            for k, v in action_map[i].items():
                if 'device' in k:
                    for h in v:
                        action_map[i][k][h] = 1.0
                else:
                    pass

        BasicBatteryRBC.action_map.fset(self, action_map)