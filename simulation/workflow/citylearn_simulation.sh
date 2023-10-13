python simulation/simulate.py general 1 simulate-citylearn citylearn RBC -x '2020' -w ComfortReward -e 1 -st 0 2159 -et 0 2159 -c
python simulation/simulate.py general 1 simulate-citylearn citylearn RBC -x '2021' -w ComfortReward -e 1 -st 2160 4319 -et 2160 4319 -c 
python simulation/simulate.py general 1 simulate-citylearn citylearn RBC -x '2022' -w ComfortReward -e 1 -st 4320 6479 -et 4320 6479 -c
python simulation/simulate.py general 2 simulate-citylearn citylearn FullPowerHeatPumpRBC -x '2022' -w ComfortReward -e 1 -st 4320 6479 -et 4320 6479 -c
python simulation/simulate.py general 2 simulate-citylearn citylearn ZeroPowerHeatPumpRBC -x '2022' -w ComfortReward -e 1 -st 4320 6479 -et 4320 6479 -c
python simulation/simulate.py general 2 simulate-citylearn stable-baselines3 SAC -x '2022' -w ComfortReward -e 30 -st 0 2159 -st 2160 4319 -et 4320 6479 -c
python simulation/simulate.py general 2 simulate-citylearn stable-baselines3 SAC -x '2022' -w AverageComfortReward -e 30 -st 0 2159 -st 2160 4319 -et 4320 6479 -c
python simulation/simulate.py general 2 simulate-citylearn stable-baselines3 SAC -x '2022' -w MinimumComfortReward -e 30 -st 0 2159 -st 2160 4319 -et 4320 6479 -c
python simulation/simulate.py general 2 simulate-citylearn citylearn SAC -x '2022' -w ComfortReward -e 30 -st 0 2159 -st 2160 4319 -et 4320 6479
# python simulation/simulate.py general 2 simulate-citylearn citylearn MARLISA -x '2022' -w ComfortReward -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479
# python simulation/simulate.py general 2 simulate-citylearn citylearn MARLISA -x '2022' -w ComfortReward -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479
# python simulation/simulate.py general 3 simulate-citylearn stable-baselines3 SAC -x '2022' -w ComfortReward -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c
# python simulation/simulate.py general 3 simulate-citylearn citylearn MARLISA -x '2022' -w ComfortReward -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479
