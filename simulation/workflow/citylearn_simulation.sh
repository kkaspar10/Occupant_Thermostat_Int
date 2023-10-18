python -m src.simulate general 1 simulate-citylearn citylearn RBC -x '2020' -e 1 -st 0 2159 -et 0 2159 -c
python -m src.simulate general 1 simulate-citylearn citylearn RBC -x '2021' -e 1 -st 2160 4319 -et 2160 4319 -c 
python -m src.simulate general 1 simulate-citylearn citylearn RBC -x '2022' -e 1 -st 4320 6479 -et 4320 6479 -c
python -m src.simulate general 2 simulate-citylearn citylearn FullPowerHeatPumpRBC -x '2022' -e 1 -st 4320 6479 -et 4320 6479 -c
python -m src.simulate general 2 simulate-citylearn citylearn ZeroPowerHeatPumpRBC -x '2022' -e 1 -st 4320 6479 -et 4320 6479 -c
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x 'building_all-2-2022' -w AverageComfortReward -e 15 -st 0 2159 -st 2160 4319 -et 4320 6479 -c
