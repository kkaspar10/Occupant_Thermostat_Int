# LOD-1
python -m src.simulate general 1 simulate-citylearn citylearn RBC -x "building_all-final-2020" -e 1 -st 0 2159 -et 0 2159 -c
python -m src.simulate general 1 simulate-citylearn citylearn RBC -x "building_all-final-2021" -e 1 -st 2160 4319 -et 2160 4319 -c
python -m src.simulate general 1 simulate-citylearn citylearn RBC -x "building_all-final-2022" -e 1 -st 4320 6479 -et 4320 6479 -c
# LOD-2
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_1-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.6, \"higher_exponent\": 1.6}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 0
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_2-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 1
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_3-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 2
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_4-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4, \"higher_exponent\": 1.4}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 3
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_5-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4, \"higher_exponent\": 1.4}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 4
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_6-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 5
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_7-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 6
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_8-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 7
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_9-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 8
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_10-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 9
# LOD-3
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_1-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.6, \"higher_exponent\": 1.6}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 0
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_2-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 1
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_3-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 2
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_4-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4, \"higher_exponent\": 1.4}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 3
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_5-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4, \"higher_exponent\": 1.4}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 4
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_6-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 5
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_7-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 6
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_8-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 7
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_9-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 8
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -x "building_10-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 9
