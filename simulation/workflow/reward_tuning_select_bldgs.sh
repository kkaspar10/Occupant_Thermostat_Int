# LOD-2 re-run for buildings not needing reward tuning
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_2-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 1
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_5-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4, \"higher_exponent\": 1.4}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 4
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_6-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 5
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_8-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 7
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_9-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 8
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "building_10-final-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 9

#LOD 2 reward tuning for buildings still not performing as expected
#Building 1 (247942)
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_1-reward_tuning-1-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 0
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_1-reward_tuning-2-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 0
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_1-reward_tuning-3-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4, \"higher_exponent\": 1.4}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 0
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_1-reward_tuning-4-2022" -w ComfortReward -k "{\"lower_exponent\": 1.6, \"higher_exponent\": 1.6}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 0
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_1-reward_tuning-5-2022" -w ComfortReward -k "{\"lower_exponent\": 1.8, \"higher_exponent\": 1.8}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 0
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_1-reward_tuning-6-2022" -w ComfortReward -k "{\"lower_exponent\": 2.0, \"higher_exponent\": 2.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 0

#Building 3 (481052)
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_3-reward_tuning-1-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 2
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_3-reward_tuning-2-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 2
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_3-reward_tuning-3-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4, \"higher_exponent\": 1.4}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 2
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_3-reward_tuning-4-2022" -w ComfortReward -k "{\"lower_exponent\": 1.6, \"higher_exponent\": 1.6}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 2
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_3-reward_tuning-5-2022" -w ComfortReward -k "{\"lower_exponent\": 1.8, \"higher_exponent\": 1.8}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 2
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_3-reward_tuning-6-2022" -w ComfortReward -k "{\"lower_exponent\": 2.0, \"higher_exponent\": 2.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 2

#Building 4 (498771)
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_4-reward_tuning-1-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 3
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_4-reward_tuning-2-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 3
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_4-reward_tuning-3-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4, \"higher_exponent\": 1.4}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 3
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_4-reward_tuning-4-2022" -w ComfortReward -k "{\"lower_exponent\": 1.6, \"higher_exponent\": 1.6}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 3
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_4-reward_tuning-5-2022" -w ComfortReward -k "{\"lower_exponent\": 1.8, \"higher_exponent\": 1.8}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 3
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_4-reward_tuning-6-2022" -w ComfortReward -k "{\"lower_exponent\": 2.0, \"higher_exponent\": 2.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 3

#Building 7 (546814)
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_7-reward_tuning-1-2022" -w ComfortReward -k "{\"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 6
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_7-reward_tuning-2-2022" -w ComfortReward -k "{\"lower_exponent\": 1.2, \"higher_exponent\": 1.2}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 6
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_7-reward_tuning-3-2022" -w ComfortReward -k "{\"lower_exponent\": 1.4, \"higher_exponent\": 1.4}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 6
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_7-reward_tuning-4-2022" -w ComfortReward -k "{\"lower_exponent\": 1.6, \"higher_exponent\": 1.6}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 6
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_7-reward_tuning-5-2022" -w ComfortReward -k "{\"lower_exponent\": 1.8, \"higher_exponent\": 1.8}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 6
python -m src.simulate general 2 simulate-citylearn stable-baselines3 SAC -x "Building_7-reward_tuning-6-2022" -w ComfortReward -k "{\"lower_exponent\": 2.0, \"higher_exponent\": 2.0}" -e 10 -st 0 2159 -st 2160 4319 -et 4320 6479 -c -b 6