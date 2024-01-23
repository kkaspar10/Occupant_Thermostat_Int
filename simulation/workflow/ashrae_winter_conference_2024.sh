# # Find building with least LSTM error
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_all-final-2022-awc2024" -e 1 -st 5064 6479 -et 5064 6479 -c
#
#
# # LOD1
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-0" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 1 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-1" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 2 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-2" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 3 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-3" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 4 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-4" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 5 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-5" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 6 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-6" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 7 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-7" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 8 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-8" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 9 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-9" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 10 -c
# python -m src.simulate general 1 simulate-citylearn citylearn RBC -a schema_awc2024.json -x "building_3-final-2022-awc2024-10" -b 2 -e 1 -st 5064 6479 -et 5064 6479 -s 11 -c
#
#
# # energy-comfort balancing
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-ecb-0" -w DiscomfortPenaltyAndConsumptionPenalty -k "{\"override_exponent\": 2.0, \"coefficients\": [0.2, 0.8]}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 1 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-ecb-1" -w DiscomfortPenaltyAndConsumptionPenalty -k "{\"override_exponent\": 2.0, \"coefficients\": [0.4, 0.6]}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 1 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-ecb-2" -w DiscomfortPenaltyAndConsumptionPenalty -k "{\"override_exponent\": 2.0, \"coefficients\": [0.6, 0.4]}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 1 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-ecb-3" -w DiscomfortPenaltyAndConsumptionPenalty -k "{\"override_exponent\": 2.0, \"coefficients\": [0.8, 0.2]}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 1 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-ecb-3" -w DiscomfortPenaltyAndConsumptionPenalty -k "{\"override_exponent\": 2.0, \"coefficients\": [0.98, 0.02]}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 1 -c
#
#
# LOD3
# DiscomfortPenalty
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-0" -w DiscomfortPenalty -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 1 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-1" -w DiscomfortPenalty -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 2 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-2" -w DiscomfortPenalty -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 3 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-3" -w DiscomfortPenalty -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 4 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-4" -w DiscomfortPenalty -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 5 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-5" -w DiscomfortPenalty -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 6 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-6" -w DiscomfortPenalty -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 7 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-7" -w DiscomfortPenalty -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 8 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-8" -w DiscomfortPenalty -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 9 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-9" -w DiscomfortPenalty -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 10 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-10" -w DiscomfortPenalty -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 11 -c
#
# DiscomfortAndSetpointReward
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-0" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 1 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-1" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 2 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-2" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 3 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-3" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 4 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-h-4" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 5 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-5" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 6 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-6" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 7 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-7" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 8 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-8" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 9 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-9" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 10 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-10" -w DiscomfortAndSetpointReward -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 11 -c
#
# ComfortReward
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-o-0" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 1 -c
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-o-1" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 2 -c
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-o-2" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 3 -c
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-o-3" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 4 -c
python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-o-4" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 50 -st 2904 4319 -et 5064 6479 -s 5 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-5" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 6 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-6" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 7 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-7" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 8 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-8" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 9 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-9" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 10 -c
# python -m src.simulate general 3 simulate-citylearn stable-baselines3 SAC -a schema_awc2024.json -x "building_3-final-2022-awc2024-r-10" -w ComfortReward -k "{\"band\": 0.0, \"lower_exponent\": 1.0, \"higher_exponent\": 1.0}" -b 2 -e 100 -st 2904 4319 -et 5064 6479 -s 11 -c
# h = hold setpoint adjustment, use all original obs
# r = revert setpoint adjustment, use all original obs
# o = hold setpoint adjustment, use  t_out , t_spt, t_in as observations