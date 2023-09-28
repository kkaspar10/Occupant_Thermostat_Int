## The below functions describe the quantity of the SP Increase or Decrease for two cluster type: Average, Tolerant

## There are four total models saved as .pkl files in the Interaction_Models folder:
## Average_Amount_SP_Decrease_v2.pkl = Amount of SP Decrease for an Average Occupant
## Average_Amount_SP_Increase_v2.pkl = Amount of SP Increase for an Average Occupant
## Tolerant_Amount_SP_Decrease_v2.pkl = Amount of SP Decrease for a Tolerant Occupant
## Tolerant_Amount_SP_Increase_v2.pkl = Amount of SP Increase for a Tolerant Occupant

## Each model is a trained random forest model that has three predictor values and one output value

## Input_data = [[T_SP_C, T_SP(t-1)_C, T_in-T_SP(t-1)_C]]
## input variables description:
## T_SP_C = T_SP in degrees C at current timestep
## T_SP(t-1)_C = T_SP in degrees C at 1 hour before
## T_in-T_SP(t-1)_C = Difference between the indoor air temperature and the setpoint (in degrees C) one hour before the current timestep

## Target/output variable = 0, 1 (two classes)
## 0 = 0.5 degrees C Increase/Decrease in SP; 1 = 1.5 degrees C Increase/Decrease in SP

## The below sample code can be used to load and deploy the models:

# Load the model from the pickle file
# Note: use v2 of models, see note in models folder
filename = 'Average_Amount_SP_Decrease_v2.pkl'
with open(filename, 'rb') as file:
    average_Amount_SP_Decrease_model = pickle.load(file)

input_data = [[T_SP_C, T_SP(t-1)_C, T_in-T_SP(t-1)_C]] #insert/read values from timestep in question

predict_class = average_Amount_SP_Decrease_model.predict(input_data)



