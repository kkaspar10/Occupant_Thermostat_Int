## The below functions describe the probability of a SP increase or decrease for two cluster type: Average, Tolerant

## Because only the winter season is considered, and because energy efficiency naturally means the algorithm will
## tend towards lower temperatures, predicting the probability of a SP increase is the crucial modeling step.
## To effectively model this, three scenarios are considered: 'Sleep' (Hours 0:5 inclusive), 'Away' (Hours 9-16 inclusive)
## and 'Home' (Hours 6:8, 17:23 inclusive). Three equations for each occupant type are thus included below.

## In the event of inefficient decision-making, the indoor air temperature could be increased too high
## If this happens, each type of occupant (Average, Tolerant) has a curve that can be triggered. 
## Just one curve for each type of occupant is included for simplification,
## as generally more of the SP interactions were SP increases in winter.

## These functions calculate a probability of interaction. In deployment, a random number needs to be generated.
## If the random number is less than the probability, then the action is taken. For example, if the probability
## is calculated as 0.20 (20%) and the random number is 14, then that means the action is taken. Thus, a 
## random number between 1 and 100 must be generated and compared to the probability at each time step.

## All functions are a probability of T_indoor rounded to the nearest 0.5 degrees C

def calculate_prob(a, b, T_indoor_C_rounded):
    return 1 / (1 + np.exp(-(a + b * T_indoor_C_rounded)))

##Probability of a SP Increase equations:

##Occupant Type 1: Tolerant (Generally prefers lower indoor temperatures in winter)
a_tolerant_home = 28.22
b_tolerant_home = -1.64

a_tolerant_away = 15.24
b_tolerant_away = -1.03

a_tolerant_sleep = 21.16
b_tolerant_sleep = -1.43

##Occupant Type 2: Average (Generally exhibits temperature preferences in line with those typical for the region)

a_average_home = 29.36
b_average_home = -1.57

a_average_away = 15.10
b_average_away = -0.96

a_average_sleep = 26.31
b_average_sleep = -1.56


##Probability of a SP Increase equations:

a_tolerant_decrease = -23.19
b_tolerant_decrease = 0.97

a_average_decrease = -30.34
b_average_decrease = 1.13