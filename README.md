# OptionTools
It is my personal option tools in Python. Most of the codes comes from my homework at Columbia 
Engineering, and I picked up the most frequently used codes and summarized them into this repo. 

## Cores
This file contains basis classes which frequently used in Application parts.
### $MC.py
This py file contains the base classes of the binomial tree, Monte Carlo simulation, discrete and 
continuous markov chains, and Metropolis-Hasting sampling.

### $Random.py
In model calibration, we will frequently use MLE. Therefore, I write an abstract distribution class which
produce the derivatives and score function of common probability distribution. What's more, it also plays an important role when we want to sample in Metropolis-Hasting.

## Numeric
Currently, this file contains all numeric methods used in this tools.

### $NewtonRaphson.py
The implementation of famous Newton algorithm with Numba.


## Applications
### $BStools.py
This file includes most frequently used function in BS world, including calculations of
Greeks and implied Vol. 

### $LVtools.py
This file includes some local volatility models. 
Currently, I only uploaded the code of Local Vol binomial tree and its visualization. 
I will keep updating my work which includes Mckean SDE and particle method. 

### $SVtools.py
This file contains basic stochastic volatility models. The currently supported
 stochastic volatility processes include CIR, OU, and mean-reversion GBM. 

Only MC pricing tools available. I will update FFT methods sometime. 

# About me
Looking for job opportunities in quantitative finance. 
ps3136@columbia.edu
