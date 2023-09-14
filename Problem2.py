# For a simple SIR epidemic where data is given for the peak prevalence, find the appropriate value of beta, assuming that gamma is known

# Importing Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fmin

# Value of gamma
gamma = 0.1

# Data for maximum prevalence
dat_maxI = 0.3

# Function defining the governing equations
def goveqs(in_, t, beta, gamma):
    """ in :    state vector at time 't'
    t      :    time (not actually used in the current governing equations, but kept as placeholder)
    beta   :    infections per case per unit time in an SIR model
    gamma  :    recovery rate in an SIR model
    output :    vector of first derivatives for each compartment """
    S, I, R, J = in_
    N = S + I + R

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    dJdt = beta * S * I / N

    return [dSdt, dIdt, dRdt, dJdt]

def get_objective(beta):
    """
     Function to calculate the squared difference between model and data, for
     any given value of beta
     beta       :   Model parameter for rate of infection
     dat_maxI   :   Data for the maximum prevalence
     output     :   Squared difference between model and simulation """
    
    if beta < 0:
        return np.inf
    else:
        geq = lambda in_, t: goveqs(in_, t, beta, 0.1)
        seed = 1e-6
        init = [1 - seed, seed, 0, 0]
        
        # Simulate the epidemic
        t = np.arange(0, 501)
        soln = odeint(geq, init, t)
        sim_maxI = np.max(soln[:, 1])
        
        return (1 - sim_maxI / dat_maxI) ** 2

# Find value of beta to minimize the objective function
beta_sol = fmin(get_objective, 3)

# Plot the resulting epidemic to verify the maximum prevalence
geq = lambda in_, t: goveqs(in_, t, beta_sol, gamma)
seed = 1e-6
init = [1 - seed, seed, 0, 0]

# Simulate the epidemic
t = np.arange(0, 201)
soln = odeint(geq, init, t)

# Plot the prevalence
plt.plot(t, soln[:, 1], linewidth=1.5)
plt.xlabel('Time (days)')
plt.ylabel('Prevalence (proportion of population)')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Calculate the maximum prevalence
max_prevalence = np.max(soln[:, 1])
print(f"Maximum prevalence: {max_prevalence}")