# For a simple SIR epidemic with data for prevalence at four different timepoints, find appropriate values for beta and gamma by minimising sum of squares, assuming that ALL prevalent cases are reported

# Importing Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.optimize import fmin

# Data for calibration
data_time = np.array([53, 66, 79, 92])                  # Days on which prevalence was observed
data_prevs = np.array([0.006, 0.04, 0.06, 0.023]) * 3   # Prevalence observed on each day

# Function definitions
def get_objective(x, data_time, data_prevs):
    if np.any(x < 0):
        return np.inf
    else:
        beta, gamma = x

        geq = lambda in_, t: goveqs(in_, t, beta, gamma)  # Function handle to goveqs, specifying values of beta and gamma
        seed = 1e-6
        init = np.array([1 - seed, seed, 0, 0])  # Introducing a perturbation ('seed') to the disease-free equilibrium

        # Simulate the epidemic
        t = np.arange(0, 501)
        soln = odeint(geq, init, t)

        # Find the prevalence at the timepoints specified in the data
        sim_prevs = soln[data_time - 1, 1]
        dat_prevs = data_prevs

        return np.sum((1 - sim_prevs / dat_prevs) ** 2)

def goveqs(in_, t, beta, gamma):
    # Function defining the governing equations (local gradient)
    # in_   : state vector at time 't'
    # t     : time (not actually used in the current governing equations, but kept as placeholder)
    # beta  : infections per case per unit time in an SIR model
    # gamma : recovery rate in an SIR model
    # out   : vector of first derivatives for each compartment

    # Initialise the output vector
    out = np.zeros_like(in_)
    S, I, R, J = in_
    N = np.sum(in_[:3])

    # Set up the governing equations
    out[0] = -beta * S * I / N              # dS/dt
    out[1] = beta * S * I / N - gamma * I   # dI/dt
    out[2] = gamma * I                      # dR/dt
    out[3] = beta * S * I / N               # dJ/dt

    return out

# Find values of beta and gamma to minimize the objective function
x_sol = fmin(get_objective, [0.1, 0.1], args=(data_time, data_prevs))

# Plot the resulting epidemic to verify that it gives the correct max prevalence
geq = lambda in_, t: goveqs(in_, t, x_sol[0], x_sol[1])  # Function handle to goveqs, allowing us to fix beta and gamma
seed = 1e-6
init = np.array([1 - seed, seed, 0, 0])  # Introducing a perturbation ('seed') to the disease-free equilibrium
t = np.arange(0, 201)
soln = odeint(geq, init, t)

# Plot the prevalence
plt.figure()
plt.plot(t, soln[:, 1], linewidth=1.5)
plt.plot(data_time, data_prevs, '.', markersize=24)
plt.xlabel('Time (days)')
plt.ylabel('Prevalence (proportion of population)')
plt.tick_params(labelsize=14)
plt.legend(['Model simulation', 'Data'])
plt.show()
