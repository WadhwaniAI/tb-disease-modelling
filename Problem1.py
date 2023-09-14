# Simulate a simple SIR epidemic, for given values of beta and gamma, and output the max prevalence   

# Importing Necessary Packages
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Function defining the governing equations
def goveqs(in_, t, beta, gamma):
    S, I, R, J = in_
    N = S + I + R

    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    dJdt = beta * S * I / N

    return [dSdt, dIdt, dRdt, dJdt]

# Natural history parameters
beta = 0.2    # Each person infects on average 1/beta people per day
gamma = 0.1   # Mean infectious period is 10 days

#Introducing a perturbation ('seed') to the disease-free equilibrium
seed = 1e-6
init = [1 - seed, seed, 0, 0]

# Simulate the epidemic
t = np.arange(0, 271)
soln = odeint(goveqs, init, t, args=(beta, gamma))

# Extract the infectious compartment (I) values
prevalence = soln[:, 1]

# Plot the prevalence
plt.plot(t, prevalence, linewidth=1.5)
plt.xlabel('Time (days)')
plt.ylabel('Prevalence (proportion of population)')
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# Calculate the maximum prevalence
max_prevalence = np.max(prevalence)
print(f"Maximum prevalence: {max_prevalence}")