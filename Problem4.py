# For a simple SIR epidemic with data for prevalence at four different timepoints, find appropriate values for beta and gamma by maximising likelihood, assuming that 30% of prevalent cases are reported

# Importing packages
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fmin
from scipy.special import gammaln
import matplotlib.pyplot as plt

def get_objective(x, data_time, data_prevs, prop_report):
    """
    Function to calculate goodness-of-fit of prevalence data using likelihood 
    function, in cases where only a proportion is reported

    x:           Parameter vector, first element is beta, second is gamma
    data_time:   Vector showing days of observation in data
    data_prevs:  Vector showing the prevalence observed on each day in data_time
    prop_report: Proportion of prevalent cases being reported

    Returns:
        Negative log-likelihood of the data given the parameters
    """
    if min(x) < 0:
        return np.inf
    else:
        beta = x[0]
        gamma = x[1]

        # Simulate the epidemic
        t = np.arange(0, 500)
        init = np.array([1 - seed, seed, 0, 0])
        soln = odeint(goveqs, init, t, (beta, gamma))

        # Find the prevalence at the time points specified in the data
        sim_prevs = soln[data_time - 1, 1]

        # Construct Poisson likelihood terms (in log space)
        kvec = data_prevs
        lamvec = sim_prevs * prop_report
        return -np.sum(kvec * np.log(lamvec) - lamvec - gammaln(kvec + 1))

def goveqs(inp, t, beta, gamma):
    """
    Function defining the governing equations (local gradient)

    t       :   time (not actually used in the current governing equations, but kept as placeholder)
    inp     :   state vector at time 't'
    beta    :   infections per case per unit time in an SIR model
    gamma   :   recovery rate in an SIR model

    Returns:
        Vector of first derivatives for each compartment
    """
    S, I, R, J = inp
    N = S + I + R

    # Set up the governing equations
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    dJdt = beta * S * I / N

    return np.array([dSdt, dIdt, dRdt, dJdt])

# Data for calibration
data_time = np.array([53, 66, 79, 92])              # Days on which prevalence was observed
data_prevs = np.array([0.006, 0.04, 0.06, 0.023])   # Prevalence observed on each day
prop_report = 0.3                                   # Proportion of prevalent cases that are reported
seed = 1e-6                                         # Introducing a perturbation ('seed') to the disease-free equilibrium

# Find values of beta and gamma to minimize the value of this function
result = fmin(get_objective,[0.5, 0.25],args=(data_time, data_prevs, prop_report))
x_sol = result

# Plot the resulting epidemic to verify that it gives the correct max prevalence
geq = lambda inp, t: goveqs(inp, t, x_sol[0], x_sol[1])
init = np.array([1 - seed, seed, 0, 0])
t = np.arange(0, 200)
soln = odeint(geq, init, t)

# Plot the prevalence
plt.figure()
plt.plot(t, soln[:, 1] * prop_report, linewidth=1.5)
plt.xlabel("Time (days)")
plt.ylabel("Prevalence (proportion of population)")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.plot(data_time, data_prevs, '.', markersize=24)
plt.legend(["Model simulation", "Data"])
plt.show()

