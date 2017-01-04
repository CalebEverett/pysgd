import numpy as np
import importlib
from pysgd.objectives import Objective

# Define general gradient descent algorithm
def sgd(
    theta0,
    obj='stab_tang',
    adapt='constant',
    data=np.array([]),
    size=50,
    alpha=.01,
    epsilon=10**-8,
    beta1=0.9,
    beta2=0.999,
    delta_min=10**-6,
    iters=1000):

    # Initialize gradient adaptation parameters
    params = dict(
        alpha=alpha,
        epsilon=epsilon,
        beta1=beta1,
        beta2=beta2
    )

    # Initialize cost and gradient functions
    obj_fun = Objective(obj, data, size)

    # Initialize gradient adaptation.
    grad_adapt = importlib.import_module('pysgd.adaptations.' + adapt).adapt

    # Initialize theta and cost history for convergence testing and plot
    theta_hist = np.zeros((iters, theta0.shape[0]+1))
    theta_hist[0] = obj_fun.cost(theta0)

    # Initialize theta generator
    theta_gen = grad_adapt(params, obj_fun.grad)(theta0)

    # Initialize iteration variables
    delta = float("inf")
    i = 1

    # Run algorithm
    while delta > delta_min:
        # Get next theta
        theta = next(theta_gen)

        # Store cost for plotting, test for convergence
        try:
            theta_hist[i] = obj_fun.cost(theta)
        except:
            print('{} minimum change in theta not achieved in {} iterations.'
                  .format(delta_min, theta_hist.shape[0]))
            break
        delta = np.max(np.square(theta - theta_hist[i-1,:-1]))**0.5

        i += 1
    # Trim zeros and return
    theta_hist = theta_hist[:i]
    return theta_hist

