import numpy as np
# Define gradient and cost functions for testing (Stablyinski-Tang function)
def grad_fun(theta):
    return np.apply_along_axis(lambda o: 2.5 - 16*o + 2*o**3, 0, theta)

def cost_fun(theta):
    return np.append(theta, np.sum(5*theta - 16*theta**2 + theta**4) / theta.shape[0])
