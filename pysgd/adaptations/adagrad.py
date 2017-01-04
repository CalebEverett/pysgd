import numpy as np

# Define theta generator function for Adagrad
# [Duchi, J., Hazan, E., and Singer, Y. Adaptive Subgradient Methods for Online Learning and Stochastic Optimization](http://stanford.edu/~jduchi/projects/DuchiHaSi10_colt.pdf)
def adapt(params, grad_fun):

    def theta_gen_adagrad(theta):
        # Initialize hyperparameters and gradient history
        alpha = params['alpha']
        epsilon = params['epsilon']
        grad_hist = 0

        # Generate adapted theta
        while True:
            # Get gradient
            gradient = grad_fun(theta)

            # Perform gradient adaptation
            grad_hist += np.square(gradient)
            theta = theta - alpha * gradient / (epsilon + np.sqrt(grad_hist))
            yield theta

    return theta_gen_adagrad
