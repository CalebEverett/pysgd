import numpy as np

# Define theta generator function for Adam
# [Kingma, D. P., & Ba, J. L. (2015). Adam: A Method for Stochastic Optimization. International Conference on Learning Representations.](https://arxiv.org/pdf/1412.6980v8.pdf)
def adapt(params, grad_fun):

    # Define theta generator
    def theta_gen_adam(theta):
        # Initialize hyperparameters, moment and iteration variables
        alpha = params['alpha']
        beta1 = params['beta1']
        beta2 = params['beta2']
        epsilon = params['epsilon']
        moment1 = np.zeros(theta.shape[0])
        moment2 = np.zeros(theta.shape[0])
        t = 1

        # Generate new theta
        while True:
            # Get gradient
            gradient = grad_fun(theta)

            # Update moment estimates
            moment1 = beta1 * moment1 + (1 - beta1) * gradient
            moment2 = beta2 * moment2 + (1 - beta2) * np.square(gradient)

            # Yield adapted gradient
            theta = ( theta - alpha * (1 - beta2**t)**0.5 / (1 - beta1**t) *
                        moment1 / (epsilon + np.sqrt(moment2)) )
            yield theta
            t += 1

    return theta_gen_adam

    # Less efficient calculation of theta that is easier to follow
    # moment1_hat = moment1 / (1 - beta1**t)
    # moment2_hat = moment2 / (1 - beta2**t)
    # theta = theta - alpha * moment1_hat / (epsilon + np.sqrt(moment2_hat))



