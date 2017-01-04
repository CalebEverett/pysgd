# Define generator function for constant alpha
# Not terribly useful here, but allows for other gradient adaptations
def adapt(params, grad_fun):

    def theta_gen_const(theta):
        alpha = params['alpha']
        while True:
            theta = theta - alpha * grad_fun(theta)
            yield theta

    return theta_gen_const
