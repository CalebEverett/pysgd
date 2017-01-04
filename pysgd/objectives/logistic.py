import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def grad_fun(theta, batch):
    X, y = batch
    return (sigmoid(X.dot(theta)) - y).T.dot(X) / X.shape[0]

def cost_fun(theta, data):
    X = data[:,:-1]
    y = data[:,-1]
    sigm = sigmoid(X.dot(theta))
    return np.append(theta, np.sum(-y.dot(np.log(sigm)) - (1 - y).dot(np.log(1 - sigm))) / data.shape[0])
