import importlib
import numpy as np
from os.path import dirname, basename, isfile
import glob
modules = glob.glob(dirname(__file__)+"/*.py")
__all__ = [ basename(f)[:-3] for f in modules if isfile(f)]

class Objective(object):

    def __init__(self, obj, data, size):

        obj = importlib.import_module('pysgd.objectives.' + obj)

        def batches_gen(data=data, size=size):
            i = 0
            while True:
                index = slice(i*size, min((i+1)*size, data.shape[0]), 1)
                if data.shape[0] - i * size > 0:
                    yield (data[index,:-1], data[index,-1])
                    i += 1
                else:
                    np.random.shuffle(data)
                    i = 0

        self.batches = batches_gen()

        def grad_from_data(theta):
            return obj.grad_fun(theta, next(self.batches))

        def cost_from_data(theta):
            return obj.cost_fun(theta, data)

        if data.size > 1:
            self.grad = grad_from_data
            self.cost = cost_from_data
        else:
            self.grad = obj.grad_fun
            self.cost = obj.cost_fun
