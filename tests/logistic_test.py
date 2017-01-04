import pytest
import numpy as np
from pysgd import sgd
from pysgd.objectives.logistic import sigmoid

def gen_data():
    data = np.ones((100,3))
    obj_theta = np.array([100, 10])
    np.random.seed(4)
    data[:,1:-1] = np.random.randint(101, size=(100,1))
    data[:,-1] = data[:,:-1].dot(obj_theta) * np.random.uniform(0.85, 1.15, 100)
    mu = np.mean(data[:,1:-1], axis=0)
    sigma = np.std(data[:,1:-1], axis=0)
    data[:,1:-1] = (data[:,1:-1] - mu) / sigma
    yield data

@pytest.mark.parametrize('obj', ['stab_tang', 'logistic'])
@pytest.mark.parametrize('adapt', ['constant', 'adagrad', 'adam'])
def test_theta(obj, adapt):
    theta0 = dict(
        stab_tang=np.array([-0.2, -4.4]),
        linear=np.zeros(2),
        logistic=np.zeros(2)
    )

    test_data = next(gen_data())
    test_data[:,-1] = np.round(sigmoid((test_data[:,-1] - np.mean(test_data[:,-1])) / np.std(test_data[:,-1])), 0)

    data = dict(
        stab_tang=np.array([]),
        linear=test_data,
        logistic=test_data
    )

    adapts =      ['constant', 'adagrad', 'adam']

    alpha = dict(
        stab_tang=[0.01,        0.10,     0.10  ],
        linear=   [0.01,        0.10,     0.10  ],
        logistic= [0.20,       10.00,     0.10  ]
    )

    theta_hist = sgd(
        theta0=theta0[obj],
        obj=obj,
        adapt=adapt,
        data=data[obj],
        alpha=alpha[obj][adapts.index(adapt)],
        iters=10000
    )

    print(test_data[-10:])
    print(theta_hist[-5:])

    result = dict(
        stab_tang = np.array([-2.9,  -2.9]),
        linear = np.array([616.22926755,  265.339612734]),
        logistic = np.array([-0.42916511,  5.79726286])
    )
    assert np.allclose(theta_hist[-1,:-1], result[obj], atol=0.1) == True
