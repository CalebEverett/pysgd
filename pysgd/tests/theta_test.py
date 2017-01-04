import pytest
import numpy as np
from pysgd import sgd

def gen_data():
    data = np.ones((100,4))
    data[:,1:] = np.loadtxt(open('tests/logistic.txt', 'rb'), delimiter=',')
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
        logistic=np.zeros(3)
    )

    test_data = next(gen_data())

    data = dict(
        stab_tang=np.array([]),
        linear=test_data,
        logistic=test_data
    )

    adapts =      ['constant', 'adagrad', 'adam']
    alpha = dict(
        stab_tang=[0.01,        0.10,     0.10  ],
        linear=   [0.01,        0.10,     0.10  ],
        logistic= [0.20,        0.30,     0.10  ]
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
        logistic = np.array([1.71617081,  3.98250812,  3.73291532])
    )
    assert np.allclose(theta_hist[-1,:-1], result[obj], atol=0.1) == True
