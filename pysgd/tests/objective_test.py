import pytest
import numpy as np
from pysgd.objectives import Objective

@pytest.fixture(scope='module')
def test_data():
    test_data = np.ones((100,3))
    for i in range(test_data.shape[0]):
        test_data[i,1:] = np.array([i+1, i+1])
    yield test_data
    test_data = None

# Test to make sure Objective returns correct theta from data (run first so before data shuffle)
def test_with_data(test_data):
    test_obj = Objective('linear', test_data, 40)
    theta0 = np.zeros(2)
    assert test_obj.grad(theta0)[0] == -20.5
    assert test_obj.grad(theta0)[0] == -60.5
    assert test_obj.grad(theta0).size == 2
    assert test_obj.cost(theta0).size == 3
    assert test_obj.cost(theta0)[-1] == 1691.75

# Test to make sure Objective returns correct gradient without
def test_without_data():
    test_obj = Objective('stab_tang', data=np.array([]), size=0)
    theta0 = np.zeros(2)
    theta1 = test_obj.grad(theta0)
    assert theta1[0] == 2.5
    assert test_obj.grad(theta1)[0] == -6.25
    assert test_obj.cost(theta1)[-1] == -48.4375

# Test that batches return tuple of (m x n, m)
def test_batch_shape(test_data):
    test_obj = Objective('linear', test_data, 40)
    batch = next(test_obj.batches)
    assert batch[0].shape[0] == 40
    assert batch[0].shape[1] == 2
    assert batch[1].shape[0]  == 40
    with pytest.raises(IndexError):
        batch[1].shape[1]

# Test that batches returns correctly sequenced batches with size > m
def test_size_less_m(test_data):
    test_batches = Objective('linear', test_data, 40).batches
    assert next(test_batches)[0][-1,-1] == 40
    assert next(test_batches)[0][-1,-1] == 80
    assert next(test_batches)[0].shape[0] == 20
    assert next(test_batches)[0].shape[0] == 40
    assert next(test_batches)[0][-1, -1] != 80

# Test that batches returns correctly sequenced batches with size = m
def test_size_equal_m(test_data):
    test_batches = Objective('linear', test_data, 100).batches
    assert next(test_batches)[0].shape[0] == 100
    assert next(test_batches)[0].shape[0] == 100

# Test that batches returns correctly sequenced batches with size > m
def test_size_greater_m(test_data):
    test_batches = Objective('linear', test_data, 110).batches
    assert next(test_batches)[0].shape[0] == 100
    assert next(test_batches)[0].shape[0] == 100


