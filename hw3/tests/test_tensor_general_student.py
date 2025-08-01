import random
from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np
import numba
import pytest
import torch
from hypothesis import given, settings
from hypothesis.strategies import DataObject, data, integers, lists, permutations
import os

import minitorch
from minitorch import MathTestVariable, Tensor, TensorBackend, grad_check

from .strategies import assert_close, small_floats
from .tensor_strategies import assert_close_tensor, shaped_tensors, tensors

one_arg, two_arg, red_arg = MathTestVariable._comp_testing()

shared: Dict[str, TensorBackend] = {}
from minitorch.cuda_kernel_ops import CudaKernelOps


if numba.cuda.is_available():
    backend_tests = [pytest.param("cuda")]
    matmul_tests = [pytest.param("cuda")]
    shared["cuda"] = minitorch.TensorBackend(CudaKernelOps)

def load_numpy_array(arr_path):
    with open(arr_path, 'rb') as f:
        return np.load(f)

###############################################################################
# Assignment 2 Problem 1
###############################################################################
    
################################ POW ##########################################

@pytest.mark.a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("exp", [0, 1, 2, 3])
@pytest.mark.parametrize("backend", backend_tests)
def test_pow_1_student(sizes, exp, backend):
    test_dir = f'./tests/data/pow_1'
    test_str = str(exp)+'_' + '_'.join(map(str, sizes))
    data_path = os.path.join(test_dir, f'{test_str}_data.npy')
    result_path = os.path.join(test_dir, f'{test_str}_result.npy')
    grad_path = os.path.join(test_dir, f'{test_str}_grad.npy')

    data = load_numpy_array(data_path)
    result_ = load_numpy_array(result_path)
    grad_ = load_numpy_array(grad_path)    
    
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    
    result = x ** exp

    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(x.grad.to_numpy(), grad_, atol=1e-5, rtol=1e-5)


@pytest.mark.a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("exp", [0.5])
@pytest.mark.parametrize("backend", backend_tests)
def test_pow_2_student(sizes, exp, backend):
    test_dir = f'./tests/data/pow_2'
    test_str = str(exp)+'_' + '_'.join(map(str, sizes))
    data_path = os.path.join(test_dir, f'{test_str}_data.npy')
    result_path = os.path.join(test_dir, f'{test_str}_result.npy')
    grad_path = os.path.join(test_dir, f'{test_str}_grad.npy')

    data = load_numpy_array(data_path)
    result_ = load_numpy_array(result_path)
    grad_ = load_numpy_array(grad_path) 
    
    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)
    
    result = x ** exp

    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(x.grad.to_numpy(), grad_, atol=1e-5, rtol=1e-5)

@pytest.mark.a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("exp", [1, 2])
@pytest.mark.parametrize("backend", backend_tests)
def test_pow_3_student(sizes, exp, backend):
    test_dir = f'./tests/data/pow_3'
    test_str = str(exp)+'_' + '_'.join(map(str, sizes))
    data_path = os.path.join(test_dir, f'{test_str}_data.npy')
    result_path = os.path.join(test_dir, f'{test_str}_result.npy')
    grad_path = os.path.join(test_dir, f'{test_str}_grad.npy')

    data = load_numpy_array(data_path)
    result_ = load_numpy_array(result_path)
    grad_ = load_numpy_array(grad_path)

    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)

    result = (x ** exp) * 1.34

    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(x.grad.to_numpy(), grad_, atol=1e-5, rtol=1e-5)


################################ TANH ########################################

@pytest.mark.a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("backend", backend_tests)
def test_tanh_1_student(sizes, backend):
    test_dir = f'./tests/data/tanh_1'
    test_str = '_'.join(map(str, sizes))
    data_path = os.path.join(test_dir, f'{test_str}_data.npy')
    result_path = os.path.join(test_dir, f'{test_str}_result.npy')
    grad_path = os.path.join(test_dir, f'{test_str}_grad.npy')

    data = load_numpy_array(data_path)
    result_ = load_numpy_array(result_path)
    grad_ = load_numpy_array(grad_path)

    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)

    result = x.tanh()

    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(x.grad.to_numpy(), grad_, atol=1e-5, rtol=1e-5)


@pytest.mark.a2_1
@pytest.mark.parametrize("sizes", [(5,), (128,), (1, 64), (128, 256)])
@pytest.mark.parametrize("backend", backend_tests)
def test_tanh_2_student(sizes, backend):
    test_dir = f'./tests/data/tanh_2'
    test_str = '_'.join(map(str, sizes))
    data_path = os.path.join(test_dir, f'{test_str}_data.npy')
    result_path = os.path.join(test_dir, f'{test_str}_result.npy')
    grad_path = os.path.join(test_dir, f'{test_str}_grad.npy')

    data = load_numpy_array(data_path)
    result_ = load_numpy_array(result_path)
    grad_ = load_numpy_array(grad_path)

    x = minitorch.tensor(data.tolist(), backend=shared[backend], requires_grad=True)

    result = (x.tanh() + 1.0) * 0.5
    
    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(x.grad.to_numpy(), grad_, atol=1e-5, rtol=1e-5)