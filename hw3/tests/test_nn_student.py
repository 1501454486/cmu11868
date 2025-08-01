import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor
import numpy as np
import torch
from .strategies import assert_close
from .tensor_strategies import tensors


datatype = np.float32

########################################################################
# ASSIGNMENT 2 PROBLEM 2
########################################################################

import numba
import os

GENERAL_SHAPES = [(2, 5), (3, 8), (64, 128)]
_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(minitorch.CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )] 

def load_numpy_array(arr_path):
    with open(arr_path, 'rb') as f:
        return np.load(f)

################################ GELU ##########################################

@pytest.mark.a2_2
@pytest.mark.parametrize("sizes", GENERAL_SHAPES)
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_gelu_student(sizes, backend):
    test_dir = f'./tests/data/gelu'
    test_str = '_'.join(map(str, sizes))
    data_path = os.path.join(test_dir, f'{test_str}_data.npy')
    result_path = os.path.join(test_dir, f'{test_str}_result.npy')
    grad_path = os.path.join(test_dir, f'{test_str}_grad.npy')

    x = load_numpy_array(data_path)
    result_ = load_numpy_array(result_path)
    grad_ = load_numpy_array(grad_path)    

    A = minitorch.tensor(x.tolist(), backend=backend, requires_grad=True)

    result = minitorch.GELU(A)

    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(A.grad.to_numpy(), grad_, atol=1e-5, rtol=1e-5)

################################ LOGSUMEXP ##########################################

@pytest.mark.a2_2
@pytest.mark.parametrize("sizes", GENERAL_SHAPES)
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_logsumexp_student(sizes, backend):
    test_dir = f'./tests/data/logsumexp'
    test_str = '_'.join(map(str, sizes))
    data_path = os.path.join(test_dir, f'{test_str}_data.npy')
    result_path = os.path.join(test_dir, f'{test_str}_result.npy')
    grad_path = os.path.join(test_dir, f'{test_str}_grad.npy')

    x = load_numpy_array(data_path)
    result_ = load_numpy_array(result_path)
    grad_ = load_numpy_array(grad_path)    

    dim=1

    A = minitorch.tensor(x.tolist(), backend=backend, requires_grad=True)

    result = minitorch.logsumexp(A, dim=dim)

    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(A.grad.to_numpy(), grad_, atol=1e-5, rtol=1e-5)

################################ SOFTMAX LOSS ##########################################

@pytest.mark.a2_2
@pytest.mark.parametrize("batches", [1, 64, 256])
@pytest.mark.parametrize("classes", [2, 32, 128, 10000])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_softmax_loss_student(batches, classes, backend):
    test_dir = f'./tests/data/softmax_loss'
    test_str = str(batches) + '_' + str(classes)
    logits_path = os.path.join(test_dir, f'{test_str}_logits.npy')
    targets_path = os.path.join(test_dir, f'{test_str}_targets.npy')
    result_path = os.path.join(test_dir, f'{test_str}_result.npy')
    logits_grad_path = os.path.join(test_dir, f'{test_str}_logits_grad.npy')

    logits_np = load_numpy_array(logits_path)
    targets_np = load_numpy_array(targets_path)
    _result = load_numpy_array(result_path)
    logits_grad = load_numpy_array(logits_grad_path) 

    logits = minitorch.tensor_from_numpy(logits_np, backend=backend, requires_grad=True)
    targets = minitorch.tensor_from_numpy(targets_np, backend=backend, requires_grad=True)

    result = minitorch.softmax_loss(logits, targets)

    np.testing.assert_allclose(result.to_numpy(), _result, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(logits.grad.to_numpy(), logits_grad, atol=1e-5, rtol=1e-5)