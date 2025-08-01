import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

import numpy as np
import torch
import torch.nn as nn
import numba
import os

########################################################################
# ASSIGNMENT 2 PROBLEM 3
########################################################################

_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )]

def load_numpy_array(arr_path):
    with open(arr_path, 'rb') as f:
        return np.load(f)

################################ EMBEDDING ########################################

@pytest.mark.a2_3
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("num_embeddings", [3, 200])
@pytest.mark.parametrize("seq_len", [1, 50])
@pytest.mark.parametrize("embedding_dim", [256])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_embedding_student(batch_size, num_embeddings, seq_len, embedding_dim, backend):
    test_dir = f'./tests/data/embedding'
    test_str = '_'.join(map(str, (batch_size, num_embeddings, seq_len, embedding_dim)))
    
    data_path         = os.path.join(test_dir, f'{test_str}_data.npy')
    layer_weight_path = os.path.join(test_dir, f'{test_str}_layer_weight.npy')
    result_path       = os.path.join(test_dir, f'{test_str}_result.npy')
    weight_grad_path  = os.path.join(test_dir, f'{test_str}_weight_grad.npy')

    data         = load_numpy_array(data_path)
    layer_weight = load_numpy_array(layer_weight_path)
    result_      = load_numpy_array(result_path)
    weight_grad  = load_numpy_array(weight_grad_path)

    X = minitorch.tensor_from_numpy(data, backend=backend)
    layer = minitorch.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, backend=backend)
    layer.weights.value = minitorch.tensor_from_numpy(layer_weight, backend=backend, requires_grad=True)

    result = layer(X)

    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(layer.weights.value.grad.to_numpy(), weight_grad, atol=1e-5, rtol=1e-5)

################################ DROPOUT ########################################

@pytest.mark.a2_3
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_dropout_student(backend):
    np.random.seed(10)
    test_dir = f'./tests/data/dropout'
    result_ = load_numpy_array(os.path.join(test_dir, 'dropout.npy'))

    # Dropout ratio 0 means nothing gets deleted 
    data = np.random.randn(10, 10)
    x = minitorch.tensor(data.tolist(), backend=backend)
    layer = minitorch.Dropout(p_dropout=0)
    result = layer(x)
    np.testing.assert_allclose(result.to_numpy(), data, atol=1e-5, rtol=1e-5)

    # Nothing should be dropped when not training
    layer = minitorch.Dropout(p_dropout=0.5)
    layer.training = False
    result = layer(x)
    np.testing.assert_allclose(result.to_numpy(), data, atol=1e-5, rtol=1e-5)

    layer = minitorch.Dropout(p_dropout = 0.5)
    layer.training = True
    result = layer(x)
    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)


################################ LINEAR ########################################

@pytest.mark.a2_3
@pytest.mark.parametrize("sizes", [(64, 256, 128), (8, 256, 8), (128, 256, 512)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_linear_student(sizes, bias, backend):
    test_dir = f'./tests/data/linear'
    test_str = '_'.join(map(str, sizes + (bias,)))

    data_path         = os.path.join(test_dir, f'{test_str}_data.npy')
    layer_weight_path = os.path.join(test_dir, f'{test_str}_layer_weight.npy')
    result_path       = os.path.join(test_dir, f'{test_str}_result.npy')
    weight_grad_path  = os.path.join(test_dir, f'{test_str}_weight_grad.npy')
    X_grad_path       = os.path.join(test_dir, f'{test_str}_X_grad.npy')
    if bias:
        layer_bias_path = os.path.join(test_dir, f'{test_str}_layer_bias.npy')
        bias_grad_path  = os.path.join(test_dir, f'{test_str}_bias_grad.npy')
    
    data         = load_numpy_array(data_path)
    result_      = load_numpy_array(result_path)
    layer_weight = load_numpy_array(layer_weight_path) 
    weight_grad  = load_numpy_array(weight_grad_path)
    X_grad       = load_numpy_array(X_grad_path)
    if bias: 
        layer_bias = load_numpy_array(layer_bias_path) 
        bias_grad = load_numpy_array(bias_grad_path) 

    m, n, p = sizes
    X = minitorch.tensor_from_numpy(data, backend, True)
    layer = minitorch.Linear(in_size=n, out_size=p, bias=bias, backend=backend)
    layer.weights.value = minitorch.tensor_from_numpy(layer_weight, backend, requires_grad=True)
    if bias:
        layer.bias.value = minitorch.tensor_from_numpy(layer_bias, backend, requires_grad=True)
    
    result = layer(X)

    np.testing.assert_allclose(result.to_numpy(), result_, rtol=1e-5,atol=1e-5)

    result.sum().backward()

    np.testing.assert_allclose(X.grad.to_numpy(), X_grad, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(layer.weights.value.grad.to_numpy(), weight_grad, rtol=1e-5, atol=1e-5)

    if bias:
        np.testing.assert_allclose(layer.bias.value.grad.to_numpy(),bias_grad,rtol=1e-5,atol=1e-5)

################################ LAYERNORM ########################################

@pytest.mark.a2_3
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("dim", [3, 128, 256])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_layernorm_student(batch_size, dim, eps, backend):
    test_dir = f'./tests/data/layernorm'
    test_str = '_'.join(map(str, (batch_size, dim)))

    data_path         = os.path.join(test_dir, f'{test_str}_data.npy')
    result_path       = os.path.join(test_dir, f'{test_str}_result.npy')
    x_grad_path       = os.path.join(test_dir, f'{test_str}_x_grad.npy')
    weight_grad_path  = os.path.join(test_dir, f'{test_str}_weight_grad.npy')

    data         = load_numpy_array(data_path)
    result_      = load_numpy_array(result_path)
    x_grad       = load_numpy_array(x_grad_path)
    weight_grad  = load_numpy_array(weight_grad_path)

    layer = minitorch.LayerNorm1d(dim=dim, eps=eps, backend=backend)
    x_minitorch = minitorch.tensor(data.tolist(), backend=backend, requires_grad=True)
    result = layer(x_minitorch)
    np.testing.assert_allclose(result.to_numpy(), result_, atol=1e-5, rtol=1e-5)
    result.sum().backward()
    np.testing.assert_allclose(x_minitorch.grad.to_numpy(), x_grad, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(layer.weights.value.grad.to_numpy(), weight_grad, atol=1e-5, rtol=1e-5)
