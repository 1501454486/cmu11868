import pytest
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

import numpy as np
import torch
import torch.nn as nn
import numba

np.random.seed(3)


_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )]


@pytest.mark.a2_2
@pytest.mark.parametrize("batch_size", [1, 5])
@pytest.mark.parametrize("num_embeddings", [3, 200])
@pytest.mark.parametrize("seq_len", [1, 50])
@pytest.mark.parametrize("embedding_dim", [256])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_embedding(batch_size, num_embeddings, seq_len, embedding_dim, backend):
    np.random.seed(11868)
    torch.manual_seed(10)

    data = np.random.randint(0, num_embeddings, size=(batch_size, seq_len))
    X = minitorch.tensor_from_numpy(data, backend=backend)
    X_ = torch.tensor(data)

    layer_ = torch.nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)

    layer = minitorch.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim, backend=backend)
    layer.weights.value = minitorch.tensor_from_numpy(layer_.weight.detach().numpy(), backend=backend, requires_grad=True)

    result = layer(X)
    result_ = layer_(X_)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        layer.weights.value.grad.to_numpy(),
        layer_.weight.grad.detach().numpy(), 
        atol=1e-5, 
        rtol=1e-5
    )


def test_dropout():
    pass

@pytest.mark.a2_2
@pytest.mark.parametrize("sizes", [(64, 256, 128)])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_linear(sizes, bias, backend):
    np.random.seed(10)
    torch.manual_seed(10)
    
    m, n, p = sizes
    data = np.random.randn(m, n)
    X = minitorch.tensor_from_numpy(data, backend, True)
    X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_ = torch.nn.Linear(in_features=n, out_features=p, bias=bias, dtype=torch.float32)
    layer = minitorch.Linear(in_size=n, out_size=p, bias=bias, backend=backend)

    weights = layer_.weight.detach().numpy().T
    layer.weights.value = minitorch.tensor_from_numpy(weights.copy(), backend, requires_grad=True)
    if bias:    
        b = layer_.bias.detach().numpy()
        layer.bias.value = minitorch.tensor_from_numpy(b.copy(), backend, requires_grad=True)
    
    result = layer(X)
    result_ = layer_(X_)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(),
        X_.grad.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    np.testing.assert_allclose(
        layer.weights.value.grad.to_numpy(),
        layer_.weight.grad.detach().numpy().T, 
        rtol=1e-5,
        atol=1e-5
    )

    if bias:
        np.testing.assert_allclose(
            layer.bias.value.grad.to_numpy(),
            layer_.bias.grad.detach().numpy(), 
            rtol=1e-5,
            atol=1e-5
        )


@pytest.mark.parametrize("sizes", [(64, 128, 256)])
@pytest.mark.parametrize("bias", [False])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_linear_double(sizes, bias, backend):
    np.random.seed(10)
    torch.manual_seed(10)
    
    bs, n_embd, middle_dim = sizes
    data = np.random.randn(bs, n_embd)
    X = minitorch.tensor_from_numpy(data, backend, True)
    X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

    layer_1_ = torch.nn.Linear(in_features=n_embd, out_features=middle_dim, bias=bias, dtype=torch.float32)
    layer_2_ = torch.nn.Linear(in_features=middle_dim, out_features=n_embd, bias=bias, dtype=torch.float32)
    layer_1 = minitorch.Linear(in_size=n_embd, out_size=middle_dim, bias=bias, backend=backend)
    layer_2 = minitorch.Linear(in_size=middle_dim, out_size=n_embd, bias=bias, backend=backend)

    weights_1 = layer_1_.weight.detach().numpy().T
    weights_2 = layer_2_.weight.detach().numpy().T
    layer_1.weights.value = minitorch.tensor_from_numpy(weights_1.copy(), backend, requires_grad=True)
    layer_2.weights.value = minitorch.tensor_from_numpy(weights_2.copy(), backend, requires_grad=True)
    if bias:    
        b_1 = layer_1_.bias.detach().numpy()
        layer_1.bias.value = minitorch.tensor_from_numpy(b_1.copy(), backend, requires_grad=True)
        b_2 = layer_2_.bias.detach().numpy()
        layer_2.bias.value = minitorch.tensor_from_numpy(b_2.copy(), backend, requires_grad=True)
    
    result = layer_2(layer_1(X))
    result_ = layer_2_(layer_1_(X_))

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        X.grad.to_numpy(),
        X_.grad.detach().numpy(), 
        rtol=1e-5,
        atol=1e-5
    )

    np.testing.assert_allclose(
        layer_1.weights.value.grad.to_numpy(),
        layer_1_.weight.grad.detach().numpy().T, 
        rtol=1e-5,
        atol=1e-5
    )
    np.testing.assert_allclose(
        layer_2.weights.value.grad.to_numpy(),
        layer_2_.weight.grad.detach().numpy().T, 
        rtol=1e-5,
        atol=1e-5
    )

    if bias:
        np.testing.assert_allclose(
            layer_1.bias.value.grad.to_numpy(),
            layer_1_.bias.grad.detach().numpy(), 
            rtol=1e-5,
            atol=1e-5
        )
        np.testing.assert_allclose(
            layer_2.bias.value.grad.to_numpy(),
            layer_2_.bias.grad.detach().numpy(), 
            rtol=1e-5,
            atol=1e-5
        )

@pytest.mark.a2_2
@pytest.mark.parametrize("batch_size", [1, 64])
@pytest.mark.parametrize("dim", [3, 128, 256])
@pytest.mark.parametrize("eps", [1e-5])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_layernorm(batch_size, dim, eps, backend):
    np.random.seed(5)

    x = np.random.randn(batch_size, dim)

    layer = minitorch.LayerNorm1d(
        dim=dim, eps=eps, backend=backend
    )

    layer_ = torch.nn.LayerNorm(
        normalized_shape=dim, eps=eps
    )
    x_minitorch = minitorch.tensor(x.tolist(), backend=backend)
    x_torch = torch.tensor(x.tolist(), dtype=torch.float32, requires_grad=True)

    result = layer(x_minitorch)
    result_ = layer_(x_torch)

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        x_minitorch.grad.to_numpy(),
        x_torch.grad.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )