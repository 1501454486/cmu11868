import pytest
from hypothesis import given

import minitorch
from minitorch import Tensor
import numpy as np
import torch
from .strategies import assert_close
from .tensor_strategies import tensors


datatype = np.float32

@pytest.mark.task4_3
@given(tensors(shape=(1, 1, 4, 4)))
def test_avg(t: Tensor) -> None:
    out = minitorch.avgpool2d(t, (2, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(2)]) / 4.0
    )

    out = minitorch.avgpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(2) for j in range(1)]) / 2.0
    )

    out = minitorch.avgpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], sum([t[0, 0, i, j] for i in range(1) for j in range(2)]) / 2.0
    )
    minitorch.grad_check(lambda t: minitorch.avgpool2d(t, (2, 2)), t)


@pytest.mark.task4_4
@given(tensors(shape=(2, 3, 4)))
def test_max(t: Tensor) -> None:
    # ASSIGN4.4
    t.requires_grad_(True)
    t2 = t + minitorch.rand((2, 3, 4)) * 1e-6
    argt2 = minitorch.argmax(t2, 2)
    v = minitorch.max(t2, minitorch.tensor([2]))
    v.sum().backward()
    for i in range(2):
        for j in range(3):
            m = -1e9
            ind = -1
            for k in range(4):
                if t2[i, j, k] > m:
                    m = t2[i, j, k]
                    ind = k

            assert_close(v[i, j, 0], m)
            assert t.grad is not None
            assert t.grad[i, j, ind] == 1.0
            assert t.grad[i, j, ind] == argt2[i, j, ind]
            if ind > 0:
                assert t.grad[i, j, ind - 1] == 0.0
    # END ASSIGN4.4


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_max_pool(t: Tensor) -> None:
    out = minitorch.maxpool2d(t, (2, 2))
    print(out)
    print(t)
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(2)])
    )

    out = minitorch.maxpool2d(t, (2, 1))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(2) for j in range(1)])
    )

    out = minitorch.maxpool2d(t, (1, 2))
    assert_close(
        out[0, 0, 0, 0], max([t[0, 0, i, j] for i in range(1) for j in range(2)])
    )


@pytest.mark.task4_4
@given(tensors())
def test_drop(t: Tensor) -> None:
    q = minitorch.dropout(t, 0.0)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]
    q = minitorch.dropout(t, 1.0)
    assert q[q._tensor.sample()] == 0.0
    q = minitorch.dropout(t, 1.0, ignore=True)
    idx = q._tensor.sample()
    assert q[idx] == t[idx]


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    x = q.sum(dim=3)
    assert_close(x[0, 0, 0, 0], 1.0)

    q = minitorch.softmax(t, 1)
    x = q.sum(dim=1)
    assert_close(x[0, 0, 0, 0], 1.0)

    minitorch.grad_check(lambda a: minitorch.softmax(a, dim=2), t)


@pytest.mark.task4_4
@given(tensors(shape=(1, 1, 4, 4)))
def test_log_softmax(t: Tensor) -> None:
    q = minitorch.softmax(t, 3)
    q2 = minitorch.logsoftmax(t, 3).exp()
    for i in q._tensor.indices():
        assert_close(q[i], q2[i])

    minitorch.grad_check(lambda a: minitorch.logsoftmax(a, dim=2), t)

########################################################################
# ASSIGNMENT 2 TESTS
########################################################################

import numba

GENERAL_SHAPES = [(2, 5), (3, 8), (64, 128)]
_BACKENDS = [pytest.param(
                 minitorch.TensorBackend(minitorch.CudaKernelOps), 
                 marks=pytest.mark.skipif(not numba.cuda.is_available(), reason="No GPU")
             )] 


@pytest.mark.parametrize("sizes", GENERAL_SHAPES)
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_a2_gelu(sizes, backend):
    x = np.random.randn(*sizes).astype(datatype)
    A = minitorch.tensor(x.tolist(), backend=backend, requires_grad=True)
    A_ = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    result = minitorch.GELU(A)
    result_ = torch.nn.functional.gelu(A_, approximate='tanh')

    np.testing.assert_allclose(
        result.to_numpy(),
        result_.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    result.sum().backward()
    result_.sum().backward()

    np.testing.assert_allclose(
        A.grad.to_numpy(),
        A_.grad.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )


@pytest.mark.parametrize("sizes", GENERAL_SHAPES)
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_a2_logsumexp(sizes, backend):
    dim=1
    
    x = np.random.randn(*sizes).astype(datatype)
    A = minitorch.tensor(x.tolist(), backend=backend)
    _A = torch.tensor(x, dtype=torch.float32, requires_grad=True)

    result = minitorch.logsumexp(A, dim=dim)
    _result = torch.logsumexp(_A, dim=dim, keepdim=True)

    np.testing.assert_allclose(
        result.to_numpy(),
        _result.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    result.sum().backward()
    _result.sum().backward()

    np.testing.assert_allclose(
        A.grad.to_numpy(),
        _A.grad.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )


@pytest.mark.parametrize("batches", [1, 64, 256])
@pytest.mark.parametrize("classes", [2, 32, 128, 10000])
@pytest.mark.parametrize("backend", _BACKENDS, ids=["CudaKernelOps"])
def test_a2_softmax_loss(batches, classes, backend):
    np.random.seed(10)
    # Classes=1 may be buggy

    logits_np = np.random.randn(batches, classes).astype(datatype)
    targets_np = np.random.randint(low=0, high=classes, size=(batches,))


    logits = minitorch.tensor_from_numpy(logits_np, backend=backend, requires_grad=True)
    targets = minitorch.tensor_from_numpy(targets_np, backend=backend, requires_grad=True)

    _logits = torch.tensor(logits_np, dtype=torch.float32, requires_grad=True)
    _targets = torch.tensor(targets_np, dtype=torch.long)

    result = minitorch.softmax_loss(logits, targets)
    _result_none = torch.nn.functional.cross_entropy(_logits, _targets, reduction='none')

    np.testing.assert_allclose(
        result.to_numpy(),
        _result_none.detach().numpy(),
        atol=1e-5,
        rtol=1e-5
    )

    result.sum().backward()
    _result_none.sum().backward()

    np.testing.assert_allclose(
        logits.to_numpy(), 
        _logits.detach().numpy(),
        atol=1e-5, 
        rtol=1e-5
    )