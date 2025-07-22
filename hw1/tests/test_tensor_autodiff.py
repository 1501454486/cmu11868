from typing import Callable, Iterable, List, Tuple

import pytest
import numpy as np
from hypothesis import given
from hypothesis.strategies import DataObject, data, lists, permutations

from minitorch import MathTestVariable, Tensor, grad_check, tensor, topological_sort

from .strategies import assert_close, small_floats
from .tensor_strategies import shaped_tensors, tensors

one_arg, two_arg, red_arg = MathTestVariable._comp_testing()


@given(lists(small_floats, min_size=1))
def test_create(t1: List[float]) -> None:
    "Test the ability to create an index a 1D Tensor"
    t2 = tensor(t1)
    for i in range(len(t1)):
        np.testing.assert_allclose(t1[i], t2[i], atol=1e-5, rtol=1e-5)


def test_topo_case1() -> None:
    # Test case 1
    a1, b1 = tensor([[0.88282157]], requires_grad=True), tensor([[0.90170084]], requires_grad=True)
    c1 = 3 * a1 * a1 + 4 * b1 * a1 - a1

    soln = np.array(
        [   [[0.88282157]],[[2.64846471]],
            [[2.33812177]],[[0.90170084]],
            [[3.60680336]],[[3.1841638]],
            [[5.52228558]],[[-0.88282157]],
            [[4.63946401]],
        ]
    )

    topo_order = np.array([x.to_numpy() for x in topological_sort(c1)])[::-1]
    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)
    
    
def test_topo_case2() -> None:
    # Test case 2
    
    a1, b1 = tensor([[0.20914675], [0.65264178]], requires_grad=True), tensor([[0.65394286], [.08218317]], requires_grad=True)
    c1 = 3 * ((b1 * a1) + (2.3412 * b1) * a1) + 1.5
    
    soln = [[[0.65394286],[0.08218317]],
            [[0.20914675],[0.65264178]],
            [[0.13677002],[0.05363617]],
            [[1.53101102],[0.19240724]],
            [[0.32020598],[0.125573  ]],
            [[0.456976  ],[0.17920917]],
            [[1.37092801],[0.53762752]],
            [[2.87092801],[2.03762752]]]

    topo_order = np.array([x.to_numpy() for x in topological_sort(c1)])[::-1]
    assert len(soln) == len(topo_order)
    
    # step through list as entries differ in length
    for topo, sol in zip(topo_order, soln):
        np.testing.assert_allclose(topo, sol, rtol=1e-06, atol=1e-06)
        
        
@given(tensors())
@pytest.mark.parametrize("fn", one_arg)
def test_one_derivative(
    fn: Tuple[str, Callable[[float], float], Callable[[Tensor], Tensor]], t1: Tensor
) -> None:
    "Test the gradient of a one-arg tensor function"
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(data(), tensors())
def test_permute(data: DataObject, t1: Tensor) -> None:
    "Test the permute function"
    permutation = data.draw(permutations(range(len(t1.shape))))

    def permute(a: Tensor) -> Tensor:
        return a.permute(*permutation)

    grad_check(permute, t1)


def test_grad_size() -> None:
    "Test the size of the gradient (from @WannaFy)"
    a = tensor([1], requires_grad=True)
    b = tensor([[1, 1]], requires_grad=True)

    c = (a * b).sum()

    c.backward()
    assert c.shape == (1,)
    assert a.grad is not None
    assert b.grad is not None
    assert a.shape == a.grad.shape
    assert b.shape == b.grad.shape


@given(tensors())
@pytest.mark.parametrize("fn", red_arg)
def test_grad_reduce(
    fn: Tuple[str, Callable[[Iterable[float]], float], Callable[[Tensor], Tensor]],
    t1: Tensor,
) -> None:
    "Test the grad of a tensor reduce"
    name, _, tensor_fn = fn
    grad_check(tensor_fn, t1)


@given(shaped_tensors(2))
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    ts: Tuple[Tensor, Tensor],
) -> None:
    name, _, tensor_fn = fn
    t1, t2 = ts
    grad_check(tensor_fn, t1, t2)


@given(shaped_tensors(2))
@pytest.mark.parametrize("fn", two_arg)
def test_two_grad_broadcast(
    fn: Tuple[str, Callable[[float, float], float], Callable[[Tensor, Tensor], Tensor]],
    ts: Tuple[Tensor, Tensor],
) -> None:
    "Test the grad of a two argument function"
    name, base_fn, tensor_fn = fn
    t1, t2 = ts
    grad_check(tensor_fn, t1, t2)

    # broadcast check
    grad_check(tensor_fn, t1.sum(0), t2)
    grad_check(tensor_fn, t1, t2.sum(0))


def test_fromlist() -> None:
    "Test longer from list conversion"
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t = tensor([[[2, 3, 4], [4, 5, 7]]])
    assert t.shape == (1, 2, 3)


def test_view() -> None:
    "Test view"
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    t2 = t.view(6)
    assert t2.shape == (6,)
    t2 = t2.view(1, 6)
    assert t2.shape == (1, 6)
    t2 = t2.view(6, 1)
    assert t2.shape == (6, 1)
    t2 = t2.view(2, 3)
    assert t.is_close(t2).all().item() == 1.0


@given(tensors())
def test_back_view(t1: Tensor) -> None:
    "Test the graident of view"
    import torch
    def view(a: Tensor) -> Tensor:
        a = a.contiguous()
        if isinstance(a, torch.Tensor):
            return a.view(a.size())
        return a.view(a.size)

    grad_check(view, t1)


def test_fromnumpy() -> None:
    t = tensor([[2, 3, 4], [4, 5, 7]])
    assert t.shape == (2, 3)
    n = t.to_numpy()
    t2 = tensor(n.tolist())
    for ind in t._tensor.indices():
        assert t[ind] == t2[ind]
