import random
from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np
import numba
import pytest
from hypothesis import given, settings
from hypothesis.strategies import DataObject, data, integers, lists, permutations

import minitorch
from minitorch import MathTestVariable, Tensor, TensorBackend, grad_check

from .strategies import assert_close, small_floats
from .tensor_strategies import assert_close_tensor, shaped_tensors, tensors

one_arg, two_arg, red_arg = MathTestVariable._comp_testing()

shared: Dict[str, TensorBackend] = {}
from minitorch.cuda_kernel_ops import CudaKernelOps

from project.run_sentiment_linear import Linear, Network


if numba.cuda.is_available():
    backend_tests = [pytest.param("cuda")]
    matmul_tests = [pytest.param("cuda")]
    shared["cuda"] = minitorch.TensorBackend(CudaKernelOps)

    
def test_Linear_1() -> None:
    random.seed(42)
    in_size = 50
    out_size = 5
    batch_size = 3
    
    x = [[random.random() for j in range(in_size)] for i in range(batch_size)]
    x = minitorch.tensor(x, backend=shared["cuda"])
    lin_layer = Linear(in_size, out_size)
    out = lin_layer.forward(x)
    
    ans = [[0.284114, -0.076160, 0.060211, -0.054526, 0.170600],
            [0.337623, -0.093076, 0.036470, -0.178649, 0.107644],
            [0.184583, -0.014374, -0.095709, -0.149702, 0.246605]]
    
    ans = minitorch.tensor(ans, backend=shared["cuda"])
    assert_close(out, ans)

def test_Linear_2() -> None:
    
    random.seed(128)
    in_size = 100
    out_size = 9
    batch_size = 5
    
    x = [[random.random() for j in range(in_size)] for i in range(batch_size)]
    x = minitorch.tensor(x, backend=shared["cuda"])
    lin_layer = Linear(in_size, out_size)
    out = lin_layer.forward(x)
    
    ans = [ [0.137780, -0.313636, -0.028814, -0.101995, -0.323188, -0.005756, 0.051685, 0.088392, -0.221262],
            [0.089450, -0.299106, -0.076019, -0.087124, -0.232612, -0.126865, 0.145259, 0.049999, -0.111446],
            [0.079981, -0.301324, 0.084022, -0.151545, -0.393993, -0.249443, 0.202341, 0.001381, -0.257413],
            [0.275477, -0.371416, 0.004781, -0.072271, -0.445334, -0.156278, 0.047011, -0.001491, -0.219601],
            [0.034959, -0.286523, -0.031863, 0.053074, -0.265015, -0.232448, 0.088677, 0.083853, 0.022447]]
    
    ans = minitorch.tensor(ans, backend=shared["cuda"])
    assert_close(out, ans)
    

def test_Network_1() -> None:
    random.seed(21)
    hidden_dim = 3
    embed_dim = 5
    batch_size = 3
    
    x = [[[random.random() for k in range(embed_dim)] for j in range(15)] for i in range(batch_size)]
    x = minitorch.tensor(x, backend=shared["cuda"])
    lin_layer = Network(embedding_dim=embed_dim, hidden_dim=hidden_dim)
    out = lin_layer.forward(x)

    ans = [0.496940, 0.496504, 0.496926]
    ans = minitorch.tensor(ans, backend=shared["cuda"])
    assert_close(out, ans)
    

def test_Network_2() -> None:
    random.seed(235)
    hidden_dim = 100
    embed_dim = 50
    batch_size = 16
    
    x = [[[random.random() for k in range(embed_dim)] for j in range(15)] for i in range(batch_size)]
    x = minitorch.tensor(x, backend=shared["cuda"])
    lin_layer = Network(embedding_dim=embed_dim, hidden_dim=hidden_dim)
    out = lin_layer.forward(x)

    ans = [0.505370, 0.511656, 0.507613, 0.516586, 0.516189, 0.506369, 0.505869, 0.502742, 0.508764, 0.514705, 0.503345, 0.507725, 0.509373, 0.516602, 0.519662, 0.504855]
    ans = minitorch.tensor(ans, backend=shared["cuda"])
    assert_close(out, ans)
    

def test_Network_3() -> None:
    random.seed(89)
    hidden_dim = 200
    embed_dim = 150
    batch_size = 5
    
    x = [[[random.random() for k in range(embed_dim)] for j in range(15)] for i in range(batch_size)]
    x = minitorch.tensor(x, backend=shared["cuda"])
    lin_layer = Network(embedding_dim=embed_dim, hidden_dim=hidden_dim)
    out = lin_layer.forward(x)

    ans = [0.505171, 0.498700, 0.506095, 0.519629, 0.513037]
    ans = minitorch.tensor(ans, backend=shared["cuda"])
    assert_close(out, ans)