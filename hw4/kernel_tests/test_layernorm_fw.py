import ctypes
import numpy as np
import time

import torch
import torch.nn.functional as F

from pycuda import gpuarray, driver
import pycuda.autoinit

from test_utils import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)

# Load the shared library
lib = ctypes.CDLL("minitorch/cuda_kernels/layernorm_kernel.so")


@kt.case(atol=1e-2, rtol=1e-3, ntest=5)
def test_launch_layernorm():
    batch_size, seq_len = kt.bs_sl()
    bsz_seq = batch_size * seq_len
    hidden_dim = kt.hidden_dim
    print(
        "(batch_token_num, hidden_dim): "
        f"({bsz_seq}, {hidden_dim})"
    )
    
    custom_res = kt.rand((bsz_seq, hidden_dim))
    inp = kt.rand((bsz_seq, hidden_dim))
    gamma = kt.rand((hidden_dim))
    beta = kt.rand((hidden_dim))
    var = kt.rand((bsz_seq))
    means = kt.rand((bsz_seq))
    
    def custom():
      inp_mt = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
      gamma_mt = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
      beta_mt = minitorch.tensor(beta.clone().tolist(), backend=backend, requires_grad=True)

      start_time = time.time()
      out_mt = inp_mt.layernorm(gamma_mt, beta_mt)
      end_time = time.time()
      out = torch.tensor(out_mt._tensor._storage).cuda()
      return [out], end_time - start_time
    
    def baseline():
      inp_mt = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
      gamma_mt = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
      beta_mt = minitorch.tensor(beta.clone().tolist(), backend=backend, requires_grad=True)

      start_time = time.time()

      x = inp_mt.contiguous()
      batch, dim = x.shape

      mean = x.mean(dim=1).view(batch, 1)
      variance = x.var(dim=1).view(batch, 1)
      x = (x - mean) / ((variance + kt.epsilon) ** 0.5)
      x = gamma_mt * x + beta_mt
      end_time = time.time()

      base = torch.tensor(x._tensor._storage).cuda()
      return [
          base.contiguous(),
      ], end_time - start_time
    
    return custom, baseline


kt.init(device='cuda:0', nhead=8)
kt.run(
  'test_launch_layernorm'
)