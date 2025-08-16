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


@kt.case(atol=1e-3, rtol=1e-2, ntest=5)
def test_launch_layernorm_bw():
    batch_size, seq_len = kt.bs_sl()
    bsz_seq = batch_size * seq_len
    hidden_dim = kt.hidden_dim
    print(
        "(batch_token_num, hidden_dim): "
        f"({bsz_seq}, {hidden_dim})"
    )

    ln_input = kt.rand((bsz_seq, hidden_dim))
    out_grad = kt.rand((bsz_seq, hidden_dim))
    gamma = kt.rand((hidden_dim))
    beta = kt.rand((hidden_dim))

    inp_numpy = ln_input.cpu().numpy()
    var_mt = minitorch.tensor(inp_numpy.var(axis=1).tolist(), backend=backend)
    means_mt = minitorch.tensor(inp_numpy.mean(axis=1).tolist(), backend=backend)
    var = torch.tensor(var_mt._tensor._storage.astype(np.float32)).cuda()
    mean = torch.tensor(means_mt._tensor._storage.astype(np.float32)).cuda()

    def custom():
      inp_mt = minitorch.tensor(ln_input.clone().tolist(), backend=backend, requires_grad=True)
      gamma_mt = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
      beta_mt = minitorch.tensor(beta.clone().tolist(), backend=backend, requires_grad=True)
      out_mt = inp_mt.layernorm(gamma_mt, beta_mt)
      out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True) 

      start_time = time.time()
      out_mt.backward(out_grad_mt)
      end_time = time.time()
      
      inp_grad = torch.tensor(inp_mt.grad.to_numpy(), dtype=torch.float32).cuda()
      gamma_grad = torch.tensor(gamma_mt.grad.to_numpy(), dtype=torch.float32).cuda()
      betta_grad = torch.tensor(beta_mt.grad.to_numpy(), dtype=torch.float32).cuda()
        
      return [gamma_grad, betta_grad, inp_grad], end_time - start_time

    def baseline():
      f_input = minitorch.tensor(ln_input.clone().tolist(), backend=backend, requires_grad=True)
      f_gamma = minitorch.tensor(gamma.clone().tolist(), backend=backend, requires_grad=True)
      f_out_grad = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True) 

      start_time = time.time()

      f_means = f_input.mean(dim=1)
      f_vars = f_input.var(dim=1)
      f_stds = minitorch.tensor(np.sqrt(f_vars.to_numpy()).reshape(-1, 1).tolist(), backend=backend, requires_grad=True)

      xhat = (f_input - f_means) / f_stds
      dxhat = f_out_grad * f_gamma
      f_betta_grad = f_out_grad.sum(dim=0)
      f_gamma_grad = (f_out_grad * xhat).sum(dim=0)
      dinp = dxhat.sum(dim=1) + xhat * (dxhat * xhat).sum(dim=1)
      dinp = dxhat - dinp / hidden_dim
      dinp = dinp / f_stds

      end_time = time.time()

      inp_grad = torch.tensor(dinp.to_numpy(), dtype=torch.float32).cuda()
      gamma_grad = torch.tensor(f_gamma_grad.to_numpy(), dtype=torch.float32).cuda()
      betta_grad = torch.tensor(f_betta_grad.to_numpy(), dtype=torch.float32).cuda()

      return kt.norm_res_list(gamma_grad, betta_grad, inp_grad), end_time - start_time

    return custom, baseline


kt.init(device='cuda:0', nhead=8)
kt.run(
  'test_launch_layernorm_bw'
)