import numpy as np
import time

import torch

from test_utils import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)


@kt.case(atol=1e-2, rtol=1e-3, ntest=5)
def test_launch_attn_softmax_bw():
  nhead = kt.nhead
  batch_size, from_len = kt.bs_sl()
  _, to_len = kt.bs_sl(batch_size)

  print(
      "(batch_size, nhead, from_len, to_len): "
      f"({batch_size}, {nhead}, {from_len}, {to_len})"
  )

  out_grad = kt.rand((batch_size, nhead, from_len, to_len))
  inp = kt.rand((batch_size, nhead, from_len, to_len))

  def custom():
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)
    inp_mt = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
    mask_mt = minitorch.tensor(np.zeros((batch_size, 1, 1, to_len)).tolist(), backend=backend, requires_grad=True)
    soft_inp_mt = inp_mt.attn_softmax(mask_mt)

    start_time = time.time()
    soft_inp_mt.backward(out_grad_mt)
    end_time = time.time()

    inp_grad = torch.tensor(inp_mt.grad.to_numpy(), dtype=torch.float32).cuda()
    return [
        inp_grad,
    ], end_time - start_time

  def baseline():
    out_grad_mt = minitorch.tensor(out_grad.clone().tolist(), backend=backend, requires_grad=True)
    inp_mt = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
    soft_inp_mt = minitorch.nn.softmax(inp_mt, dim=3)

    start_time = time.time()
    tsum = out_grad_mt * soft_inp_mt
    tsum = tsum.sum(dim=3).view(tsum.shape[0], tsum.shape[1], tsum.shape[2], 1)
    res = soft_inp_mt * (out_grad_mt - tsum)
    end_time = time.time()

    res = torch.tensor(res._tensor._storage).float().cuda()
    return kt.norm_res_list(res), end_time - start_time

  return custom, baseline


kt.init(device="cuda:0", nhead=8)
kt.run(
  'test_launch_attn_softmax_bw'
)
