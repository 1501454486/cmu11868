import numpy as np
import time

import torch

from test_utils import TestDecorator
kt = TestDecorator()

import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps
backend = minitorch.TensorBackend(CudaKernelOps)


@kt.case(atol=1e-3, rtol=1e-3, ntest=5)
def test_launch_attn_softmax():
  batch_size, from_len = kt.bs_sl()
  is_dec_self_attn = False

  if is_dec_self_attn:
      to_len = from_len
      is_dec_self_attn_infer = np.random.choice([True, False])
  else:
      _, to_len = kt.bs_sl(batch_size)
      is_dec_self_attn_infer = False

  if is_dec_self_attn_infer:
      to_len = from_len
      from_len = 1
      beam_size = np.random.choice([3, 4, 5])
      batch_size *= beam_size

  nhead = kt.nhead
  print(
      "(batch_size, nhead, from_len, to_len, is_dec_self_attn,"
      f" is_dec_self_attn_infer): ({batch_size}, {nhead}, {from_len}, {to_len},"
      f" {is_dec_self_attn}, {is_dec_self_attn_infer})"
  )

  inp = kt.rand((batch_size, nhead, from_len, to_len))
  if is_dec_self_attn:
      mask = kt.dec_self_attn_mask(to_len) * -1e8
      mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, to_len, to_len]
  else:
      mask = kt.attn_mask(batch_size, to_len) * -1e8
      mask = mask.unsqueeze(1).unsqueeze(1)  # [batch_size, 1, 1, to_len]

  def custom():
    inp_mt = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
    mask_mt = minitorch.tensor(mask.clone().tolist(), backend=backend, requires_grad=True)

    start_time = time.time()
    cust_out = inp_mt.attn_softmax(mask_mt)
    end_time = time.time()

    cust_out = torch.tensor(cust_out._tensor._storage).float().cuda()
    return [
        cust_out,
    ], end_time - start_time

  def baseline():
    inp_mt = minitorch.tensor(inp.clone().tolist(), backend=backend, requires_grad=True)
    mask_mt = minitorch.tensor(mask.clone().tolist(), backend=backend, requires_grad=True)

    start_time = time.time()
    if not is_dec_self_attn_infer:
      res = minitorch.nn.softmax(inp_mt + mask_mt, dim=3)
    else:
      res = minitorch.nn.softmax(inp_mt, dim=3)
    end_time = time.time()

    res = torch.tensor(res._tensor._storage).float().cuda()
    return kt.norm_res_list(res), end_time - start_time

  return custom, baseline


kt.init(device="cuda:0", nhead=8)
kt.run(
  'test_launch_attn_softmax'
)