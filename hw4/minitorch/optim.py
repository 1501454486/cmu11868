from typing import Sequence

from .module import Parameter
from .scalar import Scalar

import math
import numpy as np


class Optimizer:
    def __init__(self, parameters: Sequence[Parameter]):
        self.parameters = parameters
    
    def zero_grad(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue
            if hasattr(p.value, "derivative"):
                if p.value.derivative is not None:
                    p.value.derivative = None
            if hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.value.grad = None

    def _print(self) -> None:
        for param in self.parameters:
            if param.value is None:
                continue
            print(param.value.shape)
            print(param.value.grad)


class Adam(Optimizer):
    def __init__(self,
                 parameters: Sequence[Parameter],
                 lr=1e-3,
                 beta1=0.9,
                 beta2=0.999,
                 eps=1e-8):
        super().__init__(parameters=parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self._states = {}
        for p in parameters:
            self._states[id(p)] = {}

    def step(self):
        for p in self.parameters:
            if p.value is None:
                continue
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    grad = p.value.grad
                    state = self._states[id(p)]

                    # State initialization
                    if len(state) == 0:
                        # print(f'initializing state: tensor_shape = {p.value.shape}')
                        state['step'] = 0
                        state['exp_avg'] = grad.zeros()
                        state['exp_avg_sq'] = grad.zeros()

                    state['step'] += 1
                    state['exp_avg'] = state['exp_avg'] * self.beta1 + (1 - self.beta1) * grad
                    state['exp_avg_sq'] = state['exp_avg_sq'] * self.beta2 + (1 - self.beta1) * grad ** 2

                    # denom = exp_avg_sq.sqrt().add_(group['eps'])
                    denom = state['exp_avg_sq'] ** 0.5 + self.eps

                    bias_correction1 = 1. - self.beta1 ** state['step']
                    bias_correction2 = 1. - self.beta2 ** state['step']

                    step_size = self.lr * math.sqrt(
                        bias_correction2) / bias_correction1

                    p.update(p.value - step_size * state['exp_avg'] / denom)


# class Adam(Optimizer):
#     def __init__(
#             self,
#             parameters: Sequence[Parameter],
#             lr: float = 0.1,
#             beta1: float = 0.9,
#             beta2: float = 0.999,
#             eps: float = 1e-8,
#             weight_decay: float = 0.0,
#         ):
#         super().__init__(parameters)
#         self.lr = lr
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.eps = eps
#         self.weight_decay = weight_decay
#         self.t = 0
#
#         self.m = {}
#         self.v = {}
#
#     def step(self) -> None:
#         print("STEP NUM PARAMS", len(self.parameters))
#         for w in self.parameters:
#             if w.value is None:
#                 continue
#             elif hasattr(w.value, "grad"):
#                 if w.value.grad is not None:
#                     res = None
#                     self.t += 1
#                     ### BEGIN SOLUTION
#                     if self.weight_decay != 0:
#                         w_grad = w.value.grad + self.weight_decay * w.value
#                     else:
#                         w_grad = w.value.grad
#
#                     if id(w) in self.m:
#                         self.m[id(w)] = (
#                             self.beta1 * self.m[id(w)] + (1 - self.beta1) * w_grad
#                         ).detach()
#                         self.v[id(w)].data = (
#                             self.beta2 * self.v[id(w)] + (1 - self.beta2) * (w_grad ** 2)
#                         ).detach()
#                     else:
#                         self.m[id(w)] = ((1 - self.beta1) * w_grad).detach()
#                         self.v[id(w)] = (1 - self.beta2) * (w_grad ** 2).detach()
#
#                     m_hat = self.m[id(w)] / (1 - self.beta1 ** self.t)
#                     v_hat = self.v[id(w)] / (1 - self.beta2 ** self.t)
#
#                     res = w.value - (self.lr * m_hat) / (v_hat ** 0.5 + self.eps)
#                     assert(res.requires_grad())
#                     assert not(np.isnan(res.to_numpy()).any())
#                     ###
#
#                     w.update(res)


class SGD(Optimizer):
    def __init__(self, parameters: Sequence[Parameter], lr: float = 1.0):
        super().__init__(parameters)
        self.lr = lr

    def step(self) -> None:
        for p in self.parameters:
            if p.value is None:
                continue
            elif hasattr(p.value, "grad"):
                if p.value.grad is not None:
                    p.update(p.value - self.lr * p.value.grad)
