import torch
import torch.nn as nn
import conv2d_optim_wmma as conv2d
import math
from torch.nn.modules.utils import _pair
from typing import Union
from torch.nn.common_types import _size_2_t


class Conv2DFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, params):
        stride  = _pair(params[0])
        padding = _pair(params[1])
        output  = conv2d.forward(input.contiguous(), weight.contiguous(),
                                 stride, padding)
        ctx.save_for_backward(input, weight, params)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, params = ctx.saved_tensors
        stride  = _pair(params[0])
        padding = _pair(params[1])
        grad_input, grad_weight = conv2d.backward(
            input.contiguous(), grad_output.contiguous(),
            weight.contiguous(), stride, padding)
        return grad_input, grad_weight, None


class Conv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        bias: bool = False,
        device=None,
        dtype=None
    ):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Conv2d, self).__init__()
        self.params = torch.Tensor([_pair(stride), _pair(padding)])
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels,
                        *_pair(kernel_size), **factory_kwargs))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, input):
        return Conv2DFunction.apply(input, self.weight, self.params)
