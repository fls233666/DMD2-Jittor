# 网络中的基础组件代码

import math
import numpy as np
import jittor as jt
from jittor import nn

def weight_init(shape, mode, fan_in, fan_out):
    if mode == "xavier_uniform":
        return np.sqrt(6 / (fan_in + fan_out)) * (jt.rand(shape) * 2 - 1)

    if mode == "xavier_normal":
        return np.sqrt(2 / (fan_in + fan_out)) * jt.randn(shape)

    if mode == "kaiming_uniform":
        return np.sqrt(3 / fan_in) * (jt.rand(shape) * 2 - 1)

    if mode == "kaiming_normal":
        return np.sqrt(1 / fan_in) * jt.randn(shape)

    raise ValueError(f'Invalid init mode "{mode}"')

class Linear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, init_mode="kaiming_normal", init_weight=1, init_bias=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        init_kwargs = dict(mode=init_mode, fan_in=in_features, fan_out=out_features)
        self.weight = nn.weight_init([out_features, in_features], **init_kwargs) * init_weight
        self.bias = nn.weight_init([out_features], **init_kwargs) * init_bias if bias else None

    def execute(self, x):
        x = x @ self.weight.to(x.dtype).t()
        if self.bias is not None:
            x = self.bias.to(x.dtype)
        return x
    
