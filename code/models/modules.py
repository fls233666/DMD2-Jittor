"""Basic network components for the Jittor DMD2/EDM implementation."""

import numpy as np
import jittor as jt
from jittor import nn


def silu(x):
    # Apply the SiLU activation used by EDM/DMD2 blocks.
    return x * jt.sigmoid(x)


def weight_init(shape, mode, fan_in, fan_out):
    # Initialize weights with the same modes as the official EDM implementation.
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
    # Implement a custom fully connected layer with EDM-style initialization.
    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        init_kwargs = dict(
            mode=init_mode,
            fan_in=in_features,
            fan_out=out_features,
        )
        self.weight = weight_init([out_features, in_features], **init_kwargs)
        self.weight = self.weight * init_weight

        self.bias = None
        if bias:
            self.bias = weight_init([out_features], **init_kwargs) * init_bias

    def execute(self, x):
        # Apply the affine transformation y = xW^T + b.
        weight = self.weight
        bias = self.bias

        if hasattr(weight, "to"):
            weight = weight.to(x.dtype)
        x = x @ weight.transpose(0, 1)

        if bias is not None:
            if hasattr(bias, "to"):
                bias = bias.to(x.dtype)
            x = x + bias

        return x


class Conv2d(nn.Module):
    # Implement EDM-style convolution with optional up/down sampling.
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel,
        bias=True,
        up=False,
        down=False,
        resample_filter=(1, 1),
        fused_resample=False,
        init_mode="kaiming_normal",
        init_weight=1,
        init_bias=0,
    ):
        super().__init__()
        assert not (up and down), "Conv2d cannot upsample and downsample together."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.up = up
        self.down = down
        self.fused_resample = fused_resample

        if kernel:
            init_kwargs = dict(
                mode=init_mode,
                fan_in=in_channels * kernel * kernel,
                fan_out=out_channels * kernel * kernel,
            )
            self.weight = weight_init(
                [out_channels, in_channels, kernel, kernel],
                **init_kwargs,
            )
            self.weight = self.weight * init_weight

            self.bias = None
            if bias:
                self.bias = weight_init([out_channels], **init_kwargs)
                self.bias = self.bias * init_bias
        else:
            self.weight = None
            self.bias = None

        self._resample_filter = None
        if up or down:
            filt = jt.array(resample_filter).float32()
            filt = jt.matmul(filt.reshape(-1, 1), filt.reshape(1, -1))
            filt = filt.reshape(1, 1, filt.shape[0], filt.shape[1])
            filt = filt / (jt.sum(filt) ** 2)
            self._resample_filter = filt.stop_grad()

    def execute(self, x):
        # Apply convolution, optional resampling, and bias addition.
        weight = self.weight
        bias = self.bias
        filt = self._resample_filter

        if weight is not None and hasattr(weight, "to"):
            weight = weight.to(x.dtype)
        if bias is not None and hasattr(bias, "to"):
            bias = bias.to(x.dtype)
        if filt is not None and hasattr(filt, "to"):
            filt = filt.to(x.dtype)

        weight_pad = weight.shape[-1] // 2 if weight is not None else 0
        filt_pad = (filt.shape[-1] - 1) // 2 if filt is not None else 0

        if self.fused_resample and self.up and weight is not None:
            x = self._execute_fused_upsample(x, weight, filt, weight_pad, filt_pad)
        elif self.fused_resample and self.down and weight is not None:
            x = self._execute_fused_downsample(x, weight, filt, weight_pad, filt_pad)
        else:
            x = self._execute_resample_then_conv(x, weight, filt, weight_pad, filt_pad)

        if bias is not None:
            x = x + bias.reshape(1, -1, 1, 1)

        return x

    def _execute_fused_upsample(self, x, weight, filt, weight_pad, filt_pad):
        # Apply fused upsampling followed by learnable convolution.
        filt = (filt * 4).repeat([self.in_channels, 1, 1, 1])
        x = nn.conv_transpose2d(
            x,
            filt,
            bias=None,
            stride=2,
            padding=max(filt_pad - weight_pad, 0),
            groups=self.in_channels,
        )
        x = nn.conv2d(
            x,
            weight,
            bias=None,
            padding=max(weight_pad - filt_pad, 0),
        )
        return x

    def _execute_fused_downsample(self, x, weight, filt, weight_pad, filt_pad):
        # Apply learnable convolution followed by fused downsampling.
        x = nn.conv2d(
            x,
            weight,
            bias=None,
            padding=weight_pad + filt_pad,
        )
        filt = filt.repeat([self.out_channels, 1, 1, 1])
        x = nn.conv2d(
            x,
            filt,
            bias=None,
            stride=2,
            groups=self.out_channels,
        )
        return x

    def _execute_resample_then_conv(self, x, weight, filt, weight_pad, filt_pad):
        # Apply non-fused resampling and optional learnable convolution.
        if self.up:
            up_filt = (filt * 4).repeat([self.in_channels, 1, 1, 1])
            x = nn.conv_transpose2d(
                x,
                up_filt,
                bias=None,
                stride=2,
                padding=filt_pad,
                groups=self.in_channels,
            )

        if self.down:
            down_filt = filt.repeat([self.in_channels, 1, 1, 1])
            x = nn.conv2d(
                x,
                down_filt,
                bias=None,
                stride=2,
                padding=filt_pad,
                groups=self.in_channels,
            )

        if weight is not None:
            x = nn.conv2d(
                x,
                weight,
                bias=None,
                padding=weight_pad,
            )

        return x


class GroupNorm(nn.Module):
    # Normalize features by channel groups with learnable affine parameters.
    def __init__(
        self,
        num_channels,
        num_groups=32,
        min_channels_per_group=4,
        eps=1e-5,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_groups = min(num_groups, num_channels // min_channels_per_group)
        self.num_groups = max(self.num_groups, 1)
        self.eps = eps

        assert num_channels % self.num_groups == 0, (
            f"num_channels={num_channels} must be divisible by "
            f"num_groups={self.num_groups}"
        )

        self.weight = jt.ones([num_channels])
        self.bias = jt.zeros([num_channels])

    def execute(self, x):
        # Apply group normalization to an NCHW-like tensor.
        batch = x.shape[0]
        channels = x.shape[1]
        groups = self.num_groups

        assert channels == self.num_channels, (
            f"Expected input channel {self.num_channels}, got {channels}"
        )

        x_grouped = x.reshape([batch, groups, channels // groups] + list(x.shape[2:]))
        reduce_dims = list(range(2, len(x_grouped.shape)))

        mean = x_grouped.mean(dims=reduce_dims, keepdims=True)
        var = ((x_grouped - mean) ** 2).mean(dims=reduce_dims, keepdims=True)

        x_norm = (x_grouped - mean) / jt.sqrt(var + self.eps)
        x_norm = x_norm.reshape(x.shape)

        affine_shape = [1, channels] + [1] * (len(x.shape) - 2)
        weight = self.weight.reshape(affine_shape)
        bias = self.bias.reshape(affine_shape)

        return x_norm * weight + bias


class PositionalEmbedding(nn.Module):
    # Encode scalar noise labels with deterministic sinusoidal frequencies.
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()

        assert num_channels % 2 == 0, f"num_channels must be even, got {num_channels}"

        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def execute(self, x):
        # Map scalar inputs to sinusoidal positional embeddings.
        half_channels = self.num_channels // 2
        freqs = jt.arange(0, half_channels).float32()

        denom = half_channels - (1 if self.endpoint else 0)
        denom = max(denom, 1)

        freqs = freqs / denom
        freqs = (1.0 / self.max_positions) ** freqs

        x = x.reshape(-1, 1).float32()
        freqs = freqs.reshape(1, -1).float32()
        x = jt.matmul(x, freqs)

        return jt.concat([jt.cos(x), jt.sin(x)], dim=1)


class FourierEmbedding(nn.Module):
    # Encode scalar noise labels with random Fourier features.
    def __init__(self, num_channels, scale=16):
        super().__init__()

        assert num_channels % 2 == 0, f"num_channels must be even, got {num_channels}"

        self.num_channels = num_channels
        self.scale = scale
        self._freqs = (jt.randn([num_channels // 2]) * scale).stop_grad()

    def execute(self, x):
        # Map scalar inputs to random Fourier embeddings.
        x = x.reshape(-1, 1).float32()
        freqs = (2 * np.pi * self._freqs).reshape(1, -1).float32()
        x = jt.matmul(x, freqs)

        return jt.concat([jt.cos(x), jt.sin(x)], dim=1)


def attention_op(q, k):
    # Compute scaled dot-product attention weights.
    assert len(q.shape) == 3, f"q must be 3D [N, C, Q], got {q.shape}"
    assert len(k.shape) == 3, f"k must be 3D [N, C, K], got {k.shape}"
    assert q.shape[0] == k.shape[0], (
        f"batch/head dimension mismatch: q={q.shape}, k={k.shape}"
    )
    assert q.shape[1] == k.shape[1], (
        f"channel dimension mismatch: q={q.shape}, k={k.shape}"
    )

    channels = k.shape[1]
    q_float = q.float32()
    k_float = (k / np.sqrt(channels)).float32()

    weights = jt.matmul(q_float.transpose(0, 2, 1), k_float)
    weights = nn.softmax(weights, dim=2)

    return weights


class AttentionOp:
    # Provide an official-style apply wrapper around attention_op.
    @staticmethod
    def apply(q, k):
        # Compute attention weights through the functional implementation.
        return attention_op(q, k)
