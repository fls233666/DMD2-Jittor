"""Basic network components for the Jittor DMD2/EDM implementation."""

import os
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


def _cudnn_conv_backward_w(x, grad_output, weight_shape, padding_h, padding_w):
    if not jt.flags.use_cuda:
        return None

    cudnn = getattr(jt, "cudnn", None)
    ops = getattr(cudnn, "ops", None)
    if ops is None or not hasattr(ops, "cudnn_conv_backward_w"):
        raise RuntimeError(
            "DMD2_SAFE_*_DW_MODE=cudnn requires Jittor cuDNN ops, but "
            "jt.cudnn.ops.cudnn_conv_backward_w is unavailable. Set conv_opt=0 "
            "before importing jittor, or use DMD2_SAFE_*_DW_MODE=code."
        )

    return ops.cudnn_conv_backward_w(
        x,
        grad_output,
        int(weight_shape[2]),
        int(weight_shape[3]),
        1,
        1,
        int(padding_h),
        int(padding_w),
        1,
        1,
        1,
    )


class SafePointwiseConv2d(jt.Function):
    # Override 1x1 conv backward to avoid Jittor fused dW kernels on large maps.
    def execute(self, x, weight):
        self.x = x
        self.weight = weight
        return nn.conv2d(x, weight, bias=None, padding=0)

    def grad(self, grad_output):
        if grad_output is None:
            return None, None

        x = self.x
        weight = self.weight

        if os.environ.get("DMD2_SAFE_POINTWISE_CONV_INPUT_GRAD", "1") == "0":
            dx = None
        elif os.environ.get("DMD2_SAFE_POINTWISE_CONV_CODE_GRAD", "1") != "0":
            dx = self._input_grad_code(grad_output, weight, x.shape)
        else:
            dx = nn.conv_transpose2d(grad_output, weight, bias=None, padding=0)

        if os.environ.get("DMD2_SAFE_POINTWISE_CONV_WEIGHT_GRAD", "1") == "0":
            dweight = None
        else:
            dw_mode = os.environ.get("DMD2_SAFE_POINTWISE_CONV_DW_MODE", "cudnn")
            if dw_mode == "cudnn":
                dweight = _cudnn_conv_backward_w(
                    x=x,
                    grad_output=grad_output,
                    weight_shape=weight.shape,
                    padding_h=0,
                    padding_w=0,
                )
                if dweight is None:
                    dweight = self._weight_grad_code(x, grad_output, weight.shape)
            elif dw_mode == "code":
                dweight = self._weight_grad_code(x, grad_output, weight.shape)
            elif dw_mode == "matmul":
                dweight = self._weight_grad_matmul_chunked(
                    x=x,
                    grad_output=grad_output,
                    weight_shape=weight.shape,
                )
            elif dw_mode == "reduce5d":
                dweight = self._weight_grad_reduce5d_chunked(
                    x=x,
                    grad_output=grad_output,
                    weight_shape=weight.shape,
                )
            else:
                dweight = self._weight_grad_reduce_chunked(
                    x=x,
                    grad_output=grad_output,
                    weight_shape=weight.shape,
                )

        return dx, dweight

    @staticmethod
    def _weight_grad_reduce_chunked(x, grad_output, weight_shape):
        batch, in_channels, height, width = x.shape
        out_channels = grad_output.shape[1]

        chunk_size = int(os.environ.get("DMD2_SAFE_POINTWISE_CONV_DW_CHUNK", "32"))
        chunk_size = max(1, chunk_size)

        chunks = []
        for start in range(0, in_channels, chunk_size):
            size = min(chunk_size, in_channels - start)
            channel_grads = []
            for offset in range(size):
                channel = start + offset
                x_channel = x.reindex(
                    [batch, 1, height, width],
                    ["i0", str(channel), "i2", "i3"],
                )
                grad_channel = (grad_output * x_channel).sum(dims=[0, 2, 3])
                channel_grads.append(grad_channel.reshape([out_channels, 1, 1, 1]))

            if len(channel_grads) == 1:
                chunks.append(channel_grads[0])
            else:
                chunks.append(jt.concat(channel_grads, dim=1))

        if len(chunks) == 1:
            return chunks[0].reshape(list(weight_shape))
        return jt.concat(chunks, dim=1).reshape(list(weight_shape))

    @staticmethod
    def _weight_grad_reduce5d_chunked(x, grad_output, weight_shape):
        batch, in_channels, height, width = x.shape
        out_channels = grad_output.shape[1]

        chunk_size = int(os.environ.get("DMD2_SAFE_POINTWISE_CONV_DW_CHUNK", "32"))
        chunk_size = max(1, chunk_size)

        grad_view = grad_output.reshape([batch, out_channels, 1, height, width])
        chunks = []
        for start in range(0, in_channels, chunk_size):
            size = min(chunk_size, in_channels - start)
            x_view = x.reindex(
                [batch, 1, size, height, width],
                ["i0", f"i2+{start}", "i3", "i4"],
            )
            grad_chunk = (grad_view * x_view).sum(dims=[0, 3, 4])
            chunks.append(grad_chunk.reshape([out_channels, size, 1, 1]))

        if len(chunks) == 1:
            return chunks[0].reshape(list(weight_shape))
        return jt.concat(chunks, dim=1).reshape(list(weight_shape))

    @staticmethod
    def _weight_grad_matmul_chunked(x, grad_output, weight_shape):
        batch, in_channels, height, width = x.shape
        out_channels = weight_shape[0]
        spatial_size = height * width
        flat_size = batch * spatial_size

        x_flat = x.reindex(
            [flat_size, in_channels],
            [
                f"i0/{spatial_size}",
                "i1",
                f"(i0%{spatial_size})/{width}",
                f"i0%{width}",
            ],
        )
        grad_flat = grad_output.reindex(
            [flat_size, out_channels],
            [
                f"i0/{spatial_size}",
                "i1",
                f"(i0%{spatial_size})/{width}",
                f"i0%{width}",
            ],
        )
        dweight = SafePointwiseConv2d._weight_grad_chunked(
            x_flat=x_flat,
            grad_flat=grad_flat,
            out_channels=out_channels,
            in_channels=in_channels,
            flat_size=flat_size,
        )
        return dweight.reshape(list(weight_shape))

    @staticmethod
    def _input_grad_code(grad_output, weight, x_shape):
        return jt.code(
            list(x_shape),
            grad_output.dtype,
            [grad_output, weight],
            cpu_src="""
                for (int b = 0; b < out_shape0; ++b)
                for (int ic = 0; ic < out_shape1; ++ic)
                for (int h = 0; h < out_shape2; ++h)
                for (int w = 0; w < out_shape3; ++w) {
                    float acc = 0.0f;
                    for (int oc = 0; oc < in0_shape1; ++oc) {
                        acc += (float)@in0(b, oc, h, w) * (float)@in1(oc, ic, 0, 0);
                    }
                    @out(b, ic, h, w) = acc;
                }
            """,
            cuda_src="""
                __global__ static void pointwise_input_grad_kernel(@ARGS_DEF) {
                    @PRECALC
                    int total = out_shape0 * out_shape1 * out_shape2 * out_shape3;
                    int step = blockDim.x * gridDim.x;
                    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
                         idx < total; idx += step) {
                        int w = idx % out_shape3;
                        int tmp = idx / out_shape3;
                        int h = tmp % out_shape2;
                        tmp /= out_shape2;
                        int ic = tmp % out_shape1;
                        int b = tmp / out_shape1;

                        float acc = 0.0f;
                        for (int oc = 0; oc < in0_shape1; ++oc) {
                            acc += (float)@in0(b, oc, h, w) * (float)@in1(oc, ic, 0, 0);
                        }
                        @out(b, ic, h, w) = acc;
                    }
                }

                int total = out_shape0 * out_shape1 * out_shape2 * out_shape3;
                int threads = 256;
                int blocks = (total + threads - 1) / threads;
                if (blocks > 65535) blocks = 65535;
                pointwise_input_grad_kernel<<<blocks, threads>>>(@ARGS);
            """,
        )

    @staticmethod
    def _weight_grad_code(x, grad_output, weight_shape):
        return jt.code(
            list(weight_shape),
            grad_output.dtype,
            [x, grad_output],
            cpu_src="""
                for (int oc = 0; oc < out_shape0; ++oc)
                for (int ic = 0; ic < out_shape1; ++ic) {
                    float acc = 0.0f;
                    for (int b = 0; b < in0_shape0; ++b)
                    for (int h = 0; h < in0_shape2; ++h)
                    for (int w = 0; w < in0_shape3; ++w) {
                        acc += (float)@in1(b, oc, h, w) * (float)@in0(b, ic, h, w);
                    }
                    @out(oc, ic, 0, 0) = acc;
                }
            """,
            cuda_src="""
                __global__ static void pointwise_weight_grad_kernel(@ARGS_DEF) {
                    @PRECALC
                    int total_weights = out_shape0 * out_shape1;
                    int flat_size = in0_shape0 * in0_shape2 * in0_shape3;
                    int step = blockDim.x * gridDim.x;

                    for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
                         out_idx < total_weights; out_idx += step) {
                        int oc = out_idx / out_shape1;
                        int ic = out_idx % out_shape1;

                        float acc = 0.0f;
                        for (int flat = 0; flat < flat_size; ++flat) {
                            int w = flat % in0_shape3;
                            int tmp = flat / in0_shape3;
                            int h = tmp % in0_shape2;
                            int b = tmp / in0_shape2;
                            acc += (float)@in1(b, oc, h, w) * (float)@in0(b, ic, h, w);
                        }

                        @out(oc, ic, 0, 0) = acc;
                    }
                }

                int total_weights = out_shape0 * out_shape1;
                int threads = 256;
                int blocks = (total_weights + threads - 1) / threads;
                if (blocks > 65535) blocks = 65535;
                pointwise_weight_grad_kernel<<<blocks, threads>>>(@ARGS);
            """,
        )

    @staticmethod
    def _weight_grad_chunked(
        x_flat,
        grad_flat,
        out_channels,
        in_channels,
        flat_size,
    ):
        chunk_size = int(os.environ.get("DMD2_SAFE_POINTWISE_CONV_DW_CHUNK", "32"))
        chunk_size = max(1, chunk_size)

        chunks = []
        for start in range(0, out_channels, chunk_size):
            size = min(chunk_size, out_channels - start)
            grad_chunk_t = grad_flat.reindex(
                [size, flat_size],
                ["i1", f"i0+{start}"],
            )
            chunks.append(jt.matmul(grad_chunk_t, x_flat))

        if len(chunks) == 1:
            return chunks[0].reshape([out_channels, in_channels])
        return jt.concat(chunks, dim=0).reshape([out_channels, in_channels])


class SafeLearnedConv2d(jt.Function):
    # Override learned conv backward to avoid Jittor's large 7D dW broadcast path.
    def execute(self, x, weight, padding_h, padding_w):
        self.x = x
        self.weight = weight
        self.padding_h = int(padding_h)
        self.padding_w = int(padding_w)
        return nn.conv2d(
            x,
            weight,
            bias=None,
            padding=(self.padding_h, self.padding_w),
        )

    def grad(self, grad_output):
        if grad_output is None:
            return None, None, None, None

        x = self.x
        weight = self.weight
        padding = (self.padding_h, self.padding_w)

        if os.environ.get("DMD2_SAFE_LEARNED_CONV_INPUT_GRAD", "1") == "0":
            dx = None
        else:
            dx = nn.conv_transpose2d(
                grad_output,
                weight,
                bias=None,
                padding=padding,
            )

        if os.environ.get("DMD2_SAFE_LEARNED_CONV_WEIGHT_GRAD", "1") == "0":
            dweight = None
        else:
            dw_mode = os.environ.get("DMD2_SAFE_LEARNED_CONV_DW_MODE", "cudnn")
            if dw_mode == "cudnn":
                dweight = _cudnn_conv_backward_w(
                    x=x,
                    grad_output=grad_output,
                    weight_shape=weight.shape,
                    padding_h=self.padding_h,
                    padding_w=self.padding_w,
                )
                if dweight is None:
                    dweight = self._weight_grad_code(
                        x=x,
                        grad_output=grad_output,
                        weight_shape=weight.shape,
                        padding_h=self.padding_h,
                        padding_w=self.padding_w,
                    )
            elif dw_mode == "matmul":
                dweight = self._weight_grad_matmul_chunked(
                    x=x,
                    grad_output=grad_output,
                    weight_shape=weight.shape,
                    padding_h=self.padding_h,
                    padding_w=self.padding_w,
                )
            elif dw_mode == "reduce":
                dweight = self._weight_grad_reduce(
                    x=x,
                    grad_output=grad_output,
                    weight_shape=weight.shape,
                    padding_h=self.padding_h,
                    padding_w=self.padding_w,
                )
            else:
                dweight = self._weight_grad_code(
                    x=x,
                    grad_output=grad_output,
                    weight_shape=weight.shape,
                    padding_h=self.padding_h,
                    padding_w=self.padding_w,
                )

        return dx, dweight, None, None

    @staticmethod
    def _weight_grad_code(
        x,
        grad_output,
        weight_shape,
        padding_h,
        padding_w,
    ):
        return jt.code(
            list(weight_shape),
            grad_output.dtype,
            [x, grad_output],
            cpu_src=f"""
                for (int oc = 0; oc < out_shape0; ++oc)
                for (int ic = 0; ic < out_shape1; ++ic)
                for (int kh = 0; kh < out_shape2; ++kh)
                for (int kw = 0; kw < out_shape3; ++kw) {{
                    float acc = 0.0f;
                    for (int b = 0; b < in1_shape0; ++b)
                    for (int oh = 0; oh < in1_shape2; ++oh)
                    for (int ow = 0; ow < in1_shape3; ++ow) {{
                        int ih = oh + kh - {int(padding_h)};
                        int iw = ow + kw - {int(padding_w)};
                        if (ih >= 0 && ih < in0_shape2 && iw >= 0 && iw < in0_shape3) {{
                            acc += (float)@in1(b, oc, oh, ow) * (float)@in0(b, ic, ih, iw);
                        }}
                    }}
                    @out(oc, ic, kh, kw) = acc;
                }}
            """,
            cuda_src=f"""
                __global__ static void learned_conv_weight_grad_kernel(@ARGS_DEF) {{
                    @PRECALC
                    int total_weights =
                        out_shape0 * out_shape1 * out_shape2 * out_shape3;
                    int flat_size = in1_shape0 * in1_shape2 * in1_shape3;
                    int step = blockDim.x * gridDim.x;

                    for (int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
                         out_idx < total_weights; out_idx += step) {{
                        int kw = out_idx % out_shape3;
                        int tmp = out_idx / out_shape3;
                        int kh = tmp % out_shape2;
                        tmp /= out_shape2;
                        int ic = tmp % out_shape1;
                        int oc = tmp / out_shape1;

                        float acc = 0.0f;
                        for (int flat = 0; flat < flat_size; ++flat) {{
                            int ow = flat % in1_shape3;
                            int tmp2 = flat / in1_shape3;
                            int oh = tmp2 % in1_shape2;
                            int b = tmp2 / in1_shape2;
                            int ih = oh + kh - {int(padding_h)};
                            int iw = ow + kw - {int(padding_w)};
                            if (
                                ih >= 0 && ih < in0_shape2 &&
                                iw >= 0 && iw < in0_shape3
                            ) {{
                                acc += (float)@in1(b, oc, oh, ow) *
                                    (float)@in0(b, ic, ih, iw);
                            }}
                        }}

                        @out(oc, ic, kh, kw) = acc;
                    }}
                }}

                int total_weights =
                    out_shape0 * out_shape1 * out_shape2 * out_shape3;
                int threads = 256;
                int blocks = (total_weights + threads - 1) / threads;
                if (blocks > 65535) blocks = 65535;
                learned_conv_weight_grad_kernel<<<blocks, threads>>>(@ARGS);
            """,
        )

    @staticmethod
    def _weight_grad_matmul_chunked(
        x,
        grad_output,
        weight_shape,
        padding_h,
        padding_w,
    ):
        batch, in_channels, _, _ = x.shape
        out_channels, _, kernel_h, kernel_w = weight_shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]
        spatial_size = out_h * out_w
        flat_size = batch * spatial_size
        kernel_area = kernel_h * kernel_w
        in_features = in_channels * kernel_area

        in_chunk = int(
            os.environ.get(
                "DMD2_SAFE_LEARNED_CONV_DW_CHUNK",
                os.environ.get("DMD2_SAFE_POINTWISE_CONV_DW_CHUNK", "32"),
            )
        )
        out_chunk = int(os.environ.get("DMD2_SAFE_LEARNED_CONV_DW_OUT_CHUNK", "32"))
        in_chunk = max(1, in_chunk)
        out_chunk = max(1, out_chunk)

        feature_chunks = []
        for feat_start in range(0, in_features, in_chunk):
            feat_size = min(in_chunk, in_features - feat_start)
            x_flat = x.reindex(
                [flat_size, feat_size],
                [
                    f"i0/{spatial_size}",
                    f"(i1+{feat_start})/{kernel_area}",
                    (
                        f"(i0%{spatial_size})/{out_w}"
                        f"+((i1+{feat_start})%{kernel_area})/{kernel_w}"
                        f"-{padding_h}"
                    ),
                    f"i0%{out_w}+(i1+{feat_start})%{kernel_w}-{padding_w}",
                ],
                overflow_value=0.0,
            )

            out_chunks = []
            for out_start in range(0, out_channels, out_chunk):
                out_size = min(out_chunk, out_channels - out_start)
                grad_chunk_t = grad_output.reindex(
                    [out_size, flat_size],
                    [
                        f"i1/{spatial_size}",
                        f"i0+{out_start}",
                        f"(i1%{spatial_size})/{out_w}",
                        f"i1%{out_w}",
                    ],
                )
                out_chunks.append(jt.matmul(grad_chunk_t, x_flat))

            if len(out_chunks) == 1:
                feature_chunks.append(out_chunks[0])
            else:
                feature_chunks.append(jt.concat(out_chunks, dim=0))

        if len(feature_chunks) == 1:
            dweight_flat = feature_chunks[0]
        else:
            dweight_flat = jt.concat(feature_chunks, dim=1)
        return dweight_flat.reshape(list(weight_shape))

    @staticmethod
    def _weight_grad_reduce(
        x,
        grad_output,
        weight_shape,
        padding_h,
        padding_w,
    ):
        batch, in_channels, _, _ = x.shape
        out_channels, _, kernel_h, kernel_w = weight_shape
        out_h, out_w = grad_output.shape[2], grad_output.shape[3]

        channel_chunks = []
        for in_channel in range(in_channels):
            row_chunks = []
            for kernel_y in range(kernel_h):
                col_chunks = []
                for kernel_x in range(kernel_w):
                    x_patch = x.reindex(
                        [batch, 1, out_h, out_w],
                        [
                            "i0",
                            str(in_channel),
                            f"i2+{kernel_y}-{padding_h}",
                            f"i3+{kernel_x}-{padding_w}",
                        ],
                        overflow_value=0.0,
                    )
                    grad_kernel = (grad_output * x_patch).sum(dims=[0, 2, 3])
                    col_chunks.append(grad_kernel.reshape([out_channels, 1, 1, 1]))
                row_chunks.append(jt.concat(col_chunks, dim=3))
            channel_chunks.append(jt.concat(row_chunks, dim=2))

        return jt.concat(channel_chunks, dim=1).reshape(list(weight_shape))


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

        self._cached_weight_t = None

    def clear_transposed_weight_cache(self):
        self._cached_weight_t = None

    def freeze_transposed_weight(self):
        # Cache W^T for frozen teacher networks to avoid repeated CUDA transposes.
        self._cached_weight_t = self.weight.transpose(0, 1).stop_grad()
        self._cached_weight_t.persistent = False
        return self._cached_weight_t

    def execute(self, x):
        # Apply the affine transformation y = xW^T + b.
        if os.environ.get("DMD2_SAFE_LINEAR", "0") != "0":
            return self._execute_conv1x1_linear(x)

        weight = self._cached_weight_t
        if weight is None:
            weight = self.weight.transpose(0, 1)
        bias = self.bias

        if hasattr(weight, "to"):
            weight = weight.to(x.dtype)
        x = x @ weight

        if bias is not None:
            if hasattr(bias, "to"):
                bias = bias.to(x.dtype)
            x = x + bias

        return x

    def _execute_conv1x1_linear(self, x):
        leading_shape = list(x.shape[:-1])
        x_flat = x.reshape([-1, self.in_features, 1, 1])

        weight = self.weight
        bias = self.bias
        if hasattr(weight, "to"):
            weight = weight.to(x.dtype)
        if bias is not None and hasattr(bias, "to"):
            bias = bias.to(x.dtype)

        weight = weight.reshape([self.out_features, self.in_features, 1, 1])
        if os.environ.get("DMD2_SAFE_LEARNED_CONV", "0") != "0":
            x_flat = SafeLearnedConv2d.apply(x_flat, weight, 0, 0)
        elif os.environ.get("DMD2_SAFE_POINTWISE_CONV", "0") != "0":
            x_flat = SafePointwiseConv2d.apply(x_flat, weight)
        else:
            x_flat = nn.conv2d(x_flat, weight, bias=None, padding=0)
        x = x_flat.reshape(leading_shape + [self.out_features])

        if bias is not None:
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
        resample_filter = tuple(resample_filter)
        self._nearest_2x_resample_filter = resample_filter == (1, 1)

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
            filt = filt / jt.sum(filt)
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

    def _can_use_nearest_2x_upsample(self, filt, filt_pad):
        if filt is None or filt_pad != 0:
            return False
        if os.environ.get("DMD2_SAFE_UPSAMPLE", "1") == "0":
            return False
        return self._nearest_2x_resample_filter

    @staticmethod
    def _nearest_2x_upsample(x):
        batch, channels, height, width = x.shape
        x = x.reshape([batch, channels, height, 1, width, 1])
        x = x.broadcast([batch, channels, height, 2, width, 2])
        return x.reshape([batch, channels, height * 2, width * 2])

    def _execute_fused_upsample(self, x, weight, filt, weight_pad, filt_pad):
        # Apply fused upsampling followed by learnable convolution.
        if self._can_use_nearest_2x_upsample(filt, filt_pad):
            x = self._nearest_2x_upsample(x)
            conv_pad = weight_pad
        else:
            filt = (filt * 4).repeat([self.in_channels, 1, 1, 1])
            x = nn.conv_transpose2d(
                x,
                filt,
                bias=None,
                stride=2,
                padding=max(filt_pad - weight_pad, 0),
                groups=self.in_channels,
            )
            conv_pad = max(weight_pad - filt_pad, 0)

        x = self._execute_learned_conv(x, weight, conv_pad)
        return x

    def _execute_fused_downsample(self, x, weight, filt, weight_pad, filt_pad):
        # Apply learnable convolution followed by fused downsampling.
        x = self._execute_learned_conv(x, weight, weight_pad + filt_pad)
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
            if self._can_use_nearest_2x_upsample(filt, filt_pad):
                x = self._nearest_2x_upsample(x)
            else:
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
            x = self._execute_learned_conv(x, weight, weight_pad)

        return x

    @staticmethod
    def _can_use_safe_pointwise_conv(weight, weight_pad):
        if os.environ.get("DMD2_SAFE_POINTWISE_CONV", "0") == "0":
            return False
        return list(weight.shape[-2:]) == [1, 1] and weight_pad == 0

    @staticmethod
    def _can_use_safe_learned_conv(weight, padding):
        if os.environ.get("DMD2_SAFE_LEARNED_CONV", "0") == "0":
            return False
        if padding < 0:
            return False
        return len(weight.shape) == 4

    def _execute_learned_conv(self, x, weight, padding):
        if self._can_use_safe_learned_conv(weight, padding):
            return SafeLearnedConv2d.apply(x, weight, padding, padding)
        if self._can_use_safe_pointwise_conv(weight, padding):
            return SafePointwiseConv2d.apply(x, weight)
        return nn.conv2d(x, weight, bias=None, padding=padding)


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
        self.freqs = (jt.randn([num_channels // 2]) * scale).stop_grad()

    def execute(self, x):
        # Map scalar inputs to random Fourier embeddings.
        x = x.reshape(-1, 1).float32()
        freqs = (2 * np.pi * self.freqs).reshape(1, -1).float32()
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


class SafeAttentionValueOp(jt.Function):
    # Avoid Jittor strided-batched GEMM in attention value backward on ImageNet64.
    def execute(self, v, weights):
        self.v = v
        self.weights = weights
        return jt.matmul(v, weights.transpose(0, 2, 1))

    def grad(self, grad_output):
        if grad_output is None:
            return None, None

        v = self.v
        weights = self.weights
        batch_heads = v.shape[0]

        dv_chunks = []
        dweights_chunks = []
        for idx in range(batch_heads):
            grad_i = grad_output[idx]
            v_i = v[idx]
            weights_i = weights[idx]

            dv_i = jt.matmul(grad_i, weights_i)
            dweights_i = jt.matmul(grad_i.transpose(0, 1), v_i)
            dv_chunks.append(dv_i.reshape([1] + list(dv_i.shape)))
            dweights_chunks.append(dweights_i.reshape([1] + list(dweights_i.shape)))

        return jt.concat(dv_chunks, dim=0), jt.concat(dweights_chunks, dim=0)


def attention_value_op(v, weights):
    if os.environ.get("DMD2_SAFE_ATTENTION_VALUE", "1") == "0":
        return jt.matmul(v, weights.transpose(0, 2, 1))
    return SafeAttentionValueOp.apply(v, weights)


class AttentionOp:
    # Provide an official-style apply wrapper around attention_op.
    @staticmethod
    def apply(q, k):
        # Compute attention weights through the functional implementation.
        return attention_op(q, k)


class AttentionValueOp:
    # Provide an apply wrapper around the safe attention value implementation.
    @staticmethod
    def apply(v, weights):
        return attention_value_op(v, weights)
