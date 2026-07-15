"""Unit tests for code/models/modules.py."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import jittor as jt


# ------------------------------------------------------------
# Runtime setup
# ------------------------------------------------------------

def _maybe_enable_cuda():
    # Enable CUDA by default when available in the course environment.
    if os.environ.get("JT_USE_CUDA", "1") == "1":
        jt.flags.use_cuda = 1


def _load_modules():
    # Load modules.py robustly without relying on the package name "code".
    this_file = Path(__file__).resolve()

    candidates = [
        this_file.parents[1] / "code" / "models" / "modules.py",
        this_file.parent / "modules.py",
        Path.cwd() / "code" / "models" / "modules.py",
        Path.cwd() / "modules.py",
    ]

    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location("dmd2_modules", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module

    raise FileNotFoundError(
        "Cannot find modules.py. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


_maybe_enable_cuda()
M = _load_modules()


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def assert_shape(x, expected_shape):
    # Check a Jittor Var shape against a Python list.
    assert list(x.shape) == list(expected_shape), (
        f"Expected shape {expected_shape}, got {list(x.shape)}"
    )


def assert_close(actual, expected, atol=1e-5, rtol=1e-5):
    # Compare arrays with a readable assertion message.
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def expect_assertion_error(fn):
    # Verify that a callable raises AssertionError.
    try:
        fn()
    except AssertionError:
        return
    raise AssertionError("Expected AssertionError, but no error was raised.")


# ------------------------------------------------------------
# silu
# ------------------------------------------------------------

def test_silu_values():
    # Test SiLU against the reference formula x * sigmoid(x).
    x = jt.array([-2.0, -1.0, 0.0, 1.0, 2.0]).float32()
    y = M.silu(x)

    x_np = x.numpy()
    expected = x_np / (1.0 + np.exp(-x_np))

    assert_close(y.numpy(), expected, atol=1e-6, rtol=1e-6)


# ------------------------------------------------------------
# weight_init
# ------------------------------------------------------------

def test_weight_init_all_modes():
    # Test all supported initialization modes return the requested shape.
    shape = [4, 3]
    fan_in = 3
    fan_out = 4

    modes = [
        "xavier_uniform",
        "xavier_normal",
        "kaiming_uniform",
        "kaiming_normal",
    ]

    for mode in modes:
        w = M.weight_init(shape, mode=mode, fan_in=fan_in, fan_out=fan_out)
        assert_shape(w, shape)
        assert np.isfinite(w.numpy()).all(), f"{mode} produced non-finite values."


def test_weight_init_invalid_mode():
    # Test invalid initialization mode raises ValueError.
    try:
        M.weight_init([2, 2], mode="invalid_mode", fan_in=2, fan_out=2)
    except ValueError:
        return

    raise AssertionError("weight_init should raise ValueError for invalid mode.")


# ------------------------------------------------------------
# Linear
# ------------------------------------------------------------

def test_linear_shape():
    # Test Linear output shape.
    layer = M.Linear(in_features=5, out_features=7)
    x = jt.randn([4, 5])
    y = layer(x)

    assert_shape(y, [4, 7])


def test_linear_manual_values():
    # Test Linear with manually assigned weights and bias.
    layer = M.Linear(in_features=3, out_features=2, bias=True)

    layer.weight = jt.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]
    ).float32()
    layer.bias = jt.array([0.5, -0.5]).float32()

    x = jt.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 0.0, 1.0],
        ]
    ).float32()

    y = layer(x)

    expected = np.array(
        [
            [6.5, 14.5],
            [5.5, 13.5],
        ],
        dtype=np.float32,
    )

    assert_close(y.numpy(), expected, atol=1e-5, rtol=1e-5)


def test_linear_without_bias():
    # Test Linear works when bias=False.
    layer = M.Linear(in_features=3, out_features=2, bias=False)
    x = jt.randn([8, 3])
    y = layer(x)

    assert_shape(y, [8, 2])
    assert layer.bias is None


# ------------------------------------------------------------
# Conv2d
# ------------------------------------------------------------

def test_conv2d_basic_shape():
    # Test normal convolution keeps spatial resolution for kernel=3.
    layer = M.Conv2d(in_channels=3, out_channels=8, kernel=3)
    x = jt.randn([2, 3, 16, 16])
    y = layer(x)

    assert_shape(y, [2, 8, 16, 16])


def test_conv2d_kernel_zero_identity_shape():
    # Test kernel=0 without resampling returns the original tensor shape.
    layer = M.Conv2d(in_channels=3, out_channels=3, kernel=0)
    x = jt.randn([2, 3, 16, 16])
    y = layer(x)

    assert_shape(y, [2, 3, 16, 16])


def test_conv2d_up_only_shape():
    # Test non-learnable upsampling doubles spatial resolution.
    layer = M.Conv2d(in_channels=3, out_channels=3, kernel=0, up=True)
    x = jt.randn([2, 3, 16, 16])
    y = layer(x)

    assert_shape(y, [2, 3, 32, 32])


def test_conv2d_down_only_shape():
    # Test non-learnable downsampling halves spatial resolution.
    layer = M.Conv2d(in_channels=3, out_channels=3, kernel=0, down=True)
    x = jt.randn([2, 3, 16, 16])
    y = layer(x)

    assert_shape(y, [2, 3, 8, 8])


def test_conv2d_fused_up_shape():
    # Test fused upsampling followed by learnable convolution.
    layer = M.Conv2d(
        in_channels=3,
        out_channels=8,
        kernel=3,
        up=True,
        fused_resample=True,
    )
    x = jt.randn([2, 3, 16, 16])
    y = layer(x)

    assert_shape(y, [2, 8, 32, 32])


def test_conv2d_fused_down_shape():
    # Test learnable convolution followed by fused downsampling.
    layer = M.Conv2d(
        in_channels=3,
        out_channels=8,
        kernel=3,
        down=True,
        fused_resample=True,
    )
    x = jt.randn([2, 3, 16, 16])
    y = layer(x)

    assert_shape(y, [2, 8, 8, 8])


def test_conv2d_rejects_up_and_down_together():
    # Test Conv2d rejects simultaneous upsampling and downsampling.
    expect_assertion_error(
        lambda: M.Conv2d(
            in_channels=3,
            out_channels=3,
            kernel=3,
            up=True,
            down=True,
        )
    )


def test_conv2d_resample_filter_shape():
    # Test the fixed resampling filter is stored with NCHW kernel layout and
    # normalized like the official EDM PyTorch implementation.
    layer = M.Conv2d(
        in_channels=3,
        out_channels=3,
        kernel=0,
        up=True,
        resample_filter=(1, 3, 3, 1),
    )
    filt = layer._resample_filter

    assert filt is not None
    assert_shape(filt, [1, 1, 4, 4])
    assert_close(filt.numpy().sum(), 1.0, atol=1e-6, rtol=1e-6)


# ------------------------------------------------------------
# GroupNorm
# ------------------------------------------------------------

def test_groupnorm_shape_and_group_count():
    # Test GroupNorm output shape and automatic group count.
    layer = M.GroupNorm(num_channels=32)
    x = jt.randn([2, 32, 8, 8])
    y = layer(x)

    assert_shape(y, [2, 32, 8, 8])
    assert layer.num_groups == 8


def test_groupnorm_zero_mean_unit_variance():
    # Test GroupNorm approximately normalizes each sample group.
    layer = M.GroupNorm(num_channels=32)
    x = jt.randn([2, 32, 8, 8])
    y = layer(x)

    y_np = y.numpy()
    groups = layer.num_groups
    y_grouped = y_np.reshape(2, groups, 32 // groups, 8, 8)

    mean = y_grouped.mean(axis=(2, 3, 4))
    var = y_grouped.var(axis=(2, 3, 4))

    assert np.max(np.abs(mean)) < 1e-4
    assert np.max(np.abs(var - 1.0)) < 1e-3


def test_groupnorm_small_channels():
    # Test GroupNorm handles small channel counts.
    layer = M.GroupNorm(num_channels=8)
    x = jt.randn([2, 8, 4, 4])
    y = layer(x)

    assert_shape(y, [2, 8, 4, 4])
    assert layer.num_groups == 2


def test_groupnorm_channel_mismatch():
    # Test GroupNorm rejects mismatched input channels.
    layer = M.GroupNorm(num_channels=8)
    x = jt.randn([2, 16, 4, 4])

    expect_assertion_error(lambda: layer(x))


# ------------------------------------------------------------
# PositionalEmbedding
# ------------------------------------------------------------

def test_positional_embedding_shape():
    # Test PositionalEmbedding output shape.
    layer = M.PositionalEmbedding(num_channels=64)
    x = jt.array([0.0, 1.0, 2.0]).float32()
    y = layer(x)

    assert_shape(y, [3, 64])


def test_positional_embedding_zero_value():
    # Test PositionalEmbedding at x=0 gives cos=1 and sin=0.
    layer = M.PositionalEmbedding(num_channels=8)
    x = jt.array([0.0]).float32()
    y = layer(x).numpy()

    expected = np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.float32)
    assert_close(y, expected, atol=1e-6, rtol=1e-6)


def test_positional_embedding_endpoint_shape():
    # Test endpoint=True branch.
    layer = M.PositionalEmbedding(num_channels=16, endpoint=True)
    x = jt.randn([5])
    y = layer(x)

    assert_shape(y, [5, 16])


def test_positional_embedding_rejects_odd_channels():
    # Test PositionalEmbedding requires even channel count.
    expect_assertion_error(lambda: M.PositionalEmbedding(num_channels=7))


# ------------------------------------------------------------
# FourierEmbedding
# ------------------------------------------------------------

def test_fourier_embedding_shape():
    # Test FourierEmbedding output shape.
    layer = M.FourierEmbedding(num_channels=64)
    x = jt.array([0.0, 1.0, 2.0]).float32()
    y = layer(x)

    assert_shape(y, [3, 64])


def test_fourier_embedding_zero_value():
    # Test FourierEmbedding at x=0 gives cos=1 and sin=0.
    layer = M.FourierEmbedding(num_channels=8)
    x = jt.array([0.0]).float32()
    y = layer(x).numpy()

    expected = np.array([[1, 1, 1, 1, 0, 0, 0, 0]], dtype=np.float32)
    assert_close(y, expected, atol=1e-6, rtol=1e-6)


def test_fourier_embedding_freq_shape():
    # Test FourierEmbedding stores half-channel random frequencies.
    layer = M.FourierEmbedding(num_channels=32)

    assert_shape(layer.freqs, [16])
    assert "freqs" in layer.state_dict(), "FourierEmbedding freqs should be checkpointed."


def test_fourier_embedding_rejects_odd_channels():
    # Test FourierEmbedding requires even channel count.
    expect_assertion_error(lambda: M.FourierEmbedding(num_channels=7))


# ------------------------------------------------------------
# attention_op and AttentionOp
# ------------------------------------------------------------

def test_attention_op_shape():
    # Test attention_op returns [N, Q, K].
    q = jt.randn([4, 16, 25])
    k = jt.randn([4, 16, 25])
    w = M.attention_op(q, k)

    assert_shape(w, [4, 25, 25])


def test_attention_op_softmax_sum():
    # Test attention weights sum to 1 along the key dimension.
    q = jt.randn([2, 8, 10])
    k = jt.randn([2, 8, 12])
    w = M.attention_op(q, k)

    sums = w.sum(dim=2).numpy()
    expected = np.ones_like(sums)

    assert_close(sums, expected, atol=1e-5, rtol=1e-5)


def test_attention_op_wrapper_matches_function():
    # Test AttentionOp.apply matches attention_op.
    q = jt.randn([2, 8, 16])
    k = jt.randn([2, 8, 16])

    w1 = M.attention_op(q, k)
    w2 = M.AttentionOp.apply(q, k)

    assert_close(w1.numpy(), w2.numpy(), atol=1e-6, rtol=1e-6)


def test_attention_op_rejects_bad_shape():
    # Test attention_op rejects non-3D inputs.
    q = jt.randn([2, 8, 16])
    k = jt.randn([2, 8, 4, 4])

    expect_assertion_error(lambda: M.attention_op(q, k))


def test_attention_op_rejects_channel_mismatch():
    # Test attention_op rejects mismatched channel dimensions.
    q = jt.randn([2, 8, 16])
    k = jt.randn([2, 9, 16])

    expect_assertion_error(lambda: M.attention_op(q, k))


# ------------------------------------------------------------
# Script entry
# ------------------------------------------------------------

def main():
    # Run all tests without requiring pytest.
    test_silu_values()

    test_weight_init_all_modes()
    test_weight_init_invalid_mode()

    test_linear_shape()
    test_linear_manual_values()
    test_linear_without_bias()

    test_conv2d_basic_shape()
    test_conv2d_kernel_zero_identity_shape()
    test_conv2d_up_only_shape()
    test_conv2d_down_only_shape()
    test_conv2d_fused_up_shape()
    test_conv2d_fused_down_shape()
    test_conv2d_rejects_up_and_down_together()
    test_conv2d_resample_filter_shape()

    test_groupnorm_shape_and_group_count()
    test_groupnorm_zero_mean_unit_variance()
    test_groupnorm_small_channels()
    test_groupnorm_channel_mismatch()

    test_positional_embedding_shape()
    test_positional_embedding_zero_value()
    test_positional_embedding_endpoint_shape()
    test_positional_embedding_rejects_odd_channels()

    test_fourier_embedding_shape()
    test_fourier_embedding_zero_value()
    test_fourier_embedding_freq_shape()
    test_fourier_embedding_rejects_odd_channels()

    test_attention_op_shape()
    test_attention_op_softmax_sum()
    test_attention_op_wrapper_matches_function()
    test_attention_op_rejects_bad_shape()
    test_attention_op_rejects_channel_mismatch()

    print("All modules.py tests passed.")


if __name__ == "__main__":
    main()
