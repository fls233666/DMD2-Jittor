"""Smoke tests for code/models/unet.py.

Run from the project root:

    python tests/test_unet.py

Optional:

    python tests/test_unet.py --no-cuda
    python tests/test_unet.py --skip-attn
    python tests/test_unet.py --full-imagenet-smoke
"""

import argparse
import sys
from pathlib import Path


def _add_project_paths():
    # Make the test runnable from either the project root or the tests directory.
    test_file = Path(__file__).resolve()
    project_root = test_file.parents[1]
    models_dir = project_root / "code" / "models"

    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(models_dir))


_add_project_paths()

import jittor as jt

from unet import UNetBlock, DhariwalUNet, get_imagenet_dhariwal_unet


def _set_runtime(use_cuda=True):
    # Configure Jittor runtime.
    jt.flags.use_cuda = 1 if use_cuda else 0

    try:
        jt.set_global_seed(2026)
    except Exception:
        pass


def _sync():
    # Force lazy Jittor operations to execute.
    try:
        jt.sync_all()
    except Exception:
        pass


def _shape(x):
    return list(x.shape)


def _assert_shape(name, value, expected):
    actual = _shape(value)
    print(f"[{name}] shape = {actual}")
    assert actual == expected, f"{name}: expected {expected}, got {actual}"


def _randn(shape):
    return jt.randn(shape).float32()


def test_unet_block_basic():
    # Basic residual block: channel and spatial size stay unchanged.
    batch = 2
    x = _randn([batch, 16, 32, 32])
    emb = _randn([batch, 64])

    block = UNetBlock(
        in_channels=16,
        out_channels=16,
        emb_channels=64,
        attention=False,
        dropout=0,
    )
    y = block(x, emb)
    _sync()

    _assert_shape("UNetBlock basic", y, [batch, 16, 32, 32])


def test_unet_block_channel_change():
    # Residual block with a 1x1 skip projection: channel size changes.
    batch = 2
    x = _randn([batch, 16, 32, 32])
    emb = _randn([batch, 64])

    block = UNetBlock(
        in_channels=16,
        out_channels=32,
        emb_channels=64,
        attention=False,
        dropout=0,
    )
    y = block(x, emb)
    _sync()

    _assert_shape("UNetBlock channel change", y, [batch, 32, 32, 32])


def test_unet_block_downsample():
    # Downsample block: spatial size should be divided by 2.
    batch = 2
    x = _randn([batch, 16, 32, 32])
    emb = _randn([batch, 64])

    block = UNetBlock(
        in_channels=16,
        out_channels=16,
        emb_channels=64,
        down=True,
        attention=False,
        dropout=0,
    )
    y = block(x, emb)
    _sync()

    _assert_shape("UNetBlock downsample", y, [batch, 16, 16, 16])


def test_unet_block_upsample():
    # Upsample block: spatial size should be multiplied by 2.
    batch = 2
    x = _randn([batch, 16, 16, 16])
    emb = _randn([batch, 64])

    block = UNetBlock(
        in_channels=16,
        out_channels=16,
        emb_channels=64,
        up=True,
        attention=False,
        dropout=0,
    )
    y = block(x, emb)
    _sync()

    _assert_shape("UNetBlock upsample", y, [batch, 16, 32, 32])


def test_unet_block_attention():
    # Attention branch smoke test. Keep resolution small to reduce memory cost.
    batch = 2
    x = _randn([batch, 16, 8, 8])
    emb = _randn([batch, 64])

    block = UNetBlock(
        in_channels=16,
        out_channels=16,
        emb_channels=64,
        attention=True,
        num_heads=2,
        dropout=0,
    )
    y = block(x, emb)
    _sync()

    _assert_shape("UNetBlock attention", y, [batch, 16, 8, 8])


def test_dhariwal_unet_unconditional_small():
    # Small unconditional DhariwalUNet smoke test for fast debugging.
    batch = 2
    resolution = 32

    model = DhariwalUNet(
        img_resolution=resolution,
        in_channels=3,
        out_channels=3,
        label_dim=0,
        model_channels=16,
        channel_mult=(1, 2),
        channel_mult_emb=4,
        num_blocks=1,
        attn_resolutions=(),
        dropout=0,
        label_dropout=0,
    )

    x = _randn([batch, 3, resolution, resolution])
    noise_labels = _randn([batch])

    y = model(x, noise_labels)
    _sync()

    _assert_shape("DhariwalUNet unconditional small", y, [batch, 3, resolution, resolution])


def test_dhariwal_unet_conditional_small():
    # Small class-conditional DhariwalUNet smoke test.
    batch = 2
    resolution = 32
    label_dim = 10

    model = DhariwalUNet(
        img_resolution=resolution,
        in_channels=3,
        out_channels=3,
        label_dim=label_dim,
        model_channels=16,
        channel_mult=(1, 2),
        channel_mult_emb=4,
        num_blocks=1,
        attn_resolutions=(),
        dropout=0,
        label_dropout=0,
    )

    x = _randn([batch, 3, resolution, resolution])
    noise_labels = _randn([batch])
    class_labels = _randn([batch, label_dim])

    y = model(x, noise_labels, class_labels)
    _sync()

    _assert_shape("DhariwalUNet conditional small", y, [batch, 3, resolution, resolution])


def test_dhariwal_unet_bottleneck_small():
    # return_bottleneck=True should stop after the encoder.
    batch = 2
    resolution = 32

    model = DhariwalUNet(
        img_resolution=resolution,
        in_channels=3,
        out_channels=3,
        label_dim=0,
        model_channels=16,
        channel_mult=(1, 2),
        channel_mult_emb=4,
        num_blocks=1,
        attn_resolutions=(),
        dropout=0,
        label_dropout=0,
    )

    x = _randn([batch, 3, resolution, resolution])
    noise_labels = _randn([batch])

    y = model(x, noise_labels, return_bottleneck=True)
    _sync()

    _assert_shape("DhariwalUNet bottleneck small", y, [batch, 32, 16, 16])


def test_get_imagenet_dhariwal_unet_constructor():
    # Constructor-only test for the official ImageNet-64 configuration.
    # Do not run forward by default because it is much heavier.
    model = get_imagenet_dhariwal_unet(
        img_resolution=64,
        in_channels=3,
        out_channels=3,
        label_dim=1000,
        dropout=0.0,
    )

    assert model.img_resolution == 64
    assert model.in_channels == 3
    assert model.out_channels == 3
    assert model.label_dim == 1000

    print("[get_imagenet_dhariwal_unet] constructor passed")


def test_full_imagenet_smoke():
    # Optional heavy forward smoke test for the official ImageNet-64 configuration.
    # This may require significant GPU memory.
    batch = 1
    resolution = 64
    label_dim = 1000

    model = get_imagenet_dhariwal_unet(
        img_resolution=resolution,
        in_channels=3,
        out_channels=3,
        label_dim=label_dim,
        dropout=0.0,
    )

    x = _randn([batch, 3, resolution, resolution])
    noise_labels = _randn([batch])
    class_labels = _randn([batch, label_dim])

    y = model(x, noise_labels, class_labels)
    _sync()

    _assert_shape("ImageNet-64 DhariwalUNet full smoke", y, [batch, 3, resolution, resolution])


def run_all(skip_attn=False, full_imagenet_smoke=False):
    # Run tests in a stable order from local blocks to full network.
    test_unet_block_basic()
    test_unet_block_channel_change()
    test_unet_block_downsample()
    test_unet_block_upsample()

    if skip_attn:
        print("[UNetBlock attention] skipped")
    else:
        test_unet_block_attention()

    test_dhariwal_unet_unconditional_small()
    test_dhariwal_unet_conditional_small()
    test_dhariwal_unet_bottleneck_small()
    test_get_imagenet_dhariwal_unet_constructor()

    if full_imagenet_smoke:
        test_full_imagenet_smoke()
    else:
        print("[ImageNet-64 DhariwalUNet full smoke] skipped")

    print("All selected unet.py tests passed.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cuda", action="store_true", help="Run tests on CPU.")
    parser.add_argument(
        "--skip-attn",
        action="store_true",
        help="Skip the AttentionOp branch test.",
    )
    parser.add_argument(
        "--full-imagenet-smoke",
        action="store_true",
        help="Run the heavy ImageNet-64 DhariwalUNet forward test.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    use_cuda = not args.no_cuda
    _set_runtime(use_cuda=use_cuda)

    print(f"Jittor CUDA enabled: {use_cuda}")
    run_all(
        skip_attn=args.skip_attn,
        full_imagenet_smoke=args.full_imagenet_smoke,
    )
