"""Discriminator-style heads for the Jittor DMD2/EDM implementation."""

from jittor import nn

try:
    from .modules import GroupNorm, silu
except ImportError:
    from modules import GroupNorm, silu


def _spatial_kernel(size):
    # Convert int/list/tuple spatial size to a 2D kernel tuple.
    if isinstance(size, int):
        return (size, size)
    return tuple(size)


class BottleneckDiscriminator(nn.Module):
    # Predict a real/fake logit from the EDM UNet bottleneck representation.
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        bottleneck_resolution=8,
        num_groups=32,
    ):
        super().__init__()

        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        kernel = _spatial_kernel(max(int(bottleneck_resolution) // 2, 1))

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bottleneck_resolution = bottleneck_resolution

        self.conv0 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.norm0 = GroupNorm(
            num_channels=hidden_channels,
            num_groups=num_groups,
        )
        self.conv1 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel,
            stride=kernel,
            padding=0,
        )
        self.norm1 = GroupNorm(
            num_channels=hidden_channels,
            num_groups=num_groups,
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def execute(self, x):
        # Apply the classification head and return logits with shape [B, 1].
        x = self.conv0(x)
        x = silu(self.norm0(x))
        x = self.conv1(x)
        x = silu(self.norm1(x))
        x = self.conv2(x)

        if x.shape[2] != 1 or x.shape[3] != 1:
            x = x.mean(dims=[2, 3], keepdims=True)

        return x.reshape(x.shape[0], -1)


class EDMRealismHead(BottleneckDiscriminator):
    # Alias that documents the role used by DMD2 guidance training.
    pass
