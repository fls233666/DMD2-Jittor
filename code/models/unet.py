"""U-Net blocks for the Jittor DMD2/EDM implementation."""

import numpy as np
import jittor as jt
from jittor import nn

try:
    from .modules import Linear, Conv2d, GroupNorm, PositionalEmbedding, AttentionOp
except ImportError:
    from modules import Linear, Conv2d, GroupNorm, PositionalEmbedding, AttentionOp

try:
    from .modules import silu
except ImportError:
    try:
        from modules import silu
    except ImportError:
        def silu(x):
            # Apply the SiLU activation used by EDM/DMD2 blocks.
            return x * jt.sigmoid(x)


def _is_training(module):
    # Return the current training flag while staying compatible with Jittor.
    if hasattr(module, "is_training"):
        flag = module.is_training
        return flag() if callable(flag) else bool(flag)

    if hasattr(module, "training"):
        return bool(module.training)

    return True


def _dropout(x, p, training):
    # Apply dropout only during training.
    if p is None or p == 0 or not training:
        return x

    try:
        return nn.dropout(x, p=p, is_train=training)
    except TypeError:
        try:
            return nn.dropout(x, p=p, training=training)
        except TypeError:
            return nn.dropout(x, p=p)


class UNetBlock(nn.Module):
    # Implement the EDM/DMD2 residual U-Net block with optional attention.
    def __init__(
        self,
        in_channels,
        out_channels,
        emb_channels,
        up=False,
        down=False,
        attention=False,
        num_heads=None,
        channels_per_head=64,
        dropout=0,
        skip_scale=1,
        eps=1e-5,
        resample_filter=(1, 1),
        resample_proj=False,
        adaptive_scale=True,
        init=dict(),
        init_zero=dict(init_weight=0),
        init_attn=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.skip_scale = skip_scale
        self.adaptive_scale = adaptive_scale

        if not attention:
            self.num_heads = 0
        elif num_heads is not None:
            self.num_heads = num_heads
        else:
            self.num_heads = max(out_channels // channels_per_head, 1)

        self.norm0 = GroupNorm(num_channels=in_channels, eps=eps)
        self.conv0 = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel=3,
            up=up,
            down=down,
            resample_filter=resample_filter,
            **init,
        )

        affine_out_channels = out_channels * (2 if adaptive_scale else 1)
        self.affine = Linear(
            in_features=emb_channels,
            out_features=affine_out_channels,
            **init,
        )

        self.norm1 = GroupNorm(num_channels=out_channels, eps=eps)
        self.conv1 = Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel=3,
            **init_zero,
        )

        self.skip = None
        if out_channels != in_channels or up or down:
            kernel = 1 if resample_proj or out_channels != in_channels else 0
            self.skip = Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                up=up,
                down=down,
                resample_filter=resample_filter,
                **init,
            )

        if self.num_heads:
            self.norm2 = GroupNorm(num_channels=out_channels, eps=eps)
            self.qkv = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels * 3,
                kernel=1,
                **(init_attn if init_attn is not None else init),
            )
            self.proj = Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel=1,
                **init_zero,
            )

    def execute(self, x, emb):
        # Apply residual convolutional branch.
        orig = x
        x = self.conv0(silu(self.norm0(x)))

        params = self.affine(emb).reshape(emb.shape[0], -1, 1, 1)
        if hasattr(params, "to"):
            params = params.to(x.dtype)

        if self.adaptive_scale:
            scale = params[:, : self.out_channels]
            shift = params[:, self.out_channels :]
            x = silu(shift + self.norm1(x) * (scale + 1))
        else:
            x = silu(self.norm1(x + params))

        x = _dropout(x, self.dropout, _is_training(self))
        x = self.conv1(x)

        residual = self.skip(orig) if self.skip is not None else orig
        x = (x + residual) * self.skip_scale

        # Apply optional self-attention branch.
        if self.num_heads:
            x = self._execute_attention(x)

        return x

    def _execute_attention(self, x):
        # Compute self-attention over flattened spatial positions.
        batch, channels = x.shape[0], x.shape[1]
        spatial_shape = list(x.shape[2:])
        spatial_size = int(np.prod(spatial_shape))

        qkv = self.qkv(self.norm2(x))
        qkv = qkv.reshape(
            [batch * self.num_heads, channels // self.num_heads, 3, spatial_size]
        )

        q = qkv[:, :, 0, :]
        k = qkv[:, :, 1, :]
        v = qkv[:, :, 2, :]

        weights = AttentionOp.apply(q, k)
        attn = jt.matmul(v, weights.transpose(0, 2, 1))
        attn = attn.reshape(x.shape)

        x = (self.proj(attn) + x) * self.skip_scale
        return x


class DhariwalUNet(nn.Module):
    # Implement the ADM/Dhariwal U-Net used by the ImageNet branch of DMD2.
    def __init__(
        self,
        img_resolution,
        in_channels,
        out_channels,
        label_dim=0,
        augment_dim=0,
        model_channels=192,
        channel_mult=(1, 2, 3, 4),
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=(32, 16, 8),
        dropout=0.10,
        label_dropout=0,
    ):
        super().__init__()

        self.img_resolution = img_resolution
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.label_dim = label_dim
        self.augment_dim = augment_dim
        self.label_dropout = label_dropout

        emb_channels = model_channels * channel_mult_emb

        init = dict(
            init_mode="kaiming_uniform",
            init_weight=np.sqrt(1 / 3),
            init_bias=np.sqrt(1 / 3),
        )
        init_zero = dict(
            init_mode="kaiming_uniform",
            init_weight=0,
            init_bias=0,
        )
        block_kwargs = dict(
            emb_channels=emb_channels,
            channels_per_head=64,
            dropout=dropout,
            init=init,
            init_zero=init_zero,
        )

        # Mapping.
        self.map_noise = PositionalEmbedding(num_channels=model_channels)
        self.map_augment = None
        if augment_dim:
            self.map_augment = Linear(
                in_features=augment_dim,
                out_features=model_channels,
                bias=False,
                **init_zero,
            )

        self.map_layer0 = Linear(
            in_features=model_channels,
            out_features=emb_channels,
            **init,
        )
        self.map_layer1 = Linear(
            in_features=emb_channels,
            out_features=emb_channels,
            **init,
        )

        self.map_label = None
        if label_dim:
            self.map_label = Linear(
                in_features=label_dim,
                out_features=emb_channels,
                bias=False,
                init_mode="kaiming_normal",
                init_weight=np.sqrt(label_dim),
            )

        # Encoder.
        self._enc_items = []
        cout = in_channels
        for level, mult in enumerate(channel_mult):
            res = img_resolution >> level

            if level == 0:
                cin = cout
                cout = model_channels * mult
                self._add_enc(
                    f"{res}x{res}_conv",
                    Conv2d(
                        in_channels=cin,
                        out_channels=cout,
                        kernel=3,
                        **init,
                    ),
                )
            else:
                self._add_enc(
                    f"{res}x{res}_down",
                    UNetBlock(
                        in_channels=cout,
                        out_channels=cout,
                        down=True,
                        **block_kwargs,
                    ),
                )

            for idx in range(num_blocks):
                cin = cout
                cout = model_channels * mult
                self._add_enc(
                    f"{res}x{res}_block{idx}",
                    UNetBlock(
                        in_channels=cin,
                        out_channels=cout,
                        attention=(res in attn_resolutions),
                        **block_kwargs,
                    ),
                )

        skip_channels = [
            getattr(self, attr_name).out_channels
            for _, attr_name in self._enc_items
        ]

        # Decoder.
        self._dec_items = []
        for level, mult in reversed(list(enumerate(channel_mult))):
            res = img_resolution >> level

            if level == len(channel_mult) - 1:
                self._add_dec(
                    f"{res}x{res}_in0",
                    UNetBlock(
                        in_channels=cout,
                        out_channels=cout,
                        attention=True,
                        **block_kwargs,
                    ),
                )
                self._add_dec(
                    f"{res}x{res}_in1",
                    UNetBlock(
                        in_channels=cout,
                        out_channels=cout,
                        **block_kwargs,
                    ),
                )
            else:
                self._add_dec(
                    f"{res}x{res}_up",
                    UNetBlock(
                        in_channels=cout,
                        out_channels=cout,
                        up=True,
                        **block_kwargs,
                    ),
                )

            for idx in range(num_blocks + 1):
                cin = cout + skip_channels.pop()
                cout = model_channels * mult
                self._add_dec(
                    f"{res}x{res}_block{idx}",
                    UNetBlock(
                        in_channels=cin,
                        out_channels=cout,
                        attention=(res in attn_resolutions),
                        **block_kwargs,
                    ),
                )

        self.out_norm = GroupNorm(num_channels=cout)
        self.out_conv = Conv2d(
            in_channels=cout,
            out_channels=out_channels,
            kernel=3,
            **init_zero,
        )

    def _add_enc(self, name, module):
        # Register one encoder module while keeping the official traversal order.
        attr_name = f"enc_{len(self._enc_items)}_{name}"
        setattr(self, attr_name, module)
        self._enc_items.append((name, attr_name))

    def _add_dec(self, name, module):
        # Register one decoder module while keeping the official traversal order.
        attr_name = f"dec_{len(self._dec_items)}_{name}"
        setattr(self, attr_name, module)
        self._dec_items.append((name, attr_name))

    def _map_embeddings(self, x, noise_labels, class_labels=None, augment_labels=None):
        # Map noise, class labels, and optional augmentation labels to one embedding.
        emb = self.map_noise(noise_labels)

        if self.map_augment is not None and augment_labels is not None:
            emb = emb + self.map_augment(augment_labels)

        emb = silu(self.map_layer0(emb))
        emb = self.map_layer1(emb)

        if self.map_label is not None:
            if class_labels is None:
                class_labels = jt.zeros([x.shape[0], self.label_dim])

            tmp = class_labels
            if _is_training(self) and self.label_dropout:
                mask = jt.rand([x.shape[0], 1]) >= self.label_dropout
                tmp = tmp * mask.float32()

            emb = emb + self.map_label(tmp)

        emb = silu(emb)
        return emb

    def execute(
        self,
        x,
        noise_labels,
        class_labels=None,
        augment_labels=None,
        return_bottleneck=False,
    ):
        # Mapping.
        emb = self._map_embeddings(
            x=x,
            noise_labels=noise_labels,
            class_labels=class_labels,
            augment_labels=augment_labels,
        )

        # Encoder.
        skips = []
        for _, attr_name in self._enc_items:
            block = getattr(self, attr_name)

            if isinstance(block, UNetBlock):
                x = block(x, emb)
            else:
                x = block(x)

            skips.append(x)

        if return_bottleneck:
            return x

        # Decoder.
        for _, attr_name in self._dec_items:
            block = getattr(self, attr_name)

            if x.shape[1] != block.in_channels:
                x = jt.concat([x, skips.pop()], dim=1)

            x = block(x, emb)

        x = self.out_conv(silu(self.out_norm(x)))
        return x


def get_imagenet_dhariwal_unet(
    img_resolution=64,
    in_channels=3,
    out_channels=3,
    label_dim=1000,
    dropout=0.0,
):
    # Create the ImageNet-64 DhariwalUNet configuration used by DMD2/EDM.
    return DhariwalUNet(
        img_resolution=img_resolution,
        in_channels=in_channels,
        out_channels=out_channels,
        label_dim=label_dim,
        model_channels=192,
        channel_mult=(1, 2, 3, 4),
        channel_mult_emb=4,
        num_blocks=3,
        attn_resolutions=(32, 16, 8),
        dropout=dropout,
        label_dropout=0,
    )
