# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
from typing import List, Optional, Tuple, Union
from abc import abstractmethod

import torch
from torch import nn
from torch import Tensor

from sda.nn.activations import ACTIVATIONS

# Largely based on https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/diffusion/ddpm/unet.py
# MIT License
# Copyright (c) 2020 Varuna Jayasiri

conv_dict = {
            1: nn.Conv1d,
            2: nn.Conv2d,
            3: nn.Conv3d,
        }

tr_conv_dict = {
            1: nn.ConvTranspose1d,
            2: nn.ConvTranspose2d,
            3: nn.ConvTranspose3d,
        }

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class AttentionBlock(nn.Module):
    """Attention block This is similar to [transformer multi-head
    attention]

    Args:
        n_channels (int): the number of channels in the input
        n_heads (int): the number of heads in multi-head attention
        d_k: the number of dimensions in each head
        n_groups (int): the number of groups for [group normalization][torch.nn.GroupNorm].

    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: Optional[int] = None, n_groups: int = 1):
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
        # Scale for dot-product attention
        self.scale = d_k**-0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: Tensor):
        # Get shape
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum("bihd,bjhd->bijh", q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum("bijh,bjhd->bihd", attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)
        return res


class ResidualBlock(TimestepBlock):
    """Wide Residual Blocks used in modern Unet architectures.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        activation (str): Activation function to use.
        norm (bool): Whether to use normalization.
        n_groups (int): Number of groups for group normalization.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial: int,
        activation: str = "gelu",
        norm: bool = False,
        n_groups: int = 1,
        use_scale_shift_norm: bool = True,
        embed_dim: int = 32,
    ):
        super().__init__()
        self.activation: nn.Module = ACTIVATIONS.get(activation, None)()
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")
        self.use_scale_shift_norm = use_scale_shift_norm
        Conv = conv_dict.get(spatial)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=[3] * spatial, padding=[1] * spatial)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=[3] * spatial, padding=[1] * spatial)
        # If the number of input channels is not equal to the number of output channels we have to
        # project the shortcut connection
        if in_channels != out_channels:
            self.shortcut = Conv(in_channels, out_channels, kernel_size=[1] * spatial)
        else:
            self.shortcut = nn.Identity()

        if norm:
            self.norm1 = nn.GroupNorm(n_groups, in_channels)
            self.norm2 = nn.GroupNorm(n_groups, out_channels)
        else:
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()

        self.emb_layers = nn.Sequential(
            self.activation,
            nn.Linear(
                embed_dim,
                2 * out_channels if use_scale_shift_norm else out_channels,
            ),
        )

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # First convolution layer
        h = self.conv1(self.activation(self.norm1(x)))
        emb_out = self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if not self.use_scale_shift_norm:
            x = x + emb_out
        # Second convolution layer
        # h = self.conv2(self.activation(self.norm2(h)))
        h = self.norm2(h)
        if self.use_scale_shift_norm:
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = h * (1 + scale) + shift
        h = self.conv2(self.activation(h))
        # Add the shortcut connection and return
        return h + self.shortcut(x)


class DownBlock(TimestepBlock):
    """Down block This combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the first half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (nn.Module): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, spatial, activation=activation, norm=norm)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: Tensor, emb: Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class UpBlock(TimestepBlock):
    """Up block that combines [`ResidualBlock`][pdearena.modules.twod_unet.ResidualBlock] and [`AttentionBlock`][pdearena.modules.twod_unet.AttentionBlock].

    These are used in the second half of U-Net at each resolution.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        has_attn (bool): Whether to use attention block
        activation (str): Activation function
        norm (bool): Whether to use normalization
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial: int,
        has_attn: bool = False,
        activation: str = "gelu",
        norm: bool = False,
    ):
        super().__init__()
        # The input has `in_channels + out_channels` because we concatenate the output of the same resolution
        # from the first half of the U-Net
        self.res = ResidualBlock(in_channels + out_channels, out_channels, spatial, activation=activation, norm=norm)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, x: Tensor, emb: Tensor):
        x = self.res(x, emb)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """Middle block

    It combines a `ResidualBlock`, `AttentionBlock`, followed by another
    `ResidualBlock`.

    This block is applied at the lowest resolution of the U-Net.

    Args:
        n_channels (int): Number of channels in the input and output.
        has_attn (bool, optional): Whether to use attention block. Defaults to False.
        activation (str): Activation function to use. Defaults to "gelu".
        norm (bool, optional): Whether to use normalization. Defaults to False.
    """

    def __init__(self, n_channels: int, spatial: int, has_attn: bool = False, activation: str = "gelu", norm: bool = False):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, spatial, activation=activation, norm=norm)
        self.attn = AttentionBlock(n_channels) if has_attn else nn.Identity()
        self.res2 = ResidualBlock(n_channels, n_channels, spatial, activation=activation, norm=norm)

    def forward(self, x: Tensor):
        x = self.res1(x)
        x = self.attn(x)
        x = self.res2(x)
        return x


class Upsample(nn.Module):
    r"""Scale up the feature map by $2 \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int, spatial: int):
        super().__init__()
        Conv = tr_conv_dict.get(spatial)
        self.conv = Conv(n_channels, n_channels, [4] * spatial, [2] * spatial, [1] * spatial)

    def forward(self, x: Tensor):
        return self.conv(x)


class Downsample(nn.Module):
    r"""Scale down the feature map by $\frac{1}{2} \times$

    Args:
        n_channels (int): Number of channels in the input and output.
    """

    def __init__(self, n_channels: int, spatial: int):
        super().__init__()
        Conv = conv_dict.get(spatial)
        self.conv = Conv(n_channels, n_channels, [3] * spatial, [2] * spatial, [1] * spatial)

    def forward(self, x: Tensor):
        return self.conv(x)


class Unet(nn.Module):
    """Modern U-Net architecture

    This is a modern U-Net architecture with wide-residual blocks and spatial attention blocks

    Args:
        n_input_scalar_components (int): Number of scalar components in the model
        n_input_vector_components (int): Number of vector components in the model
        n_output_scalar_components (int): Number of output scalar components in the model
        n_output_vector_components (int): Number of output vector components in the model
        time_history (int): Number of time steps in the input
        time_future (int): Number of time steps in the output
        hidden_channels (int): Number of channels in the hidden layers
        activation (str): Activation function to use
        norm (bool): Whether to use normalization
        ch_mults (list): List of channel multipliers for each resolution
        is_attn (list): List of booleans indicating whether to use attention blocks
        mid_attn (bool): Whether to use attention block in the middle block
        n_blocks (int): Number of residual blocks in each resolution
        use1x1 (bool): Whether to use 1x1 convolutions in the initial and final layers
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_input_vector_components: int,
        n_output_scalar_components: int,
        n_output_vector_components: int,
        time_history: int,
        time_future: int,
        hidden_channels: int,
        activation: str,
        norm: bool = False,
        ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
        is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, False, False),
        mid_attn: bool = False,
        n_blocks: int = 2,
        use1x1: bool = False,
        embed_dim: int = 32,
        spatial: int = 2,
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_input_vector_components = n_input_vector_components
        self.n_output_scalar_components = n_output_scalar_components
        self.n_output_vector_components = n_output_vector_components
        self.time_history = time_history
        self.time_future = time_future
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim

        self.activation: nn.Module = ACTIVATIONS.get(activation, None)()
        if self.activation is None:
            raise NotImplementedError(f"Activation {activation} not implemented")

        self.time_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            self.activation,
            nn.Linear(embed_dim, embed_dim),
        )
        # Number of resolutions
        n_resolutions = len(ch_mults)

        insize = time_history * (self.n_input_scalar_components + self.n_input_vector_components * spatial)
        n_channels = hidden_channels
        Conv = conv_dict[spatial]
        # Project image into feature map
        if use1x1:
            self.image_proj = Conv(insize, n_channels, kernel_size=1)
        else:
            self.image_proj = Conv(insize, n_channels, kernel_size=[3] * spatial, padding=[1] * spatial)

        # #### First half of U-Net - decreasing resolution
        down = []
        # Number of channels
        out_channels = in_channels = n_channels
        # For each resolution
        for i in range(n_resolutions):
            # Number of output channels at this resolution
            out_channels = in_channels * ch_mults[i]
            # Add `n_blocks`
            for _ in range(n_blocks):
                down.append(
                    TimestepEmbedSequential(DownBlock(
                        in_channels,
                        out_channels,
                        spatial,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    ))
                )
                in_channels = out_channels
            # Down sample at all resolutions except the last
            if i < n_resolutions - 1:
                down.append(TimestepEmbedSequential(Downsample(in_channels, spatial)))

        # Combine the set of modules
        self.down = nn.ModuleList(down)

        # Middle block
        self.middle = TimestepEmbedSequential(MiddleBlock(out_channels, spatial,has_attn=mid_attn, activation=activation, norm=norm))

        # #### Second half of U-Net - increasing resolution
        up = []
        # Number of channels
        in_channels = out_channels
        # For each resolution
        for i in reversed(range(n_resolutions)):
            # `n_blocks` at the same resolution
            out_channels = in_channels
            for _ in range(n_blocks):
                up.append(
                    TimestepEmbedSequential(UpBlock(
                        in_channels,
                        out_channels,
                        spatial,
                        has_attn=is_attn[i],
                        activation=activation,
                        norm=norm,
                    ))
                )
            # Final block to reduce the number of channels
            out_channels = in_channels // ch_mults[i]
            up.append(TimestepEmbedSequential(UpBlock(in_channels, out_channels, spatial, has_attn=is_attn[i], activation=activation, norm=norm)))
            in_channels = out_channels
            # Up sample at all resolutions except last
            if i > 0:
                up.append(TimestepEmbedSequential(Upsample(in_channels, spatial)))

        # Combine the set of modules
        # self.up = nn.ModuleList(up)
        self.up = nn.ModuleList(up)

        if norm:
            self.norm = nn.GroupNorm(8, n_channels)
        else:
            self.norm = nn.Identity()
        out_channels = time_future * (self.n_output_scalar_components + self.n_output_vector_components * spatial)
        #
        if use1x1:
            self.final = Conv(in_channels, out_channels, kernel_size=1)
        else:
            self.final = Conv(in_channels, out_channels, kernel_size=[3] * spatial, padding=[1] * spatial)

    def forward(self, x: Tensor, emb: Tensor) -> Tensor:
        # assert x.dim() == 5
        orig_shape = x.shape
        # x = x.reshape(x.size(0), -1, *x.shape[3:])  # collapse T,C
        # x = self.image_proj(x)

        emb = self.time_embed(emb)

        h = [x]
        for m in self.down:
            x = m(x, emb)
            h.append(x)

        x = self.middle(x, emb)

        for m in self.up:
            if isinstance(m, Upsample):
                x = m(x)
            else:
                # Get the skip connection from first half of U-Net and concatenate
                s = h.pop()
                x = torch.cat((x, s), dim=1)
                #
                x = m(x, emb)

        x = self.final(self.activation(self.norm(x)))
        x = x.reshape(
            orig_shape[0], -1, (self.n_output_scalar_components + self.n_output_vector_components * 2), *orig_shape[3:]
        )
        return x


if __name__ == "__main__":
    seq_length = 4
    x_dim = 4
    embed_dim = 32

    net = Unet(
        n_input_scalar_components = 0,
        n_input_vector_components = 1,
        n_output_scalar_components = 0,
        n_output_vector_components = 1,
        time_history = seq_length,
        time_future = seq_length,
        hidden_channels = 32,
        # hidden_channels = 64,
        activation = 'GELU',
        norm = True,
        # ch_mults = (1, 2, 4, 16),
        ch_mults = (1, 2, 4),
        is_attn = (False, False, False, False),
        mid_attn = False,
        n_blocks = 2,
        use1x1 = False,
        embed_dim = embed_dim,
        spatial = 1
    )

    x = torch.randn(2, seq_length, 1)
    t = torch.randn(2, seq_length, 1)

    from sda.score import TimeEmbedding
    embedding = TimeEmbedding(embed_dim)
    t = embedding(t)

    print(net)
    y = net(x, t)
