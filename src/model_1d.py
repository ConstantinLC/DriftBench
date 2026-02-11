import math
import torch
import torch.nn as nn
from torch import einsum
import torch.nn.functional as F
from einops import rearrange
from functools import partial

### MODEL BLOCK DEFINITIONS
class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    # 1D Transposed Convolution
    return nn.ConvTranspose1d(dim, dim, 4, 2, 1)

def Downsample(dim, padding_mode='circular'):
    # 1D Convolution with stride 2
    return nn.Conv1d(dim, dim, 4, 2, 1, padding_mode=padding_mode)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, padding_mode='circular'):
        super().__init__()
        # 1D Conv
        self.proj = nn.Conv1d(dim, dim_out, 3, padding=1, padding_mode=padding_mode)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if scale_shift is not None:
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8, padding_mode='circular'):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None else None
        )

        self.block1 = Block(dim, dim_out, groups=groups, padding_mode=padding_mode)
        self.block2 = Block(dim_out, dim_out, groups=groups, padding_mode=padding_mode)
        # 1D Conv for residual
        self.res_conv = nn.Conv1d(dim, dim_out, 1, padding_mode=padding_mode) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.block1(x)

        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            # Rearrange for 1D: (batch, channel) -> (batch, channel, 1)
            h = rearrange(time_emb, "b c -> b c 1") + h

        h = self.block2(h)
        return h + self.res_conv(x)

class ConvNextBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True, padding_mode='circular'):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if time_emb_dim is not None else None
        )

        # 1D Depthwise Conv (kernel size 7)
        self.ds_conv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim, padding_mode=padding_mode)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv1d(dim, dim_out * mult, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv1d(dim_out * mult, dim_out, 3, padding=1, padding_mode=padding_mode),
        )

        self.res_conv = nn.Conv1d(dim, dim_out, 1, padding_mode=padding_mode) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if self.mlp is not None and time_emb is not None:
            condition = self.mlp(time_emb)
            # Rearrange for 1D
            h = h + rearrange(condition, "b c -> b c 1")

        h = self.net(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, padding_mode='circular'):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        # 1D Convs
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False, padding_mode=padding_mode)
        self.to_out = nn.Conv1d(hidden_dim, dim, 1, padding_mode=padding_mode)

    def forward(self, x):
        b, c, l = x.shape  # l is length instead of h, w
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) l -> b h c l", h=self.heads), qkv
        )
        q = q * self.scale

        sim = einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h l d -> b (h d) l")
        return self.to_out(out)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, padding_mode='circular'):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False, padding_mode=padding_mode)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1, padding_mode=padding_mode), 
                                    nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) l -> b h c l", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c l -> b (h c) l", h=self.heads)
        return self.to_out(out)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class FourierEmbedding(nn.Module):
    def __init__(self, out_channels, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(out_channels // 2) * scale)

    def forward(self, x):
        x = x.ger(self.freqs.to(x.dtype))
        return torch.cat([x.sin(), x.cos()], dim=-1)

class Unet1D(nn.Module):
    def __init__(
        self,
        dim,
        sigmas=torch.tensor([1]),
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
        padding_mode="circular",
    ):
        super().__init__()

        self.padding_mode = padding_mode

        # determine dimensions
        self.channels = channels
        # Move sigmas to register_buffer generally preferred, but keeping user style
        self.sigmas = sigmas.to('cuda')
        #self.register_buffer("sigmas", sigmas)

        init_dim = init_dim if init_dim is not None else dim // 3 * 2
        
        # 1D Init Conv
        self.init_conv = nn.Conv1d(channels, init_dim, 7, padding=3, padding_mode=self.padding_mode)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult, padding_mode=padding_mode)
        else:
            block_klass = partial(ResnetBlock, groups=resnet_block_groups, padding_mode=padding_mode)

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                FourierEmbedding(dim),
                nn.Linear(dim, time_dim),
                nn.SiLU(),
                nn.Linear(time_dim, time_dim),
            )
        else:
            time_dim = None
            self.time_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out, padding_mode=padding_mode))),
                        Downsample(dim_out, padding_mode=padding_mode) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim, padding_mode=padding_mode)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in, padding_mode=padding_mode))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = out_dim if out_dim is not None else channels
        self.final_conv = nn.Sequential(
            block_klass(dim, dim, time_emb_dim=time_dim), # Ensure block handles time_emb if passed
            nn.Conv1d(dim, out_dim, 1, padding_mode=self.padding_mode)
        )

    def forward(self, x, time):
        x = self.init_conv(x)

        noise_levels = torch.log(self.sigmas[time])

        t = self.time_mlp(noise_levels) if self.time_mlp is not None else None

        h = []
        interm = []

        # downsample
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            interm.append(x)

        # bottleneck
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        # upsample
        for block1, block2, attn, upsample in self.ups:
            # Concatenate along channel dim (dim 1)
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
            interm.append(x)

        # Final blocks usually don't need time embedding if it's just a projection, 
        # but if using ResBlock/ConvNextBlock at end, pass t
        final_block = self.final_conv[0]
        final_proj = self.final_conv[1]
        
        x = final_block(x, t)
        final_out = final_proj(x)
        
        return final_out