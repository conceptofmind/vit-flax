import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from typing import Callable

from einops import rearrange, reduce

from functools import partial

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

class LayerNorm(nn.Module): # layernorm, but done in the channel dimension #1
    dim: int
    eps: float = 1e-5

    @nn.compact
    def __call__(self, x):
        eps = self.eps
        g = self.param('g', nn.initializers.ones, [1, 1, 1, self.dim])
        b = self.param('b', nn.initializers.zeros, [1, 1, 1, self.dim])

        var = jnp.var(x, axis = -1, keepdims = True)
        mean = jnp.mean(x, axis = -1, keepdims = True)

        x = (x - mean) / jnp.sqrt((var + eps)) * g + b

        return x

class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        return self.fn(x, **kwargs)


class Downsample(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        conv = nn.Conv(self.dim, kernel_size = (3,3), strides = (2, 2), padding='SAME')
        x = conv(x)
        return x

class PEG(nn.Module):
    dim: int 
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        proj = nn.Conv(
            features = self.dim, 
            kernel_size = (self.kernel_size, self.kernel_size), 
            strides = (1, 1), 
            padding="SAME",
            feature_group_count = self.dim, 
            )(x)
        x = proj + x
        return x

class MLP(nn.Module):
    dim: int 
    expansion_factor: int = 4 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        inner_dim = self.dim * self.expansion_factor

        x = nn.Conv(features = inner_dim, kernel_size = (1, 1), strides = (1, 1))(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        x = nn.Conv(features = self.dim, kernel_size = (1, 1), strides = (1, 1))(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)

        return x

class ScalableSelfAttention(nn.Module):
    dim: int
    heads: int = 8 
    dim_key: int = 32
    dim_value: int = 32
    dropout: int = 0.0 
    reduction_factor: int = 1

    @nn.compact
    def __call__(self, x):

        heads = self.heads
        scale = self.dim_key ** -0.5

        to_q = nn.Conv(self.dim_key * heads, kernel_size = (1, 1), strides = (1, 1), use_bias=False)

        to_k = nn.Conv(self.dim_key * heads, 
                        kernel_size = (self.reduction_factor, self.reduction_factor), 
                        strides = (self.reduction_factor, self.reduction_factor), 
                        use_bias=False)

        to_v = nn.Conv(self.dim_value * heads, 
        kernel_size= (self.reduction_factor, self.reduction_factor), 
        strides= (self.reduction_factor, self.reduction_factor),  
        use_bias=False)

        to_out = nn.Sequential([
            nn.Conv(self.dim, kernel_size= (1, 1), strides= (1, 1)),
            nn.Dropout(rate=self.dropout, deterministic=False),
        ])

        _, height, width, _ = x.shape
        heads = self.heads

        q, k, v = to_q(x), to_k(x), to_v(x)

        # split out heads
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> b h (...) d', h=heads), (q, k, v))

        # similarity

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        # attention
        attn = nn.softmax(dots, axis = -1)

        # aggregate values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge back heads
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x=height, y=width)
        out = to_out(out)

        return out

class InteractiveWindowedSelfAttention(nn.Module):
    dim: int 
    window_size: int
    heads: int = 8 
    dim_key: int = 32 
    dim_value: int = 32 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        heads = self.heads
        scale = self.dim_key ** -0.5
        window_size = self.window_size

        local_interactive_module = nn.Conv(self.dim_value * heads, kernel_size = (3, 3), strides = (1,1), padding='SAME')

        to_q = nn.Conv(self.dim_key * heads, kernel_size = (1, 1), strides = (1, 1), use_bias=False)
        to_k = nn.Conv(self.dim_key * heads, kernel_size= (1, 1), strides = (1, 1), use_bias=False)
        to_v = nn.Conv(self.dim_value * heads, kernel_size=(1,1), strides=(1,1), use_bias=False)

        to_out = nn.Sequential([
            nn.Conv(self.dim, kernel_size = (1, 1), strides = (1, 1)),
            nn.Dropout(rate = self.dropout, deterministic=False)
        ])


        _, height, width, _ = x.shape
        heads = self.heads
        wsz = window_size

        wsz_h, wsz_w = default(wsz, height), default(wsz, width)
        assert (height % wsz_h) == 0 and (width % wsz_w) == 0, f'height ({height}) or width ({width}) of feature map is not divisible by the window size ({wsz_h}, {wsz_w})'

        q, k, v = to_q(x), to_k(x), to_v(x)

        # get output of LIM
        local_out = local_interactive_module(v)

        # divide into window (and split out heads) for efficient self attention
        q, k, v = map(lambda t: rearrange(t, 'b (x w1) (y w2) (h d) -> (b x y) h (w1 w2) d', h = heads, w1 = wsz_h, w2 = wsz_w), (q, k, v))

        # similarity
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        # attention
        attn = nn.softmax(dots, axis = -1)

        # aggregate values
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # reshape the windows back to full feature map (and merge heads)
        out = rearrange(out, '(b x y) h (w1 w2) d -> b (x w1) (y w2) (h d)', x = height // wsz_h, y = width // wsz_w, w1 = wsz_h, w2 = wsz_w)

        # add LIM output
        out = out + local_out

        out = to_out(out)

        return out

class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int = 8
    ff_expansion_factor: int = 4
    dropout: float = 0.
    ssa_dim_key: int = 32
    ssa_dim_value: int = 32
    ssa_reduction_factor: int = 1
    iwsa_dim_key: int = 32
    iwsa_dim_value: int = 32
    iwsa_window_size: int = None
    norm_output: bool = True

    @nn.compact
    def __call__(self, x):

        layers = []

        for ind in range(self.depth):
            is_first = ind == 0

            layers.append([
                PreNorm(ScalableSelfAttention(self.dim, heads=self.heads, dim_key=self.ssa_dim_key, dim_value=self.ssa_dim_value,
                                                   reduction_factor=self.ssa_reduction_factor, dropout=self.dropout)),
                PreNorm(MLP(self.dim, expansion_factor=self.ff_expansion_factor, dropout=self.dropout)),
                PEG(self.dim) if is_first else None,
                PreNorm(MLP(self.dim, expansion_factor=self.ff_expansion_factor, dropout=self.dropout)),
                PreNorm(InteractiveWindowedSelfAttention(self.dim, heads=self.heads, dim_key=self.iwsa_dim_key, dim_value=self.iwsa_dim_value,
                                                              window_size=self.iwsa_window_size,
                                                              dropout=self.dropout))
            ])

        norm = nn.LayerNorm(epsilon = 1e-5, use_bias = False) if self.norm_output else IdentityLayer()

        for ssa, ff1, peg, iwsa, ff2 in layers:
            x = ssa(x) + x
            x = ff1(x) + x

            if exists(peg):
                x = peg(x)

            x = iwsa(x) + x
            x = ff2(x) + x

        x = norm(x)

        return x

class ScalableViT(nn.Module):
    num_classes: int
    dim: int
    depth: int
    heads: int
    reduction_factor: int
    window_size: int = None
    iwsa_dim_key: int = 32
    iwsa_dim_value: int = 32
    ssa_dim_key: int = 32
    ssa_dim_value: int = 32
    ff_expansion_factor: int = 4
    channels: int = 3
    dropout: float = 0.0

    @nn.compact
    def __call__(self, img, **kwargs):


        to_patches = nn.Conv(self.dim, kernel_size = (7,7), strides=(4,4), padding='SAME')

        assert isinstance(self.depth, tuple), 'depth needs to be tuple if integers indicating number of transformer blocks at that stage'

        num_stages = len(self.depth)
        dims = tuple(map(lambda i: (2 ** i) * self.dim, range(num_stages)))

        hyperparams_per_stage = [
            self.heads,
            self.ssa_dim_key,
            self.ssa_dim_value,
            self.reduction_factor,
            self.iwsa_dim_key,
            self.iwsa_dim_value,
            self.window_size,
        ]

        hyperparams_per_stage = list(map(partial(cast_tuple, length=num_stages), hyperparams_per_stage))
        assert all(tuple(map(lambda arr: len(arr) == num_stages, hyperparams_per_stage)))

        scalable_layers = []

        for ind, (layer_dim, layer_depth, layer_heads, layer_ssa_dim_key, layer_ssa_dim_value, layer_ssa_reduction_factor, layer_iwsa_dim_key, layer_iwsa_dim_value, layer_window_size) in enumerate(zip(dims, self.depth, *hyperparams_per_stage)):
            is_last = ind == (num_stages - 1)

            scalable_layers.append([
                Transformer(dim=layer_dim, depth=layer_depth, heads=layer_heads,
                            ff_expansion_factor=self.ff_expansion_factor, dropout=self.dropout, ssa_dim_key=layer_ssa_dim_key,
                            ssa_dim_value=layer_ssa_dim_value, ssa_reduction_factor=layer_ssa_reduction_factor,
                            iwsa_dim_key=layer_iwsa_dim_key, iwsa_dim_value=layer_iwsa_dim_value,
                            iwsa_window_size=layer_window_size),
                Downsample(layer_dim * 2) if not is_last else None
            ])

        mlp_head = nn.Sequential([
            nn.LayerNorm(epsilon = 1e-5, use_bias = False),
            nn.Dense(self.num_classes)
        ])

        x = to_patches(img)

        for transformer, downsample in scalable_layers:
            x = transformer(x)

            if exists(downsample):
                x = downsample(x)

        x = reduce(x, 'b h w d-> b d', 'mean')
        x = mlp_head(x)

        return x

if __name__ == "__main__":

    import numpy

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 256, 256, 3))


    v = ScalableViT(
        num_classes = 1000,
        dim = 64,                               # starting model dimension. at every stage, dimension is doubled
        heads = (2, 4, 8, 16),                  # number of attention heads at each stage
        depth = (2, 2, 20, 2),                  # number of transformer blocks at each stage
        ssa_dim_key = (40, 40, 40, 32),         # the dimension of the attention keys (and queries) for SSA. in the paper, they represented this as a scale factor on the base dimension per key (ssa_dim_key / dim_key)
        reduction_factor = (8, 4, 2, 1),        # downsampling of the key / values in SSA. in the paper, this was represented as (reduction_factor ** -2)
        window_size = (64, 32, None, None),     # window size of the IWSA at each stage. None means no windowing needed
        dropout = 0.1,                          # attention and feedforward dropout
    )

    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2), 
                'emb_dropout': jax.random.PRNGKey(3)}

    params = v.init(init_rngs, img)
    output = v.apply(params, img, rngs=init_rngs)
    print(output.shape)

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: numpy.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")