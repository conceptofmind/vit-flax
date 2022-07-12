from threading import local
from typing import Any, Callable
import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from einops import rearrange, reduce

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

def divisible_by(val, d):
    return (val % d) == 0

class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

class Downsample(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        conv = nn.Conv(self.dim, kernel_size = (3,3), strides=(2,2), padding='SAME')
        x = conv(x)
        return x

class PEG(nn.Module): 
    dim: int 
    kernel_size: int = 3

    @nn.compact
    def __call__(self, x):
        proj = nn.Conv(self.dim, kernel_size = (self.kernel_size, self.kernel_size), strides = (1,1), padding='SAME', feature_group_count=self.dim)
        x = proj(x) + x
        return x


class MLP(nn.Module):
    dim: int 
    mult: int = 4 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        x = nn.Dense(self.dim * self.mult)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate = self.dropout, deterministic=False)(x)
        x = nn.Dense(self.dim)(x)
        return x

class Attention(nn.Module):
    dim: int 
    heads: int = 4 
    dim_head: int = 32 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, rel_pos_bias=None):

        inner_dim = self.dim_head * self.heads
        heads = self.heads
        scale = self.dim_head ** -0.5

        norm = nn.LayerNorm(epsilon=1e-5, use_bias=False)
        
        to_qkv = nn.Dense(inner_dim * 3, use_bias=False)

        to_out = nn.Dense(self.dim)

        h = self.heads

        # prenorm
        x = norm(x)
        qkv = to_qkv(x)
        qkv = jnp.split(qkv, 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)
        q = q * scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add relative positional bias for local tokens
        if exists(rel_pos_bias):
            sim = sim + rel_pos_bias
        attn = nn.softmax(sim, axis = -1)

        # merge heads

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = to_out(x)

        return x

class R2LTransformer(nn.Module):
    dim: int 
    window_size: int 
    depth: int = 4 
    heads: int = 4 
    dim_head: int = 32 
    attn_dropout: float = 0.0 
    ff_dropout: float = 0.0

    @nn.compact
    def __call__(self, local_tokens, region_tokens=None):

        layers = []

        window_size = self.window_size
        rel_positions = 2 * window_size - 1
        local_rel_pos_bias = nn.Embed(rel_positions ** 2, self.heads)

        for _ in range(self.depth):
            layers.append([
                Attention(self.dim, heads = self.heads, dim_head = self.dim_head, dropout = self.attn_dropout),
                MLP(self.dim, dropout = self.ff_dropout)
            ])

        lh, lw = local_tokens.shape[1:3]
        rh, rw = region_tokens.shape[1:3]
        window_size_h, window_size_w = lh // rh, lw // rw

        local_tokens = rearrange(local_tokens, 'b h w c -> b (h w) c')
        region_tokens = rearrange(region_tokens, 'b h w c -> b (h w) c')

        # calculate local relative positional bias
        h_range = jnp.arange(window_size_h)
        w_range = jnp.arange(window_size_w)

        grid_x, grid_y = jnp.meshgrid(h_range, w_range, indexing='ij')
        grid = jnp.stack([grid_x, grid_y])
        grid = rearrange(grid, 'c h w -> c (h w)')
        grid = (grid[:, :, None] - grid[:, None, :]) + (self.window_size - 1)

        bias_indices = jnp.sum((grid * jnp.array([1, self.window_size * 2 - 1])[:, None, None]), axis=0)
        rel_pos_bias = local_rel_pos_bias(bias_indices)
        rel_pos_bias = rearrange(rel_pos_bias, 'i j h -> () h i j')
        rel_pos_bias = jnp.pad(rel_pos_bias, [[0, 0], [0, 0], [1, 0], [1, 0]])

        # go through r2l transformer layers
        for attn, ff in layers:
            region_tokens = attn(region_tokens) + region_tokens

            # concat region tokens to local tokens

            local_tokens = rearrange(local_tokens, 'b (h w) d -> b h w d', h=lh)
            local_tokens = rearrange(local_tokens, 'b (h p1) (w p2) d -> (b h w) (p1 p2) d', p1=window_size_h, p2=window_size_w)
            region_tokens = rearrange(region_tokens, 'b n d -> (b n) () d')

            # do self attention on local tokens, along with its regional token
            region_and_local_tokens = jnp.concatenate([region_tokens, local_tokens], axis=1)
            region_and_local_tokens = attn(region_and_local_tokens, rel_pos_bias=rel_pos_bias) + region_and_local_tokens

            # feedforward
            region_and_local_tokens = ff(region_and_local_tokens) + region_and_local_tokens

            # split back local and regional tokens
            region_tokens, local_tokens = region_and_local_tokens[:, :1], region_and_local_tokens[:, 1:]
            local_tokens = rearrange(local_tokens, '(b h w) (p1 p2) d -> b (h p1 w p2) d', h=lh // window_size_h, w=lw // window_size_w, p1=window_size_h)
            region_tokens = rearrange(region_tokens, '(b n) () d -> b n d', n=rh * rw)

        local_tokens = rearrange(local_tokens, 'b (h w) c -> b h w c', h=lh, w=lw)
        region_tokens = rearrange(region_tokens, 'b (h w) c -> b h w c', h=rh, w=rw)

        return local_tokens, region_tokens

class RegionViT(nn.Module):
    dim: tuple = (64, 128, 256, 512)
    depth: tuple = (2, 2, 8, 2)
    window_size: int = 7
    num_classes: int = 1000
    tokenize_local_3_conv: bool = False
    local_patch_size: int = 4
    use_peg: bool = False
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, **kwargs):

        dim = cast_tuple(self.dim, 4)
        depth = cast_tuple(self.depth, 4)
        assert len(dim) == 4, 'dim needs to be a single value or a tuple of length 4'
        assert len(depth) == 4, 'depth needs to be a single value or a tuple of length 4'

        local_patch_size = self.local_patch_size

        region_patch_size = local_patch_size * self.window_size
        region_patch_size = local_patch_size * self.window_size

        init_dim, *_, last_dim = dim


        # layers
        region_layers = []

        for ind, dim, num_layers in zip(range(4), dim, depth):
            not_first = ind != 0
            need_downsample = not_first
            need_peg = not_first and self.use_peg

            region_layers.append([
                Downsample(dim) if need_downsample else IdentityLayer(),
                PEG(dim) if need_peg else IdentityLayer(),
                R2LTransformer(dim, depth=num_layers, window_size=self.window_size, attn_dropout=self.attn_dropout, ff_dropout=self.ff_dropout)
            ])

        # final logits
        to_logits = nn.Sequential([
            
            nn.LayerNorm(epsilon=1e-5, use_bias=False),
            nn.Dense(self.num_classes)
        ])

        _, h, w, _ = x.shape
        assert divisible_by(h, region_patch_size) and divisible_by(w, region_patch_size), 'height and width must be divisible by region patch size'
        assert divisible_by(h, local_patch_size) and divisible_by(w, local_patch_size), 'height and width must be divisible by local patch size'

        if self.tokenize_local_3_conv:
            local_encoder = nn.Conv(init_dim, kernel_size=3, strides=2, padding='SAME')(x)
            local_encoder = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(local_encoder)
            local_encoder = nn.gelu(local_encoder)
            local_encoder = nn.Conv(init_dim, kernel_size=(3,3), strides=(2,2), padding='SAME')(local_encoder)
            local_encoder = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(local_encoder)
            local_encoder = nn.gelu(local_encoder)
            local_encoder = nn.Conv(init_dim, kernel_size=(3,3), strides=(1,1), padding='SAME')(local_encoder)
            local_tokens = local_encoder
        else:
            local_encoder = nn.Conv(init_dim, kernel_size=(8,8), strides=(4,4), padding='SAME')(x)
            local_tokens = local_encoder

        x = rearrange(local_tokens, 'b (h p1) (w p2) c -> b h w (c p1 p2) ', p1=region_patch_size, p2=region_patch_size)
        region_encoder = nn.Sequential([
            nn.Conv(init_dim, kernel_size=(1,1), strides=(1,1))
        ])
        region_tokens = region_encoder(x)

        for down, peg, transformer in region_layers:
            local_tokens, region_tokens = down(local_tokens), down(region_tokens)
            local_tokens = peg(local_tokens)
            local_tokens, region_tokens = transformer(local_tokens, region_tokens)

        x = reduce(region_tokens, 'b h w c -> b c', 'mean')

        x = to_logits(region_tokens)
        return x

if __name__ == "__main__":

    import numpy

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 224, 224, 3))

    v = RegionViT(
        dim = (64, 128, 256, 512),      # tuple of size 4, indicating dimension at each stage
        depth = (2, 2, 8, 2),           # depth of the region to local transformer at each stage
        window_size = 7,                # window size, which should be either 7 or 14
        num_classes = 1000,             # number of output classes
        tokenize_local_3_conv = False,  # whether to use a 3 layer convolution to encode the local tokens from the image. the paper uses this for the smaller models, but uses only 1 conv (set to False) for the larger models
        use_peg = False,                # whether to use positional generating module. they used this for object detection for a boost in performance
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