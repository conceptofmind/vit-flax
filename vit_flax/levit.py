import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from typing import Callable

from einops import rearrange

from math import ceil

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cast_tuple(val, l = 3):
    val = val if isinstance(val, tuple) else (val,)
    return (*val, *((val[-1],) * max(l - len(val), 0)))

def always(val):
    return lambda *args, **kwargs: val

class GlobalAvgPool(nn.Module):

    @nn.compact
    def __call__(self, x):
        return jnp.mean(x, axis = (1, 2))

class MLP(nn.Module):
    dim: int 
    mult: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features = self.dim * self.mult, kernel_size = (1, 1), strides = (1, 1))(x)
        x = jax.nn.hard_swish(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        x = nn.Conv(features = self.dim, kernel_size = (1, 1), strides = (1, 1))(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        return x

def jax_unstack(x, axis = 0):
    return jnp.moveaxis(x, axis, 0)



class Attention(nn.Module): 
    dim: int 
    fmap_size: int 
    heads: int = 8 
    dim_key: int = 32 
    dim_value: int = 64 
    dropout: float = 0.0 
    dim_out: int = None 
    downsample: bool = False
     
    @nn.compact
    def __call__(self, x):

        inner_dim_key = self.dim_key * self.heads
        inner_dim_value = self.dim_value * self.heads
        dim_out = default(self.dim_out, self.dim)

        heads = self.heads
        scale = self.dim_key ** -0.5

        #x = nn.BatchNorm(use_running_average = False, momentum = 0.9, epsilon = 1e-5)(x)

        to_q = nn.Sequential([
            nn.Conv(
                features = inner_dim_key, 
                kernel_size = (1, 1), 
                strides = ((2, 2) if self.downsample else (1, 1)), 
                use_bias=False),
            nn.BatchNorm(use_running_average = False, momentum = 0.9, epsilon = 1e-05),
        ])

        to_k = nn.Sequential([
            nn.Conv(
                features = inner_dim_key, 
                kernel_size = (1, 1), 
                strides = (1, 1), 
                use_bias = False),
            nn.BatchNorm(use_running_average = False, momentum=0.9, epsilon=1e-05),
        ])

        to_v = nn.Sequential([
            nn.Conv(
                features = inner_dim_value, 
                kernel_size = (1, 1), 
                strides = (1, 1), 
                use_bias=False),
            nn.BatchNorm(use_running_average = False, momentum=0.9, epsilon=1e-05),
        ])

        to_out = nn.Sequential([
            nn.Conv(features = dim_out, kernel_size = (1, 1), strides = (1, 1)),
            nn.BatchNorm(use_running_average = False, momentum=0.9, epsilon=1e-05),
            nn.Dropout(rate = self.dropout, deterministic = False),
        ])

        # positional bias
        pos_bias = nn.Embed(self.fmap_size * self.fmap_size, heads)
        q_range = jnp.arange(0, self.fmap_size, step=(2 if self.downsample else 1))
        k_range = jnp.arange(self.fmap_size)

        q_pos = jnp.stack(jnp.meshgrid(q_range, q_range, indexing='ij'), axis=-1)
        k_pos = jnp.stack(jnp.meshgrid(k_range, k_range, indexing='ij'), axis=-1)

        q_pos, k_pos = map(lambda t: rearrange(t, 'i j c -> (i j) c'), (q_pos, k_pos))
        rel_pos = jnp.abs((q_pos[:, None, ...] - k_pos[None, :, ...]))

        x_rel, y_rel = jax_unstack(rel_pos, axis = -1)
        pos_indices = (x_rel * self.fmap_size) + y_rel


        b, height, width, n = x.shape
        q = to_q(x)

        h = self.heads
        y = q.shape[1] # height

        qkv = (q, to_k(x), to_v(x))
        q, k, v = map(lambda t: rearrange(t, 'b ... (h d) -> b h (...) d', h=h), qkv)

        # i,j = height*width
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        def apply_pos_bias(fmap):
            bias = pos_bias(pos_indices)
            bias = rearrange(bias, 'i j h -> () h i j')
            return fmap + (bias / scale)

        dots = apply_pos_bias(dots)

        attn = nn.softmax(dots, axis = -1)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h (x y) d -> b x y (h d)', h=h, y=y)
        x = nn.gelu(x)
        x = to_out(x)

        return x

class Transformer(nn.Module):
    dim: int 
    fmap_size: int 
    depth: int 
    heads: int 
    dim_key: int 
    dim_value: int 
    mlp_mult: int = 2 
    dropout: float = 0.0 
    dim_out: int = None 
    downsample: bool = False

    @nn.compact
    def __call__(self, x):

        dim_out = default(self.dim_out, self.dim)
        attn_residual = (not self.downsample) and self.dim == dim_out
        layers = []

        for _ in range(self.depth):
            layers.append([
                Attention(self.dim, fmap_size = self.fmap_size, heads=self.heads, dim_key=self.dim_key, dim_value=self.dim_value,
                          dropout=self.dropout, downsample=self.downsample, dim_out=dim_out),
                MLP(dim_out, self.mlp_mult, dropout=self.dropout)
            ])

        for attn, mlp in layers:
            attn_res = (x if attn_residual else 0)
            x = attn(x) + attn_res
            x = mlp(x) + x

        return x

class LeViT(nn.Module):
    image_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_mult: int
    stages: int = 3
    dim_key: int = 32
    dim_value: int = 64
    dropout: float = 0.0
    num_distill_classes: int = None

    @nn.compact
    def __call__(self, img, **kwargs):

        dims = cast_tuple(self.dim, self.stages)
        depths = cast_tuple(self.depth, self.stages)
        layer_heads = cast_tuple(self.heads, self.stages)

        assert all(map(lambda t: len(t) == self.stages, (dims, depths, layer_heads))), \
            'dimensions, depths, and heads must be a tuple that is less than the designated number of stages'

        conv_embedding = nn.Sequential([
            nn.Conv(features = 32, kernel_size=(3, 3), strides=(2, 2), padding='SAME'),
            nn.Conv(features = 64, kernel_size=(3, 3), strides=(2, 2), padding='SAME'),
            nn.Conv(features = 128, kernel_size=(3, 3), strides=(2, 2), padding='SAME'),
            nn.Conv(features = dims[0], kernel_size=(3, 3), strides=(2, 2), padding='SAME')
        ])

        fmap_size = self.image_size // (2 ** 4)
        backbone = []

        for ind, dim, depth, heads in zip(range(self.stages), dims, depths, layer_heads):
            is_last = ind == (self.stages - 1)
            backbone.append(Transformer(dim, fmap_size, depth, heads, self.dim_key, self.dim_value, self.mlp_mult, self.dropout))

            if not is_last:
                next_dim = dims[ind + 1]
                backbone.append(Transformer(dim, fmap_size, 1, heads * 2, self.dim_key, self.dim_value, dim_out=next_dim, downsample=True))
                fmap_size = ceil(fmap_size / 2)


        distill_head = nn.Dense(features = self.num_distill_classes) if exists(self.num_distill_classes) else always(None)
        mlp_head = nn.Dense(features = self.num_classes)

        x = conv_embedding(img)

        backbone = nn.Sequential(backbone)
        x = backbone(x)

        x = GlobalAvgPool()(x)
        out = mlp_head(x)
        distill = distill_head(x)

        if exists(distill):
            return out, distill

        return out

if __name__ == "__main__":

    import numpy

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 224, 224, 3))

    v = LeViT(
        image_size = 224,
        num_classes = 1000,
        stages = 3,             # number of stages
        dim = (256, 384, 512),  # dimensions at each stage
        depth = 4,              # transformer of depth 4 at each stage
        heads = (4, 6, 8),      # heads at each stage
        mlp_mult = 2,
        dropout = 0.1
    )

    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2)}

    params = v.init(init_rngs, img)
    output = v.apply(params, img, mutable=['batch_stats'], rngs=init_rngs)

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: numpy.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")