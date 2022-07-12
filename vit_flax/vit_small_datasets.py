import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from typing import Callable

from einops import rearrange, repeat
from torch import det

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def shift(x):
    b, h, w, c = x.shape
    shifted_x = []

    shifts = [1, -1] # [shift, axis]

    # width
    z = jnp.zeros([b, h, 1, c])
    for idx, shift in enumerate(shifts):
        if idx == 0:
            s = jnp.roll(x, shift, axis = 2)[:, :, shift:, :]
            concat = jnp.concatenate([z, s], axis = 2)

        else:
            s = jnp.roll(x, shift, axis = 2)[:, :, :shift, :]
            concat = jnp.concatenate([s, z], axis = 2)

        shifted_x.append(concat)

    # height
    z = jnp.zeros([b, 1, w, c])
    for idx, shift in enumerate(shifts):
        if idx == 0:
            s = jnp.roll(x, shift, axis=1)[:, shift:, :, :]
            concat = jnp.concatenate([z, s], axis=1)
        else:
            s = jnp.roll(x, shift, axis=1)[:, :shift, :, :]
            concat = jnp.concatenate([s, z], axis=1)

        shifted_x.append(concat)

    return shifted_x


class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features = self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        x = nn.Dense(features = self.dim)(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        return x

class LSA(nn.Module):
    dim: int 
    heads: int = 8 
    dim_head: int = 64 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        inner_dim = self.dim_head * self.heads
        heads = self.heads
        temperature = jnp.log(self.dim_head ** -0.5)

        to_qkv = nn.Dense(inner_dim * 3, use_bias=False)

        to_out = nn.Sequential([
            nn.Dense(self.dim),
            nn.Dropout(rate = self.dropout, deterministic = False),
        ])

        qkv = to_qkv(x)
        qkv = jnp.split(qkv, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * jnp.exp(temperature)

        mask = jnp.eye(dots.shape[-1], dtype=bool)
        mask_value = -jnp.finfo(dots).max
        dots = jnp.where(mask, mask_value, dots)

        attn = nn.softmax(dots, axis = -1)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(x, 'b h n d -> b n (h d)')
        out = to_out(out)

        return out

class Transformer(nn.Module):
    dim: int
    depth: int 
    heads: int
    dim_head: int 
    mlp_dim: int
    dropout: float =  0.0

    @nn.compact
    def __call__(self, x):

        layers = []

        for _ in range(self.depth):
            layers.append([
                PreNorm(LSA(self.dim, heads = self.heads, dim_head = self.dim_head, dropout = self.dropout)),
                PreNorm(FeedForward(self.dim, self.mlp_dim, dropout = self.dropout))
            ])

        for attn, ff in layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class SPT(nn.Module):
    dim: int 
    patch_size: int

    @nn.compact
    def __call__(self, x):

        to_patch_tokens = nn.Sequential([  
            nn.LayerNorm(epsilon = 1e-5, use_bias = False),
            nn.Dense(self.dim)
        ])

        shifted_x = shift(x)
        x_with_shifts = jnp.concatenate([x, *shifted_x], axis = -1)
        
        x_with_shifts = rearrange(x_with_shifts, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=self.patch_size, p2=self.patch_size)

        x = to_patch_tokens(x_with_shifts)

        return x

class ViT(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    pool: str = 'cls'
    dim_head: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0

    @nn.compact
    def __call__(self, img, **kwargs):

        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        patch_embedding = SPT(dim=self.dim, patch_size=self.patch_size)

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])

        dropout = nn.Dropout(rate=self.emb_dropout, deterministic=False)

        transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout)

        pool = self.pool

        mlp_head = nn.Sequential([
            nn.LayerNorm(epsilon = 1e-5, use_bias = False),
            nn.Dense(self.num_classes)
        ])

        x = patch_embedding(img)
        b, n, d = x.shape

        cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x += pos_embedding[:, :(n + 1)]
        x = dropout(x)

        x = transformer(x)

        if pool == 'mean':
            x = jnp.mean(x, axis = 1)
        else:
            x = x[:, 0]

        x = mlp_head(x)

        return x


if __name__ == '__main__':

    import numpy as np

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (4, 256, 256, 3))

    v = ViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1
    )

    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2), 
                'emb_dropout': jax.random.PRNGKey(3)}

    params = v.init(init_rngs, img)
    output = v.apply(params, img, rngs=init_rngs)
    print(output.shape)

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")

    spt = SPT(
        dim = 1024,
        patch_size = 16
    )

    spt_params = spt.init(init_rngs, img)
    spt_output = spt.apply(spt_params, img, rngs=init_rngs)
    print(spt_output.shape)