import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from einops import rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def posemb_sincos_2d(patches, temperature = 10000):
    _, h, w, dim = patches.shape

    y, x = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing = 'ij')
    assert (dim % 4) == 0, 'feature dimension must be multiple of 4 for sincos emb'
    omega = jnp.arange(dim // 4) / (dim // 4 - 1)
    omega = 1. / (temperature ** omega)

    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :] 
    pe = jnp.concatenate((jnp.sin(x), jnp.cos(x), jnp.sin(y), jnp.cos(y)), axis = 1)
    return pe

# classes

class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

class FeedForward(nn.Module):
    dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.dim)(x)
        return x

class Attention(nn.Module): 
    dim: int 
    heads: int = 8
    dim_head: int = 64

    @nn.compact
    def __call__(self, x):

        inner_dim = self.dim_head *  self.heads
        heads = self.heads
        scale = self.dim_head ** -0.5
        norm = nn.LayerNorm(epsilon = 1e-5, use_bias = False)

        to_qkv = nn.Dense(inner_dim * 3, use_bias = False)
        to_out = nn.Dense(self.dim, use_bias = False)

        x = norm(x)

        qkv = to_qkv(x).split(3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        attn = nn.softmax(dots, axis = -1)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(x, 'b h n d -> b n (h d)')
        return to_out(out)

class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):

        layers = []

        for _ in range(self.depth):
            layers.append([
                Attention(self.dim, heads = self.heads, dim_head = self.dim_head),
                FeedForward(self.dim, self.mlp_dim)
            ])

        for attn, ff in layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class SimpleViT(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    channels: int = 3 
    dim_head: int = 64
        
    @nn.compact
    def __call__(self, img):

        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim)

        linear_head = nn.Sequential([
            nn.LayerNorm(epsilon = 1e-5, use_bias = False),
            nn.Dense(features = self.num_classes)
        ])

        x = rearrange(img, 'b c (h p1) (w p2) -> b h w (p1 p2 c)', p1 = patch_height, p2 = patch_width)
        x = nn.Dense(features = self.dim)(x)
        pe = posemb_sincos_2d(x)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        x = transformer(x)
        x = x.mean(axis = 1)

        x = IdentityLayer()(x)
        return linear_head(x)

if __name__ == "__main__":

    import numpy as np

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 3, 256, 256))

    v = SimpleViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048
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