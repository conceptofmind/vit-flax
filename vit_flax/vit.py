import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

import numpy as np

from typing import Callable

from einops import rearrange, repeat

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

class PreNorm(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm()(x)
        return self.fn(x, **kwargs)

class Residual(nn.Module):
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

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

class Attention(nn.Module):
    dim: int
    heads: int = 8
    dim_head: int = 64
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        inner_dim = self.dim_head * self.heads
        project_out = not (self.heads == 1 and self.dim_head == self.dim)
        scale = self.dim_head ** -0.5
        
        to_qkv = nn.Dense(features = inner_dim * 3, use_bias = False)(x)
        qkv = jnp.split(to_qkv, 3, axis = -1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        attn = nn.softmax(dots, axis = -1)
        #attn = nn.Dropout(rate = self.dropout)(attn, deterministic = False)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        if project_out:
            out = nn.Dense(features = self.dim)(out)
            to_out = nn.Dropout(rate = self.dropout)(out, deterministic = False)
        else:
            to_out = IdentityLayer()(out)

        return to_out

class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int
    mlp_dim: int
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):

        layers = []

        for _ in range(self.depth):
            layers.append([
                Residual(PreNorm(Attention(self.dim, self.heads, self.dim_head, self.dropout))),
                Residual(PreNorm(FeedForward(self.dim, self.mlp_dim, self.dropout)))
            ])

        for attn, ff in layers:
            x = attn(x)
            x = ff(x)

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
    dim_head = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0
    
    @nn.compact
    def __call__(self, x):

        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0 
        assert image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert self.pool in {'cls', 'mean'}

        #pos_embed_key, token_key = jax.random.split(self.key, 2)

        #pos_embedding = jax.random.normal(pos_embed_key, shape = [1, num_patches + 1, self.dim])
        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])
        #cls_token = jax.random.normal(token_key, shape = [1, 1, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])

        x = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        x = nn.Dense(features = self.dim)(x)

        b, n, _ = x.shape

        cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x += pos_embedding[:, :(n + 1)]

        x = nn.Dropout(rate = self.emb_dropout)(x, deterministic = False)

        x = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout)(x)

        x = jnp.mean(x, axis = 1) if self.pool == 'mean' else x[:, 0]

        x = IdentityLayer()(x)

        x = nn.LayerNorm()(x)

        x = nn.Dense(features = self.num_classes)(x)

        return x

if __name__ == '__main__':

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 256, 256, 3))

    v = ViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
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