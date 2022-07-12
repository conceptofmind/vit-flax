import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from typing import Callable

from einops import rearrange, repeat

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
        scale = self.dim_head ** -0.5
        
        to_qkv = nn.Dense(features = inner_dim * 3, use_bias = False)(x)
        qkv = jnp.split(to_qkv, 3, axis = -1)
        
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        # attention
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        attn = nn.softmax(dots, axis = -1)

        # re-attention
        reattn_weights = self.param('reattn_weights', nn.initializers.zeros, [self.heads, self.heads])
        attn = einsum('b h i j, h g -> b g i j', attn, reattn_weights)
        attn =  rearrange(attn, 'b h i j -> b i j h')
        attn = nn.LayerNorm()(attn)
        attn = rearrange(attn, 'b i j h -> b h i j')
        
        # aggregate and out
        x = jnp.matmul(attn, v)
        out = rearrange(x, 'b h n d -> b n (h d)')
        out = nn.Dense(features = self.dim)(out)
        to_out = nn.Dropout(rate = self.dropout)(out, deterministic = False)

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

        for attn, mlp in layers:
            x = attn(x)# + x
            x = mlp(x)# + x

        return x

class DeepViT(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim:int
    pool: str = 'cls'
    dim_head: int = 64
    dropout: float = 0.
    emb_dropout: float = 0.
    
    @nn.compact
    def __call__(self, x):
        assert self.image_size % self.patch_size == 0
        num_patches = (self.image_size // self.patch_size) ** 2
        assert self.pool in {'cls', 'mean'}

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])

        x = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        x = nn.Dense(features = self.dim)(x)

        b, n, _ = x.shape

        cls_tokens = repeat(cls_token, '() n d -> b n d', b = b)
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

    import numpy as np

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 256, 256, 3))

    v = DeepViT(
        image_size = 256,
        patch_size = 32,
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