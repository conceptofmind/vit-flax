from typing import Any, Callable
import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from einops import rearrange, reduce, repeat

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

class Parallel(nn.Module):
    fns: Callable

    @nn.compact
    def __call__(self, x):
        return sum([fn(x) for fn in self.fns])

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

class Attention(nn.Module): 
    dim: int 
    heads: int = 8 
    dim_head: int = 64 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        inner_dim = self.dim_head *  self.heads
        project_out = not (self.heads == 1 and self.dim_head == self.dim)

        heads = self.heads
        scale = self.dim_head ** -0.5

        to_qkv = nn.Dense(inner_dim * 3, use_bias = False)
        
        to_out = nn.Sequential([
            nn.Dense(features = self.dim, use_bias = False),
            nn.Dropout(rate = self.dropout, deterministic = False),
        ]) if project_out else IdentityLayer()

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
    num_parallel_branches: int = 2 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        layers = []

        # attn_block = lambda: PreNorm(Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout))
        # ff_block = lambda: PreNorm(MLP(dim, mlp_dim, dropout=dropout))

        for _ in range(self.depth):
            layers.append([
                Parallel([PreNorm(Attention(self.dim, heads=self.heads, dim_head=self.dim_head, dropout=self.dropout)) for _ in range(self.num_parallel_branches)]),
                Parallel([PreNorm(FeedForward(self.dim, self.mlp_dim, dropout=self.dropout)) for _ in range(self.num_parallel_branches)])
            ])

        for attns, ffs in layers:
            x = attns(x) + x
            x = ffs(x) + x
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
    num_parallel_branches: int = 2
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

        patch_embedding = nn.Sequential([
            nn.Dense(self.dim)
        ])

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])

        
        dropout = nn.Dropout(rate = self.emb_dropout, deterministic=False)

        transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.num_parallel_branches, self.dropout)

        pool = self.pool

        mlp_head = nn.Sequential([
            nn.LayerNorm(epsilon = 1e-5, use_bias = False),
            nn.Dense(self.num_classes)
        ])

        img = rearrange(img, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        x = patch_embedding(img)


        b, n, d = x.shape

        cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x += pos_embedding[:, :(n + 1)]
        x = dropout(x)
        x = transformer(x)

        if pool == 'mean':
            x = jnp.mean(x, axis=1)
        else:
            x = x[:, 0]

        x = mlp_head(x)

        return x

if __name__ == '__main__':

    import numpy as np

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 256, 256, 3))

    v = ViT(
        image_size = 256,
        patch_size = 16,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 8,
        mlp_dim = 2048,
        num_parallel_branches = 2,  # in paper, they claimed 2 was optimal
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