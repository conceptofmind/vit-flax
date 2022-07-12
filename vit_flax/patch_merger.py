from typing import Any, Callable
import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from einops import rearrange, reduce

def exists(val):
    return val is not None

def default(val ,d):
    return val if exists(val) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

class PatchMerger(nn.Module): 
    dim: int
    num_tokens_out: int

    @nn.compact
    def __call__(self, x):

        scale = self.dim ** -0.5
        norm = nn.LayerNorm(epsilon = 1e-5, use_bias = False)
        key = jax.random.PRNGKey(0)
        queries = jax.random.normal(key, [self.num_tokens_out, self.dim])

        x = norm(x)
        sim = jnp.matmul(queries, jnp.transpose(x, [0, 2, 1]) * scale)
        attn = nn.softmax(sim, axis = -1)
        x = jnp.matmul(attn, x)

        return x

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
    dropout: float = 0.0
    patch_merge_layer: int = None
    patch_merge_num_tokens: int = 8

    @nn.compact
    def __call__(self, x):

        layers = []
        patch_merge_layer_index = default(self.patch_merge_layer, self.depth // 2) - 1  # default to mid-way through transformer, as shown in paper
        patch_merger = PatchMerger(dim = self.dim, num_tokens_out = self.patch_merge_num_tokens)

        for _ in range(self.depth):
            layers.append([
                PreNorm(Attention(self.dim, heads = self.heads, dim_head = self.dim_head, dropout = self.dropout)),
                PreNorm(FeedForward(self.dim, self.mlp_dim, dropout = self.dropout))
            ])

        for index, (attn, ff) in enumerate(layers):
            x = attn(x) + x
            x = ff(x) + x

            if index == patch_merge_layer_index:
                x = patch_merger(x)

        return x

class ViT(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    patch_merge_layer: int = None
    patch_merge_num_tokens: int = 8
    dim_head: int = 64
    dropout: int = 0.0
    emb_dropout: int = 0.0

    @nn.compact
    def __call__(self, img, **kwargs):

        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])

        dropout = nn.Dropout(rate = self.emb_dropout, deterministic=False)

        transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout, self.patch_merge_layer, self.patch_merge_num_tokens)

        mlp_head = nn.Sequential([
            nn.LayerNorm(epsilon = 1e-5, use_bias = False),
            nn.Dense(features = self.num_classes)
        ])

        x = rearrange(img, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        x = nn.Dense(features = self.dim)(x)
        b, n, _ = x.shape

        x += pos_embedding[:, :n]
        x = dropout(x)

        x = transformer(x)
        x = reduce(x, 'b n d -> b d', 'mean')
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
        depth = 12,
        heads = 8,
        patch_merge_layer = 6,        # at which transformer layer to do patch merging
        patch_merge_num_tokens = 8,   # the output number of tokens from the patch merge
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

    key = jax.random.PRNGKey(5)

    features = jax.random.normal(key, (4, 256, 1024))

    merger = PatchMerger(
        dim = 1024,
        num_tokens_out = 8   # output number of tokens
    )

    merger_params = merger.init(init_rngs, features)
    merger_output = merger.apply(merger_params, features)
    print(merger_output.shape)
