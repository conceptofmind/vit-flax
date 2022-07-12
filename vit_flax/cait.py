from typing import Callable

import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from einops import rearrange, repeat

from random import randrange

def exists(val):
    return val is not None

def dropout_layers(layers, dropout):
    if dropout == 0:
        return layers

    num_layers = len(layers)
    key = jax.random.PRNGKey(0)
    to_drop = jax.random.uniform(key, minval=0.0, maxval=1.0, shape=[num_layers]) < dropout

    # make sure at least one layer makes it
    if all(to_drop):
        rand_index = randrange(num_layers)
        to_drop[rand_index] = False

    layers = [layer for (layer, drop) in zip(layers, to_drop) if not drop]
    return layers

class LayerScale(nn.Module): 
    dim: int 
    fn: Callable 
    depth: int

    @nn.compact
    def __call__(self, x, **kwargs):

        if self.depth <= 18: # epsilon detailed in section 2 of paper
            init_eps = 0.1
        elif self.depth > 18 and self.depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = jnp.full([1, 1, self.dim], init_eps)

        return self.fn(x, **kwargs) * scale

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
    def __call__(self, x, context = None):

        inner_dim = self.dim_head * self.heads

        heads = self.heads
        scale = self.dim_head ** -0.5

        mix_heads_pre_attn = self.param('mix_heads_pre_attn', nn.initializers.zeros, [heads, heads])
        mix_heads_post_attn = self.param('mix_heads_post_attn', nn.initializers.zeros, [heads, heads])

        if not exists(context):
            context = x
        else:
            context = jnp.concatenate([x, context], axis = 1)

        q = nn.Dense(features = inner_dim, use_bias = False)(x)
        kv = nn.Dense(features = inner_dim * 2, use_bias = False)(context)

        k, v = jnp.split(kv, 2, axis = -1)
        qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        dots = einsum('b h i j, h g -> b g i j', dots, mix_heads_pre_attn)  # talking heads, pre-softmax
        attn = nn.softmax(dots, axis = -1)
        attn = einsum('b h i j, h g -> b g i j', attn, mix_heads_post_attn)  # talking heads, post-softmax

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        out = nn.Dense(features = self.dim)(x)
        to_out = nn.Dropout(rate = self.dropout)(out, deterministic = False)

        return to_out

class Transformer(nn.Module):
    dim: int 
    depth: int 
    heads: int 
    dim_head: int 
    mlp_dim: int 
    dropout: float = 0.0 
    layer_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, context=None):

        layers = []
        layer_dropout = self.layer_dropout

        for ind in range(self.depth):
            layers.append([
                LayerScale(self.dim, PreNorm(Attention(self.dim, self.heads, self.dim_head, dropout = self.dropout)), depth = ind + 1),
                LayerScale(self.dim, PreNorm(FeedForward(self.dim, self.mlp_dim, dropout = self.dropout)), depth = ind + 1)
            ])

        layers = dropout_layers(layers, dropout = layer_dropout)

        for attn, ff in layers:
            x = attn(x, context=context) + x
            x = ff(x) + x

        return x

class CaiT(nn.Module):
    image_size: int 
    patch_size: int 
    num_classes: int 
    dim: int 
    depth: int 
    cls_depth: int 
    heads: int 
    mlp_dim: int
    dim_head: int = 64 
    dropout: float = 0.0 
    emb_dropout: float = 0.0 
    layer_dropout: float = 0.0

    @nn.compact    
    def __call__(self, img):

        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (self.image_size // self.patch_size) ** 2

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])

        x = rearrange(img, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        x = nn.Dense(features = self.dim)(x)
        b, n, d = x.shape

        x += pos_embedding[:, :n]
        x = nn.Dropout(rate = self.emb_dropout)(x, deterministic = False)

        x = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout, self.layer_dropout)(x)

        cls_tokens = repeat(cls_token, '() n d -> b n d', b = b)
        x = Transformer(self.dim, self.cls_depth, self.heads, self.dim_head, self.mlp_dim, self.dropout, self.layer_dropout)(cls_tokens, context = x)

        mlp_head = nn.Sequential([
            nn.LayerNorm(epsilon = 1e-5, use_bias = False),
            nn.Dense(features = self.num_classes)
        ])
        x = mlp_head(x[:, 0])

        return x

if __name__ == "__main__":

    import numpy as np

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 256, 256, 3))

    v = CaiT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 12,             # depth of transformer for patch to patch attention only
        cls_depth = 2,          # depth of cross attention of CLS tokens to patch
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1,
        layer_dropout = 0.05    # randomly dropout 5% of the layers
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
