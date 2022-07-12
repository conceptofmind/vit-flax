import jax
import jax.numpy as jnp
from jax.numpy import einsum

import flax.linen as nn

from typing import Callable

from einops import rearrange, reduce

def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else ((val,) * depth)

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
    dim: int
    fn: Callable

    @nn.compact
    def __call__(self, x, **kwargs):
        x = LayerNorm(self.dim)(x)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    dim: int
    mult: int
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features = self.dim * self.mult, kernel_size = (1, 1), strides = (1, 1), use_bias = False)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        x = nn.Conv(features = self.dim, kernel_size = (1, 1), strides = (1, 1), use_bias = False)(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)
        return x

class Attention(nn.Module): 
    dim: int 
    heads: int = 8 
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):
        dim_head = self.dim // self.heads
        inner_dim = dim_head * self.heads
        heads = self.heads
        scale = dim_head ** -0.5

        b, h, w, c = x.shape
        heads = heads

        qkv = nn.Conv(features = inner_dim * 3, kernel_size = (1, 1), strides = (1, 1), use_bias = False)(x)
        qkv = jnp.split(qkv, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> b h (x y) d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        attn = nn.softmax(dots, axis = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d)', x = h, y = w)
        
        x = nn.Conv(features = self.dim, kernel_size = (1, 1), strides = (1, 1), use_bias = False)(out)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)

        return x

class Aggregate(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features = self.dim, kernel_size = (3, 3), strides = (1, 1), padding='SAME')(x)
        x = LayerNorm(self.dim)(x)
        x = nn.max_pool(x, window_shape = (3, 3), strides = (2, 2), padding='SAME')
        return x

class Transformer(nn.Module):
    dim: int 
    seq_len: int 
    depth: int 
    heads: int 
    mlp_mult: int 
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):

        layers = []

        pos_emb = self.param('pos_emb', nn.initializers.zeros, [self.seq_len])

        for _ in range(self.depth):
            layers.append([
                PreNorm(self.dim, Attention(self.dim, heads = self.heads, dropout = self.dropout)),
                PreNorm(self.dim, FeedForward(self.dim, self.mlp_mult, dropout = self.dropout))
            ])

        _, h, w, c = x.shape

        pos_emb = pos_emb[:(h * w)]
        pos_emb = rearrange(pos_emb, '(h w) -> () h w ()', h = h, w = w)
        x = x + pos_emb

        for attn, ff in layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class NesT(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    heads: int
    num_hierarchies: int
    block_repeats: int
    mlp_mult: int = 4
    dropout: float = 0.

    @nn.compact
    def __call__(self, img, **kwargs):

        assert (self.image_size % self.patch_size) == 0, 'Image dimensions must be divisible by the patch size.'
        fmap_size = self.image_size // self.patch_size
        blocks = 2 ** (self.num_hierarchies - 1)

        seq_len = (fmap_size // blocks) ** 2   # sequence length is held constant across heirarchy
        hierarchies = list(reversed(range(self.num_hierarchies)))
        mults = [2 ** i for i in reversed(hierarchies)]

        layer_heads = list(map(lambda t: t * self.heads, mults))
        layer_dims = list(map(lambda t: t * self.dim, mults))
        last_dim = layer_dims[-1]

        layer_dims = [*layer_dims, layer_dims[-1]]
        dim_pairs = zip(layer_dims[:-1], layer_dims[1:])

        block_repeats = cast_tuple(self.block_repeats, self.num_hierarchies)

        nest_layers = []

        for level, heads, (dim_in, dim_out), block_repeat in zip(hierarchies, layer_heads, dim_pairs, block_repeats):
            is_last = level == 0
            depth = block_repeat

            nest_layers.append([
                Transformer(dim_in, seq_len, depth, heads, self.mlp_mult, self.dropout),
                Aggregate(dim_out) if not is_last else IdentityLayer()
            ])

        x = rearrange(img, 'b (h p1) (w p2) c -> b h w (p1 p2 c) ', p1 = self.patch_size, p2 = self.patch_size)
        x = nn.Conv(features = layer_dims[0], kernel_size = (1, 1), strides = (1, 1))(x)

        num_hierarchies = len(nest_layers)

        for level, (transformer, aggregate) in zip(reversed(range(num_hierarchies)), nest_layers):
            block_size = 2 ** level
            x = rearrange(x, 'b (b1 h) (b2 w) c -> (b b1 b2) h w c', b1 = block_size, b2 = block_size)
            x = transformer(x)
            x = rearrange(x, '(b b1 b2) h w c -> b (b1 h) (b2 w) c', b1 = block_size, b2 = block_size)
            x = aggregate(x)

        x = LayerNorm(last_dim)(x)
        x = reduce(x, 'b h w c -> b c', 'mean')
        x = nn.Dense(features = self.num_classes)(x)

        return x

if __name__ == '__main__':

    import numpy

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 224, 224, 3))

    v = NesT(
        image_size = 224,
        patch_size = 4,
        dim = 96,
        heads = 3,
        num_hierarchies = 3,        # number of hierarchies
        block_repeats = (2, 2, 8),  # the number of transformer blocks at each heirarchy, starting from the bottom
        num_classes = 1000
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