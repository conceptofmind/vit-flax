import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from typing import Callable

from einops import rearrange

from math import ceil
from einops import rearrange, repeat

from math import sqrt

import tensorflow as tf

def cast_tuple(val, num):
    return val if isinstance(val, tuple) else (val,) * num

def conv_output_size(image_size, kernel_size, stride, padding = 0):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

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
    dropout: float = 0.

    @nn.compact
    def __call__(self, x):

        inner_dim = self.dim_head * self.heads
        project_out = not (self.heads == 1 and self.dim_head == self.dim)

        scale = self.dim_head ** -0.5

        to_qkv = nn.Dense(features=inner_dim * 3, use_bias=False)

        qkv = to_qkv(x)
        qkv = jnp.split(qkv, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        attn = nn.softmax(dots, axis = -1)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        if project_out:
            x = nn.Dense(features=self.dim)(x)
            x = nn.Dropout(rate=self.dropout)(x, deterministic=False)
        else:
            x = []

        return x

class Transformer(nn.Module):
    dim: int 
    depth: int 
    heads: int 
    dim_head: int 
    mlp_dim: int 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        layers = []

        for _ in range(self.depth):
            layers.append([
                PreNorm(Attention(self.dim, heads = self.heads, dim_head = self.dim_head, dropout = self.dropout)),
                PreNorm(FeedForward(self.dim, self.mlp_dim, dropout = self.dropout))
            ])

        for attn, mlp in layers:
            x = attn(x) + x
            x = mlp(x) + x

        return x


# depthwise convolution, for pooling
class DepthWiseConv2d(nn.Module): 
    dim_in: int 
    dim_out: int 
    kernel_size: int 
    stride: int 
    bias: bool = True

    @nn.compact
    def __call__(self, x):

        x = nn.Conv(
            features = self.dim_out, 
            kernel_size = (self.kernel_size, self.kernel_size), 
            strides = (self.stride, self.stride), 
            padding = 'SAME', 
            feature_group_count = self.dim_in, 
            use_bias = self.bias)(x)

        x = nn.Conv(features = self.dim_out, kernel_size = (1, 1), strides = (1, 1), use_bias = self.bias)(x)

        return x

# pooling layer
class Pool(nn.Module):
    dim: int

    @nn.compact        
    def __call__(self, x):

        downsample = DepthWiseConv2d(self.dim, self.dim * 2, kernel_size = 3, stride = 2)
        cls_ff = nn.Dense(features = self.dim * 2)

        cls_token, tokens = x[:, :1], x[:, 1:]
        cls_token = cls_ff(cls_token)

        tokens = rearrange(tokens, 'b (h w) c -> b h w c', h=int(sqrt(tokens.shape[1])))
        tokens = downsample(tokens)
        tokens = rearrange(tokens, 'b h w c -> b (h w) c')

        x = jnp.concatenate([cls_token, tokens], axis = 1)

        return x

# class Unfold(nn.Module):
#     kernel_size: int
#     stride: int

#     @nn.compact        
#     def __call__(self, x):
#         kernel_size = [1, self.kernel_size, self.kernel_size, 1]
#         stride = [1, self.stride, self.stride, 1]
#         rates = [1, 1, 1, 1]

#         x = jax.lax.conv_general_dilated_patches(
#             x, 
#             filter_shape = kernel_size, 
#             window_strides = stride, padding='VALID')(x)
#         x = rearrange(x, 'b h w c -> b (h w) c')
#         return x

class PiT(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    depth: int
    heads: int
    mlp_dim: int
    dim_head: int = 64
    dropout: float = 0.0
    emb_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, **kwargs):

        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        assert isinstance(self.depth,
                          tuple), 'depth must be a tuple of integers, specifying the number of blocks before each downsizing'

        heads = cast_tuple(self.heads, len(self.depth))

        output_size = conv_output_size(self.image_size, self.patch_size, self.patch_size // 2)
        num_patches = output_size ** 2

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])

        dropout = nn.Dropout(rate = self.emb_dropout, deterministic = False)

        transformer_layers = []

        for ind, (layer_depth, layer_heads) in enumerate(zip(self.depth, heads)):
            not_last = ind < (len(self.depth) < 1)

            transformer_layers.append(Transformer(self.dim, layer_depth, layer_heads, self.dim_head, self.mlp_dim, self.dropout))

            if not_last:
                transformer_layers.append(Pool(self.dim))
                self.dim *= 2

        mlp_head = nn.Sequential([
            nn.LayerNorm(epsilon = 1e-5, use_bias = False),
            nn.Dense(features = self.num_classes)
        ])

        x = tf.image.extract_patches(
            x, 
            sizes=[1, self.patch_size, self.patch_size, 1], 
            strides=[1, self.patch_size//2, self.patch_size//2, 1], 
            rates=[1,1,1,1], 
            padding='VALID')

        x = rearrange(x, 'b h w c -> b (h w) c')

        x = nn.Dense(features = self.dim)(x)

        b, n, d = x.shape

        cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x += pos_embedding[:, :(n + 1)]
        x = dropout(x)

        transformer_layers = nn.Sequential(transformer_layers)
        x = transformer_layers(x)
        x = mlp_head(x[:, 0])

        return x

if __name__ == "__main__":

    import numpy

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 224, 224, 3))

    v = PiT(
        image_size = 224,
        patch_size = 14,
        dim = 256,
        num_classes = 1000,
        depth = (3, 3, 3),     # list of depths, indicating the number of rounds of each stage before a downsample
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
        jax.tree_leaves(jax.tree_map(lambda x: numpy.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")
