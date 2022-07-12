import tensorflow as tf

import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

import numpy as np

from typing import Callable, Any

from einops import rearrange, repeat

#from vit import Transformer

from einops import rearrange, repeat
import math

def exists(val):
    return val is not None

def conv_output_size(image_size, kernel_size, stride, padding):
    return int(((image_size - kernel_size + (2 * padding)) / stride) + 1)

class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
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
                PreNorm(Attention(self.dim, self.heads, self.dim_head, self.dropout)),
                PreNorm(FeedForward(self.dim, self.mlp_dim, self.dropout))
            ])

        for attn, ff in layers:
            x = attn(x) + x
            x = ff(x) + x

        return x

class RearrangeUnfoldTransformer(nn.Module):
    is_first: bool 
    is_last: bool 
    kernel_size: int 
    stride: int
    dim: int 
    heads: int 
    depth: int 
    dim_head: int 
    mlp_dim: int 
    dropout: float
        
    @nn.compact
    def __call__(self, x):

        is_first = self.is_first
        is_last = self.is_last
        kernel_size = [1, self.kernel_size, self.kernel_size, 1]
        stride = [1, self.stride, self.stride, 1]
        rates = [1, 1, 1, 1]

        # transformer
        dim = self.dim
        heads = self.heads
        depth = self.depth
        dim_head = self.dim_head
        mlp_dim = self.mlp_dim
        dropout = self.dropout

        if not is_last:
            transformer_layer = Transformer(dim=dim, heads=heads, depth=depth, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout)

        if not is_first:
            x = rearrange(x, 'b (h w) c -> b h w c', h=int(math.sqrt(x.shape[1])))
        x = tf.image.extract_patches(x, sizes=kernel_size, strides=stride, rates=rates, padding='SAME')
        x = jnp.array(x)
        x = rearrange(x, 'b h w c -> b (h w) c')
        if not is_last:
            x = transformer_layer(x)

        return x

class T2TViT(nn.Module):
    image_size: int 
    num_classes: int 
    dim: int
    depth: int = None 
    heads: int = None 
    mlp_dim: int = None 
    pool: str = 'cls' 
    channels: int = 3 
    dim_head: int = 64 
    dropout: float = 0.0 
    emb_dropout: float = 0.0
    transformer: Any = None 
    t2t_layers: tuple = ((7, 4), (3, 2), (3, 2))
        
    @nn.compact
    def __call__(self, img, **kwargs):

        assert self.pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        layers = []
        layer_dim = self.channels
        output_image_size = self.image_size

        for i, (kernel_size, stride) in enumerate(self.t2t_layers):
            layer_dim *= kernel_size ** 2
            is_first = i == 0
            is_last = i == (len(self.t2t_layers) - 1)
            output_image_size = conv_output_size(output_image_size, kernel_size, stride, stride // 2)

            layers.append(RearrangeUnfoldTransformer(is_first, is_last, kernel_size, stride,
                                                  dim=layer_dim, heads=1, depth=1, dim_head=layer_dim, mlp_dim=layer_dim, dropout=self.dropout)
            )

        layers.append(nn.Dense(self.dim))
        patch_embedding = nn.Sequential(layers)

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, output_image_size ** 2 + 1, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])


        dropout = nn.Dropout(rate = self.emb_dropout, deterministic=False)

        if not exists(self.transformer):
            assert all([exists(self.depth), exists(self.heads), exists(self.mlp_dim)]), 'depth, heads, and mlp_dim must be supplied'
            transformer = Transformer(self.dim, self.depth, self.heads, self.dim_head, self.mlp_dim, self.dropout)
        else:
            transformer = self.transformer

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
            x = jnp.mean(x, axis=1)
        else:
            x = x[:, 0]

        x = mlp_head(x)

        return x

if __name__ == '__main__':

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 224, 224, 3))

    v = T2TViT(
        dim = 512,
        image_size = 224,
        depth = 5,
        heads = 8,
        mlp_dim = 512,
        num_classes = 1000,
        t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
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