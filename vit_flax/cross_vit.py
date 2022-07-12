import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

from typing import Any, Callable

from einops import rearrange, repeat

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

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
    def __call__(self, x, context=None, kv_include_self=False, training=True):
        inner_dim = self.dim_head * self.heads

        heads = self.heads
        scale = self.dim_head ** -0.5

        context = default(context, x)

        if kv_include_self:
            context = jnp.concatenate([x, context], axis = 1) # cross attention requires CLS token includes itself as key / value

        q = nn.Dense(features = inner_dim, use_bias = False)(x)
        kv = nn.Dense(features = inner_dim * 2, use_bias = False)(context)
        k, v = jnp.split(kv, 2, axis = -1)
        qkv = (q, k, v)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale

        attn = nn.softmax(dots, axis = -1)

        x = einsum('b h i j, b h j d -> b h i d', attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')

        x = nn.Dense(features = self.dim)(x)
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)

        return x

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

        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)

        return x

# projecting CLS tokens, in the case that small and large patch tokens have different dimensions
class ProjectInOut(nn.Module):
 
    dim_in: int 
    dim_out: int
    fn: Callable

    @nn.compact
    def __call__(self, x, *args, **kwargs):
        fn = self.fn

        need_projection = self.dim_in != self.dim_out

        if need_projection:
            project_in = nn.Dense(features = self.dim_out)
            project_out = nn.Dense(features = self.dim_in)

        # args check
        if need_projection:
            x = project_in(x)

        x = fn(x, *args, **kwargs)

        if need_projection:
            x = project_out(x)

        return x

# cross attention transformer
class CrossTransformer(nn.Module):
    sm_dim: int 
    lg_dim: int 
    depth: int 
    heads: int 
    dim_head: int 
    dropout: float

    @nn.compact
    def __call__(self, inputs):

        layers = []

        for _ in range(self.depth):
            layers.append([
                ProjectInOut(self.sm_dim, self.lg_dim, PreNorm(Attention(self.lg_dim, heads = self.heads, dim_head = self.dim_head, dropout = self.dropout))),
                ProjectInOut(self.lg_dim, self.sm_dim, PreNorm(Attention(self.sm_dim, heads = self.heads, dim_head = self.dim_head, dropout = self.dropout)))
                        ])

        sm_tokens, lg_tokens = inputs
        (sm_cls, sm_patch_tokens), (lg_cls, lg_patch_tokens) = map(lambda t: (t[:, :1], t[:, 1:]), (sm_tokens, lg_tokens))

        for sm_attend_lg, lg_attend_sm in layers:
            sm_cls = sm_attend_lg(sm_cls, context = lg_patch_tokens, kv_include_self = True) + sm_cls
            lg_cls = lg_attend_sm(lg_cls, context = sm_patch_tokens, kv_include_self = True) + lg_cls

        sm_tokens = jnp.concatenate([sm_cls, sm_patch_tokens], axis = 1)
        lg_tokens = jnp.concatenate([lg_cls, lg_patch_tokens], axis = 1)

        return sm_tokens, lg_tokens

# multi-scale encoder
class MultiScaleEncoder(nn.Module):
    depth: int
    sm_dim: int
    lg_dim: int
    sm_enc_params: Any
    lg_enc_params: Any
    cross_attn_heads: int
    cross_attn_depth: int
    cross_attn_dim_head: int = 64
    dropout: float = 0.

    @nn.compact
    def __call__(self, inputs):

        layers = []

        for _ in range(self.depth):
            layers.append([Transformer(dim = self.sm_dim, dropout = self.dropout, **self.sm_enc_params),
                                Transformer(dim = self.lg_dim, dropout = self.dropout, **self.lg_enc_params),
                                CrossTransformer(sm_dim = self.sm_dim, lg_dim = self.lg_dim,
                                                 depth = self.cross_attn_depth, heads = self.cross_attn_heads, dim_head = self.cross_attn_dim_head, dropout = self.dropout)
                                ])

        sm_tokens, lg_tokens = inputs
        for sm_enc, lg_enc, cross_attend in layers:
            sm_tokens, lg_tokens = sm_enc(sm_tokens), lg_enc(lg_tokens)
            sm_tokens, lg_tokens = cross_attend([sm_tokens, lg_tokens])

        return sm_tokens, lg_tokens

# patch-based image to token embedder
class ImageEmbedder(nn.Module):
    dim: int
    image_size: int
    patch_size: int
    dropout:float = 0.

    @nn.compact
    def __call__(self, x):

        assert self.image_size % self.patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (self.image_size // self.patch_size) ** 2

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])

        x = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size)
        x = nn.Dense(features = self.dim)(x)

        b, n, d = x.shape

        cls_tokens = repeat(cls_token, '() n d -> b n d', b = b)
        x = jnp.concatenate([cls_tokens, x], axis = 1)
        x += pos_embedding[:, :(n + 1)]
        x = nn.Dropout(rate = self.dropout)(x, deterministic = False)

        return x

# cross ViT class
class CrossViT(nn.Module):
    image_size: int
    num_classes: int
    sm_dim: int
    lg_dim: int
    sm_patch_size: int = 12
    sm_enc_depth: int = 1
    sm_enc_heads: int = 8
    sm_enc_mlp_dim: int = 2048
    sm_enc_dim_head: int = 64
    lg_patch_size: int = 16
    lg_enc_depth: int = 4
    lg_enc_heads: int = 8
    lg_enc_mlp_dim: int = 2048
    lg_enc_dim_head: int = 64
    cross_attn_depth: int = 2
    cross_attn_heads: int = 8
    cross_attn_dim_head: int = 64
    depth: int = 3
    dropout: float = 0.1
    emb_dropout: float = 0.1

    @nn.compact
    def __call__(self, img):

        multi_scale_encoder = MultiScaleEncoder(
            depth = self.depth,
            sm_dim = self.sm_dim,
            lg_dim = self.lg_dim,
            cross_attn_heads = self.cross_attn_heads,
            cross_attn_dim_head = self.cross_attn_dim_head,
            cross_attn_depth = self.cross_attn_depth,
            sm_enc_params = dict(
                depth = self.sm_enc_depth,
                heads = self.sm_enc_heads,
                mlp_dim = self.sm_enc_mlp_dim,
                dim_head = self.sm_enc_dim_head
            ),
            lg_enc_params = dict(
                depth = self.lg_enc_depth,
                heads = self.lg_enc_heads,
                mlp_dim = self.lg_enc_mlp_dim,
                dim_head = self.lg_enc_dim_head
            ),
            dropout = self.dropout
        )

        sm_tokens = ImageEmbedder(dim = self.sm_dim, image_size = self.image_size, patch_size = self.sm_patch_size, dropout = self.emb_dropout)(img)
        lg_tokens = ImageEmbedder(dim = self.lg_dim, image_size = self.image_size, patch_size = self.lg_patch_size, dropout = self.emb_dropout)(img)

        sm_tokens, lg_tokens = multi_scale_encoder([sm_tokens, lg_tokens])

        sm_cls, lg_cls = map(lambda t: t[:, 0], (sm_tokens, lg_tokens))

        sm_logits = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(sm_cls)
        sm_logits = nn.Dense(features = self.num_classes)(sm_logits)

        lg_logits = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(lg_cls)
        lg_logits = nn.Dense(features = self.num_classes)(lg_logits)

        x = sm_logits + lg_logits

        return x

if __name__ == '__main__':

    import numpy

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 256, 256, 3))

    v = CrossViT(
            image_size = 256,
            num_classes = 1000,
            depth = 4,               # number of multi-scale encoding blocks
            sm_dim = 192,            # high res dimension
            sm_patch_size = 16,      # high res patch size (should be smaller than lg_patch_size)
            sm_enc_depth = 2,        # high res depth
            sm_enc_heads = 8,        # high res heads
            sm_enc_mlp_dim = 2048,   # high res feedforward dimension
            lg_dim = 384,            # low res dimension
            lg_patch_size = 64,      # low res patch size
            lg_enc_depth = 3,        # low res depth
            lg_enc_heads = 8,        # low res heads
            lg_enc_mlp_dim = 2048,   # low res feedforward dimensions
            cross_attn_depth = 2,    # cross attention rounds
            cross_attn_heads = 8,    # cross attention heads
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