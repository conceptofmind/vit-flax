import flax.linen as nn

import jax
import jax.numpy as jnp
from jax.numpy import einsum

import numpy as np

from typing import Callable

from einops import rearrange, repeat, reduce

def cast_tuple(val, length = 1):
    return val if isinstance(val, tuple) else ((val,) * length)

# cross embed layer
class CrossEmbedLayer(nn.Module):
    dim: int 
    kernel_sizes: int 
    stride: int = 2

    @nn.compact
    def __call__(self, x):

        kernel_sizes = sorted(self.kernel_sizes)
        num_scales = len(kernel_sizes)

        # calculate the dimension at each scale
        dim_scales = [int(self.dim / (2 ** i)) for i in range(1, num_scales)]
        dim_scales = [*dim_scales, self.dim - sum(dim_scales)]

        convs = []
        for kernel, dim_scale in zip(kernel_sizes, dim_scales):
            convs.append(nn.Conv(dim_scale, 
                            kernel_size=(kernel, kernel), 
                            strides=(self.stride, self.stride), 
                            padding='SAME'))

        fmaps = tuple(map(lambda conv: conv(x), convs))
        x = jnp.concatenate(fmaps, axis=-1)
        return x

# dynamic positional bias
class DynamicPositionBias(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.dim)(x)
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        x = nn.relu(x)
        x = nn.Dense(self.dim)(x)
        x = nn.LayerNorm(epsilon = 1e-5, use_bias = False)(x)
        x = nn.relu(x)
        x = nn.Dense(1)(x)
        x = rearrange(x, '... () -> ...')
        return x

# transformer classes

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

class MLP(nn.Module):
    dim: int
    mult: int = 4
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):
        x = LayerNorm(self.dim)(x)
        x = nn.Conv(self.dim*self.mult, kernel_size=(1,1), strides=(1,1))(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout, deterministic=False)(x)
        x = nn.Conv(self.dim, kernel_size=(1,1), strides=(1,1))(x)
        return x

class Attention(nn.Module): 
    dim: int
    attn_type: str
    window_size: int 
    dim_head: int = 32 
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        assert self.attn_type in {'short', 'long'}, 'attention type must be one of local or distant'
        heads = self.dim // self.dim_head

        scale = self.dim_head ** -0.5
        inner_dim = self.dim_head * heads

        attn_type = self.attn_type
        window_size = self.window_size

        norm = LayerNorm(self.dim)
        to_qkv = nn.Conv(inner_dim * 3, kernel_size=(1,1), strides=(1,1), use_bias=False)
        to_out = nn.Conv(self.dim, kernel_size=(1,1), strides=(1,1))

        # positions
        dpb = DynamicPositionBias(self.dim // 4)

        # calculate and store indices for retrieving bias
        pos = jnp.arange(window_size)
        grid = jnp.stack(jnp.meshgrid(pos, pos, indexing='ij'))
        grid = rearrange(grid, 'c i j -> (i j) c')
        rel_pos = grid[:, None] - grid[None, :]
        rel_pos += window_size - 1
        rel_pos_indices = jnp.sum(rel_pos * jnp.array([2 * window_size - 1, 1]), axis=-1)

        _, height, width, _ = x.shape

        wsz = window_size

        # prenorm
        x = norm(x)
        
        # rearrange for short or long distance attention

        if attn_type == 'short':
            x = rearrange(x, 'b (h s1) (w s2) d -> (b h w) s1 s2 d', s1=wsz, s2=wsz)
        elif attn_type == 'long':
            x = rearrange(x, 'b (l1 h) (l2 w) d -> (b h w) l1 l2 d', l1=wsz, l2=wsz)

        # queries / keys / values
        qkv = to_qkv(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b x y (h d) -> b h (x y) d', h=heads), (q, k, v))
        q = q * scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # add dynamic positional bias
        pos = jnp.arange(-wsz, wsz + 1)
        rel_pos = jnp.stack(jnp.meshgrid(pos, pos, indexing='ij'))
        rel_pos = rearrange(rel_pos, 'c i j -> (i j) c')
        biases = dpb(rel_pos)
        rel_pos_bias = biases[rel_pos_indices]

        sim = sim + rel_pos_bias

        # attend
        attn = nn.softmax(sim, axis = -1)

        # merge heads
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b x y (h d) ', x=wsz, y=wsz)
        out = to_out(out)
        # rearrange back for long or short distance attention
        if self.attn_type == 'short':
            out = rearrange(out, '(b h w) s1 s2 d -> b (h s1) (w s2) d', h=height // wsz, w=width // wsz)
        elif self.attn_type == 'long':
            out = rearrange(out, '(b h w) l1 l2 d -> b (l1 h) (l2 w) d', h=height // wsz, w=width // wsz)

        return out

class Transformer(nn.Module): 
    dim: int
    local_window_size: int 
    global_window_size: int
    depth: int = 4 
    dim_head: int = 32 
    attn_dropout: float = 0.0 
    ff_dropout: float = 0.0

    @nn.compact
    def __call__(self, x):

        layers = []

        for _ in range(self.depth):
            layers.append([
                Attention(self.dim, attn_type='short', window_size=self.local_window_size, dim_head=self.dim_head, dropout=self.attn_dropout),
                MLP(self.dim, dropout=self.ff_dropout),
                Attention(self.dim, attn_type='long', window_size=self.global_window_size, dim_head=self.dim_head, dropout=self.attn_dropout),
                MLP(self.dim, dropout=self.ff_dropout)
            ])

        for short_attn, short_ff, long_attn, long_ff in layers:
            x = short_attn(x) + x
            x = short_ff(x) + x
            x = long_attn(x) + x
            x = long_ff(x) + x

        return x

class CrossFormer(nn.Module):
    dim: tuple = (64, 128, 256, 512)
    depth: tuple = (2, 2, 8, 2)
    global_window_size: tuple = (8, 4, 2, 1)
    local_window_size: int = 7
    cross_embed_kernel_sizes: tuple = ((4, 8, 16, 32), (2, 4), (2, 4), (2, 4))
    cross_embed_strides: tuple = (4, 2, 2, 2)
    num_classes: int = 1000
    attn_dropout: float = 0.0
    ff_dropout: float = 0.0

    @nn.compact
    def __call__(self, x, **kwargs):

        dim = cast_tuple(self.dim, 4)
        depth = cast_tuple(self.depth, 4)
        global_window_size = cast_tuple(self.global_window_size, 4)
        local_window_size = cast_tuple(self.local_window_size, 4)
        cross_embed_kernel_sizes = cast_tuple(self.cross_embed_kernel_sizes, 4)
        cross_embed_strides = cast_tuple(self.cross_embed_strides, 4)

        assert len(dim) == 4
        assert len(depth) == 4
        assert len(global_window_size) == 4
        assert len(local_window_size) == 4
        assert len(cross_embed_kernel_sizes) == 4
        assert len(cross_embed_strides) == 4

        # layers
        crossformer_layers = []

        for dim_out, layers, global_wsz, local_wsz, cel_kernel_sizes, cel_stride in zip(dim, depth,
                                                                                        global_window_size, local_window_size,
                                                                                        cross_embed_kernel_sizes, cross_embed_strides):
            crossformer_layers.append([
                CrossEmbedLayer(dim_out, cel_kernel_sizes, stride=cel_stride),
                Transformer(dim_out, local_window_size=local_wsz, global_window_size=global_wsz, depth=layers,
                            attn_dropout=self.attn_dropout, ff_dropout=self.ff_dropout)
            ])

        # final logits
        to_logits = nn.Sequential([
            nn.Dense(self.num_classes)
        ])

        for cel, transformer in crossformer_layers:
            x = cel(x)
            x = transformer(x)

        x = reduce(x, 'b h w c -> b c', 'mean')

        x = to_logits(x)

        return x


if __name__ == '__main__':

    import numpy as np

    key = jax.random.PRNGKey(0)

    img = jax.random.normal(key, (1, 224, 224, 3))

    v = CrossFormer(
        num_classes = 1000,                # number of output classes
        dim = (64, 128, 256, 512),         # dimension at each stage
        depth = (2, 2, 8, 2),              # depth of transformer at each stage
        global_window_size = (8, 4, 2, 1), # global window sizes at each stage
        local_window_size = 7,             # local window size (can be customized for each stage, but in paper, held constant at 7 for all stages)
    )

    init_rngs = {'params': jax.random.PRNGKey(1), 
                'dropout': jax.random.PRNGKey(2)}

    params = v.init(init_rngs, img)
    output = v.apply(params, img, rngs=init_rngs)
    print(output.shape)

    n_params_flax = sum(
        jax.tree_leaves(jax.tree_map(lambda x: np.prod(x.shape), params))
    )
    print(f"Number of parameters in Flax model: {n_params_flax}")