import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange, repeat
from typing import Any

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class IdentityLayer(nn.Module):

  @nn.compact
  def __call__(self, x):
    return x

class ViT(nn.Module):
    image_size: int
    patch_size: int
    num_classes: int
    dim: int
    transformer: Any
    pool: str = 'cls'
    
    @nn.compact
    def __call__(self, x):

        image_height, image_width = pair(self.image_size)
        patch_height, patch_width = pair(self.patch_size)

        assert image_height % patch_height == 0 
        assert image_width % patch_width == 0

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        assert self.pool in {'cls', 'mean'}

        pos_embedding = self.param('pos_embedding', nn.initializers.zeros, [1, num_patches + 1, self.dim])
        cls_token = self.param('cls', nn.initializers.zeros, [1, 1, self.dim])

        x = rearrange(x, 'b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width)
        x = nn.Dense(features = self.dim)(x)

        b, n, _ = x.shape

        cls_tokens = repeat(cls_token, '() n d -> b n d', b=b)
        x = jnp.concatenate([cls_tokens, x], axis=1)
        x += pos_embedding[:, :(n + 1)]

        x = nn.Dropout(rate = self.emb_dropout)(x, deterministic = False)

        x = self.transformer(x)

        x = jnp.mean(x, axis = 1) if self.pool == 'mean' else x[:, 0]

        x = IdentityLayer()(x)

        x = nn.LayerNorm()(x)

        x = nn.Dense(features = self.num_classes)(x)

        return x
