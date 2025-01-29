import jax.numpy as jnp
from flax import linen as nn
from einops import rearrange, pack, unpack
from typing import Sequence

class AxialPositionalEmbedding(nn.Module):
  dim: int
  axial_shape: Sequence[int]
  axial_dims: Sequence[int] | None = None

  @nn.compact
  def __call__(self, x):
    max_seq_len = jnp.prod(jnp.array(self.axial_shape))
    summed = self.axial_dims is None

    if summed:
      axial_dims = ((self.dim,) * len(self.axial_shape))
    else:
      axial_dims = self.axial_dims

    batch, seq_len, _ = x.shape
    embs = []

    for ind, (shape, axial_dim) in enumerate(zip(self.axial_shape, axial_dims)):
      ax_shape = [1] * len(self.axial_shape)
      ax_shape[ind] = shape
      ax_shape = (1, *ax_shape, axial_dim)
      ax_emb = self.param(f'ax_emb_{ind}', nn.initializers.normal(stddev = 1.0), ax_shape)
      expand_shape = (batch, *self.axial_shape, axial_dim)
      emb = jnp.broadcast_to(ax_emb, expand_shape)
      emb = emb.reshape(batch, max_seq_len, axial_dim)
      embs.append(emb)

    if summed:
      pos_emb = jnp.sum(jnp.stack(embs), axis = 0)
    else:
      pos_emb = jnp.concatenate(embs, axis = -1)

    return pos_emb[:, :seq_len]

class AxialPositionalEmbeddingImage(nn.Module):
    dim: int
    axial_shape: Sequence[int]
    axial_dims: Sequence[int] | None = None

    @nn.compact
    def __call__(self, img):
      img = rearrange(img, 'b c h w -> b h w c')
      img, packed_shape = pack([img], 'b * c')

      pos_emb = AxialPositionalEmbedding(
        dim         = self.dim,
        axial_shape = self.axial_shape,
        axial_dims  = self.axial_dims
      )(img)

      pos_emb, = unpack(pos_emb, packed_shape, 'b * c')
      pos_emb = rearrange(pos_emb, 'b h w c -> b c h w')
      return pos_emb