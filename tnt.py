# general useful functionalities
from typing import Any, Callable, Sequence, Optional
from functools import partial

# jax/flax imports
import jax
from jax import numpy as jnp, vmap, pmap, jit, grad, random, lax
from jax.nn import initializers

import flax
from flax.core import freeze, unfreeze
from flax.training import train_state  # just for convenience
from flax import linen as nn  # explains itself
from flax import serialization
from flax import struct

# optimizer
import optax  # offers optimizers, don't use flax.optim, flax is built very modular and in a collaborative way, flax themselves state that people should now use optax for optimizers instead of flax.optim

# array manipulation
from einops import rearrange, repeat


@struct.dataclass
class Config:
    # general (model) options/parameters
    dtype: Any = jnp.float32
    num_classes: int = 10  # mnist dataset has 10 classes
    tnt_blocks: int = 6  # amount of tnt blocks which will be stacked
    attention_dropout: float = .0
    mlp_dropout: float = .0

    # inits
    kernel_init: Callable = initializers.xavier_uniform()
    bias_init: Callable = initializers.normal(stddev=1e-6)
    pos_emb_init: Callable = initializers.normal(stddev=0.02)
    emb_init: Callable = initializers.zeros
    learning_rate = 0.001

    # image parameters
    img_shape = (150, 150, 1)

    # patch parameters
    outer_size: int = 15
    outer_emb_dim: int = 192  # 80
    outer_heads: int = 3  # 8
    outer_head_dim: int = 64
    outer_expansion_rate = 4  # 3  # the MLP block will use a Dense Layer to project the feature vector to a new feature vector patch_expansion_rate times the feature vector's original size before using another Dense layer to squeeze the new feature vector to it's original size

    # pixel parameters (pixel=smaller patch, so not a real image pixel)
    inner_size: int = 5
    inner_emb_dim: int = 12  # 16
    inner_heads: int = 2
    inner_head_dim: int = 64
    inner_expansion_rate = 4  # 3  # see patch_expansion_rate comment above


class Embedding(nn.Module):
    config: Config
    inner: bool

    @nn.compact
    def __call__(self, images):
        config = self.config

        if self.inner:
            outer_tokens = rearrange(images, "b (h o1) (w o2) c -> (b h w) o1 o2 c",
                                     o1=config.outer_size,
                                     o2=config.outer_size
                                     )
            inner_tokens = rearrange(outer_tokens, "o (h i1) (w i2) c -> o (h w) (i1 i2 c)",
                                     i1=config.inner_size,
                                     i2=config.inner_size
                                     )
            output = nn.Dense(
                features=config.inner_emb_dim,
                kernel_init=config.kernel_init, bias_init=config.bias_init,
                dtype=config.dtype,
                name="inner_embeddings"
            )(inner_tokens)
        else:
            amount_outer_tokens = (config.img_shape[0] // config.outer_size) ** 2  # for one image
            output = self.param("outer_embeddings", config.emb_init, (amount_outer_tokens + 1, config.outer_emb_dim))
        return output


class PositionalEmbedding(nn.Module):
    config: Config
    inner: bool

    @nn.compact
    def __call__(self, embeddings):
        config = self.config

        shape = embeddings.shape[-2:]
        if self.inner:
            pos_emb = self.param("inner_positional_embeddings", config.pos_emb_init, shape)  # amount_inner_tokens, inner_emb_dim
        else:
            pos_emb = self.param("outer_positional_embeddings", config.pos_emb_init, shape)   # amount_outer_tokens + class token, outer_emb_dim
        return pos_emb


class MultiHeadSelfAttention(nn.Module):
    config: Config
    inner: bool

    @nn.compact
    def __call__(self, embeddings):
        config = self.config

        if self.inner:
            name = "inner_"
            features = (3, config.inner_heads, config.inner_head_dim)  # 3 because -> different parameters for q, k, v
        else:
            name = "outer_"
            features = (3, config.outer_heads, config.outer_head_dim)  # 3 because -> different parameters for q, k, v

        qkv = nn.DenseGeneral(features=features,
                              axis=-1,
                              kernel_init=config.kernel_init, use_bias=False,
                              name=name + "qkv"
                              )(embeddings)
        # split them up at axis -3 (see above comments) to divide the array to q k v and remove redundant dimensions
        qkv = jnp.split(qkv, axis=-3, indices_or_sections=3)  # tuple with 3 arrays, but contains redundant dimension
        q, k, v = map(lambda x: jnp.squeeze(x), qkv)
        attention_weights = jnp.einsum("...nid, ...njd -> ...nij", q, k) / jnp.sqrt(features[-1]).astype(config.dtype)
        v = jnp.einsum("...nij, ...njd -> ...nid", attention_weights, v)

        attention_output = nn.DenseGeneral(features=embeddings.shape[-1],
                                           axis=(-2, -1),
                                           kernel_init=config.kernel_init, use_bias=False,
                                           name=name + "attention_output"
                                           )(v)
        if config.attention_dropout:
            attention_output = nn.Dropout(config.attention_dropout)(attention_output)
        return attention_output


class MLP(nn.Module):
    config: Config
    inner: bool

    @nn.compact
    def __call__(self, embeddings):
        """
        Applies an MLP (non-linear transformations) to the embeddings with following layers
         Dense(expansion) -> GeLU activation function -> (Dropout) -> Dense(squeeze).
        :param embeddings:
        :return: embeddings
        """
        config = self.config

        emb_dim = embeddings.shape[-1]
        if self.inner:
            name = "inner_"
            expansion_rate = config.inner_expansion_rate
        else:
            name = "outer_"
            expansion_rate = config.outer_expansion_rate

        embeddings = nn.Dense(features=emb_dim * expansion_rate,
                              kernel_init=config.kernel_init, bias_init=config.bias_init,
                              dtype=config.dtype,
                              name=name + "expansion"
                              )(embeddings)
        embeddings = nn.gelu(embeddings)
        if config.mlp_dropout:
            embeddings = nn.Dropout(config.mlp_dropout)(embeddings)
        embeddings = nn.Dense(features=emb_dim,
                              kernel_init=config.kernel_init, bias_init=config.bias_init,
                              dtype=config.dtype,
                              name=name + "squeeze"
                              )(embeddings)
        return embeddings


class TNTBlock(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, outer_emb, inner_emb):
        config = self.config

        # inner block
        x = nn.LayerNorm(dtype=config.dtype)(inner_emb)
        x = MultiHeadSelfAttention(config=config)(x)
        x = x + inner_emb

        inner_output = nn.LayerNorm(dtype=config.dtype)(x)
        inner_output = MLP(config=config, inner=True)(inner_output)
        inner_output = inner_output + x

        # Addition: inner block + outer embeddings
        projected_inner_output = nn.LayerNorm(dtype=config.dtype)(inner_output)
        projected_inner_output = rearrange(projected_inner_output, "... n d -> ... (n d)")
        projected_inner_output = nn.Dense(features=outer_emb.shape[-1],  # ==config.outer_emb_dim
                                          kernel_init=config.kernel_init, bias_init=config.bias_init,
                                          dtype=config.dtype,
                                          name="inner_to_outer_embedding_projection"
                                          )(projected_inner_output)



class TransformerEncoder(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, outer_emb, inner_emb):
        config = self.config
        for _ in range(config.tnt_blocks):
            patch_emb, pixel_emb = TNTBlock(config=config)(patch_emb=outer_emb, pixel_emb=inner_emb)
        return inner_emb


class TransformerInTransformer(nn.Module):
    config: Config

    @nn.compact
    def __call__(self, images):
        config = self.config
        b, h, w, c = ((1, ) + images.shape)[-4:]

        # retrieve embeddings
        outer_emb = Embedding(config=config, inner=False)(images)
        inner_emb = Embedding(config=config, inner=True)(images)

        # create positional embeddings
        outer_pos_emb = PositionalEmbedding(config=config, inner=False)(outer_emb)  # amount_outer_tokens + class token, outer_emb_dim
        inner_pos_emb = PositionalEmbedding(config=config, inner=True)(inner_emb)  # amount_inner_tokens, inner_emb_dim

        # add positional embeddings
        outer_emb = repeat(outer_emb, "n d -> b n d", b=b) + rearrange(outer_pos_emb, "n d -> () n d")
        inner_emb = inner_emb + rearrange(inner_pos_emb, "n d -> () n d")

        attention = MultiHeadSelfAttention(config=config, inner=True)(inner_emb)
        return outer_emb, inner_emb


if __name__ == '__main__':
    cfg = Config()
    key1, key2 = random.split(random.PRNGKey(0))
    model = TransformerInTransformer(config=cfg)
    sample = jnp.ones(shape=(2, ) + cfg.img_shape)
    params = model.init(key1, sample)
    attention = model.apply(params, sample)
    print(attention.shape)
