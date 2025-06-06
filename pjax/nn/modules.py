from abc import ABC, abstractmethod
from copy import deepcopy
from functools import partial
from typing import Callable, Sequence

import jax
import numpy as np
from jax import numpy as jnp

import pjax

from ..core.frozen_dict import FrozenDict, freeze


class Parameter:
    """Learnable parameter for neural network modules.

    Parameters:
        shape: parameter tensor shape.
        dtype: parameter data type.
        init_fn: initialization function ``(key, shape, dtype) -> array``.
    """

    def __init__(self, shape: Sequence[int], dtype: jnp.dtype = jnp.float32, init_fn: Callable | None = None):
        self.shape = shape
        self.dtype = dtype
        self.init_fn = init_fn

    def init(self, key) -> jnp.ndarray:
        """Initialize parameter with random key."""
        if self.init_fn is not None:
            return self.init_fn(key, self.shape, self.dtype)
        return jnp.zeros(self.shape, self.dtype)


class Weight(Parameter):
    """Learnable weight parameter with He normal initialization."""

    def __init__(self, shape: Sequence[int], dtype: jnp.dtype = jnp.float32, init_fn: Callable | None = None):
        def default_init_fn(key, shape, dtype):
            return jax.nn.initializers.he_normal()(key, shape, dtype)

        super().__init__(shape, dtype, init_fn or default_init_fn)


class Bias(Parameter):
    """Learnable bias parameter initialized to zeros."""

    def __init__(self, shape: Sequence[int], dtype: jnp.dtype = jnp.float32):
        super().__init__(shape, dtype)


class Module(ABC):
    """Base class for neural network modules.

    Modules can contain parameters and submodules, and support parameter
    initialization and application. This abstract base class provides the
    foundation for building neural network layers and models.

    Methods:
        init: Initialize all parameters in module and submodules using random keys.
        apply: Apply module computation using provided parameter values.
        __call__: Define forward computation (must be implemented by subclasses).
    """

    def init(self, key: jax.Array) -> FrozenDict:
        """Initialize all parameters in module and submodules."""
        params = {}
        for name, value in self.__dict__.items():
            key, subkey = jax.random.split(key)
            if isinstance(value, Parameter):
                params[name] = value.init(subkey)
            if isinstance(value, Module):
                subparams = value.init(subkey)
                for k, v in subparams.items():
                    params[f"{name}.{k}"] = v
        return freeze(params)

    def apply(self, params: FrozenDict, *args, **kwargs):
        """Apply module to inputs using provided parameters."""
        module = deepcopy(self)
        # replace parameters with values from params
        for name, value in params.items():
            ref = module
            path, param = name.rsplit(".", 1)
            for attr in path.split("."):
                ref = getattr(ref, attr)
            setattr(ref, param, value)
        return module(*args, **kwargs)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """Define forward computation for the module."""
        pass


class Embedding(Module):
    """Learnable lookup table mapping integers to vectors.

    Creates an embedding layer that maps discrete tokens (represented as integers)
    to dense vector representations.

    Args:
        num_embeddings: size of the vocabulary/number of possible input values.
        features: dimensionality of the output embedding vectors.

    Attributes:
        weight: learnable embedding matrix of shape ``(num_embeddings, features)``.
    """

    def __init__(self, num_embeddings: int, features: int):
        super().__init__()
        self.weight = Weight((num_embeddings, features), init_fn=jax.random.normal)

    def __call__(self, input: pjax.Array):
        return pjax.index(self.weight, input)


class Linear(Module):
    """Linear (fully connected) layer without bias.

    Applies a linear transformation to input data: :math:`\\text{output} = \\text{input} \\times \\text{weight}`.

    Args:
        in_features: number of input features.
        out_features: number of output features.

    Attributes:
        weight: learnable weight matrix of shape ``(in_features, out_features)``.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.weight = Weight((in_features, out_features))

    def __call__(self, input):
        return pjax.matmul(input, self.weight)


class ReLU(Module):
    """Rectified Linear Unit with bias.

    Applies ReLU activation function with a learnable bias parameter.
    Supports multiple inputs for element-wise activation.

    Args:
        features: number of features/neurons in the layer.

    Attributes:
        bias: learnable bias vector of shape ``(features,)``.
    """

    def __init__(self, features):
        super().__init__()
        self.bias = Bias((features,))

    def __call__(self, *inputs):
        return pjax.sum_relu(self.bias, *inputs)


class MultiHeadAttention(Module):
    """Multi-head attention mechanism for transformer architectures.

    Implements the multi-head attention mechanism as described in "Attention
    Is All You Need" (Vaswani et al., 2017). Splits the input into multiple
    attention heads and applies scaled dot-product attention.

    Args:
        model_features: dimensionality of input and output features.
        qkv_features: dimensionality of query, key, and value vectors per head.
        heads: number of attention heads.

    Attributes:
        query_layer: linear layer for computing queries.
        key_layer: linear layer for computing keys.
        value_layer: linear layer for computing values.
        out_layer: output projection layer.
        heads: number of attention heads.
    """

    def __init__(self, model_features, qkv_features, heads):
        super().__init__()
        self.query_layer = Linear(model_features, heads * qkv_features)
        self.key_layer = Linear(model_features, heads * qkv_features)
        self.value_layer = Linear(model_features, heads * qkv_features)
        self.out_layer = Linear(heads * qkv_features, model_features)
        self.heads = heads

    def __call__(self, input):
        """Compute multi-head attention.

        Args:
            input: tensor of shape ``(batch_size, seq_len, model_features)``.

        Returns:
            tensor of shape ``(batch_size, seq_len, model_features)``.
        """
        q = self.query_layer(input)
        k = self.key_layer(input)
        v = self.value_layer(input)

        # obtain the query, key, and value arrays of shape (batch_size, heads, seq_len, qkv_features)
        batch_size, seq_len = input.shape[:2]

        def split_heads(x):
            return pjax.transpose(pjax.reshape(x, (batch_size, seq_len, self.heads, -1)), (0, 2, 1, 3))

        q, k, v = map(split_heads, (q, k, v))

        # compute the attention scores
        scale = 1.0 / jnp.sqrt(q.shape[-1])
        qk = pjax.matmul(q, pjax.transpose(k, (0, 1, 3, 2)))
        qk = pjax.add(qk, pjax.array(scale))  # simple scaling via addition
        qk = pjax.relu(qk)

        # compute the weighted sum of values
        o = pjax.matmul(qk, v)

        # merge the heads
        o = pjax.reshape(pjax.transpose(o, (0, 2, 1, 3)), (batch_size, seq_len, -1))

        return self.out_layer(o)


class Conv2D(Module):
    """2D convolutional layer via patch extraction and linear projection.

    Implements 2D convolution by extracting patches from the input tensor
    and applying a linear transformation. This approach leverages the patch
    extraction functionality for efficient convolution computation.

    Args:
        in_channels: number of input channels.
        out_channels: number of output channels.
        kernel_shape: shape of the convolution kernel as (height, width).
        strides: stride values as (stride_height, stride_width).
        padding: padding strategy, either "SAME", "VALID", or explicit padding values.

    Attributes:
        conv_patch: configured patch extraction function.
        linear: linear layer for patch transformation.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Sequence[int] = (3, 3),
        strides: Sequence[int] = (1, 1),
        padding: str | Sequence[int] = "SAME",
    ):
        super().__init__()
        self.conv_patch = partial(pjax.conv_patch, kernel_shape=kernel_shape, strides=strides, padding=padding)
        self.linear = Linear(int(in_channels * np.prod(kernel_shape)), out_channels)

    def __call__(self, input):
        """Apply convolution to input tensor.

        Args:
            input: input tensor of shape ``(..., H, W, C)``.

        Returns:
            output tensor after convolution and projection.
        """
        patches = self.conv_patch(input)
        out = self.linear(patches)
        return out
