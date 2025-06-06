"""Shape transformation operations that don't require projection operators.

Each transformation implements forward and inverse operations:
    ``transform(input, /, *, **kwargs) -> transformed_output``
    ``inverse(input, transformed_output, /, *, **kwargs) -> original_input``

These are invertible shape manipulations that preserve data while changing
tensor layout or structure. Used internally by the computation graph.
"""

import jax
import numpy as np
from jax import numpy as jnp

from .computation import make_shape_transform


def index_transform(a, idx, /):
    """Index a tensor."""
    assert isinstance(a, jnp.ndarray)
    return a[idx]


def index_inverse(a, idx, z, /):
    return a.at[idx].set(z)


index = make_shape_transform(
    "index",
    transform=index_transform,
    inverse=index_inverse,
)


def advanced_index_transform(a, /, *, idx):
    """Advanced indexing of a tensor.

    Here, ``idx`` can be any valid JAX indexing expression (e.g., a slice).
    Note it is a keyword argument, so for different indexing expressions, recompilation is needed.
    """
    assert isinstance(a, jnp.ndarray)
    return a[idx]


def advanced_index_inverse(a, z, /, *, idx):
    """Inverse of advanced indexing: restores the original tensor."""
    return a.at[idx].set(z)


advanced_index = make_shape_transform(
    "advanced_index", transform=advanced_index_transform, inverse=advanced_index_inverse
)


def reshape_transform(a, /, *, shape):
    """Reshape a tensor to a given shape."""
    return jnp.reshape(a, shape)


def reshape_inverse(a, z, /, *, shape):
    """Inverse of reshape transform: restores the original shape."""
    return jnp.reshape(z, a.shape)


reshape = make_shape_transform("reshape", transform=reshape_transform, inverse=reshape_inverse)


def transpose_transform(a, /, *, axes):
    """Transpose a tensor along specified axes."""
    return jnp.transpose(a, axes)


def transpose_inverse(a, z, /, *, axes):
    """Inverse of transpose transform: reverses the transposition."""
    return jnp.transpose(z, np.argsort(axes))


transpose = make_shape_transform("transpose", transform=transpose_transform, inverse=transpose_inverse)


def repeat_transform(a, /, *, repeats, axis):
    """Repeat elements of a tensor along a specified axis."""
    return jnp.repeat(a, repeats, axis=axis)


def repeat_inverse(a, z, /, *, repeats, axis):
    """Inverse of repeat transform: averages the repeated values."""
    axis = axis if axis >= 0 else a.ndim + axis
    assert a.shape[axis] * repeats == z.shape[axis]
    reshaped = z.reshape(z.shape[:axis] + (repeats, -1) + z.shape[axis + 1 :])
    return reshaped.mean(axis=axis)


repeat = make_shape_transform("repeat", transform=repeat_transform, inverse=repeat_inverse)


def concatenate_transform(*args, axis=0):
    """Concatenate multiple tensors along a specified axis."""
    return jnp.concatenate(args, axis=axis)


def concatenate_inverse(*args, axis=0):
    """Inverse of concatenate transform: splits the tensor back into original parts."""
    args, z = args[:-1], args[-1]
    split_indices = np.cumsum([x.shape[axis] for x in args[:-1]])
    return jnp.split(z, split_indices, axis=axis)


concatenate = make_shape_transform("concatenate", transform=concatenate_transform, inverse=concatenate_inverse)


def zero_pad_transform(a, /, *, pad_width):
    """Apply zero-padding to a tensor."""
    return jnp.pad(a, pad_width, mode="constant")


def zero_pad_inverse(a, z, /, *, pad_width):
    """Inverse of zero-pad transform: removes padding."""
    return z[tuple(slice(i, -j if j else None) for i, j in pad_width)]


zero_pad = make_shape_transform("zero_pad", transform=zero_pad_transform, inverse=zero_pad_inverse)


def conv_patch_transform(a, /, *, kernel_shape, strides, padding):
    """Extract patches from an array for convolution-like operations.

    Returns a tensor of shape ``(N, H_out, W_out, C_in * H_k * W_k)``,
    where ``H_out`` and ``W_out`` are the output height and width after convolution,
    and ``H_k`` and ``W_k`` are the kernel height and width.
    """
    return jax.lax.conv_general_dilated_patches(
        a,
        filter_shape=kernel_shape,
        window_strides=strides,
        padding=padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )


def conv_patch_inverse(a, z, /, *, kernel_shape, strides, padding):
    """Inverse of convolution patch extraction: reconstructs the original array."""
    N, H_in, W_in, C_in = a.shape
    _, H_out, W_out, _ = z.shape
    H_k, W_k = kernel_shape

    # Calculate inverse padding
    inv_padding = (
        (H_k - 1 - padding[0][0], H_k - 1 - padding[0][1]),
        (W_k - 1 - padding[1][0], W_k - 1 - padding[1][1]),
    )

    # Reshape z to (N*C_in, H_out, W_out, H_k*W_k)
    feature_patches = (
        z.reshape(N, H_out, W_out, C_in, H_k * W_k).transpose(0, 3, 1, 2, 4).reshape(N * C_in, H_out, W_out, H_k * W_k)
    )

    # Kernel to sum contributions from patches
    sum_kernel = jax.nn.one_hot(jnp.flip(jnp.arange(H_k * W_k)), H_k * W_k, dtype=z.dtype).reshape(
        H_k, W_k, H_k * W_k, 1
    )

    # Sum patch contributions via transposed convolution
    # (N*C_in, H_out, W_out, H_k*W_k) * (H_k, W_k, H_k*W_k, 1) -> (N*C_in, H_in, W_in, 1)
    summed_patches = jax.lax.conv_transpose(
        feature_patches,
        sum_kernel,
        strides=strides,
        padding=inv_padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    # Calculate the number of patches contributing to each pixel
    # (1, H_out, W_out, 1) * (H_k, W_k, 1, 1) -> (1, H_in, W_in, 1)
    num_summed = jax.lax.conv_transpose(
        jnp.ones((1, H_out, W_out, 1), dtype=z.dtype),
        jnp.ones((H_k, W_k, 1, 1), dtype=z.dtype),
        strides=strides,
        padding=inv_padding,
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    # Normalize and reshape back to original shape
    out_flat = summed_patches / num_summed
    out = out_flat.reshape(N, C_in, H_in, W_in).transpose(0, 2, 3, 1)  # Reshape and transpose to (N, H_in, W_in, C_in)
    return out


conv_patch = make_shape_transform("conv_patch", transform=conv_patch_transform, inverse=conv_patch_inverse)
