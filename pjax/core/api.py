import math
from functools import partial
from typing import Sequence

from jax import numpy as jnp

from . import no_ops, ops
from .computation import Computation, vmap

Axis = Sequence[int] | int | None


def broadcast_to(a: Computation, shape: Sequence[int]) -> Computation:
    """Broadcasts an array to a new shape by repeating elements as needed.

    Args:
        a: array to broadcast.
        shape: target shape.

    Returns:
        broadcasted array.
    """
    for i, s in enumerate(shape[::-1]):
        if a.ndim <= i:
            a = expand_dims(a, 0)

        if a.shape[-(i + 1)] == 1:
            a = repeat(a, s, -(i + 1))

        if a.shape[-(i + 1)] != s:
            raise ValueError(f"cannot broadcast array of shape {a.shape} to shape {shape}.")

    return a


def broadcast_arrays(*args: Computation) -> list[Computation]:
    """Broadcasts multiple arrays to a common shape by repeating elements.

    Args:
        *args: arrays to broadcast.

    Returns:
        list of broadcasted arrays.
    """
    max_ndim = max([arg.ndim for arg in args])
    max_shape = [max([s[-i] for s in [arg.shape for arg in args] if i <= len(s)]) for i in range(1, max_ndim + 1)][::-1]
    return [broadcast_to(arg, max_shape) for arg in args]


def _unary_op(op, a: Computation, axis: Axis = None, keepdims: bool = False) -> Computation:
    """Applies a unary operation to an array over specified axes.

    Args:
        op: unary operation to apply.
        a: array to which the operation will be applied.
        axis: axis or axes over which to apply the operation. If ``None``, the operation is applied over all axes.
        keepdims: if ``True``, the reduced dimensions are kept with size 1.

    Returns:
        result of the unary operation.
    """
    if axis is None:
        axis = tuple(range(a.ndim))
    if isinstance(axis, int):
        axis = (axis,)

    def fn(a):
        out = op(reshape(a, -1))
        if keepdims:
            out = reshape(out, [1] * a.ndim)
        return out

    for ax in range(a.ndim):
        if ax not in axis:
            if keepdims:
                fn = vmap(fn, in_axes=ax, out_axes=ax)
            else:
                fn = vmap(fn, in_axes=ax, out_axes=-1)

    return fn(a)


def identity(a: Computation) -> Computation:
    """Identity function that returns the input unchanged.

    Args:
        a: input array.

    Returns:
        unchanged input array.
    """
    return ops.identity(a)


def sum_(a: Computation, axis: Axis = None, keepdims: bool = False) -> Computation:
    """Computes the sum of an array along the specified axis or axes.

    Args:
        a: array to compute the sum.
        axis: axis or axes along which to compute the sum. If ``None``, sum of all elements is computed.
        keepdims: if ``True``, the reduced axes are kept with length 1.

    Returns:
        sum of the array along the specified axis.
    """
    return _unary_op(ops.sum_, a, axis=axis, keepdims=keepdims)


def add(a: Computation, b: Computation) -> Computation:
    """Element-wise addition of two arrays.

    Args:
        a: first array.
        b: second array with broadcastable shape.

    Returns:
        array containing the element-wise sum of the two arrays.
    """
    return sum_(stack(broadcast_arrays(a, b)), axis=0)


def max_(a: Computation, axis: Axis = None, keepdims: bool = False) -> Computation:
    """Computes the maximum of an array along the specified axis or axes.

    Args:
        a: array to compute the maximum.
        axis: axis or axes along which to compute the maximum. If ``None``, maximum of all elements is computed.
        keepdims: if ``True``, the reduced axes are kept with length 1.

    Returns:
        maximum of the array along the specified axis.
    """
    return _unary_op(ops.max, a, axis=axis, keepdims=keepdims)


def maximum(a: Computation, b: Computation) -> Computation:
    """Element-wise maximum of two arrays.

    Args:
        a: first array.
        b: second array with broadcastable shape.

    Returns:
        array containing the element-wise maximum of the two arrays.
    """
    return max_(stack(broadcast_arrays(a, b)), axis=0)


def dot(a: Computation, b: Computation) -> Computation:
    """Dot product of two arrays.

    Args:
        a: array of shape ``(..., N)``.
        b: array of shape ``(N,)`` or ``(..., N, M)``. In the latter case, the leading dimensions must be broadcastable.

    Returns:
        array containing the dot product of the two arrays, with leading dimensions stacked.
    """
    fn = partial(ops.dot)  # without partial, we get recompilation errors

    if b.ndim > 1:
        fn = vmap(fn, in_axes=(None, -1))

    for _ in range(max(b.ndim - 2, 0)):
        fn = vmap(fn, in_axes=(None, 0))

    for _ in range(max(a.ndim - 1, 0)):
        fn = vmap(fn, in_axes=(0, None))

    return fn(a, b)


def matmul(a: Computation, b: Computation) -> Computation:
    """Matrix product of two arrays with element-wise scaling of the result.

    Args:
        a: array of shape ``(N,)`` or ``(..., K, N)``.
        b: array of shape ``(N,)`` or ``(..., N, M)``. In the latter case, the leading dimensions must be broadcastable.

    Returns:
        array containing the matrix product of the two arrays, with leading dimensions broadcasted. If ``b.ndim == 1`` shape is ``a.shape[:-1]``. Otherwise, shape is ``(..., K, M)``.
    """
    a_, b_ = a, b

    # expand dimensions if necessary
    if a.ndim == 1:
        a_ = expand_dims(a, 0)
    if b.ndim == 1:
        b_ = expand_dims(b, 1)

    # broadcast leading dimensions
    max_ndim = max(a_.ndim, b_.ndim)
    max_shape = tuple([max([s[-i] for s in [a_.shape, b_.shape] if i <= len(s)]) for i in range(1, max_ndim + 1)])[::-1]
    a_ = broadcast_to(a_, max_shape[:-2] + a_.shape[-2:])
    b_ = broadcast_to(b_, max_shape[:-2] + b_.shape[-2:])

    # vectorize dot product
    fn = partial(ops.dot)  # without partial, we get recompilation errors
    fn = vmap(fn, in_axes=(None, -1))
    fn = vmap(fn, in_axes=(-2, None))

    for _ in range(max(a_.ndim - 2, 0)):
        fn = vmap(fn)
    out = fn(a_, b_)

    # squeeze dimensions if necessary
    if a.ndim == 1:
        out = squeeze(out, axis=-2)
    if b.ndim == 1:
        out = squeeze(out, axis=-1)
    return out


def relu(a: Computation) -> Computation:
    """Rectified Linear Unit activation function.

    Applies the ReLU function element-wise: :math:`\\text{ReLU}(x) = \\max(0, x)`.

    Args:
        a: input array to apply ReLU activation.

    Returns:
        array with ReLU activation applied element-wise.
    """
    return ops.sum_relu(a)


def sum_relu(*args: Computation) -> Computation:
    """Sum multiple arrays and apply ReLU activation.

    Computes :math:`\\text{ReLU}(\\sum \\text{args})` where all input arrays are first broadcasted
    to compatible shapes before summation.

    Args:
        *args: variable number of arrays to sum and activate.

    Returns:
        array containing :math:`\\text{ReLU}(\\sum \\text{broadcasted arrays})`.
    """
    return ops.sum_relu(*broadcast_arrays(*args))


def quantize(a: Computation, levels=2, scale=1.0) -> Computation:
    """Quantize an array to equally spaced levels on the interval ``[-scale, scale]``.

    Args:
        a: array to quantize.
        levels: number of levels.
        scale: scale of the quantization.

    Returns:
        quantized array.
    """
    return ops.quantize(a, levels=levels, scale=scale)


def margin_loss(logits: Computation, labels: Computation) -> Computation:
    """Compute margin loss for classification tasks.

    The margin loss enforces that the predicted score for the correct class
    exceeds scores for incorrect classes by a margin. The projection ensures
    that the logit for the true class is negative if the label is 0 and
    greater than the margin for positive labels.

    Args:
        logits: predicted class scores/logits.
        labels: ground truth class labels.

    Returns:
        margin loss value.
    """
    return ops.margin_loss(logits, labels)


def cross_entropy(logits: Computation, labels: Computation) -> Computation:
    """Compute cross-entropy loss for classification.

    Computes the cross-entropy loss between predicted logits and true labels.
    This is commonly used as a loss function for multi-class classification.

    Args:
        logits: predicted class logits/scores.
        labels: ground truth one-hot encoded labels.

    Returns:
        cross-entropy loss value.
    """
    return ops.cross_entropy(logits, labels)


def index(a: Computation, idx: Computation) -> Computation:
    """Index an array using advanced or basic indexing.

    Provides array indexing functionality with support for both basic indexing
    (when ``idx`` is a tensor) and advanced indexing. Advanced indexing is used
    when the index is not a tensor, but requires recompilation for different
    indexing expressions.

    Args:
        a: array to index.
        idx: index or indices to use for accessing elements.

    Returns:
        indexed array.
    """
    if isinstance(idx, (jnp.ndarray, Computation)):
        return no_ops.index(a, idx)
    return no_ops.advanced_index(a, idx=idx)


def reshape(a: Computation, shape: Sequence[int]) -> Computation:
    """Reshape a tensor to a new shape.

    Changes the shape of an array without altering its data. The total number
    of elements must remain the same.

    Args:
        a: array to reshape.
        shape: new shape as a sequence of integers.

    Returns:
        reshaped array.
    """
    return no_ops.reshape(a, shape=shape)


def transpose(a: Computation, axes: Sequence[int]) -> Computation:
    """Transpose a tensor by permuting its axes.

    Reorders the dimensions of an array according to the specified axes
    permutation. No operation is performed if axes are already in order.

    Args:
        a: array to transpose.
        axes: sequence of integers specifying the new axis order.

    Returns:
        transposed array.
    """
    if tuple(axes) == tuple(range(a.ndim)):
        return a
    return no_ops.transpose(a, axes=axes)


def swapaxes(a: Computation, axis1: int, axis2: int) -> Computation:
    """Swap two axes of a tensor.

    Interchanges two axes of an array by transposing the specified dimensions.

    Args:
        a: array to modify.
        axis1: first axis to swap.
        axis2: second axis to swap.

    Returns:
        array with swapped axes.
    """
    transpose_axes = list(range(a.ndim))
    transpose_axes[axis1], transpose_axes[axis2] = transpose_axes[axis2], transpose_axes[axis1]
    return transpose(a, axes=transpose_axes)


def moveaxis(a: Computation, source: int, destination: int) -> Computation:
    """Move an axis to a new position.

    Moves the axis at the source position to the destination position,
    shifting other axes as needed.

    Args:
        a: array to modify.
        source: original position of the axis.
        destination: new position for the axis.

    Returns:
        array with the axis moved.
    """
    destination = destination if destination >= 0 else a.ndim + destination
    transpose_axes = list(range(a.ndim))
    transpose_axes.pop(source)
    transpose_axes.insert(destination, source)
    return transpose(a, axes=transpose_axes)


def expand_dims(a: Computation, axis: int) -> Computation:
    """Insert dimensions of length 1 into array.

    Expands the shape of an array by inserting new axes of length 1
    at the specified position.

    Args:
        a: array to expand.
        axis: position where the new axis is placed.

    Returns:
        array with expanded dimensions.
    """
    value = a if isinstance(a, jnp.ndarray) else a.value
    expand_shape = jnp.expand_dims(value, axis).shape
    return reshape(a, shape=expand_shape)


def squeeze(a: Computation, axis=None) -> Computation:
    """Remove single-dimensional entries from the shape of an array.

    Removes dimensions of size 1 from the shape of an array. If axis is not
    specified, all dimensions of size 1 are removed.

    Args:
        a: array to squeeze.
        axis: axis or axes to squeeze. If ``None``, all single-dimensional axes are removed.

    Returns:
        array with single-dimensional entries removed.
    """
    if axis is None:
        axis = tuple(i for i, s in enumerate(a.shape) if s == 1)
    elif isinstance(axis, int):
        axis = (axis,)

    axis = [i if i >= 0 else a.ndim + i for i in axis]
    new_shape = tuple(s for i, s in enumerate(a.shape) if i not in axis)
    return reshape(a, shape=new_shape)


def repeat(a: Computation, repeats, axis) -> Computation:
    """Repeat a tensor along an axis.

    Repeats elements of an array along the specified axis by the given number
    of times.

    Args:
        a: array to repeat.
        repeats: number of repetitions.
        axis: axis along which to repeat.

    Returns:
        array with repeated elements.
    """
    if repeats == 1:
        return a
    return no_ops.repeat(a, repeats=repeats, axis=axis)


def concatenate(arrays: Sequence[Computation], axis=0) -> Computation:
    """Concatenate a list of arrays along an axis.

    Joins a sequence of arrays along an existing axis. All arrays must have
    the same shape except in the dimension corresponding to axis.

    Args:
        arrays: sequence of arrays to concatenate.
        axis: axis along which to concatenate.

    Returns:
        concatenated array.
    """
    if len(arrays) == 1:
        return arrays[0]
    return no_ops.concatenate(*arrays, axis=axis)


def stack(arrays: Sequence[Computation], axis=0) -> Computation:
    """Stack a list of arrays along a new axis.

    Creates a new axis and stacks the input arrays along it. All arrays must
    have the same shape.

    Args:
        arrays: sequence of arrays to stack.
        axis: axis along which to stack the arrays.

    Returns:
        stacked array with one additional dimension.
    """
    return concatenate([expand_dims(a, axis) for a in arrays], axis=axis)


def zero_pad(a: Computation, pad_width) -> Computation:
    """Pad an array with zeros.

    Args:
        a: the array to pad.
        pad_width: specify the pad width for each dimension of an array. Padding widths
            may be separately specified for *before* and *after* the array. Options are:

            - ``int``: pad each array dimension with the same number of values
                both before and after.
            - ``(before, after)``: pad each array with ``before`` elements before, and ``after``
                elements after
            - ``((before_1, after_1), (before_2, after_2), ... (before_N, after_N))``: specify
                distinct ``before`` and ``after`` values for each array dimension.
    """

    if isinstance(pad_width, int):
        pad_width = (pad_width, pad_width)

    if isinstance(pad_width[0], int):
        pad_width = (pad_width,) * a.ndim

    if len(pad_width) != a.ndim:
        raise ValueError(
            f"pad_width must have the same length as the number of dimensions of the array, got {len(pad_width)} and {a.ndim}."
        )

    return no_ops.zero_pad(a, pad_width=pad_width)


def conv_patch(a: Computation, kernel_shape, strides, padding) -> Computation:
    """Extract patches from an array.

    Args:
        a: the array to extract patches from of shape ``(..., H, W, C)``.
        kernel_shape: kernel shape ``(H_k, W_k)``.
        strides: ``(stride_h, stride_w)``.
        padding: padding before and after each dimension:
            ``(pad_h_before, pad_h_after), (pad_w_before, pad_w_after)`` or ``"VALID"`` or ``"SAME"``.

    Returns:
        the patches of shape ``(N, H_out, W_out, C * H_k * W_k)``,
        where ``H_out = (H + pad_h_before + pad_h_after - H_k) // stride_h + 1``,
        and ``W_out = (W + pad_w_before + pad_w_after - W_k) // stride_w + 1``.
    """

    if padding == "VALID":
        padding = ((0, 0), (0, 0))

    if padding == "SAME":
        H, W = a.shape[1:3]
        H_s, W_s = strides
        H_k, W_k = kernel_shape
        H_p = max(0, math.ceil((H - 1) / H_s) * H_s + H_k - H)
        W_p = max(0, math.ceil((W - 1) / W_s) * W_s + W_k - W)
        pad_h = (H_p // 2, (H_p + 1) // 2)
        pad_w = (W_p // 2, (W_p + 1) // 2)
        padding = (pad_h, pad_w)

    if a.ndim < 3:
        raise ValueError(f"conv_patch requires an array of at least 3 dimensions, got {a.ndim}.")

    if a.ndim < 4:
        a = a[None]

    fn = lambda a: no_ops.conv_patch(a, kernel_shape=kernel_shape, strides=strides, padding=padding)
    for _ in range(a.ndim - 4):
        fn = vmap(fn)

    return fn(a)
