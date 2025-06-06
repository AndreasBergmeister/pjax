"""Core projection operators for primitive functions.

Each primitive function has two components:
    ``operation(input_args, /) -> output``
    ``projection(input_args, output, /) -> projected_inputs``

The operation computes the forward pass, while the projection operator
computes the orthogonal projection onto the function's graph.
"""

import jax
from jax import numpy as jnp

from .. import config
from .computation import make_computation


def identity_op(a, /):
    """Identity operation returning input unchanged."""
    return a


def identity_proj(a, z, /):
    """Project onto identity function graph."""
    mid = (a + z) / 2
    return (mid,)


identity = make_computation("identity", identity_op, identity_proj)


def sum_op(a, /):
    """Sum operation for 1D arrays."""
    assert a.ndim == 1
    return a.sum()


def sum_proj(a, z, /):
    """Project onto sum function graph."""
    assert a.ndim == 1 and z.ndim == 0
    t = (z - a.sum()) / (a.size + z.size)
    return (a + t,)


sum_ = make_computation("sum", sum_op, sum_proj)


def max_op(a, /):
    """Maximum operation for 1D arrays."""
    assert a.ndim == 1
    return jnp.max(a)


def max_proj(a, z, /):
    """Project onto maximum function graph."""
    assert a.ndim == 1 and z.ndim == 0
    n = a.size

    # sort array
    idx = jnp.argsort(a)
    a_sorted = jnp.array(a)[idx]

    # compute candidate maxima
    z_k_flipped = (jnp.cumsum(jnp.flip(a_sorted)) + z) / jnp.arange(2, n + 2)
    z_k = jnp.flip(z_k_flipped)

    # compute candidate arrays
    i_ge_k = jnp.triu(jnp.ones((n, n), dtype=bool))
    a_k = jnp.where(i_ge_k, z_k[:, None], a_sorted[None, :])

    # compute distances
    dist = ((a_k - a_sorted) ** 2).sum(axis=1) + (z_k - z) ** 2

    # select valid candidates
    dist_valid = jnp.where(jnp.max(a_k, axis=1) <= z_k, dist, jnp.inf)

    # select candidate minimizing distance
    k = jnp.argmin(dist_valid)
    return (a_k[k][jnp.argsort(idx)],)


max = make_computation("max", max_op, max_proj)


def dotproduct_op(a, b, /):
    """Dot product operation for 1D arrays."""
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("dotproduct requires 1D arrays.")
    if a.size != b.size:
        raise ValueError(f"dotproduct requires arrays of the same size. Got {a.size} and {b.size}.")
    return a.dot(b)


def bilinear_proj(a, b, z, /):
    """Project onto bilinear function graph using Newton's method."""
    p = a @ b
    q = a @ a + b @ b

    def f(t):
        return ((1 + t**2) * p + t * q) / (1 - t**2) ** 2 - z + t

    f_prime = jax.grad(f)

    def newton_step(t):
        return t - f(t) / f_prime(t)

    t = jax.lax.fori_loop(0, config.bilinear_projection_num_newton_steps, lambda _, t: newton_step(t), 0.0, unroll=True)

    a_new = (a + t * b) / (1 - t**2)
    b_new = (b + t * a) / (1 - t**2)

    return a_new, b_new


dot = make_computation("dot", dotproduct_op, bilinear_proj)


def sum_relu_op(*args):
    """Sum inputs and apply ReLU activation."""
    return jax.nn.relu(sum(args))


def sum_relu_proj(*args):
    """Project onto sum-ReLU function graph."""
    inputs, output = args[:-1], args[-1]

    # compute intermediate values
    s = sum(inputs)
    mid_1 = s / len(inputs)
    mid_2 = (output - s) / (len(inputs) + 1)

    # solution 1
    new_inputs_1 = [jnp.where(s < 0, x, x - mid_1) for x in inputs]
    dist_1 = jnp.where(s < 0, output**2, output**2 + len(inputs) * mid_1**2)

    # solution 2
    new_output_2 = jnp.maximum(0, output - mid_2)
    new_inputs_2 = [jnp.where(new_output_2 == 0, x - mid_1, x + mid_2) for x in inputs]
    dist_2 = jnp.where(output == 0, output**2 + len(inputs) * mid_1**2, (len(inputs) + 1) * mid_2**2)

    # select solution minimizing the distance
    return tuple(jnp.where(dist_1 < dist_2, x_1, x_2) for x_1, x_2 in zip(new_inputs_1, new_inputs_2))


sum_relu = make_computation("sum_relu", sum_relu_op, sum_relu_proj)


def quantize_op(a, /, *, levels=2, scale=1.0):
    """Quantize values to discrete levels."""
    assert levels >= 2
    step = 2 * scale / (levels - 1)
    z = jnp.round((a + scale) / step)
    z = jnp.clip(z, 0, levels - 1)
    z = z * step - scale
    return z


def quantize_proj(a, z, /, *, levels=2, scale=1.0):
    """Project onto quantization function graph."""

    quantization_points = jnp.linspace(-scale, scale, levels)
    midpoints = (quantization_points[:-1] + quantization_points[1:]) / 2
    lower_bounds = jnp.concatenate([jnp.array([-jnp.inf]), midpoints])
    upper_bounds = jnp.concatenate([midpoints, jnp.array([jnp.inf])])

    a_candidates = jnp.select(
        [a[..., None] < lower_bounds, a[..., None] > upper_bounds], [lower_bounds, upper_bounds], a[..., None]
    )

    z_candidates = jnp.broadcast_to(quantization_points, a_candidates.shape)

    dist = (a_candidates - a[..., None]) ** 2 + (z_candidates - z[..., None]) ** 2

    min_idx = jnp.argmin(dist, axis=-1, keepdims=True)
    a_new = jnp.take_along_axis(a_candidates, min_idx, axis=-1).squeeze(-1)
    return (a_new,)


quantize = make_computation("quantize", quantize_op, quantize_proj)


def margin_loss_op(logits, labels, /):
    """Margin loss operation."""
    return logits


def margin_loss_proj(logits, labels, *args):
    """Project onto margin loss constraint."""

    cond_0 = (labels <= 0) & (logits > 0)
    cond_1 = (labels > 0) & (logits < labels)

    new_logits = jnp.select([cond_0, cond_1], [0.0, labels], logits)

    return new_logits, labels


margin_loss = make_computation("margin_loss", margin_loss_op, margin_loss_proj)


def cross_entropy_op(logits, labels, /):
    """Cross-entropy operation."""
    return logits


def corss_entropy_prox(logits, labels, _, /):
    """Project onto cross-entropy constraint using iterative methods."""

    method = config.cross_entropy_method
    lmbda = config.cross_entropy_lambda
    steps = config.cross_entropy_num_steps

    if method == "newton":

        def newton(logits, labels):
            s = jax.nn.softmax(logits)
            x = logits
            for _ in range(steps):
                G = (x - logits) / lmbda + s - labels
                H = jnp.eye(x.size) / lmbda + jnp.diag(s) - jnp.outer(s, s)
                dx = jax.scipy.linalg.solve(H, G, assume_a="pos")
                x = x - dx
            return x

        x = jax.vmap(newton)(logits, labels)

    elif method == "fixed_point":
        x = logits
        for _ in range(steps):
            x = logits + lmbda * (labels - jax.nn.softmax(x))

    else:
        raise ValueError(f"Unknown method: {method}")

    return x, labels


cross_entropy = make_computation("cross_entropy", cross_entropy_op, corss_entropy_prox)
