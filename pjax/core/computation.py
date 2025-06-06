from __future__ import annotations

import inspect
from functools import partial, reduce, wraps
from typing import Any, Callable, Sequence

import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node_class

import pjax

from .frozen_dict import FrozenDict, freeze


@register_pytree_node_class
class Computation:
    """Symbolic computation node in the PJAX computation graph.

    Each instance wraps a JAX array and tracks its parents, name, and vmap (vectorization) metadata.
    ``Computation`` nodes form a directed acyclic graph, enabling symbolic differentiation, vectorization,
    and graph transformations. Most user-facing PJAX objects are ``Computation`` instances or subclasses.

    Attributes:
        value: underlying JAX array value.
        name: string identifier for the node.
        parents: tuple of parent ``Computation`` nodes.
        vmap_ids: tuple of vmap (vectorization) ids for each axis.
        static_hash_data: static data for hashing and graph identity.
    """

    def __init__(
        self,
        value: Any,
        name: str,
        parents: Sequence[Computation],
        vmap_ids: Sequence[int | None] = (),
        static_hash_data: Any = None,
    ):
        if not isinstance(value, jnp.ndarray):
            raise ValueError(f"value must be a jax array, got {type(value)}.")

        self.value = value
        self.name = name
        self.parents = tuple(parents)
        self.static_hash_data = static_hash_data
        if not vmap_ids:
            self.vmap_ids = (None,) * value.ndim
        else:
            if len(vmap_ids) != value.ndim:
                raise ValueError("The number of vmap ids must match the number of dimensions of the value.")
            self.vmap_ids = tuple(vmap_ids)
        self.vmap_sizes = {id: value.shape[axis] for id, axis in zip(vmap_ids, range(value.ndim)) if id is not None}

    def tree_flatten(self) -> tuple[tuple[Any], FrozenDict]:
        """Flattens the computation node for JAX pytree compatibility."""
        init_args = inspect.signature(self.__init__).parameters
        aux_data = {attr: getattr(self, attr) for attr in init_args if attr != "value"}
        return (self.value,), freeze(aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data: FrozenDict, children: tuple[Any]) -> Computation:
        """Reconstructs a computation node from pytree aux_data and children."""
        return cls(value=children[0], **aux_data)

    def _set_vmap_ids(self, vmap_ids: Sequence[int | None]):
        """Returns a copy of this computation with updated vmap (vectorization) ids."""
        children, aux_data = self.tree_flatten()
        aux_data = aux_data.set("vmap_ids", vmap_ids)
        return self.tree_unflatten(aux_data, children)

    def add_vmap_id(self, vmap_id: int, axis: int):
        """Adds a vmap id to the computation along the specified axis."""
        if axis < 0:
            axis = self.ndim + axis

        # replace the axis-th None with the vmap_id
        vmap_ids = list(self.vmap_ids)
        non_indices = [i for i, id in enumerate(vmap_ids) if id is None]
        vmap_ids[non_indices[axis]] = vmap_id
        return self._set_vmap_ids(vmap_ids)

    def remove_vmap_id(self, vmap_id: int):
        """Removes a vmap id from the computation."""
        if vmap_id not in self.vmap_ids:
            raise ValueError(f"vmap id {vmap_id} not found in vmap ids {self.vmap_ids}.")

        vmap_ids = [id if id != vmap_id else None for id in self.vmap_ids]
        return self._set_vmap_ids(vmap_ids)

    def __lt__(self, other: object) -> bool:
        return hash(self) < hash(other)

    def __eq__(self, other: object) -> bool:
        return hash(self) == hash(other)

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the underlying array, excluding vmap axes."""
        return tuple([dim for dim, id in zip(self.value.shape, self.vmap_ids) if id is None])

    @property
    def ndim(self) -> int:
        """Returns the number of non-vmap dimensions."""
        return len(self.shape)

    @property
    def size(self) -> int:
        """Returns the total number of elements (excluding vmap axes)."""
        return reduce(lambda x, y: x * y, self.shape)

    @property
    def dtype(self) -> jnp.dtype:
        """Returns the dtype of the underlying JAX array."""
        return self.value.dtype

    def __hash__(self) -> int:
        if not hasattr(self, "_hash"):
            self._hash = hash((self.name, self.parents, self.static_hash_data))
        return self._hash

    def __len__(self) -> int:
        return self.size

    def __repr__(self) -> str:
        return self.name

    def __getitem__(self, idx: Any) -> "Computation":
        """Returns a new computation representing an index operation on this node."""
        from .api import index

        return index(self, idx)


@register_pytree_node_class
class Parameter(Computation):
    """Learnable parameter node in the PJAX computation graph."""

    def __init__(self, value: jnp.ndarray, name: str | None = None, vmap_ids: Sequence[int | None] = ()):
        if name is None:
            name = f"Parameter(shape={value.shape}, dtype={value.dtype})"
        super().__init__(value, name, (), vmap_ids, static_hash_data=None)
        self.projection = lambda x, y: (y,)  # ignore value


def parameter(a: jnp.ndarray) -> Parameter:
    """Wraps a JAX ndarray as a learnable ``Parameter`` node in the computation graph.

    Args:
        a: JAX ndarray to wrap.

    Returns:
        ``Parameter`` node containing the value.
    """
    return Parameter(a)


@register_pytree_node_class
class Array(Computation):
    """Constant (non-learnable) array in the PJAX computation graph."""

    def __init__(self, value: jnp.ndarray, name: str | None = None, vmap_ids: Sequence[int | None] = ()):
        if name is None:
            name = f"Array(shape={value.shape}, dtype={value.dtype})"
        super().__init__(value, name, (), vmap_ids, static_hash_data=None)
        self.projection = lambda *args: (value,) * (len(args) - 1)


def array(value: jnp.ndarray, name: str | None = None) -> Array:
    """Wraps a JAX ndarray as an ``Array`` node.

    Args:
        value: array to wrap.
        name: optional name for the node.

    Returns:
        ``Array`` node containing the array.
    """
    return Array(value, name)


@register_pytree_node_class
class Operation(Computation):
    """Primitive operation node in the PJAX computation graph.

    ``Operation`` nodes represent the application of a function to one or more parent nodes.
    They store the operation, its projection (for optimization), and the resulting value.
    """

    def __init__(
        self,
        name: str,
        operation: Callable,
        projection: Callable,
        parents: Sequence[Computation],
        value: Any,
        vmap_ids: Sequence[int | None] = (),
        static_hash_data: Any = None,
    ):
        super().__init__(value, name, parents, vmap_ids, static_hash_data)
        self.operation = operation
        self.projection = projection


@register_pytree_node_class
class ShapeTransform(Computation):
    """Invertible shape transformation node in the PJAX computation graph.

    ``ShapeTransform`` nodes represent operations that change the shape of their input(s) in an invertible way,
    such as reshape or transpose.
    They store both the forward and inverse transformations for graph manipulation and optimization.
    """

    def __init__(
        self,
        name: str,
        transform: Callable,
        inverse: Callable,
        parents: Sequence[Computation],
        value: Any,
        vmap_ids: Sequence[int | None] = (),
        static_hash_data: Any = None,
    ):
        super().__init__(value, name, parents, vmap_ids, static_hash_data)
        self.transform = transform
        self.inverse = inverse


computation_tree_map = partial(jax.tree.map, is_leaf=lambda x: isinstance(x, Computation))
computation_tree_leaves = partial(jax.tree.leaves, is_leaf=lambda x: isinstance(x, Computation))


def arrays_to_inputs(args: Any) -> Any:
    """Converts all JAX arrays in a nested structure to PJAX ``Array`` nodes."""
    if not all(isinstance(arg, Computation) or isinstance(arg, jnp.ndarray) for arg in computation_tree_leaves(args)):
        raise ValueError("All positional arguments must be jax arrays or computations.")

    args = jax.tree.map(lambda x: freeze(x) if isinstance(x, dict) else x, args, is_leaf=lambda x: isinstance(x, dict))
    return computation_tree_map(lambda arg: array(arg) if isinstance(arg, jnp.ndarray) else arg, args)


vmap_ids_order = []


def vmap(fun: Callable, in_axes: int | Sequence[int | None] = 0, out_axes: int | Sequence[int | None] = 0) -> Callable:
    """Vectorizes a function along the specified axes, supporting both JAX arrays and PJAX ``Computation`` objects.

    Args:
        fun: function to vectorize.
        in_axes: axis or axes to vectorize over.
        out_axes: output axes after vectorization.

    Returns:
        vectorized function.
    """

    @wraps(fun)
    def wrapper(*args):
        # if none of the arguments are computations, return the value
        if not any(isinstance(v, Computation) for v in computation_tree_leaves(args)):
            return jax.vmap(fun, in_axes, out_axes=out_axes)(*args)

        args = arrays_to_inputs(args)
        in_axes_ = (in_axes,) * len(args) if isinstance(in_axes, int) else tuple(in_axes)

        if len(in_axes_) != len(args):
            raise ValueError("The number of in_axes must match the number of positional arguments.")

        # get unique id for this vmap instance
        vmap_id = hash((fun, in_axes_, out_axes, *args))

        if vmap_id not in vmap_ids_order:
            vmap_ids_order.append(vmap_id)

        args = [
            computation_tree_map(lambda x: x.add_vmap_id(vmap_id, axis) if axis is not None else x, arg)
            for arg, axis in zip(args, in_axes_)
        ]

        # compute and post-process output
        out = fun(*args)
        out = computation_tree_map(lambda x: x.remove_vmap_id(vmap_id), out)
        out = computation_tree_map(lambda x, axis: pjax.moveaxis(x, 0, axis), out, out_axes)
        return out

    return wrapper


def handle_vmaped(args: Sequence[Computation]):
    """Aligns vmap axes across all arguments, repeating non-vmaped arguments as needed."""
    vmap_axes_sizes = {}
    for arg in args:
        for i, (id, size) in enumerate(arg.vmap_sizes.items()):
            if id in vmap_axes_sizes:
                if vmap_axes_sizes[id] != size:
                    raise ValueError("All vmaped arguments must have the same size along the vmaped axis.")
            else:
                vmap_axes_sizes[id] = size

    all_vmap_ids = sorted(vmap_axes_sizes.keys(), key=lambda id: vmap_ids_order.index(id))

    if not vmap_axes_sizes:
        return args, []

    def vmap_along_new_axis(arg, vmap_id, axis_size):
        assert isinstance(arg, Computation)
        arg = pjax.reshape(arg, (1,) + arg.shape)
        arg = pjax.repeat(arg, axis_size, axis=0)
        arg = arg.add_vmap_id(vmap_id, 0)
        return arg

    for vmap_id in all_vmap_ids:
        args = [
            arg if vmap_id in arg.vmap_ids else vmap_along_new_axis(arg, vmap_id, vmap_axes_sizes[vmap_id])
            for arg in args
        ]

    return args, all_vmap_ids


def get_vmap_axes(args: Sequence[Computation], vmap_id: int):
    """Returns the axes corresponding to a given vmap id for each argument."""

    def get_vmap_axis(arg):
        preds = arg.vmap_ids[: arg.vmap_ids.index(vmap_id)]
        return len([id for id in preds if id is None or vmap_ids_order.index(id) > vmap_ids_order.index(vmap_id)])

    return tuple([get_vmap_axis(arg) for arg in args])


def hash_kwds(kwds: dict) -> int:
    """Computes a robust hash for a dictionary of keyword arguments."""
    if not kwds:
        return 0  # Return a neutral hash value if no keywords

    try:
        # Sort items by key for consistent hash order and hash directly
        # Convert items view to tuple for hashing
        kw_items = tuple(sorted(kwds.items()))
        return hash(kw_items)
    except TypeError:
        # Fallback: hash string representation for unhashable values
        # Sort items by key for consistent hash order
        kw_items_repr = tuple((k, repr(v)) for k, v in sorted(kwds.items()))
        return hash(kw_items_repr)


def make_computation(name: str, operation: Callable, projection: Callable) -> Callable:
    """Factory for creating a PJAX primitive operation with projection support.

    Args:
        name: name of the operation.
        operation: forward function.
        projection: projection operator.

    Returns:
        wrapped function that creates a computation node.
    """

    @wraps(operation)
    def wrapper(*args, **kwds):
        # if none of the positional arguments are computations, return the value
        if not any(isinstance(arg, Computation) for arg in args):
            return operation(*args, **kwds)

        # handle inputs
        args = arrays_to_inputs(args)
        operation_ = partial(operation, **kwds)
        projection_ = partial(projection, **kwds)

        # vmap the operation and projection if necessary
        args, all_vmap_ids = handle_vmaped(args)

        for vmap_id in all_vmap_ids[::-1]:
            in_axes = get_vmap_axes(args, vmap_id)
            operation_ = jax.vmap(operation_, in_axes=in_axes)
            projection_ = jax.vmap(projection_, in_axes=in_axes + (0,), out_axes=in_axes)

        # create the computation
        value = operation_(*[arg.value for arg in args])
        vmap_ids = all_vmap_ids + [None] * (value.ndim - len(all_vmap_ids))
        static_data_hash = hash_kwds(kwds)
        return Operation(
            name,
            operation_,
            projection_,
            parents=args,
            value=value,
            vmap_ids=vmap_ids,
            static_hash_data=static_data_hash,
        )

    return wrapper


def make_shape_transform(name: str, transform: Callable, inverse: Callable) -> Callable:
    """Factory for creating a PJAX shape transformation operation with inverse.

    Args:
        name: name of the transformation.
        transform: forward transformation function.
        inverse: inverse transformation function.

    Returns:
        wrapped function that creates a shape transformation node.
    """

    @wraps(transform)
    def wrapper(*args, **kwds):
        # if none of the positional arguments are computations, return the value
        if not any(isinstance(arg, Computation) for arg in args):
            return transform(*args, **kwds)

        # handle inputs
        args = arrays_to_inputs(args)
        transform_ = partial(transform, **kwds)
        inverse_ = partial(inverse, **kwds)

        # vmap the transform and inverse if necessary
        args, all_vmap_ids = handle_vmaped(args)

        for vmap_id in all_vmap_ids[::-1]:
            in_axes = get_vmap_axes(args, vmap_id)
            transform_ = jax.vmap(transform_, in_axes=in_axes)
            inverse_ = jax.vmap(inverse_, in_axes=in_axes + (0,), out_axes=in_axes if len(in_axes) > 1 else in_axes[0])

        # create the computation
        value = transform_(*[arg.value for arg in args])
        vmap_ids = all_vmap_ids + [None] * (value.ndim - len(all_vmap_ids))
        static_data_hash = hash_kwds(kwds)
        return ShapeTransform(
            name, transform_, inverse_, parents=args, value=value, vmap_ids=vmap_ids, static_hash_data=static_data_hash
        )

    return wrapper
