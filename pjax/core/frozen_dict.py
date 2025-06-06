from typing import Mapping

from frozendict import frozendict
from jax.tree_util import register_pytree_node_class


@register_pytree_node_class
class FrozenDict(frozendict):
    """A pytree-compatible immutable dictionary keeping insertion order."""

    def tree_flatten(self):
        sorted_keys = sorted(self.keys())
        return [self[k] for k in sorted_keys], (self.keys(), sorted_keys)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        keys, sorted_keys = aux_data
        sorted_dict = dict(zip(sorted_keys, children))
        return cls({k: sorted_dict[k] for k in keys})


def freeze(d: Mapping):
    """Recursively freeze a dictionary."""
    if isinstance(d, FrozenDict) or not isinstance(d, Mapping):
        return d

    return FrozenDict({k: freeze(v) for k, v in d.items()})
