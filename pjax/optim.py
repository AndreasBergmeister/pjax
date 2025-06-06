import warnings
from abc import ABC, abstractmethod
from functools import cache, partial
from typing import Sequence

import jax
import networkx as nx
from jax import numpy as jnp

from .core.computation import Array, Computation, Operation, Parameter, ShapeTransform
from .core.frozen_dict import FrozenDict, freeze


def plot_graph(graph: nx.DiGraph):
    """Plot a computation graph."""
    import matplotlib.pyplot as plt

    graph = prune_shape_transforms(graph)
    A, B = nx.bipartite.sets(prune_shape_transforms(graph))

    # color node; green for partition A, red for partition B, blue for non-partition nodes
    node_color = ["green" if node in A else "red" if node in B else "blue" for node in graph.nodes]

    pos = nx.spring_layout(graph, iterations=1000, seed=0)
    plt.figure(figsize=(10, 8))  # Increase the window size here
    nx.draw(graph, pos, with_labels=True, node_color=node_color, node_size=250, font_size=8)
    plt.show()


def get_graph(computation: Computation):
    """Get the computation graph of a computation."""
    graph = nx.DiGraph()

    def add_to_graph(computation):
        if computation in graph:
            return
        graph.add_node(computation)
        for parent in computation.parents:
            add_to_graph(parent)
            graph.add_edge(parent, computation)

    add_to_graph(computation)
    return graph


@cache
def prune_shape_transforms(graph: nx.DiGraph):
    """Return a copy of the graph without ``ShapeTransform``s.

    Parents of ``ShapeTransform``s are connected to their children.
    """
    graph = graph.copy()
    for node in list(graph.nodes):
        if isinstance(node, ShapeTransform):
            for parent in graph.predecessors(node):
                for child in graph.successors(node):
                    graph.add_edge(parent, child)
            graph.remove_node(node)
    return graph


def get_value(computation: Computation, inputs: FrozenDict):
    """Compute the output value of a computation."""
    if isinstance(computation, Array):
        return computation.value
    if isinstance(computation, Parameter):
        return inputs[computation][0]
    if isinstance(computation, Operation):
        return computation.operation(*inputs[computation])
    if isinstance(computation, ShapeTransform):
        return computation.transform(*[get_value(parent, inputs) for parent in computation.parents])


def gather_and_average_outputs(computation: Computation, inputs: FrozenDict, graph: nx.DiGraph):
    """Gather input values from children and average them."""
    if graph.out_degree(computation) == 0:
        return None

    output_values = [get_childs_input(child, computation, inputs, graph) for child in graph.successors(computation)]
    output_values = jnp.stack([item for sublist in output_values for item in sublist])
    return sum(output_values) / len(output_values)


def get_childs_input(child: Computation, parent: Computation, inputs: FrozenDict, graph: nx.DiGraph):
    """Get the input value(s) of a child from one of its parents.

    This returns a list, as the same parent's output can be input to the child multiple times.
    """
    # get indices of parent in computation.parents
    idxs = [i for i, p in enumerate(child.parents) if p == parent]

    if isinstance(child, Operation):
        return [inputs[child][i] for i in idxs]

    if isinstance(child, ShapeTransform):
        # shape transforms do not have an entry in `inputs` dict
        # we reconstruct the input by recursing to children and inverting the transformation
        transform_inputs = [get_value(parent, inputs) for parent in child.parents]
        output = gather_and_average_outputs(child, inputs, graph)
        values = child.inverse(*transform_inputs, output)

        if isinstance(values, jnp.ndarray):
            return [values]
        return [values[i] for i in idxs]

    raise ValueError(f"unexpected computation type {type(child)} to get input.")


def projection(computation: Operation | Parameter | Array, inputs: FrozenDict, graph: nx.DiGraph):
    """Compute the projection for a computation."""
    if isinstance(computation, Array):
        return inputs

    output = gather_and_average_outputs(computation, inputs, graph)
    projection_inputs = computation.projection(*inputs[computation], output)
    inputs = inputs.set(computation, list(projection_inputs))
    return inputs


def multiple_projection(inputs: FrozenDict, partition: set[Operation | Parameter | Array], graph: nx.DiGraph):
    for computation in partition:
        inputs = projection(computation, inputs, graph)

    # update input of children
    pruned_graph = prune_shape_transforms(graph)
    for computation in set([child for computation in partition for child in pruned_graph.successors(computation)]):
        if isinstance(computation, Operation):
            inputs = inputs.set(computation, [get_value(parent, inputs) for parent in computation.parents])

    return inputs


class Optimizer(ABC):
    """Base class for projection-based optimizers.

    Optimizers partition the computation graph and apply
    iterative projections to find feasible solutions.

    Args:
        steps_per_update: number of optimization steps per update call.
        change_projection_order: whether to reverse the order of projections.

    Attributes:
        steps_per_update: number of steps to perform in each update.
        change_projection_order: flag to control projection ordering.
    """

    def __init__(self, steps_per_update=50, change_projection_order=False):
        self.steps_per_update = steps_per_update
        self.change_projection_order = change_projection_order

    def update(self, fun, params: FrozenDict, steps_per_update=None):
        steps_per_update = steps_per_update or self.steps_per_update

        # initialize variables
        params = {name: Parameter(value, name=name) for name, value in params.items()}
        computation = fun(params)
        graph = get_graph(computation)

        # initialize vars
        inputs = {}
        for node in graph.nodes:
            if isinstance(node, Parameter):
                inputs[node] = [node.value]
            if isinstance(node, Operation):
                inputs[node] = [parent.value for parent in node.parents]
        inputs = freeze(inputs)

        # get bipartition
        pruned_graph = prune_shape_transforms(graph)
        partitions = self._partition(pruned_graph)

        # get projections
        projections = [partial(multiple_projection, partition=partition, graph=graph) for partition in partitions]
        if self.change_projection_order:
            projections = projections[::-1]

        # optimize
        def step(inputs, _):
            new_inputs = self._step(inputs, *projections)
            # compute loss
            diffs = [jnp.mean((x - y) ** 2) for x, y in zip(jax.tree.leaves(inputs), jax.tree.leaves(new_inputs))]
            loss = sum(diffs) / len(diffs)
            return new_inputs, loss

        inputs, losses = jax.lax.scan(step, inputs, length=steps_per_update)

        # update params
        new_params = {}
        for name, computation in params.items():
            if computation in inputs:
                new_params[name] = inputs[computation][0]
            else:
                warnings.warn(f"unused parameter {name}.")
                new_params[name] = params[name].value
        return freeze(new_params), losses.mean()

    @abstractmethod
    def _partition(self, pruned_graph) -> Sequence:
        """Partition the graph into subgraphs."""
        pass

    @abstractmethod
    def _step(self, vars, *projections):
        """Optimization step."""
        pass


class BipartiteOptimizer(Optimizer):
    """Base class for bipartite graph optimizers.

    Specializes the base optimizer for bipartite computation graphs. Partitions
    the graph into two sets and applies alternating projections between them.

    Inherits all parameters from ``Optimizer`` base class.
    """

    def _partition(self, pruned_graph):
        try:
            partition_a, partition_b = nx.bipartite.sets(pruned_graph)
        except nx.NetworkXError:
            raise ValueError("Computation graph is not bipartite.")

        # make sure output node is in partition a
        if any(pruned_graph.out_degree(node) == 0 for node in partition_b):
            partition_a, partition_b = partition_b, partition_a

        return partition_a, partition_b


class AlternatingProjections(BipartiteOptimizer):
    """Alternating projections optimizer for bipartite graphs.

    Implements the classical alternating projections algorithm that alternately
    projects onto two constraint sets.

    Inherits all parameters from ``BipartiteOptimizer``.
    """

    def _step(self, vars, projection_a, projection_b):
        return projection_b(projection_a(vars))


class AlternatingReflections(BipartiteOptimizer):
    """Alternating reflections optimizer using reflection operators.

    Uses reflections :math:`R = 2P - I` instead of projections.

    Inherits all parameters from ``BipartiteOptimizer``.
    """

    def _step(self, vars, projection_a, projection_b):
        def reflection(projection, vars):
            return jax.tree.map(lambda x, y: 2.0 * x - y, projection(vars), vars)

        return reflection(projection_b, reflection(projection_a, vars))


class DouglasRachford(BipartiteOptimizer):
    """Douglas-Rachford optimizer for bipartite constraint satisfaction.

    Implements the Douglas-Rachford algorithm using reflection operators:
    :math:`D_\\lambda = (1 - \\lambda)I + \\lambda R_B R_A`, where :math:`R = 2P - I`
    and :math:`\\lambda` is the relaxation parameter.

    Args:
        steps_per_update: number of optimization steps per update call.
        change_projection_order: whether to reverse the order of projections.
        relaxation: relaxation parameter :math:`\\lambda \\in (0, 1]`, controls step size and convergence.

    Attributes:
        relaxation: the relaxation parameter for the Douglas-Rachford iteration.
    """

    def __init__(self, steps_per_update=50, change_projection_order=False, relaxation=0.5):
        super().__init__(steps_per_update, change_projection_order)
        self.relaxation = relaxation

    def _step(self, vars, projection_a, projection_b):
        def reflection(projection, vars):
            return jax.tree.map(lambda x, y: 2.0 * x - y, projection(vars), vars)

        return jax.tree.map(
            lambda x, y: (1.0 - self.relaxation) * x + self.relaxation * y,
            vars,
            reflection(projection_b, reflection(projection_a, vars)),
        )


class DifferenceMap(BipartiteOptimizer):
    """Difference map optimizer for challenging feasibility problems.

    Implements the difference map algorithm:
    :math:`D(x) = x + \\beta [P_A(f_B(x)) - P_B(f_A(x))]`, where :math:`f_A` and :math:`f_B` are auxiliary maps.

    When :math:`\\beta = 1`: :math:`D(x) = x + P_A(2 P_B(x) - x) - P_B(x)`.

    Args:
        steps_per_update: number of optimization steps per update call.
        change_projection_order: whether to reverse the order of projections.
        beta: algorithm parameter controlling auxiliary map behavior.

    Attributes:
        beta: the difference map parameter, typically set to 1.0.
    """

    def __init__(self, steps_per_update=50, change_projection_order=False, beta=1.0):
        super().__init__(steps_per_update, change_projection_order)
        self.beta = beta

    def _step(self, vars, projection_a, projection_b):
        if self.beta == 1.0:
            p_b = projection_b(vars)
            inter = jax.tree.map((lambda b, x: 2.0 * b - x), p_b, vars)
            p_a = projection_a(inter)
            return jax.tree.map((lambda x, a, b: x + a - b), vars, p_a, p_b)

        p_a = projection_a(vars)
        p_b = projection_b(vars)
        f_a = jax.tree.map((lambda a, x: (1.0 - 1.0 / self.beta) * a + 1.0 / self.beta * x), p_a, vars)
        f_b = jax.tree.map((lambda b, x: (1.0 + 1.0 / self.beta) * b - 1.0 / self.beta * x), p_b, vars)
        p_f_a = projection_a(f_b)
        p_f_b = projection_b(f_a)
        return jax.tree.map((lambda x, a, b: x + self.beta * (a - b)), vars, p_f_a, p_f_b)


class CyclicOptimizer(Optimizer):
    """Base class for cyclic optimizers.

    Extends optimization to non-bipartite graphs by partitioning nodes into layers
    and applying projections cyclically. Uses breadth-first search to create
    topologically ordered partitions starting from output nodes.

    Inherits all parameters from ``Optimizer`` base class.
    """

    def _partition(self, pruned_graph):
        # first partition is all output nodes
        partitions = [set(node for node in pruned_graph if pruned_graph.out_degree(node) == 0)]
        # perform bfs to get partitions
        while any(pruned_graph.in_degree(node) != 0 for node in partitions[-1]):
            partition = set([predecessor for node in partitions[-1] for predecessor in pruned_graph.predecessors(node)])
            partitions.append(partition)
        return partitions


class CyclicProjections(CyclicOptimizer):
    """Cyclic alternating projections optimizer.

    Applies projections sequentially across all partitions in topological order.

    Inherits all parameters from ``CyclicOptimizer``.
    """

    def _step(self, vars, *projections):
        for projection in projections:
            vars = projection(vars)
        return vars


class CyclicDouglasRachford(CyclicOptimizer):
    """Cyclic Douglas-Rachford optimizer.

    Generalizes Douglas-Rachford to non-bipartite graphs by applying the algorithm
    cyclically: D(x) = D_n ... D_1(x), where D_i = 0.5 (I + R_{i+1 mod n} R_i).

    Inherits all parameters from ``CyclicOptimizer``.
    """

    def _step(self, vars, *projections):
        def reflection(projection, vars):
            return jax.tree.map(lambda x, y: 2.0 * x - y, projection(vars), vars)

        def d_r(vars, projection_a, projection_b):
            return jax.tree.map(
                lambda x, y: 0.5 * (x + y), vars, reflection(projection_b, reflection(projection_a, vars))
            )

        for i in range(len(projections)):
            vars = d_r(vars, projections[i], projections[(i + 1) % len(projections)])

        return vars
