import inspect
import itertools
from collections.abc import Iterable
from typing import Any, Callable, Generator, NamedTuple, TypeVar

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from .config import *


def safe_call(fn: Callable, *args, **kwargs) -> Any:
    """
    Calls the function with the given arguments, but only passes the arguments
    that the function accepts.

    Args:
    - fn: the function to call
    - args: positional arguments to pass to the function
    - kwargs: keyword arguments to pass to the function
    """
    params = inspect.signature(fn).parameters
    return fn(*args, **{
        k: v for k, v in kwargs.items()
        if k in params
    })


def tree_generator_zip(*args: Iterable) -> Generator:
    """
    Zips together generators that yield trees of the same structure.

    Args:
    - args: tree of generators to zip together
    Return:
    - generator that yields trees of the same structure
    """
    generators, treedef = jtu.tree_flatten(args, inspect.isgenerator)
    for r in zip(*generators):
        yield jtu.tree_unflatten(treedef, r)


def tree_zip(*args: Iterable) -> Generator:
    """
    Zips the leaves of a tree together.

    Args:
    - args: tree of lists to zip together
    Return:
    - generator that yields trees of the same structure
    """
    lists, treedef = jtu.tree_flatten(args, is_leaf_list)
    for r in zip(*lists):
        yield jtu.tree_unflatten(treedef, r)


def adj_idx(
        a_sizes: tuple[int, ...],
        b_sizes: tuple[int, ...] | None = None,
        drop_diagonal: bool = False,
        drop_off_block: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the indices of the adjacency matrix of a block matrix.
    
    Args:
    - a_sizes: sizes of the blocks in the first dimension
    - b_sizes: sizes of the blocks in the second dimension
    - drop_diagonal: whether to drop the diagonal of each block
    - drop_off_block: whether to drop the off-diagonal blocks
    Return:
    - i: row indices of the adjacency matrix
    - j: column indices of the adjacency matrix
    - m: indices of the blocks
    """
    if b_sizes is None:
        b_sizes = a_sizes
    assert np.allclose(a_sizes, b_sizes) or not drop_diagonal
    i, j, m = [], [], []
    off_a, off_b = 0, 0
    for k, (a, b) in enumerate(zip(a_sizes, b_sizes)):
        adj = np.ones((a, b))
        if drop_off_block:
            adj = np.triu(adj)
        if drop_diagonal:
            adj -= np.eye(a)
        _i, _j = np.where(adj)
        i.append(_i + off_a)
        j.append(_j + off_b)
        m.append(np.ones(_i.size, dtype=int) * k)
        off_a += a
        off_b += b
    i = np.concatenate(i, axis=0)
    j = np.concatenate(j, axis=0)
    m = np.concatenate(m, axis=0)
    return i, j, m


def triu_idx(sizes: tuple[int, ...], k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the indices of the upper triangular part of a block matrix.
    
    Args:
    - sizes: sizes of the blocks
    - k: diagonal offset
    Return:
    - i: row indices of the upper triangular part
    - j: column indices of the upper triangular part
    - m: indices of the blocks
    """
    i, j, m = [], [], []
    off = 0
    for n, size in enumerate(sizes):
        _i, _j = np.where(np.triu(np.ones((size, size)), k=k))
        i.append(_i + off)
        j.append(_j + off)
        m.append(np.ones(_i.size, dtype=int) * n)
        off += size
    i = np.concatenate(i, axis=0)
    j = np.concatenate(j, axis=0)
    m = np.concatenate(m, axis=0)
    return i, j, m


_split = jax.jit(jnp.split, static_argnums=1, static_argnames='axis')
def iterate_segments(x: jax.Array, segments: np.ndarray, axis: int = 0) -> list[jax.Array]:
    segments = tuple(np.cumsum(segments))
    return _split(x, segments, axis=axis)


def totuple(x: Iterable) -> tuple:
    """
    Converts an iterable recursively to a tuple.
    """
    try:
        return tuple(totuple(x_) for x_ in x)
    except:
        return x


def flatten(x: Iterable) -> Generator:
    """
    Flattens an iterable recursively.
    
    Args:
    - x: iterable to flatten
    Return:
    - generator that yields the flattened iterable
    """
    try:
        for i in x:
            for r in flatten(i):
                yield r
    except:
        yield x


def argsort(items: Iterable) -> list[int]:
    """
    Returns the indices that would sort a list.
    
    Args:
    - items: list to sort
    Return:
    - indices that would sort the list
    """
    return sorted(range(len(items)), key=items.__getitem__, reverse=True)


def get_attrs(items: Iterable, *attr: str) -> tuple:
    """
    Returns the attributes of a list of objects.
    
    Args:
    - items: list of objects
    - attr: attributes to get
    Return:
    - tuple of attributes
    """
    if len(attr) == 1:
        return tuple(getattr(i, attr[0]) for i in items)
    else:
        return tuple(
            tuple(getattr(i, a) for i in items)
            for a in attr
        )


def itemgetter(*items: Any) -> Callable[[Iterable[Any]], tuple]:
    """
    Implementation of itemgetter that always returns a tuple.
    
    Args:
    - items: items to get
    Return:
    - function that returns a tuple of the items
    """
    def g(obj):
        return tuple(obj[item] for item in items)
    return g


def chain(args: Iterable[Iterable]) -> tuple:
    """
    Chains a list of lists into a single list.
    
    Args:
    - args: list of lists
    Return:
    - single tuple
    """
    return tuple(itertools.chain(*args))


def np_segment_sum(data: np.ndarray, segment_ids: np.ndarray) -> np.ndarray:
    """
    Computes the sum of a numpy array along segments.
    
    Args:
    - data: array to sum
    - segment_ids: segment ids
    Return:
    - array of sums
    """
    data = np.asarray(data)
    s = np.zeros((np.max(segment_ids)+1,) + data.shape[1:], dtype=data.dtype)
    np.add.at(s, segment_ids, data)
    return s


def make_depth_equality_check(target_depth: int) -> Callable[[dict], bool]:
    """
    Returns a function that checks if the depth of a dict of dicts is equal to a target depth.
    
    Args:
    - target_depth: target depth
    Return:
    - function that checks if the depth of a dict of dicts is equal to a target depth
    """
    # Function that computes the depth of a dict of dicts with a target_depth+1 as max.
    def capped_depth(x, depth=0):
        if not isinstance(x, dict) or depth > target_depth:
            return depth

        # We use a loop instead of generator setup with max() to be more efficient.
        max_depth = 0
        for key in x:
            cap_depth = capped_depth(x[key], depth + 1)
            # If any child is deeper than the target we can stop.
            if cap_depth > target_depth:
                return cap_depth
            # update maximum depth
            if cap_depth > max_depth:
                max_depth = cap_depth
        return max_depth
    
    def depth_equal(x):
        return capped_depth(x) == target_depth
    return depth_equal


def is_leaf_list(x: Any) -> bool:
    """
    Checks if an object is a list of leaves.
    
    Args:
    - x: object to check
    Return:
    - True if the object is a list of leaves, False otherwise
    """
    if not isinstance(x, Iterable):
        return True
    if isinstance(x, (jax.Array, np.ndarray)):
        return True
    if isinstance(x, dict):
        return not any(isinstance(x_, Iterable) for x_ in x.values())
    else:
        return not any(isinstance(x_, (tuple, list)) for x_ in x)


def to_np(x: Any) -> Any:
    """
    Converts a pytree to numpy.
    
    Args:
    - x: pytree to convert
    Return:
    - pytree with numpy arrays
    """
    return jtu.tree_map(np.asarray, x)


def to_jnp(x: Any) -> Any:
    """
    Converts a pytree to jax.numpy.
    
    Args:
    - x: pytree to convert
    Return:
    - pytree with jax.numpy arrays
    """
    return jtu.tree_map(jnp.asarray, x)


T = TypeVar("T")
class EMAState(NamedTuple):
    data: T
    weight: jax.Array


# EMA for usage in JAX
def ema_make(tree: T) -> EMAState:
    """
    Creates an EMA state for a pytree.
    
    Args:
    - tree: pytree to create the EMA state for
    Return:
    - EMA state
    """
    return EMAState(jtu.tree_map(jnp.zeros_like, tree), jnp.zeros(()))


@jax.jit
def ema_update(data: EMAState, value: T, decay: float = 0.9) -> EMAState:
    """
    Updates an EMA state with a new value.
    
    Args:
    - data: EMA state
    - value: value to update the EMA state with
    - decay: decay rate
    Return:
    - updated EMA state
    """
    tree, weight = data
    return jtu.tree_map(lambda a, b: a*decay + b, tree, value), weight*decay + 1


@jax.jit
def ema_value(data: EMAState, backup: T = None) -> T:
    """
    Computes the EMA value of an EMA state.
    
    Args:
    - data: EMA state
    - backup: backup value to use if the weight is 0
    Return:
    - EMA value
    """
    tree, weight = data
    if backup is None:
        backup = tree
    is_nan = weight == 0
    return jtu.tree_map(lambda x, y: jnp.where(is_nan, y, x/weight), tree, backup)


def comp_clouds(x: np.ndarray, y: np.ndarray, c1: np.ndarray, c2: np.ndarray) -> bool:
    """
    Compares two point clouds by comparing their sets pairwise distances. 
    Note: This is not exact and may result in false positives.
    
    Args:
    - x: first point cloud
    - y: second point cloud
    - c1: first point cloud's weights
    - c2: second point cloud's weights
    Return:
    - True if the point clouds are equal, False otherwise
    """
    # This is a non-perfect comparison between point clouds
    # It may fail if one of the geometries has two identical eigenvalues
    s1 = set(np.linalg.norm(x[:, None] - x, axis=-1).reshape(-1))
    s2 = set(np.linalg.norm(y[:, None] - y, axis=-1).reshape(-1))
    if s1 != s2:
        return False
    c1 = np.array(c1)[:, None]
    c2 = np.array(c2)[:, None]
    s1 = set(np.linalg.norm((x*c1)[:, None] - (x*c1), axis=-1).reshape(-1))
    s2 = set(np.linalg.norm((y*c2)[:, None] - (y*c2), axis=-1).reshape(-1))
    return s1 == s2
