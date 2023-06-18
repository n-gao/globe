import functools
from collections import defaultdict, namedtuple
from typing import Any, Callable, Generator

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

SystemConfigs = namedtuple('SystemConfigs', ['spins', 'charges'])


@functools.lru_cache()
def nuclei_per_graph(config: SystemConfigs) -> tuple[int]:
    """
    Returns the number of nuclei per graph.
    
    Args:
    - config: SystemConfigs object
    Returns:
    - tuple of ints
    """
    return tuple(len(l) for l in config.charges)


def unique(items: tuple) -> tuple:
    """
    Returns the unique items in a tuple and the indices of the first occurence of each item.
    
    Args:
    - items: tuple of hashable items
    Returns:
    - tuple of unique items
    - tuple of tuples of indices
    - tuple of indices
    - tuple of first indices
    """
    unique = defaultdict(list)
    for i, x in enumerate(items):
        unique[x].append(i)
    return unique.keys(),\
        tuple(unique.values()),\
        np.concatenate(tuple(np.ones_like(c)*i for i, c in enumerate(unique.values()))),\
        [x[0] for x in unique.values()]


def to_individual_tuples(config: SystemConfigs) -> tuple[tuple, ...]:
    """
    Returns the unique tuples of spins and charges in a SystemConfigs object.
    
    Args:
    - config: SystemConfigs object
    Returns:
    - tuple of tuples
    """
    return tuple((s, c) for s, c in zip(*config))


def unique_configs(config: SystemConfigs) -> tuple:
    """
    Returns the unique tuples of spins and charges in a SystemConfigs object and the indices of the first occurence of each tuple.
    
    Args:
    - config: SystemConfigs object
    Returns:
    - tuple of tuples
    - tuple of tuples of indices
    - tuple of indices
    - tuple of first indices
    """
    return unique(to_individual_tuples(config))


def merge_slices(*slices: slice) -> list[slice]:
    """
    Merges adjacent slices. 
    Assumes the slices to be ordered by their starting index and to be non-overlapping.
    
    Args:
    - slices: slices to merge
    Returns:
    - list of slices
    """
    slices = list(slices)
    i = 0
    while i < len(slices) + 1:
        while i+1 < len(slices) and slices[i].stop == slices[i+1].start:
            slices[i] = slice(slices[i].start, slices[i+1].stop)
            del slices[i+1]
        i += 1
    return slices


def group_by_config(
        config: SystemConfigs,
        data: Any,
        size_fn: Callable[[SystemConfigs], int],
        axis: int = 0,
        return_config: bool = False
    ) -> Generator[Any, None, None] | Generator[tuple[Any, SystemConfigs], None, None]:
    """
    Groups data by the unique tuples of spins and charges in a SystemConfigs object.
    
    Args:
    - config: SystemConfigs object
    - data: data to group
    - size_fn: function to compute the size of each group
    - axis: axis along which to group
    - return_config: whether to return the SystemConfigs object of each group
    Returns:
    - generator of tuples of data and SystemConfigs objects if return_config is True, else generator of data
    """
    confs, idx, _, _ = unique_configs(config)

    chunks = [size_fn(s, c) for s, c in zip(config.spins, config.charges)]
    offsets = np.cumsum([0] + chunks)[:-1]
    chunks = np.array(chunks)

    slice_off = (slice(None),) * axis
    
    for (spins, charges), m in zip(confs, idx):
        n = size_fn(spins, charges)
        slices = merge_slices(*[slice(o, o+n) for o in offsets[m]])
        if len(slices) == 1:
            result = jtu.tree_map(
                lambda x: x[(*slice_off, slices[0])].reshape(x.shape[:axis] + (len(m), n) + x.shape[axis+1:]),
                data
            )
        else:
            result = jtu.tree_map(
                lambda x: jnp.concatenate([
                    x[(*slice_off, s)]
                    for s in slices
                ], axis=axis).reshape(x.shape[:axis] + [len(m)] + x.shape[axis:]),
                data
            )
        if return_config:
            yield result, (spins, charges)
        else:
            yield result

def split_by_groups(
    groups: list[list[int]],
    config: SystemConfigs,
    data: Any,
    size_fn: Callable[[SystemConfigs], int],
    axis: int = 0,
    return_config: bool = False
) -> Generator[Any, None, None] | Generator[tuple[Any, SystemConfigs], None, None]:
    """
    Splits data into groups based on the indices provided in `groups`.

    Args:
    - groups: list of lists of indices to group the data by
    - config: SystemConfigs object
    - data: data to group
    - size_fn: function to compute the size of each system
    - axis: axis along which to group
    - return_config: whether to return the SystemConfigs object of each group

    Returns:
    - generator of tuples of data and SystemConfigs objects if return_config is True, else generator of data
    """
    chunks = [size_fn(s, c) for s, c in zip(config.spins, config.charges)]
    offsets = np.cumsum([0] + chunks)[:-1]
    chunks = np.array(chunks)

    slice_off = (slice(None),) * axis
    
    for idx in groups:
        idx = np.array(idx)
        slices = merge_slices(*[slice(o, o+n) for o, n in zip(offsets[idx], chunks[idx])])
        result = jtu.tree_map(
            lambda x: jnp.concatenate([
                x[(*slice_off, s)]
                for s in slices
            ], axis=axis),
            data
        )
        if return_config:
            conf = SystemConfigs(
                tuple(config.spins[i] for i in idx),
                tuple(config.charges[i] for i in idx)
            )
            yield result, conf
        else:
            yield result


def inverse_group_idx(config: SystemConfigs) -> np.ndarray:
    """
    Returns the inverse index of unique indices in `config`.

    Args:
    - config: SystemConfigs object

    Returns:
    - numpy array of inverse indices
    """
    _, idx, _, _ = unique_configs(config)
    _, inv_idx = np.unique(np.concatenate(idx), return_index=True)
    return inv_idx
