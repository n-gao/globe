from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Iterable, Mapping, Tuple

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
from jax.scipy.special import logit

from globe.utils.config import SystemConfigs, group_by_config, split_by_groups


class ParamType(Enum):
    ORBITAL = 'orbital'
    ORBITAL_NUCLEI = 'orbital_nuclei'
    NUCLEI = 'nuclei'
    GLOBAL = 'global'


def identity(x):
    return x


def inverse_softplus(x):
    return jnp.log(jnp.exp(x) - 1)


INVERSE_TRANSFORMS = {
    jnn.softplus: inverse_softplus,
    jnn.tanh: jnp.arctanh,
    jnn.sigmoid: logit,
    identity: identity
}


@dataclass
class ParamSpec:
    param_type: ParamType
    shape: Tuple[int, ...]
    mean: float
    std: float
    transform: Callable = identity
    group: str = None
    segments: int = 1
    keep_distr: bool = False
    use_bias: bool = True


SpecTree = ParamSpec | Iterable['SpecTree'] | Mapping[Any, 'SpecTree']
ParamTree = jax.Array | Iterable['ParamTree'] | Mapping[Any, 'ParamTree']


def group_parameters(params: ParamTree, spectree: SpecTree, config: SystemConfigs, groups: list[str] | None = None) -> ParamTree | list[ParamTree]:
    """
    Groups parameters according to their type and configuration.

    Args:
    - params: a tree of parameters.
    - spectree: a tree of parameter specifications.
    - config: a SystemConfigs object containing the configuration of the system.
    - groups: a list of group names to split the parameters into.

    Returns:
    - A tree of parameters grouped according to their type and configuration.
    """
    def make_generator(param: jax.Array, spec: ParamSpec) -> jax.Array | list[jax.Array]:
        """
        Generates a list of parameters or a single parameter according to its type and configuration.

        Args:
        - param: a parameter.
        - spec: a parameter specification.

        Returns:
        - A list of parameters or a single parameter according to its type and configuration.
        """
        if spec.param_type == ParamType.NUCLEI:
            segment_fn = lambda _, c: len(c)
        elif spec.param_type == ParamType.GLOBAL:
            segment_fn = lambda _, __: 1
        elif spec.param_type == ParamType.ORBITAL:
            segment_fn = lambda s, _: max(s)
        elif spec.param_type == ParamType.ORBITAL_NUCLEI:
            segment_fn = lambda s, c: max(s) * len(c)
        else:
            raise ValueError()
        if groups is not None:
            return split_by_groups(groups, config, param, segment_fn)
        else:
            return group_by_config(config, param, segment_fn)
    return jtu.tree_map(make_generator, params, spectree)
