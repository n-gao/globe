import functools
from dataclasses import replace
from typing import Callable

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from globe.nn import Activation, ActivationWithGain, Dense, ParamTree, ReparametrizedModule
from globe.nn.parameters import ParamSpec, ParamType, SpecTree, inverse_softplus
from globe.systems.element import CORE_OFFSETS, MAX_CORE_ORB, VALENCY
from globe.utils import (SystemConfigs, chain, flatten, group_by_config,
                           itemgetter, np_segment_sum, tree_generator_zip)
from globe.utils.config import (SystemConfigs, group_by_config,
                                  inverse_group_idx, nuclei_per_graph)


def isotropic_envelope(x: jax.Array, sigma: jax.Array, pi: jax.Array, pi_scale: jax.Array) -> jax.Array:
    """
    Computes the isotropic envelope of the orbitals.

    Args:
    - x (Array): The input tensor of shape (..., n_elec, n_nuc, 4) containing the electron-nucleus distances.
    - sigma (Array): The tensor of shape (n_nuc, n_det) containing the sigma parameters for each nucleus and determinant.
    - pi (Array): The tensor of shape (n_nuc, n_det) containing the pi parameters for each nucleus and determinant.
    - pi_scale (Array): The tensor of shape (n_nuc, n_det) containing the pi scale parameters for each nucleus and determinant.

    Returns:
    - The tensor of shape (..., n_elec, n_det) containing the isotropic envelope of the orbitals.
    """
    # We must reshape here because we get sigma and pi as (n_nuc*n_orbs, n_det) from the GNN
    sigma = sigma.reshape(x.shape[-2], -1)
    pi = pi.reshape(x.shape[-2], -1)
    pi_scale = pi_scale.reshape(x.shape[-2], -1)
    pi = pi*pi_scale
    # sum_m exp(- nmd * md) * md -> nd
    return jnp.sum(jnp.exp(-x[..., -1:] * sigma) * pi, axis=-2)


def are_orbitals_shared(params: ParamTree) -> bool:
    return 'eq' in params


def is_full_det(params: ParamTree) -> bool:
    if 'eq' in params:
        return 'neq' in params
    return 'up_neq' in params


def number_of_determinants(params: ParamTree) -> int:
    key = 'eq' if are_orbitals_shared(params) else 'up_eq'
    return params[key]['envelope']['sigma'].shape[-1]


def group_orbital_params(params: ParamTree, config: SystemConfigs) -> ParamTree:
    """
    Groups the orbital parameters by configuration.

    Args:
    - params: The parameters of the orbitals.
    - config: The configuration of the system.

    Returns:
    - The parameters of the orbitals grouped by configuration.
    """
    spins = np.array(config.spins)
    def group_params_by_config(param):
        if param.shape[0] == spins.max(-1).sum():
            return group_by_config(config, param, lambda s, _: max(s))
        else:
            return group_by_config(config, param, lambda s, c: max(s)*len(c))
    return jtu.tree_map(group_params_by_config, params)


def eval_orbitals(orbital_fn, params: ParamTree, h_one: jax.Array, r_im: jax.Array, config: SystemConfigs) -> list[tuple[jax.Array, ...]]:
    """
    Evaluates the orbitals.

    Args:
    - orbital_fn: The orbital function.
    - params: The parameters of the orbitals.
    - h_one: The one-electron features.
    - r_im: The electron-nucleus distances.
    - config: The configuration of the system.

    Returns:
    - The evaluated orbitals.
    """
    result = []
    for h, (r, (spins, charges)), param in tree_generator_zip(
        group_by_config(config, h_one, lambda s, _: np.sum(s)),
        group_by_config(config, r_im, lambda s, c: np.sum(s)*len(c), return_config=True),
        group_orbital_params(params, config)
    ):
        result.append(orbital_fn(param, h, r, spins, charges))
    return result


def make_orbital_fn(orbital_fn: Callable[..., jax.Array], shared_orbitals: bool, full_det: bool, init_fn = None) -> Callable[..., tuple[jax.Array, ...]]:
    """
    Constructs a function that computes the orbital matrix.

    Args:
    - orbital_fn: The orbital function.
    - shared_orbitals: Whether the orbitals are shared.
    - full_det: Whether the full determinant is used.
    - init_fn: A function to initialize orbital function parameters.

    Returns:
    - A function that evaluates the orbital matrix.
    """
    @functools.partial(jax.vmap, in_axes=(0, 0, 0, None, None))
    def _orbitals(params, h_one, r_im, spins, charges):
        args = init_fn() if init_fn is not None else {}
        n_nuc = len(charges)
        n_elec = h_one.shape[0]
        r_im = r_im.reshape(n_elec, n_nuc, -1)
        
        if shared_orbitals:
            uu, dd = jnp.split(
                    orbital_fn(h_one, r_im, **params['eq'], **args)\
                    .reshape(n_elec, max(spins), -1),
                spins[:1],
                axis=0
            )
            if full_det:
                ud, du = jnp.split(
                        orbital_fn(h_one, r_im, **params['neq'], **args)\
                        .reshape(n_elec, max(spins), -1),
                    spins[:1],
                    axis=0
                )
        else:
            h_up, h_down = jnp.split(h_one, spins[:1])
            r_up, r_down = jnp.split(r_im, spins[:1])
            uu = orbital_fn(h_up, r_up, **params['up_eq'], **args)\
                .reshape(spins[0], max(spins), -1)
            dd = orbital_fn(h_down, r_down, **params['down_eq'], **args)\
                .reshape(spins[1], max(spins), -1)
            if full_det:
                ud = orbital_fn(h_up, r_up, **params['up_neq'], **args)\
                    .reshape(spins[0], max(spins), -1)
                du = orbital_fn(h_down, r_down, **params['down_neq'], **args)\
                    .reshape(spins[1], max(spins), -1)
        
        if full_det:
            orbitals = jnp.concatenate([
                jnp.concatenate([uu[:, :spins[0]], ud[:, :spins[1]]], axis=1),
                jnp.concatenate([du[:, :spins[0]], dd[:, :spins[1]]], axis=1),
            ], axis=0)
            orbitals = (orbitals,)
        else:
            orbitals = (uu[:, :spins[0]], dd[:, :spins[1]])
        return tuple(o.transpose(2, 0, 1) for o in orbitals)
    return _orbitals


class OrbitalModule(ReparametrizedModule):
    """
    Base class for all orbital modules.

    Attributes:
    None

    Methods:
    param_spec(shared_orbitals: bool, full_det: bool, orbital_inp: int, determinants: int) -> Dict[str, Dict[str, ParamSpec]]:
        Returns a dictionary of parameter specifications for the orbital module.

    __call__(self, params: ParamTree, h_one: jax.Array, r_im: jax.Array, config: SystemConfigs) -> jax.Array:
        Computes the output of the orbital module.
    """
    @staticmethod
    def param_spec(shared_orbitals: bool, full_det: bool, orbital_inp: int, determinants: int) -> SpecTree:
        """
        Returns a dictionary of parameter specifications for the orbital module.

        Args:
        shared_orbitals: bool
            Whether the orbitals are shared among determinants.
        full_det: bool
            Whether the full determinant is used.
        orbital_inp: int
            The input dimension of the orbitals.
        determinants: int
            The number of determinants.

        Returns:
        SpecTree:
            A dictionary of parameter specifications for the orbital module.
        """
        keys = ('eq', 'neq') if full_det else ('eq',)
        restricted_spec = {
            k: dict(
                orbital_embedding=ParamSpec(
                    ParamType.ORBITAL,
                    shape=(determinants, orbital_inp,),
                    mean=0,
                    std=1/np.sqrt(orbital_inp),
                    segments=determinants,
                    keep_distr=True,
                    group='orb_embedding'
                ),
                orbital_bias=ParamSpec(
                    ParamType.ORBITAL,
                    shape=(determinants,),
                    mean=0,
                    std=0.1,
                    keep_distr=True,
                    group='orb_bias'
                ),
                envelope=dict(
                    sigma=ParamSpec(
                        ParamType.ORBITAL_NUCLEI,
                        shape=(determinants,),
                        mean=1.0,
                        std=0.1,
                        transform=jnn.softplus,
                        keep_distr=k == 'neq',
                        group='env_sigma'
                    ),
                    pi=ParamSpec(
                        ParamType.ORBITAL_NUCLEI,
                        shape=(determinants,),
                        mean=0.0,
                        std=0.5 if k == 'eq' else 1e-2,
                        transform=jnn.tanh,
                        keep_distr=k == 'neq',
                        group='env_pi',
                        use_bias=False,
                    ),
                    pi_scale=ParamSpec(
                        ParamType.ORBITAL_NUCLEI,
                        shape=(determinants,),
                        mean=inverse_softplus(1.0),
                        std=0.1,
                        use_bias=True,
                        transform=jnn.softplus,
                        keep_distr=k == 'neq',
                        group='env_pi_scale'
                    )
                )
            ) for k in keys
        }
        if shared_orbitals:
            return restricted_spec
        else:
            # For unrestricted we need double the number of parameters
            return {
                f'{spin}_{k}': restricted_spec[k]
                for k in restricted_spec
                for spin in ('up', 'down')
            }

    def __call__(self, params, h_one, r_im, config):
        raise NotImplementedError()


class AdditiveOrbitals(OrbitalModule):
    """
    Class representing additive orbitals.
    phi_i(r_j) = sigma(h_j[:, None] + w_i)W_i + b_i
    
    Attributes:
    activation: Activation function to use.
    separate_k: Whether to separate the determinants dimension from the input dimension.
    """
    activation: Activation
    separate_k: bool

    @staticmethod
    def param_spec(shared_orbitals, full_det, orbital_inp, determinants, segment, separate_k):
        result = OrbitalModule.param_spec(shared_orbitals, full_det, orbital_inp, determinants, segment)
        for k in result:
            result[k]['orbital_embedding'] = replace(
                    result[k]['orbital_embedding'],
                    shape=(orbital_inp,) if separate_k else (determinants, orbital_inp),
                    std=0.1 if 'neq' in k else 1.0
                )
        return result

    @nn.compact
    def __call__(self, params, h_one, r_im, config):
        spins = config.spins
        shared_orbitals = are_orbitals_shared(params)
        full_det = is_full_det(params)
        n_det = number_of_determinants(params)
        act = ActivationWithGain(self.activation)

        # This cumbersome initialization should ensure that 
        # the off diagonals are 0 if full_det=True
        weight_shape = (n_det, h_one.shape[-1])
        # fan_out instead of fan_in because we have h_one.shape[-1] as last dimension here.
        # We initialize it with a small standard deviation to get results closer to HF.
        weight_init = functools.partial(jnn.initializers.variance_scaling, 0.1, 'fan_out', 'truncated_normal')
        if shared_orbitals:
            if full_det:
                initializers = [weight_init(), jnn.initializers.zeros]
            else:
                initializers = [weight_init()]
        else:
            if full_det:
                initializers = [weight_init(), weight_init(), jnn.initializers.zeros, jnn.initializers.zeros]
            else:
                initializers = [weight_init(), weight_init()]
        weights = tuple(self.param(
            f'w_{i}',
            init_fn,
            weight_shape
        ) for i, init_fn in enumerate(initializers))

        # Define orbital functions
        def _init_fn():
            return {'weight_iter': iter(weights)}

        def _orbital_fn(h, r, orbital_embedding, orbital_bias, envelope, weight_iter):
            if self.separate_k:
                orbital_embedding = orbital_embedding.reshape(-1, h.shape[-1])
            else:
                orbital_embedding = orbital_embedding.reshape(-1, n_det, h.shape[-1])
            orbital_bias = orbital_bias.reshape(-1, n_det)
            norm = np.sqrt(2.0)
            # We normalize before because it's cheaper
            if self.separate_k:
                orb_inp = act(h[:, None]/norm + orbital_embedding/norm)
            else:
                orb_inp = act(h[:, None, None]/norm + orbital_embedding/norm)
            W = next(weight_iter)
            # n number of electrons, o number of orbitals, k number of determinants, d feature dim
            # If we have separate weights for upup, updown and downup, downdown, we must repeat these.
            inp_str = 'nod' if self.separate_k else 'nokd'
            orb = jnp.einsum(f'{inp_str},kd->nok', orb_inp, W) + orbital_bias
            orb *= isotropic_envelope(r, **envelope).reshape(orb.shape)
            return orb
        
        orbital_fn = make_orbital_fn(_orbital_fn, shared_orbitals, full_det, _init_fn)
        return eval_orbitals(orbital_fn, params, h_one, r_im, config)


class ProductOrbitals(OrbitalModule):
    """
    Class representing product orbitals.
    phi_i(r_j) = h_j^T*w_i + b_i
    """

    @nn.compact
    def __call__(self, params, h_one, r_im, config):
        shared_orbitals = are_orbitals_shared(params)
        full_det = is_full_det(params)
        n_det = number_of_determinants(params)

        def _orbital_fn(h, r, orbital_embedding, orbital_bias, envelope):
            orbital_embedding = orbital_embedding.reshape(-1, n_det, h.shape[-1])
            orbital_bias = orbital_bias.reshape(-1, n_det)
            # n -> n_elec, o -> n_orb, k -> n_det, d -> inp_dim
            orb = jnp.einsum('nd,okd->nok', h, orbital_embedding) + orbital_bias
            orb *= isotropic_envelope(r, **envelope).reshape(orb.shape)
            return orb
        
        orbital_fn = make_orbital_fn(_orbital_fn, shared_orbitals, full_det)
        return eval_orbitals(orbital_fn, params, h_one, r_im, config)


class EnvelopeOnlyOrbitals(OrbitalModule):
    """
    Class representing envelope-only orbitals, i.e., phi_i(r_j) = (sum_k h_ik) * envelope_i(r_j) 
    """
    @staticmethod
    def param_spec(shared_orbitals, full_det, orbital_inp, determinants, segment):
        result = OrbitalModule.param_spec(shared_orbitals, full_det, orbital_inp, determinants, segment)
        for k in result:
            del result[k]['orbital_embedding']
        return result

    @nn.compact
    def __call__(self, params, h_one, r_im, config):
        shared_orbitals = are_orbitals_shared(params)
        full_det = is_full_det(params)
        n_det = number_of_determinants(params)

        dense = Dense(n_det)

        def _orbital_fn(h, r, orbital_bias, envelope):
            orb = dense(h).reshape(-1, 1, n_det) + orbital_bias
            orb *= isotropic_envelope(r, **envelope).reshape(orb.shape[0], -1, n_det)
            return orb
        
        orbital_fn = make_orbital_fn(_orbital_fn, shared_orbitals, full_det)
        return eval_orbitals(orbital_fn, params, h_one, r_im, config)


def _get_orbital_edges(nuc: jax.Array, valency: tuple[int, ...]) -> tuple[jax.Array, jax.Array, int]:
    """
    Given a set of nuclei and their valencies, returns the indices of the edges
    connecting the nuclei in the molecular graph.

    Args:
    - nuc: A 2D array of shape (n_atoms, 3) representing the positions of the nuclei.
    - valency: A tuple of length n_atoms representing the valency of each atom.

    Returns:
    - A tuple of three elements:
        - idx_i: A 1D array of shape (n_edges,) representing the indices of the first node of each edge.
        - idx_j: A 1D array of shape (n_edges,) representing the indices of the second node of each edge.
        - N: An integer representing the total number of edges in the molecular graph.
    """
    idx_i, idx_j = jnp.triu_indices(len(valency))
    n_iter = int(np.ceil(sum(valency) / 2))
    nuc = jnp.array(nuc, dtype=jnp.float32)
    valency = jnp.array(valency, dtype=jnp.float32)
    dists = jnp.linalg.norm(nuc[:, None] - nuc, axis=-1)
    dists += jnp.eye(len(valency)) * 15
    dists = dists[idx_i, idx_j]
    counts = jnp.zeros_like(dists, dtype=jnp.int32)

    centered = nuc - nuc.mean(0)
    edge_pos = (centered[:, None] + centered)/2
    xyz = edge_pos[idx_i, idx_j]
    r = jnp.linalg.norm(xyz, axis=-1)

    def _select_next(carry, _):
        val, counts = carry
        valid = (val[idx_i] > 0) * (val[idx_j] > 0)
        scores = valid / (dists + counts.astype(dists.dtype)/2)
        mask = jnp.ones_like(scores, dtype=bool)
        for crit in [scores, r, xyz[..., 0], xyz[..., 1], xyz[..., 2]]:
            crit = jnp.where(mask, crit, -np.inf)
            max_val = crit.max()
            mask = jnp.abs(crit - max_val) < 1e-5
        idx = jnp.argmax(mask)

        count = counts[idx]
        counts = counts.at[idx].add(1)
        val = val.at[idx_i[idx]].add(-1).at[idx_j[idx]].add(-1)
        return (val, counts), (idx_i[idx], idx_j[idx], count)

    (valency, counts), (first, second, N) = jax.lax.scan(_select_next, (valency, counts), np.arange(n_iter))
    return first, second, N


def get_valence_orbitals(nuclei: jax.Array, valency: tuple[int, ...], config: SystemConfigs) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Given a set of nuclei and their valencies, returns the indices of the edges
    connecting the nuclei in the molecular graph.

    Args:
    - nuclei: A 2D array of shape (n_atoms, 3) representing the positions of the nuclei.
    - valency: A tuple of length n_atoms representing the valency of each atom.
    - config: A SystemConfigs object representing the configuration of the system.

    Returns:
    - A tuple of four elements:
        - idx_i: A 1D array of shape (n_edges,) representing the indices of the first node of each edge.
        - idx_j: A 1D array of shape (n_edges,) representing the indices of the second node of each edge.
        - type: A 1D array of shape (n_edges,) representing the bond type.
        - counts: A 1D array of shape (n_graphs,) representing the number of orbitals per graph.
    """
    n_nuclei = nuclei_per_graph(config)
    offsets = np.cumsum([0, *n_nuclei[:-1]])
    valency = np.array(valency)

    # We process all graphs with the same signature simultaneously
    idx_i, idx_j, N = [], [], []
    for nuc, val, off in zip(
            group_by_config(config, nuclei, lambda _, c: len(c)),
            group_by_config(config, valency, lambda _, c: len(c)),
            group_by_config(config, offsets, lambda *_: 1)):
        i, j, n = jax.vmap(_get_orbital_edges, in_axes=(0, None))(nuc, tuple(val[0]))
        idx_i.append(i + off)
        idx_j.append(j + off)
        N.append(n)
    # Also keep track of the number of orbitals per graph
    counts = jtu.tree_map(lambda x: np.full((x.shape[0],), x.shape[1], dtype=np.int32), N)
    # We have to rearrange the output here to match the input order
    chained = tuple(map(chain, (idx_i, idx_j, N, counts)))
    reverse_idx = inverse_group_idx(config)
    result = [itemgetter(*reverse_idx)(r) for r in chained]
    return (*tuple(map(jnp.concatenate, result[:-1])), np.stack(result[-1]))


def get_core_orbitals(nuclei: jax.Array, config: SystemConfigs) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Given a set of nuclei, returns the core orbitals for all molecular graphs.

    Args:
    - nuclei: A 2D array of shape (n_atoms, 3) representing the positions of the nuclei.
    - config: A SystemConfigs object representing the configuration of the system.

    Returns:
    - A tuple of five elements:
        - core_i: A 1D array of shape (n_core_orbitals,) representing the indices of the nuclei associated with each core orbital.
        - core_loc: A 2D array of shape (n_core_orbitals, 3) representing the positions of the core orbitals.
        - core_type: A 1D array of shape (n_core_orbitals,) representing the type of each core orbital.
        - core_N: A 1D array of shape (n_core_orbitals,) representing the index of each core orbital within its associated nucleus.
        - core_N_orb: A 1D array of shape (n_graphs,) representing the number of core orbitals per graph.
    """
    n_nuclei = nuclei_per_graph(config)
    flat_charges = np.array(tuple(flatten(config.charges)))
    valency = np.array(itemgetter(*flat_charges)(VALENCY))
    n_core_orbitals = (flat_charges - valency) // 2
    core_loc = nuclei.repeat(n_core_orbitals, axis=0)
    core_type = np.concatenate(([
        np.arange(o) + CORE_OFFSETS[c]
        for o, c in zip(n_core_orbitals, flat_charges)
    ]))

    core_i = np.concatenate([np.full(c, i) for i, c in enumerate(n_core_orbitals)])
    core_N = np.concatenate([np.arange(c) for c in n_core_orbitals])
    core_N_orb = np_segment_sum(n_core_orbitals, np.repeat(np.arange(len(n_nuclei)), n_nuclei))
    return core_i, core_loc, core_type, core_N, core_N_orb


_concat = jax.jit(functools.partial(jnp.concatenate, axis=0))
def get_orbitals(nuclei: jax.Array, config: SystemConfigs) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """
    Given a set of nuclei and a SystemConfigs object, returns the orbitals for all molecular graphs.

    Args:
    - nuclei: A 2D array of shape (n_atoms, 3) representing the positions of the nuclei.
    - config: A SystemConfigs object representing the configuration of the system.

    Returns:
    - A tuple of four elements:
        - orb_loc: A 2D array of shape (n_orbitals, 3) representing the positions of the orbitals.
        - orb_type: A 1D array of shape (n_orbitals,) representing the type of each orbital.
        - orb_assoc: A 2D array of shape (n_orbitals, 2) representing the indices of the nuclei associated with each orbital.
        - N_orbs: A 1D array of shape (n_graphs,) representing the number of orbitals per graph.
    """
    flat_charges = np.array(tuple(flatten(config.charges)))
    valency = np.array(itemgetter(*flat_charges)(VALENCY))

    core_i, core_loc, core_type, core_N, core_N_orb = get_core_orbitals(nuclei, config)
    core_ij = jnp.stack([core_i, core_i], -1)

    val_i, val_j, val_type, val_N_orb = get_valence_orbitals(nuclei, tuple(valency), config)
    val_ij = jnp.stack([val_i, val_j], -1)
    val_loc = (nuclei[val_i] + nuclei[val_j]) / 2

    # For the valence orbital type we check the charges of both atoms
    # and then use their unique combination.
    f_c = jnp.array(flat_charges, dtype=jnp.int32)
    # val_type = jnp.array(VAL_OFFSET, dtype=jnp.int32)[(f_c[val_i], f_c[val_j])] + val_type + MAX_CORE_ORB
    val_type = val_type + MAX_CORE_ORB

    N_orbs = core_N_orb + val_N_orb
    orb_loc = []
    orb_type = []
    orb_assoc = []
    off_c, off_v = 0, 0
    for c, v in zip(core_N_orb, val_N_orb):
        orb_loc.append(core_loc[off_c:off_c+c])
        orb_loc.append(val_loc[off_v:off_v+v])
        orb_type.append(core_type[off_c:off_c+c])
        orb_type.append(val_type[off_v:off_v+v])
        orb_assoc.append(core_ij[off_c:off_c+c])
        orb_assoc.append(val_ij[off_v:off_v+v])
        off_c, off_v = off_c + c, off_v + v
    orb_loc = _concat(orb_loc)
    orb_type = _concat(orb_type)
    orb_assoc = _concat(orb_assoc)
    
    assert sum(N_orbs) == len(orb_type) == len(orb_assoc)
    return orb_loc, orb_type, orb_assoc, N_orbs
