import functools
from typing import Sequence, Tuple, Type

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import jraph
import numpy as np

from globe.nn import (Activation, ActivationWithGain, Dense, Dense_no_bias,
                        ReparametrizedModule, residual)
from globe.nn.orbitals import OrbitalModule
from globe.nn.parameters import ParamSpec, ParamType
from globe.nn.utils import pair_mask, n_pair_same, sort_by_same_spin
from globe.nn.wave_function import WaveFunction
from globe.utils import adj_idx
from globe.utils.config import SystemConfigs, nuclei_per_graph

netjit = functools.partial(jax.jit, static_argnums=3, static_argnames='config')
netvmap = functools.partial(jax.vmap, in_axes=(None, 0, None, None, None))


class FeatureConstruction(ReparametrizedModule):
    """
    A class representing a feature construction module in a Fermi Net neural network.

    Attributes:
    embedding_dim: int
        The dimension of the embedding space.
    adaptive_weights: bool
        Whether to use adaptive weights or not.
    activation: Activation
        The activation function to use.

    Methods:
    param_spec(embedding_dim: int, adaptive_weights: bool) -> Dict[str, ParamSpec]:
        Returns a dictionary of parameter specifications for this module.

    __call__(self, electrons: jax.Array, atoms: jax.Array, params: Dict[str, jax.Array], config: SystemConfigs) -> Tuple[jax.Array, jax.Array]:
        Computes the single and pairwise electron features given the input electrons, atoms, parameters and system configurations.

    """
    embedding_dim: int
    adaptive_weights: bool
    activation: Activation

    @staticmethod
    def param_spec(embedding_dim, adaptive_weights):
        result = {
            'atom_embeddings': ParamSpec(
                ParamType.NUCLEI,
                shape=(embedding_dim,),
                mean=0,
                std=1.0,
                keep_distr=True
            )
        }
        if adaptive_weights:
            result['atom_weights'] = ParamSpec(
                ParamType.NUCLEI,
                shape=(4, embedding_dim,),
                mean=0,
                std=1/np.sqrt(4),
                segments=4,
                keep_distr=True
            )
        return result
    
    @nn.compact
    def __call__(self, electrons, atoms, params, config: SystemConfigs):
        spins, n_nuclei = config.spins, nuclei_per_graph(config)
        
        # We rearrange the pairwise terms such that all the block diagonals are coming first and all
        # the off diagonals last. We do this here with the indices to avoid reindexing during runtime.
        e_e_i, e_e_j, e_e_mask = adj_idx(np.sum(spins, -1))
        e_e_diag = np.concatenate([np.eye(s).reshape(-1, 1) for s in np.sum(spins, -1)], axis=0)
        e_e_i, e_e_j, e_e_mask, e_e_diag = jtu.tree_map(
            functools.partial(sort_by_same_spin, spins=spins, drop_diagonal=False),
            (e_e_i, e_e_j, e_e_mask, e_e_diag)
        )

        r_ij = electrons[e_e_i] - electrons[e_e_j]
        r_ij_norm = jnp.linalg.norm(
            r_ij + e_e_diag,
            keepdims=True,
            axis=-1
        )
        r_ij = jnp.concatenate([r_ij, r_ij_norm * (1.0 - e_e_diag)], axis=-1)
        scaling = jnp.log1p(r_ij_norm) / r_ij_norm
        h_two = r_ij * scaling
        # Let's split h_two into the offdiagonal and diagonal bits
        # to avoid splitting and concatenating
        n = n_pair_same(spins)
        h_two = (h_two[:n], h_two[n:])
        r_ij = (r_ij[:n], r_ij[n:])
        
        e_n_i, e_n_k, _ = adj_idx(np.sum(spins, -1), n_nuclei)
        
        r_im = electrons[e_n_i] - atoms[e_n_k]
        r_im_norm = jnp.linalg.norm(r_im, axis=-1, keepdims=True)
        scaling = jnp.log(1+r_im_norm)/r_im_norm
        r_im = jnp.concatenate([
            r_im,
            r_im_norm
        ], axis=-1)
        h_one = r_im * scaling

        atom_embeddings = params['atom_embeddings']
        if self.adaptive_weights:
            kernel = params['atom_weights'][e_n_k]
        else:
            kernel = self.param(
                'weight_kernel',
                jnn.initializers.lecun_normal(),
                (4, self.embedding_dim)
            )
        h_one = jnp.einsum('...d,...dk->...k', h_one, kernel)
        h_one = jnp.tanh(h_one + atom_embeddings[e_n_k])
        
        h_one = jraph.segment_sum(
            h_one,
            e_n_i,
            electrons.shape[0],
            True
        ) / np.maximum(np.array(n_nuclei).repeat(np.sum(spins, -1), 0)[..., None], 1)
        # Add mask for pairwise terms
        r_ij = (
            r_ij,
            (e_e_mask[:n], e_e_mask[n:])
        )
        return h_one, h_two, r_ij, r_im


def aggregate_features(h_one: jax.Array, h_two: tuple[jax.Array, jax.Array], spins: np.ndarray, restricted: bool) -> Tuple[jax.Array, jax.Array]:
    """
    Aggregates the single and pairwise electron features.

    Args:
    h_one: Array
        The single electron features.
    h_two: Tuple[Array, Array]
        The pairwise electron features.
    spins: Array
        The number of electrons in each atom.
    restricted: bool
        Whether to use restricted symmetry or not.

    Returns:
    Tuple[Array, Array]
        The aggregated single and global features.
    """
    spins = np.array(spins)
    n_graphs = len(spins)
    segments = np.repeat(jnp.arange(spins.size), spins.reshape(-1))

    # We use segment_sum since segment_mean requires two segment_sums
    g_inp = jraph.segment_sum(
        h_one,
        segments,
        spins.size,
        True
    ).reshape(n_graphs, 2, -1) / np.maximum(spins[..., None], 1)
    if restricted:
        g_inp = jnp.stack([
            jnp.concatenate([g_inp[:, 0], g_inp[:, 1]], axis=-1),
            jnp.concatenate([g_inp[:, 1], g_inp[:, 0]], axis=-1)
        ], axis=1)
        g_inp = g_inp.reshape(2*n_graphs, -1)
    else:
        g_inp = g_inp.reshape(n_graphs, -1)

    pair = []
    for h, diag in zip(h_two, (True, False)):
        pair.append(jraph.segment_mean(
            h,
            pair_mask(spins, diag),
            spins.sum(),
            True
        ))
    if not restricted:
        pair = jnp.stack(pair, axis=1)
        swap_mask = np.repeat(np.arange(spins.size) % 2, spins.reshape(-1)).astype(bool)
        pair = jnp.where(swap_mask[:, None, None], pair[:, (1, 0)], pair)
        pair = (pair.reshape(spins.sum(), -1),)
    return jnp.concatenate([h_one, *pair], axis=-1), g_inp


class FermiLayer(nn.Module):
    """
    A class representing a Fermi layer in a Fermi Net neural network.

    Attributes:
    single_out: int
        The number of output features for the single update.
    pair_out: int
        The number of output features for the pairwise update.
    activation: Activation
        The activation function to use.
    restricted: bool
        Whether to use restricted symmetry or not.

    Methods:
    __call__(self, h_one: jax.Array, h_two: jax.Array, spins: jax.Array) -> jax.Array:
        Computes the output of the Fermi layer given the input features h_one, h_two and spins.
    """
    single_out: int
    pair_out: int
    activation: Activation
    restricted: bool
    
    @nn.compact
    def __call__(self, h_one, h_two, spins):
        activation = ActivationWithGain(self.activation)

        # Single update
        one_in, global_in = aggregate_features(h_one, h_two, spins, self.restricted)
        h_one_new = Dense(self.single_out)(one_in)
        global_new = Dense_no_bias(self.single_out)(global_in)
        if self.restricted:
            global_new = jnp.repeat(global_new, np.reshape(spins, -1), axis=0)
        else:
            global_new = jnp.repeat(global_new, np.sum(spins, -1), axis=0)
        h_one_new += global_new
        h_one_new = activation(h_one_new / jnp.sqrt(2))
        h_one = residual(h_one, h_one_new)
        
        # Pairwise update
        if self.pair_out > 0:
            if self.restricted:
                # Since we rearranged our pairwise terms such that the diagonals are first,
                # we only need to split the array into the first and second half as these correspond to the
                # diagonal and off diagonal terms.
                h_two_new = tuple(Dense(self.pair_out)(h) for h in h_two)
            else:
                dense = Dense(self.pair_out)
                h_two_new = jtu.tree_map(dense, h_two_new)
            if h_two_new[0].shape != h_two[0].shape:
                h_two = jtu.tree_map(jnp.tanh, h_two_new)
            else:
                h_two_new = jtu.tree_map(activation, h_two_new)
                h_two = jtu.tree_map(residual, h_two, h_two_new)
        return h_one, h_two


class FermiNet(WaveFunction):
    """
    A class representing a Fermi Net neural network.

    Attributes:
    hidden_dims: Sequence[Tuple[int, int]]
        A sequence of tuples representing the number of hidden units for each layer.
    embedding_dim: int
        The dimensionality of the electron embedding.
    embedding_adaptive_weights: bool
        Whether to use adaptive weights for the electron embedding.
    restricted: bool
        Whether to use restricted orbitals or not.
    """
    hidden_dims: Sequence[Tuple[int, int]] = (
        (256, 32), (256, 32), (256, 32), (256, 32))
    embedding_dim: int = 64
    embedding_adaptive_weights: bool = True
    restricted: bool = True

    @staticmethod
    def param_spec(
        shared_orbitals: bool,
        full_det: bool,
        determinants: int,
        orbital_type: Type[OrbitalModule],
        orbital_config: dict,
        adaptive_jastrow: bool,
        adaptive_sum_weights: bool,

        hidden_dims: Sequence[tuple[int, int]],
        embedding_adaptive_weights: bool,
        embedding_dim: int):
        return {
            **WaveFunction.spec(**locals(), out_dim=hidden_dims[-1][0]),
            'embeddings': FeatureConstruction.spec(embedding_dim, embedding_adaptive_weights)
        }

    def setup(self):
        super().setup()
        self.input_construction = nn.jit(FeatureConstruction, static_argnums=4)(
            self.embedding_dim,
            self.embedding_adaptive_weights,
            self.activation
        )
        # Do not compute an update for the last pairwise layer
        hidden_dims = [list(h) for h in self.hidden_dims]
        hidden_dims[-1][1] = 0
        self.fermi_layers = [
            FermiLayer(
                single_out=single,
                pair_out=pair,
                activation=self.activation,
                restricted=self.restricted
            )
            for single, pair in hidden_dims
        ]

    def _encode(self, electrons, atoms, config: SystemConfigs, params):
        h_one, h_two, r_ij, r_im = self.input_construction(
            electrons,
            atoms,
            params['embeddings'],
            config
        )

        # Fermi interaction
        for fermi_layer in self.fermi_layers:
            h_one, h_two = fermi_layer(h_one, h_two, config.spins)

        return h_one, r_ij, r_im
