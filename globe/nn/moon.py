import functools
from dataclasses import field
from typing import Tuple

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import jraph
import numpy as np

import globe.nn.gnn as GNN_MODULES
from globe.nn import (Activation, ActivationWithGain, Dense, Dense_no_bias,
                      FixedScalingFactor, ReparametrizedModule, residual)
from globe.nn.gnn import MlpRbf, NormEnvelope
from globe.nn.parameters import ParamSpec, ParamTree, ParamType, SpecTree
from globe.nn.utils import n_pair_same, sort_by_same_spin, spin_mask
from globe.nn.wave_function import WaveFunction
from globe.utils import adj_idx
from globe.utils.config import SystemConfigs, nuclei_per_graph


class Embedding(ReparametrizedModule):
    """
    Embedding module for the Moon model.

    Args:
    embedding_dim (int): The dimension of the embedding.
    e_out_dim (int): The dimension of the electron output.
    e_e_int_dim (int): The dimension of the electron-electron interaction.
    local_frames (bool): Whether to use local frames.
    edge_embedding (str): The type of edge embedding.
    edge_embedding_params (dict): The parameters for the edge embedding.
    activation (Activation): The activation function.

    Methods:
    param_spec(embedding_dim, e_e_int_dim, adaptive_weights, edge_embedding, edge_embedding_params, adaptive_norm):
        Returns a dictionary of parameter specifications for the module.

    __call__(self, electrons: jax.Array, atoms: jax.Array, atom_frames: jax.Array, params: dict, config: SystemConfigs) -> jax.Array:
        Computes the output of the module given the inputs and parameters.
    """
    embedding_dim: int
    e_out_dim: int
    e_e_int_dim: int
    local_frames: bool
    edge_embedding: str
    edge_embedding_params: dict
    activation: Activation
    
    @staticmethod
    def param_spec(
        embedding_dim: int,
        e_e_int_dim: int,
        adaptive_weights: bool,
        edge_embedding: str,
        edge_embedding_params: dict,
        adaptive_norm: bool
    ) -> SpecTree:
        result = {
            'atom_embeddings': ParamSpec(
                ParamType.NUCLEI,
                shape=(embedding_dim,),
                mean=0,
                std=1.0,
                keep_distr=True
            ),
            'edge_embedding':  getattr(GNN_MODULES, edge_embedding).spec(**edge_embedding_params)
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
        if adaptive_norm:
            result['norm_envelope'] = NormEnvelope.spec()
        return result

    @nn.compact
    def __call__(
        self,
        electrons: jax.Array,
        atoms: jax.Array,
        atom_frames: jax.Array,
        params: ParamTree,
        config: SystemConfigs
    ) -> jax.Array:
        """
        Computes the output of the module given the inputs and parameters.

        Args:
        electrons (jax.Array): The electron positions.
        atoms (jax.Array): The atom positions.
        atom_frames (jax.Array): The atom frames.
        params (dict): The module parameters.
        config (SystemConfigs): The system configurations.

        Returns:
        jax.Array: The output of the module.
        """
        atom_embeddings = params['atom_embeddings']
        adaptive_norm = 'norm_envelope' in params
        spins, n_nuclei = config.spins, nuclei_per_graph(config)
        flat_charges = np.concatenate(config.charges).astype(float)
        act = ActivationWithGain(self.activation)
        n_ele, n_nuc = electrons.shape[0], atoms.shape[0]

        drop_diag, drop_off_block = True, False
        e_e_i, e_e_j, e_e_mask = adj_idx(np.sum(spins, -1), drop_diagonal=drop_diag, drop_off_block=drop_off_block)
        e_e_diag = np.concatenate([np.eye(s).reshape(-1) for s in np.sum(spins, -1)])
        if drop_off_block:
            e_e_diag = np.concatenate([
                np.eye(s)[np.triu_indices(s)] for s in np.sum(spins, -1)
            ])
        if drop_diag:
            e_e_diag = np.zeros_like(e_e_i)
        e_e_diag = e_e_diag[:, None]
        e_e_i, e_e_j, e_e_mask, e_e_diag = jtu.tree_map(
            functools.partial(sort_by_same_spin, spins=spins, drop_diagonal=drop_diag, drop_off_block=drop_off_block),
            (e_e_i, e_e_j, e_e_mask, e_e_diag)
        )
        # setup electron - electron interactions
        r_ij = electrons[e_e_i] - electrons[e_e_j]
        r_ij_norm = jnp.linalg.norm(
            r_ij + e_e_diag,
            keepdims=True,
            axis=-1
        )
        r_ij = jnp.concatenate([r_ij, r_ij_norm * (1.0 - e_e_diag)], axis=-1)
        pair_scaling = jnp.log1p(r_ij_norm) / r_ij_norm
        # Let's split h_two into the offdiagonal and diagonal bits
        # to avoid splitting and concatenating
        n = n_pair_same(spins, drop_diagonal=drop_diag, drop_off_block=drop_off_block)
        ij_scale, r_ij, e_e_i, e_e_j, e_e_mask = jtu.tree_map(
            lambda x: (x[:n], x[n:]),
            (pair_scaling, r_ij, e_e_i, e_e_j, e_e_mask)
        )

        # Setup electron - atom interactions
        e_n_i, e_n_k, _ = adj_idx(np.sum(spins, -1), n_nuclei)
        r_im = electrons[e_n_i] - atoms[e_n_k]
        if self.local_frames:
            r_im = jnp.einsum('...m,...mn->...n', r_im, atom_frames[e_n_k])
        r_im_norm = jnp.linalg.norm(r_im, axis=-1, keepdims=True)
        scaling = jnp.log1p(r_im_norm) / r_im_norm
        r_im = jnp.concatenate([
            r_im,
            r_im_norm
        ], axis=-1)
        h_one = r_im * scaling

        # Setup nuclei - nuclei interactions
        n_n_i, n_n_j, _ = adj_idx(n_nuclei, n_nuclei)
        r_mn = atoms[n_n_i] - atoms[n_n_j]
        r_mn = jnp.concatenate([
            r_mn,
            jnp.linalg.norm(atoms[n_n_i] - atoms[n_n_j], axis=-1, keepdims=True)
        ], axis=-1)

        # Compute some normalization factors
        norm_env = NormEnvelope()
        e_e_contr = tuple(norm_env(r) for r in r_ij)
        e_n_contr = norm_env(r_im, e_n_k, params['norm_envelope'] if adaptive_norm else None)
        e_e_norm = 1 / (jraph.segment_sum(e_n_contr * flat_charges[e_n_k] / 2, e_n_i, n_ele, True) + 1)
        e_n_norm = 1 / (jraph.segment_sum(e_n_contr, e_n_i, n_ele, True) + 1)
        n_e_contr = norm_env(r_mn, n_n_j, params['norm_envelope'] if adaptive_norm else None)
        n_e_norm = 1 / (jraph.segment_sum(n_e_contr * flat_charges[n_n_j] / 2, n_n_i, n_nuc, True) + 1)
        n_n_contr = n_e_contr
        n_n_norm = 1 / (jraph.segment_sum(n_n_contr, n_n_i, n_nuc, True) + 1)

        # Electron - electron interactions
        ee_dist_emb = MlpRbf()
        ee_inp = []
        for idx_i, idx_j, scale, r, contr in zip(e_e_i, e_e_j, ij_scale, r_ij, e_e_contr):
            feat = []
            if not drop_off_block and not self.local_frames:
                g = Dense_no_bias(self.e_e_int_dim)(electrons)
                g = g[idx_i] - g[idx_j]
                feat.append(g)
            feat.append(Dense(self.e_e_int_dim)(r[..., -1:]))
            feat = sum(feat) * scale
            inp = act(feat) * Dense_no_bias(self.e_e_int_dim)(ee_dist_emb(r))
            inp = FixedScalingFactor()(inp, contr[:, None])
            ee_inp.append(inp)
        
        ee_inp = jnp.concatenate(ee_inp, axis=0)
        mask = np.concatenate([e_e_i[0], e_e_i[1]+n_ele])
        if drop_off_block:
            ee_inp = jnp.tile(ee_inp, (2, 1))
            mask = np.concatenate([mask, e_e_j[0], e_e_j[1]+n_ele])
        
        ee_emb = jraph.segment_sum(
            ee_inp,
            mask,
            2*n_ele,
            not drop_off_block
        )
        ee_emb = jnp.concatenate([ee_emb[:n_ele], ee_emb[n_ele:]], axis=-1)
        ee_emb = FixedScalingFactor()(ee_emb * e_e_norm[:, None], e_e_norm[:, None])
        ee_emb = Dense(self.embedding_dim)(ee_emb)
        ee_emb = act(ee_emb)
        self.sow('intermediates', 'ee_emb', ee_emb)

        # Nuclei and electron embeddings
        edge_emb_layer = getattr(GNN_MODULES, self.edge_embedding).create(**self.edge_embedding_params)
        e_n_emb = edge_emb_layer(r_im, e_n_k, params['edge_embedding'])

        if 'atom_weights' in params:
            kernel = params['atom_weights'][e_n_k]
        else:
            kernel = self.param(
                'kernel',
                jnn.initializers.normal(1/np.sqrt(4)),
                (4, self.embedding_dim)
            )
        elec_nuc_emb = jnp.einsum('...d,...dk->...k', h_one, kernel)
        self.sow('intermediates', 'en_prod', elec_nuc_emb)
        elec_nuc_emb += ee_emb[e_n_i] + atom_embeddings[e_n_k]
        elec_nuc_emb /= np.sqrt(2)
        self.sow('intermediates', 'elec_nuc_emb', elec_nuc_emb)
        elec_nuc_emb = act(elec_nuc_emb)

        weights = Dense_no_bias(2*self.embedding_dim)(e_n_emb).reshape(-1, 2, self.embedding_dim).transpose(1, 0, 2)
        agg_inp = elec_nuc_emb * weights
        agg_inp = FixedScalingFactor()(agg_inp, e_n_contr[:, None]).reshape(-1, self.embedding_dim)
        mask = spin_mask(spins)[e_n_i]
        e_n_k[~mask] += n_nuc
        agg_emb = jraph.segment_sum(
            agg_inp,
            np.concatenate([e_n_i, e_n_k + n_ele]),
            n_ele + 2*n_nuc,
            False
        )
        elec_emb = agg_emb[:n_ele] * e_n_norm[:, None]
        elec_emb = act(Dense(self.e_out_dim)(elec_emb))
        nuc_emb = (
            agg_emb[n_ele:n_ele+n_nuc] * n_e_norm[:, None],
            agg_emb[n_ele+n_nuc:] * n_e_norm[:, None],
        )
        
        # Compute nuclei nuclei edge embeddings
        n_n_emb = edge_emb_layer(r_mn, n_n_j, params['edge_embedding'])
        
        # Attach the graph mask for the jastrow factor
        r_ij = (r_ij, e_e_mask)
        return elec_emb, nuc_emb, r_ij, r_im, (e_n_emb, e_n_contr, e_n_norm), (n_n_emb, n_n_contr, n_n_norm)


class Diffusion(nn.Module):
    """
    Diffusion module for the Moon model.

    Args:
    out_dim (int): The dimension of the output.
    activation (Activation): The activation function.

    Methods:
    __call__(self, elec_emb: jax.Array, nuc_emb: Tuple[jax.Array, jax.Array], edge_weights: Tuple[jax.Array, jax.Array, jax.Array], config: SystemConfigs) -> jax.Array:
        Computes the output of the module given the inputs and parameters.
    """
    out_dim: int
    activation: Activation

    @nn.compact
    def __call__(self, elec_emb: jax.Array, nuc_emb: tuple[jax.Array, jax.Array], edge_weights: tuple[jax.Array, jax.Array, jax.Array], config: SystemConfigs) -> jax.Array:
        """
        Computes the output of the module given the inputs and parameters.

        Args:
        elec_emb (jax.Array): The electron embeddings.
        nuc_emb (Tuple[jax.Array, jax.Array]): The nuclear embeddings.
        edge_weights (Tuple[jax.Array, jax.Array, jax.Array]): The edge embeddings, contributions and normalization.
        config (SystemConfigs): The system configurations.

        Returns:
        jax.Array: The output of the module.
        """
        spins, n_nuclei = config.spins, nuclei_per_graph(config)
        activation = ActivationWithGain(self.activation)
        e_n_i, e_n_k, _ = adj_idx(np.sum(spins, -1), n_nuclei)

        out_emb = Dense(self.out_dim)(elec_emb)
        mask = spin_mask(spins)
        e_n_mask = mask[e_n_i]

        up_inp, down_inp = nuc_emb
        inp = jnp.where(e_n_mask[:, None], up_inp[e_n_k], down_inp[e_n_k])

        edge_emb, contr, norm = edge_weights

        weights = Dense_no_bias(self.out_dim)(edge_emb)

        to_elec = inp * weights
        to_elec = FixedScalingFactor()(to_elec, contr[:, None])
        out_emb += jraph.segment_sum(
            to_elec,
            e_n_i,
            np.sum(spins),
            True
        ) * norm[:, None]
        out_emb = FixedScalingFactor()(out_emb)
        out_emb = activation(out_emb)
        out_emb = activation(Dense(self.out_dim)(out_emb))
        return residual(elec_emb, out_emb)


class Update(ReparametrizedModule):
    """
    Update module for the Moon model.

    Args:
    out_dim (int): The dimension of the output.
    activation (Activation): The activation function.

    Methods:
    __call__(self, nuc_emb: Tuple[jax.Array, jax.Array], params: dict) -> Tuple[jax.Array, jax.Array]:
        Computes the output of the module given the inputs and parameters.

    """
    out_dim: int
    activation: Activation

    @staticmethod
    def param_spec(out_dim: int, adaptive_bias: bool) -> SpecTree:
        if adaptive_bias:
            return {
                'bias': ParamSpec(
                    ParamType.NUCLEI,
                    shape=(out_dim,),
                    mean=0,
                    std=0.1,
                    group='biases'
                )
            }
        return {}

    @nn.compact
    def __call__(self, nuc_emb: tuple[jax.Array, jax.Array], params: ParamTree) -> tuple[jax.Array, jax.Array]:
        """
        Computes the output of the module given the inputs and parameters.

        Args:
        nuc_emb (Tuple[jax.Array, jax.Array]): The nuclear embeddings (spin up, spin down per nucleus).
        params (ParamTree): The module parameters.

        Returns:
        Tuple[jax.Array, jax.Array]: The output of the module.
        """
        same_dense = Dense_no_bias(self.out_dim)
        diff_dense = Dense_no_bias(self.out_dim)
        activation = ActivationWithGain(self.activation)
        up_in, down_in = nuc_emb
        if 'bias' in params:
            bias = params['bias']
        else:
            bias = self.param('bias', jnn.initializers.zeros, (self.out_dim,))
        return tuple(
            residual(a, activation((same_dense(a) + diff_dense(b)) / np.sqrt(2.0) + bias))
            for a, b in ((up_in, down_in), (down_in, up_in))
        )


class Interaction(ReparametrizedModule):
    """
    Interaction module for the Moon model.

    Args:
    out_dim (int): The dimension of the output.
    activation (Activation): The activation function.

    Methods:
    param_spec(out_dim: int, adaptive_bias: bool) -> dict:
        Returns the parameter specification for the module.

    __call__(self, nuc_emb: Tuple[jax.Array, jax.Array], params: dict, edge_weights: Tuple[jax.Array, jax.Array, jax.Array], config: SystemConfigs) -> Tuple[jax.Array, jax.Array]:
        Computes the output of the module given the inputs and parameters.

    """
    out_dim: int
    activation: Activation

    @staticmethod
    def param_spec(out_dim: int, adaptive_bias: bool) -> SpecTree:
        if adaptive_bias:
            return {
                'bias': ParamSpec(
                    ParamType.NUCLEI,
                    shape=(out_dim,),
                    mean=0,
                    std=0.1,
                    group='biases'
                )
            }
        return {}

    @nn.compact
    def __call__(self, nuc_emb: tuple[jax.Array, jax.Array], params: ParamTree, edge_weights: tuple[jax.Array, jax.Array, jax.Array], config: SystemConfigs) -> tuple[jax.Array, jax.Array]:
        """
        Computes the output of the module given the inputs and parameters.

        Args:
        nuc_emb (Tuple[jax.Array, jax.Array]): The nuclear embeddings (spin up, spin down per nucleus).
        params (ParamTree): The module parameters.
        edge_weights (Tuple[jax.Array, jax.Array, jax.Array]): The edge weights. (edge embeddings, contribution, normalization)
        config (SystemConfigs): The system configurations.

        Returns:
        Tuple[jax.Array, jax.Array]: The output of the module.
        """
        edge_emb, contr, norm = edge_weights
        n_nuclei = nuclei_per_graph(config)
        n_n_i, n_n_j, _ = adj_idx(n_nuclei)
        weights = Dense_no_bias(self.out_dim)(edge_emb)
        
        dense = Dense_no_bias(self.out_dim)
        activation = ActivationWithGain(self.activation)
        up_in, down_in = nuc_emb
        inp = jnp.stack([
            jnp.concatenate([up_in, down_in], axis=-1),
            jnp.concatenate([down_in, up_in], axis=-1)
        ], axis=-2)
        if 'bias' in params:
            bias = params['bias']
        else:
            bias = self.param('bias', jnn.initializers.zeros, (self.out_dim,))
            
        msgs = dense(inp)[n_n_j] * weights[..., None, :]
        msgs = FixedScalingFactor()(msgs, contr[:, None, None])
        new_emb = jraph.segment_sum(
            msgs,
            n_n_i,
            sum(n_nuclei),
            True
        ) * norm[:, None, None]
        new_emb = activation(new_emb + bias[..., None, :])
        return residual(up_in, new_emb[:, 0, :]), residual(down_in, new_emb[:, 1, :])


class MoonLayer(ReparametrizedModule):
    """
    MoonLayer module for the Moon model.

    Args:
    dim (int): The ou of the module.
    activation (Activation): The activation function.
    use_interaction (bool): Whether to use interaction.
    update_before_int (int): The number of updates before interaction.
    update_after_int (int): The number of updates after interaction.
    adaptive_bias (bool): Whether to use adaptive bias.

    Methods:
    param_spec(dim: int, use_interaction: bool, update_before_int: int, update_after_int: int, adaptive_bias: bool) -> dict:
        Returns the parameter specification for the module.

    __call__(self, elec_emb: jax.Array, nuc_emb: jax.Array, e_n_weights: jax.Array, n_n_weights: jax.Array, params: dict, config: SystemConfigs) -> Tuple[jax.Array, jax.Array]:
        Computes the output of the module given the inputs and parameters.

    """
    dim: int
    activation: Activation

    @staticmethod
    def param_spec(
        dim: int,
        use_interaction: bool,
        update_before_int: int,
        update_after_int: int,
        adaptive_bias: bool
    ) -> SpecTree:
        result = {
            'upd_before': {
                str(i): Update.spec(dim, adaptive_bias)
                for i in range(update_before_int)
            },
            'upd_after': {
                str(i): Update.spec(dim, adaptive_bias)
                for i in range(update_after_int)
            }
        }
        if use_interaction:
            result['interaction'] = Interaction.spec(dim,adaptive_bias)
        return result

    @nn.compact
    def __call__(
        self,
        elec_emb: jax.Array,
        nuc_emb: jax.Array,
        e_n_weights: jax.Array,
        n_n_weights: jax.Array,
        params: ParamTree,
        config: SystemConfigs
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        """
        Computes the output of the module given the inputs and parameters.

        Args:
        elec_emb (jax.Array): The electronic embeddings.
        nuc_emb (jax.Array): The nuclear embeddings.
        e_n_weights (jax.Array): The edge weights between electrons and nuclei. (edge embeddings, contribution, normalization)
        n_n_weights (jax.Array): The edge weights between nuclei. (edge embeddings, contribution, normalization)
        params (ParamTree): The module parameters.
        config (SystemConfigs): The system configurations.

        Returns:
        Tuple[jax.Array, Tuple[jax.Array, jax.Array]]: The output of the module. (electrons, (nuc up, nuc down))
        """
        for p in params['upd_before'].values():
            nuc_emb = Update(self.dim, self.activation)(nuc_emb, p)
        if 'interaction' in params:
            nuc_emb = Interaction(self.dim, self.activation)(nuc_emb, params['interaction'], n_n_weights, config)
        for p in params['upd_after'].values():
            nuc_emb = Update(self.dim, self.activation)(nuc_emb, p)
        elec_emb = Diffusion(self.dim, self.activation)(elec_emb, nuc_emb, e_n_weights, config)
        return elec_emb, nuc_emb


class Moon(WaveFunction):
    """
    Moon model for electronic structure calculations.

    Args:
    hidden_dims (Tuple[int, ...]): The dimensions of the hidden layers.
    use_interaction (bool): Whether to use interaction.
    update_before_int (int): The number of updates before interaction.
    update_after_int (int): The number of updates after interaction.
    adaptive_update_bias (bool): Whether to use adaptive bias.
    edge_embedding (str): The type of edge embedding to use.
    edge_embedding_params (dict): The parameters for the edge embedding.
    local_frames (bool): Whether to use local frames.
    embedding_dim (int): The dimension of the embeddings.
    embedding_e_out_dim (int): The output dimension of the electronic embeddings.
    embedding_int_dim (int): The dimension of the interaction embeddings.
    embedding_adaptive_weights (bool): Whether to use adaptive weights for the embeddings.
    embedding_adaptive_norm (bool): Whether to use adaptive normalization for the embeddings.
    """
    hidden_dims: Tuple[int, ...] = (256, 256)
    use_interaction: bool = True
    update_before_int: int = 1
    update_after_int: int = 1
    adaptive_update_bias: bool = True

    edge_embedding: str = 'MLPEdgeEmbedding'
    edge_embedding_params: dict = field(default_factory=lambda*_: dict(
        # MLPEdgeEmbedding
        out_dim=8,
        hidden_dim=16,
        activation='silu',
        adaptive_weights=True,
        init_sph=False,
    ))

    local_frames: bool = False
    embedding_dim: int = 64
    embedding_e_out_dim: int = 256
    embedding_int_dim: int = 32
    embedding_adaptive_weights: bool = True
    embedding_adaptive_norm: bool = True

    @staticmethod
    def param_spec(
        shared_orbitals,
        full_det,
        determinants,
        orbital_type,
        orbital_config,
        adaptive_jastrow,
        adaptive_sum_weights,

        hidden_dims,
        use_interaction,
        update_before_int,
        update_after_int,
        adaptive_update_bias,

        edge_embedding,
        edge_embedding_params,

        embedding_dim,
        embedding_e_out_dim,
        embedding_int_dim,
        embedding_adaptive_weights,
        embedding_adaptive_norm):
        return {
            **WaveFunction.spec(**locals(), out_dim=hidden_dims[-1]),
            'embeddings': Embedding.spec(
                embedding_dim=embedding_dim,
                e_out_dim=embedding_e_out_dim,
                e_e_int_dim=embedding_int_dim,
                edge_embedding=edge_embedding,
                edge_embedding_params=edge_embedding_params,
                adaptive_weights=embedding_adaptive_weights,
                adaptive_norm=embedding_adaptive_norm
            ),
            'layers': {
                str(i): MoonLayer.spec(
                    dim,
                    use_interaction,
                    update_before_int,
                    update_after_int,
                    adaptive_update_bias
                )
                for i, dim in enumerate(hidden_dims)
            }
        }

    def setup(self):
        super().setup()
        self.embedding = Embedding(
            self.embedding_dim,
            self.embedding_e_out_dim,
            self.embedding_int_dim,
            local_frames=self.local_frames,
            edge_embedding=self.edge_embedding,
            edge_embedding_params=self.edge_embedding_params,
            activation=self.activation)
        self.layers = [
            MoonLayer(dim, self.activation)
            for dim in self.hidden_dims
        ]

    def _encode(self, electrons, atoms, config, params):
        elec_emb, nuc_emb, r_ij, r_im, e_n_weights, n_n_weights = self.embedding(
            electrons,
            atoms,
            params['atom_frames'],
            params['embeddings'],
            config
        )
        for layer, p in zip(self.layers, params['layers'].values()):
            elec_emb, nuc_emb = layer(elec_emb, nuc_emb, e_n_weights, n_n_weights, p, config)

        return elec_emb, r_ij, r_im
