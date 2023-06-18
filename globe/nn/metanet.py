import functools
from dataclasses import field
from itertools import groupby
from operator import itemgetter
from typing import Optional, Sequence, Tuple, Type

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import jraph
import numpy as np

import globe.nn.gnn as GNN_MODULES
from globe.nn import (Activation, ActivationWithGain, AutoMLP, Dense,
                      Dense_no_bias, Embed, ParamTree, ReparametrizedModule,
                      ScalingFactor, residual)
from globe.nn.gnn import NormEnvelope
from globe.nn.parameters import ParamSpec, ParamType, SpecTree
from globe.systems.element import MAX_ORB, VALENCY
from globe.utils import (SystemConfigs, adj_idx, flatten, nuclei_per_graph,
                         safe_call)

MAX_CHARGE = max(VALENCY.keys()) + 1


_var_init = functools.partial(
    jnn.initializers.variance_scaling,
    mode='fan_in',
    distribution='truncated_normal'
)


class MessagePassing(nn.Module):
    """
    A message passing module that takes in node, edge, and global features and updates node features.

    Args:
        msg_dim: The dimensionality of the messages.
        out_dim: The dimensionality of the output node features.
        activation: The activation function to use.
    """
    msg_dim: int
    out_dim: int
    activation: Activation

    @nn.compact
    def __call__(
            self,
            s_embed: jax.Array,
            r_embed: jax.Array,
            e_embed: jax.Array,
            senders: jax.Array,
            receivers: jax.Array,
            edge_contr: jax.Array,
            norm: jax.Array
        ) -> jax.Array:
        """
        Compute the message passing operation.

        Args:
            s_embed: The sender node embeddings.
            r_embed: The receiver node embeddings.
            e_embed: The edge embeddings.
            senders: The sender indices.
            receivers: The receiver indices.
            edge_contr: The edge contributions.
            norm: The normalization factors.

        Returns:
            The updated node embeddings.
        """
        act = ActivationWithGain(self.activation)
        inp = Dense(self.msg_dim)(s_embed)[senders] + Dense(self.msg_dim)(r_embed)[receivers]
        inp = act(inp / np.sqrt(2))
        inp = ScalingFactor()(
            inp,
            inp * Dense_no_bias(self.msg_dim)(e_embed),
            edge_contr[:, None],
            edge_contr[:, None]
        )

        msg = jraph.segment_sum(
            inp,
            receivers,
            num_segments=r_embed.shape[0]
        ) * norm[:, None]
        msg = ScalingFactor()(inp, msg, edge_contr[:, None], norm[:, None])
        return act(Dense_no_bias(self.out_dim)(msg))


class Residual(nn.Module):
    """
    A residual module that applies two dense layers with activation function and adds the input to the output.

    Args:
        dim: The dimensionality of the output.
        activation: The activation function to use.
    """
    dim: int
    activation: Activation

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Compute the residual operation.

        Args:
            x: The input array.

        Returns:
            The output array.
        """
        act = ActivationWithGain(self.activation)
        y = act(Dense(self.dim)(x))
        y = act(Dense(self.dim)(y))
        return ScalingFactor()(x, residual(x, y))


class Update(nn.Module):
    """
    A module that updates node features by applying a residual operation.

    Args:
        out_dim: The dimensionality of the output node features.
        activation: The activation function to use.
    """
    out_dim: int
    activation: Activation

    @nn.compact
    def __call__(self, n_embed: jax.Array, msg: jax.Array) -> jax.Array:
        """
        Compute the update operation.

        Args:
            n_embed: The node embeddings.
            msg: The messages.

        Returns:
            The updated node embeddings.
        """
        act = ActivationWithGain(self.activation)
        y = act(Dense(self.out_dim)(n_embed))
        y = (y + msg) / np.sqrt(2)
        # Actual update
        y = Residual(self.out_dim, self.activation)(y)
        y = act(Dense(self.out_dim)(y))
        y = residual(n_embed, y)
        y = Residual(self.out_dim, self.activation)(y)
        y = Residual(self.out_dim, self.activation)(y)
        return ScalingFactor()(n_embed, y)


class MessagePassingNetwork(nn.Module):
    """
    A module that performs message passing between nodes in a graph.

    Args:
        layers: A sequence of tuples, where each tuple contains the message and output dimensions of a message passing layer.
        activation: The activation function to use.

    Returns:
        A list of node embeddings after each message passing layer.
    """
    layers: Sequence[Tuple[int, int]]
    activation: Activation

    @nn.compact
    def __call__(
            self,
            s_embed: jax.Array,
            r_embed: jax.Array,
            e_embed: jax.Array,
            edge_contr: jax.Array,
            senders: jax.Array,
            receivers: jax.Array
        ) -> list[jax.Array]:
        """
        Compute the message passing operation.

        Args:
            s_embed: The sender node embeddings.
            r_embed: The receiver node embeddings.
            e_embed: The edge embeddings.
            edge_contr: The edge contributions.
            senders: The sender indices.
            receivers: The receiver indices.

        Returns:
            A list of node embeddings after each message passing layer.
        """
        if r_embed is None:
            s_embed, r_embed = r_embed, s_embed
        
        norm = 1 / (jraph.segment_sum(edge_contr, receivers, r_embed.shape[0], True) + 1)
        embeddings = [r_embed]
        for layer in self.layers:
            msg = MessagePassing(layer[0], layer[1], self.activation)(
                r_embed if s_embed is None else s_embed,
                r_embed,
                e_embed,
                senders,
                receivers,
                edge_contr,
                norm
            )
            r_embed = Update(layer[1], self.activation)(r_embed, msg)
            embeddings.append(r_embed)
        return embeddings


class OrbitalEmbedding(nn.Module):
    """
    A module that computes the embedding of each orbital in a graph.

    Args:
        embedding_dim: The dimensionality of the output orbital embeddings.
        edge_embedding: The module that computes the edge embeddings.
        norm_envelope: The module that computes the edge contributions.
        message_passing: The module that performs message passing between nodes in a graph.
        activation: The activation function to use.
        aggregate_before_out: Whether to aggregate the node embeddings before the output.

    Returns:
        The embedding of each orbital in the graph.
    """
    embedding_dim: int
    edge_embedding: nn.Module
    norm_envelope: NormEnvelope
    message_passing: MessagePassingNetwork
    activation: Activation
    aggregate_before_out: bool

    @nn.compact
    def __call__(
        self,
        nuclei: jax.Array,
        n_embed: jax.Array,
        orb_loc: jax.Array,
        orb_type: jax.Array,
        N_orb: int,
        config: SystemConfigs
    ) -> jax.Array:
        """
        Compute the embedding of each orbital in the graph.

        Args:
            nuclei: The nuclei positions.
            n_embed: The node embeddings.
            orb_loc: The orbital locations.
            orb_type: The orbital types.
            N_orb: The number of orbitals.
            config: The batch configuration.

        Returns:
            The embedding of each orbital in the graph.
        """
        n_nuclei = nuclei_per_graph(config)
        # Construct graph
        senders, receivers, _ = adj_idx(n_nuclei, N_orb)
        # Initialize orbital embedding + locations
        orb_emb = Embed(MAX_ORB, self.embedding_dim)(orb_type)
        # Compute edge features
        edges = orb_loc[receivers] - nuclei[senders]
        edges = jnp.concatenate([
            edges,
            jnp.linalg.norm(edges, axis=-1, keepdims=True)
        ], axis=-1)
        e_embed = self.edge_embedding(edges, receivers)
        edge_contr = self.norm_envelope(edges, receivers)

        orb_emb = self.message_passing(n_embed, orb_emb, e_embed, edge_contr, senders, receivers)
        if self.aggregate_before_out:
            orb_emb = jnp.concatenate(orb_emb, axis=-1)
        else:
            orb_emb = orb_emb[-1]
        orb_emb = Residual(self.embedding_dim, self.activation)(orb_emb)
        orb_emb = Residual(self.embedding_dim, self.activation)(orb_emb)
        return orb_emb


class OutModule(nn.Module):
    """
    A module that takes in various embeddings and produces the final parameter output.

    Args:
        param_specs: A tuple of ParamSpec objects that define the shape of the output parameters.
        mlp_factory: A factory for creating MLPs.
        enable_segments: Whether to enable segments.

    Returns:
        A tuple of arrays, each corresponding to a parameter spec.
    """
    param_specs: Tuple[ParamSpec, ...]
    mlp_factory: Type[nn.Module]
    enable_segments: bool

    @nn.compact
    def __call__(
            self,
            nuc_emb: jax.Array,
            graph_emb: jax.Array,
            orb_emb: jax.Array,
            orb_nuc_emb: jax.Array,
            orb_type: jax.Array,
            orb_idx: jax.Array,
            nuc_idx: jax.Array,
            edge_emb: jax.Array,
            edge_contr: jax.Array,
            config: SystemConfigs) -> Tuple[jax.Array, ...]:
        """
        Computes the final output given various embeddings.

        Args:
            nuc_emb: The nuclear embeddings.
            graph_emb: The graph embeddings.
            orb_emb: The orbital embeddings.
            orb_nuc_emb: The orbital-nuclear embeddings.
            orb_type: The orbital types.
            orb_idx: The orbital indices in orb-nuc interactions.
            nuc_idx: The nuclear indices in orb-nuc interactions.
            edge_emb: The edge embeddings.
            edge_contr: The edge contributions.
            config: The batch configuration.

        Returns:
            A tuple of arrays, each corresponding to a parameter spec.
        """
        proto_spec = self.param_specs[0]
        param_type = proto_spec.param_type
        segments = proto_spec.segments if self.enable_segments else 1
        transform = proto_spec.transform

        flat_charges = np.array(tuple(flatten(config.charges)))
        out_dims = [np.prod(x.shape) for x in self.param_specs]
        out_dim = np.sum(out_dims)
        seg_out = out_dim // segments
        use_bias = proto_spec.use_bias
        assert all(x % segments == 0 for x in out_dims)


        # Define biases
        Embed = functools.partial(nn.Embed, embedding_init=jnn.initializers.normal(1.0))
        if use_bias:
            if param_type is ParamType.ORBITAL:
                bias = Embed(MAX_ORB, seg_out)(orb_type)
            if param_type in (ParamType.NUCLEI, ParamType.ORBITAL_NUCLEI):
                bias = Embed(MAX_CHARGE, seg_out)(flat_charges)
                if param_type is ParamType.ORBITAL_NUCLEI:
                    bias = bias[nuc_idx]
            if param_type is ParamType.GLOBAL:
                bias = self.param('std_bias', jnn.initializers.normal(1.0), (seg_out,))
        else:
            bias = jnp.zeros((seg_out,))
        if param_type is ParamType.ORBITAL_NUCLEI:
            orb_nuc_bias = Embed(MAX_CHARGE*MAX_ORB, seg_out)(flat_charges[nuc_idx]*MAX_ORB + orb_type[orb_idx])
        
        # Define input
        if param_type is ParamType.ORBITAL:
            inp = orb_emb
        elif param_type is ParamType.ORBITAL_NUCLEI:
            inp = orb_nuc_emb
            bias = (ScalingFactor()(
                orb_nuc_bias,
                orb_nuc_bias * Dense_no_bias(seg_out)(orb_nuc_emb),
                edge_contr[:, None],
                edge_contr[:, None]
            ) + bias) / np.sqrt(2)
        elif param_type is ParamType.NUCLEI:
            inp = nuc_emb
        elif param_type is ParamType.GLOBAL:
            inp = graph_emb
        else:
            raise ValueError()
        
        # Perform inference via segments
        inp = (inp[..., None] + self.param(
            'segments_embedding',
            jnn.initializers.normal(1.0 if segments > 1 else 0.1),
            (inp.shape[-1], segments)
        )) / np.sqrt(2 if segments > 1 else 1)
        # We transpose the last two dimensions to ensure that
        # the segments are properly assigned to the correct sub_specs
        inp = jnp.swapaxes(inp, -2, -1)
        result =  self.mlp_factory(
            seg_out,
            out_kernel_init=_var_init(1.0),
            final_bias=False,
            intermediate_bias=False
        )(inp)
        # add bias
        result = (result + bias[..., None, :]).reshape(-1, out_dim) / np.sqrt(2)

        # Scale to correct std
        result *= self.param(
            'weight_kernel',
            lambda *_: jnp.concatenate([
                jnp.full((np.prod(s.shape),), s.std, dtype=np.float32)
                for s in self.param_specs
            ])
        )

        # Add respective means - these are not trainable
        result += jnp.concatenate([
            jnp.full((np.prod(s.shape),), s.mean, dtype=jnp.float32)
            for s in self.param_specs
        ])

        # Transform
        result = safe_call(transform, result, nuc_idx=nuc_idx, orb_idx=orb_idx, config=config)

        # Split into subspecs and shape correctly
        return tuple(
            x.reshape(-1, *s.shape)
            for x, s in zip(jnp.split(result, np.cumsum(out_dims[:-1]), axis=-1), self.param_specs)
        )


class GraphToParameters(nn.Module):
    """
    A module that takes in a graph and produces the final parameter output.

    Args:
        param_spec: A dictionary of ParamSpec objects that define the shape of the output parameters.
        mlp_factory: A factory for creating MLPs.
        edge_embedding: A nn.Module that embeds the edges.
        norm_envelope: A NormEnvelope that computes the normalization of the edges.
        activation: An activation function.
        enable_groups: Whether to enable groups.
        enable_segments: Whether to enable segments.

    Returns:
        A tuple of arrays, each corresponding to a parameter spec.
    """
    param_spec: SpecTree
    mlp_factory: Type[nn.Module]
    edge_embedding: nn.Module
    norm_envelope: NormEnvelope
    activation: Activation
    enable_groups: bool
    enable_segments: bool

    @nn.compact
    def __call__(
        self,
        orb_loc: jax.Array,
        orb_emb: jax.Array,
        orb_type: jax.Array,
        nuc_loc: jax.Array,
        nuc_emb: jax.Array,
        N_orb: int,
        config: SystemConfigs
    ) -> ParamTree:
        """
        Args:
            orb_loc: An array of shape (N_orb, 3) containing the location of each orbital.
            orb_emb: An array of shape (N_orb, emb_dim) containing the embedding of each orbital.
            orb_type: An array of shape (N_orb,) containing the type of each orbital.
            nuc_loc: An array of shape (N_nuc, 3) containing the location of each nucleus.
            nuc_emb: An array of shape (N_nuc, emb_dim) containing the embedding of each nucleus.
            N_orb: An integer representing the number of orbitals.
            config: A SystemConfig object containing the configuration of the system.

        Returns:
            A dictionary of the same structure as param_spec with tensors as leaves.
        """
        n_nuclei = nuclei_per_graph(config)
        act = ActivationWithGain(self.activation)
        emb_dim = orb_emb.shape[-1]

        # Prepare orb - nuc embeddings
        nuc_idx, orb_idx, _ = adj_idx(n_nuclei, N_orb)
        orb_nuc_emb = Dense(emb_dim)(orb_emb)[orb_idx] + Dense(emb_dim)(nuc_emb)[nuc_idx]
        orb_nuc_emb = act(orb_nuc_emb / np.sqrt(2))

        # prepare graph embeddings
        graph_emb = jraph.segment_sum(
            nuc_emb,
            np.arange(len(n_nuclei)).repeat(n_nuclei),
            len(n_nuclei),
            True
        ) / np.array(n_nuclei)[:, None]

        # Update each embedding type individually
        orb_nuc_emb = Residual(emb_dim, self.activation)(orb_nuc_emb)
        graph_emb = Residual(emb_dim, self.activation)(graph_emb)
        orb_emb = Residual(emb_dim, self.activation)(orb_emb)
        nuc_emb = Residual(emb_dim, self.activation)(nuc_emb)

        # Normalization
        # we must first normalize otherwise the orb-nuc pairs
        # could be very far apart and renormalized to large (instable) values
        norm_scale = jnn.initializers.constant(1.0)
        orb_nuc_emb = nn.LayerNorm(scale_init=norm_scale)(orb_nuc_emb)
        graph_emb = nn.LayerNorm(scale_init=norm_scale)(graph_emb)
        orb_emb = nn.LayerNorm(scale_init=norm_scale)(orb_emb)
        nuc_emb = nn.LayerNorm(scale_init=norm_scale)(nuc_emb)

        # Decay orb - nuc embeddings by distance
        edges = orb_loc[orb_idx] - nuc_loc[nuc_idx]
        edges = jnp.concatenate([
            edges,
            jnp.linalg.norm(edges, axis=-1, keepdims=True)
        ], axis=-1)
        edge_emb = self.edge_embedding(edges, orb_idx)
        edge_contr = self.norm_envelope(edges, orb_idx)
        orb_nuc_emb = ScalingFactor()(
            orb_nuc_emb,
            orb_nuc_emb * Dense_no_bias(emb_dim)(edge_emb),
            edge_contr[:, None],
            edge_contr[:, None]
        )

        # Gneerate parameters
        def gen_parameter(param_specs: Tuple[ParamSpec, ...]):
            return OutModule(
                mlp_factory=self.mlp_factory,
                param_specs=param_specs,
                enable_segments=self.enable_segments
            )(
                nuc_emb,
                graph_emb,
                orb_emb,
                orb_nuc_emb,
                orb_type,
                orb_idx,
                nuc_idx,
                edge_emb,
                edge_contr,
                config
            )

        # We need to group parameters
        # 1. flatten tree
        specs, tree_def = jtu.tree_flatten(self.param_spec)
        result = [None] * len(specs)
        for _, group in group_specs(specs):
            group = list(group)
            idx = [x[0] for x in group]
            specs = [x[2] for x in group]
            if self.enable_groups:
                # some sanity checks
                assert all(specs[0].param_type == x.param_type for x in specs)
                assert all(specs[0].transform == x.transform for x in specs)
                assert all(specs[0].segments == x.segments for x in specs)
                assert all(specs[0].use_bias == x.use_bias for x in specs)
                ys = gen_parameter(specs)
            else:
                ys = sum((gen_parameter((s,)) for s in specs), ())
            for i, y in zip(idx, ys):
                result[i] = y
        
        return jtu.tree_unflatten(tree_def, result)


def group_specs(specs: list[ParamSpec]) -> list[Tuple[int, str, ParamSpec]]:
    """
    Groups a list of ParamSpec objects by their group attribute.

    Args:
        specs: A list of ParamSpec objects.

    Returns:
        A list of tuples, each containing 
        the original position of the ParamSpec object,
        the group name,
        and the ParamSpec object.
    """
    # 1. Assign original positions
    specs = [
        (i, f'None_{i}' if x.group is None else x.group, x)
        for i, x in enumerate(specs)
    ]
    # 2. Select group name as key
    keyfunc = itemgetter(1)
    # 3. Group by group name
    param_groups = groupby(sorted(specs, key=keyfunc), keyfunc)
    # 4. Iterate through groups and generate parameters
    return param_groups


class MetaNet(ReparametrizedModule):
    """
    A module that implements the MetaNet architecture.

    Args:
        param_spec: A dictionary of ParamSpec objects that define the shape of the output parameters.
        layers: A sequence of tuples, each containing the message and out dimension for each layer.
        embedding_dim: An integer representing the dimensionality of the embeddings.
        edge_embedding: A string representing the type of edge embedding to use.
        edge_embedding_params: A dictionary containing the parameters for the edge embedding.
        orb_edge_params: A dictionary containing the parameters for the orbital edge.
        out_mlp_depth: An integer representing the depth of the output MLP.
        out_scale: A string representing the scaling of hidden units in AutoMLP of the output.
        aggregate_before_out: A boolean indicating whether to aggregate before output.
        activation: An activation function.
        enable_groups: A boolean indicating whether to enable groups.
        enable_segments: A boolean indicating whether to enable segments.
    """
    param_spec: dict
    layers: Sequence[Tuple[int, int]] = ((64, 128), (64, 128), (64, 128))
    embedding_dim: int = 128
    edge_embedding: str = 'MLPEdgeEmbedding'
    edge_embedding_params: dict = field(default_factory=lambda*_: dict(
        # MLP Embedding
        out_dim=16,
        hidden_dim=64,
        activation='silu',
        adaptive_weights=True,
        init_sph=False,
        # Spherical harmonics embedding
        n_rad=6,
        max_l=3,
    ))
    orb_edge_params: dict = field(default_factory=lambda*_: dict(
        param_type=ParamType.ORBITAL
    ))
    out_mlp_depth: int = 3
    out_scale: str = 'log'
    aggregate_before_out: bool = True
    activation: Activation = nn.silu
    enable_groups: bool = False
    enable_segments: bool = False

    @nn.compact
    def __call__(self, nuclei, struc_params, config: SystemConfigs):
        n_nuclei = nuclei_per_graph(config)
        graph_mask = np.arange(len(n_nuclei)).repeat(n_nuclei)
        flat_charges = np.array(tuple(flatten(config.charges)))
        axes = struc_params['axes']
        orb_loc, orb_type, _, N_orbs = struc_params['orbitals']
        nuclei = jnp.einsum('...d,...dk->...k', nuclei, axes[graph_mask])

        ## Initialize edge embeddings
        # Nuclei edge embedding
        edge_module = getattr(GNN_MODULES, self.edge_embedding)
        edge_params = NucOrbParams(edge_module.spec(**self.edge_embedding_params))(flat_charges)
        edge_embedding = functools.partial(edge_module.create(**self.edge_embedding_params), params=edge_params)
        
        # Nuclei norm
        norm_params = NucOrbParams(NormEnvelope.spec())(flat_charges)
        norm_envelope = functools.partial(NormEnvelope(), params=norm_params)

        # Orbital edge embedding
        module_params = {**self.edge_embedding_params, **self.orb_edge_params}
        orb_edge_params = NucOrbParams(edge_module.spec(**module_params))(orbital_types=orb_type)
        orb_edge_embedding = functools.partial(edge_module.create(**module_params), params=orb_edge_params)
        
        # Orbital norm
        orb_norm_params = NucOrbParams(NormEnvelope.spec(param_type=ParamType.ORBITAL))(orbital_types=orb_type)
        orb_norm_envelope = functools.partial(NormEnvelope(param_type=ParamType.ORBITAL), params=orb_norm_params)

        # Compute Nuclei - nuclei edges
        # Construct fully connected graph
        senders, receivers, _ = adj_idx(n_nuclei, drop_diagonal=True)
        edges = nuclei[senders] - nuclei[receivers]
        edges = jnp.concatenate([
            edges,
            jnp.linalg.norm(edges, axis=-1, keepdims=True)
        ], axis=-1)
        e_embed = edge_embedding(edges, receivers, params=edge_params)
        edge_contr = norm_envelope(edges, receivers, params=norm_params)

        # Node embedding
        n_embed = Embed(
            MAX_CHARGE,
            self.embedding_dim
        )(flat_charges)

        # Message passing and update
        embeddings = MessagePassingNetwork(
            self.layers,
            self.activation
        )(n_embed, None, e_embed, edge_contr, senders, receivers)

        # Output
        if self.aggregate_before_out:
            n_embed = jnp.concatenate(embeddings, axis=-1)
        else:
            n_embed = embeddings[-1]
        n_embed = Residual(self.embedding_dim, self.activation)(n_embed)
        n_embed = Residual(self.embedding_dim, self.activation)(n_embed)

        # Create orbital embeddings
        orbitals = OrbitalEmbedding(
            self.embedding_dim,
            edge_embedding=orb_edge_embedding,
            norm_envelope=orb_norm_envelope,
            message_passing=MessagePassingNetwork(
                self.layers,
                self.activation),
            activation=self.activation,
            aggregate_before_out=self.aggregate_before_out,
        )(nuclei, n_embed, orb_loc, orb_type, N_orbs, config)
        assert len(orbitals) == sum(N_orbs) == len(orb_type)

        parameters = GraphToParameters(
            self.param_spec,
            functools.partial(
                AutoMLP,
                n_layers=self.out_mlp_depth,
                activation=self.activation
            ),
            edge_embedding=orb_edge_embedding,
            norm_envelope=orb_norm_envelope,
            activation=self.activation,
            enable_groups=self.enable_groups,
            enable_segments=self.enable_segments
        )(
            orb_loc,
            orbitals,
            orb_type,
            nuclei,
            n_embed,
            N_orbs,
            config
        )
        return parameters


class NucOrbParams(ReparametrizedModule):
    """
    A module that generates static parameters for nuclei and orbitals.

    Args:
        param_spec: A dictionary of ParamSpec objects that define the shape of the output parameters.

    Returns:
        A dictionary of parameters for nuclei and orbitals.
    """
    param_spec: SpecTree

    @nn.compact
    def __call__(self, charges: Optional[jax.Array] = None, orbital_types: Optional[jax.Array] = None) -> ParamTree:
        """
        Generates parameters for nuclei and orbitals.

        Args:
            charges: An optional array of charges for nuclei.
            orbital_types: An optional array of orbital types.

        Returns:
            A dictionary of parameters for nuclei and orbitals.
        """
        if charges is not None:
            charges = jnp.array(charges).reshape(-1)
        if orbital_types is not None:
            orbital_types = jnp.array(orbital_types).reshape(-1)
        assert orbital_types is not None or charges is not None
        
        def gen_param(param_spec: ParamSpec) -> jax.Array:
            out_dim = np.prod(param_spec.shape)
            assert param_spec.param_type in (ParamType.NUCLEI, ParamType.ORBITAL)
            out = Embed(
                MAX_CHARGE if param_spec.param_type is ParamType.NUCLEI else MAX_ORB,
                out_dim,
                embedding_init=lambda key, shape, dtype=jnp.float_: jax.random.normal(key, shape, dtype) * param_spec.std + param_spec.mean
            )(charges if param_spec.param_type is ParamType.NUCLEI else orbital_types)
            return param_spec.transform(out).reshape(-1, *param_spec.shape)
        
        return jax.tree_util.tree_map(
            gen_param,
            self.param_spec
        )


class PlaceHolderGNN(ReparametrizedModule):
    """
    A module that generates static parameters for nuclei and orbitals.

    Args:
        param_spec: A dictionary of ParamSpec objects that define the shape of the output parameters.

    Returns:
        A dictionary of parameters for nuclei and orbitals.
    """
    param_spec: SpecTree

    @nn.compact
    def __call__(self, nuclei: jax.Array, axes: jax.Array, config: SystemConfigs) -> ParamTree:
        """
        Generates parameters for nuclei and orbitals.

        Args:
            nuclei: An array of nuclei.
            axes: An array of axes.
            config: A SystemConfigs object.

        Returns:
            A dictionary of parameters for nuclei and orbitals.
        """
        n_nuclei = nuclei_per_graph(config)
        n_graphs = len(config.spins)
        nuclei = nuclei.reshape(-1, 3)
        
        n_orb = np.array(config.spins).max(-1).sum()
        n_orb_nuc = (n_orb * np.array(n_nuclei)).sum()
        n_nuclei = np.sum(n_nuclei)

        i = 0
        def gen_parameter(param_spec: ParamSpec):
            nonlocal i
            i += 1
            if param_spec.param_type == ParamType.ORBITAL:
                shape = (n_orb, *param_spec.shape)
            elif param_spec.param_type == ParamType.ORBITAL_NUCLEI:
                shape = (n_orb_nuc, *param_spec.shape)
            elif param_spec.param_type == ParamType.NUCLEI:
                shape = (n_nuclei, *param_spec.shape)
            elif param_spec.param_type == ParamType.GLOBAL:
                shape = (n_graphs, *param_spec.shape)
            else:
                raise ValueError()
            
            return param_spec.transform(self.param(
                f'orb_bias_{i}',
                lambda key, shape: jax.random.normal(key, shape)*param_spec.std + param_spec.mean,
                shape
            ))
        
        return jax.tree_util.tree_map(
            gen_parameter,
            self.param_spec
        )
