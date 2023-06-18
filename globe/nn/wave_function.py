import functools
from dataclasses import field
from typing import Optional, Type

import flax.linen as nn
import jax
import jax.numpy as jnp
import jraph
import numpy as np

from globe.nn import Activation, AutoMLP, ParamTree, ReparametrizedModule
from globe.nn.orbitals import OrbitalModule, ProductOrbitals
from globe.nn.parameters import ParamSpec, ParamType, SpecTree
from globe.utils.config import (SystemConfigs, group_by_config, inverse_group_idx,
                                  nuclei_per_graph)


def log_sum_det(xs: tuple[jax.Array], w: jax.Array) -> tuple[jax.Array, jax.Array]:
    """
    Computes the logarithm of the weighted sum of the determinants of a list of matrices.

    Args:
    - xs: A list of matrices.
    - w: A weight vector.

    Returns:
    A tuple containing:
    - sign_out: The sign of the output.
    - log_out: The logarithm of the absolute value of the output.
    """
    det1 = functools.reduce(
        lambda a, b: a*b,
        [x.reshape(-1) for x in xs if x.shape[-1] == 1],
        jnp.ones(())
    )

    sign_in, logdet = functools.reduce(
        lambda a, b: (a[0]*b[0], a[1]+b[1]),
        [jnp.linalg.slogdet(x) for x in xs if x.shape[-1] > 1],
        (jnp.ones(()), jnp.zeros(()))
    )

    maxlogdet = jax.lax.stop_gradient(jnp.max(logdet))
    det = sign_in * det1 * jnp.exp(logdet - maxlogdet)

    result = jnp.vdot(det, w)
    sign_out = jnp.sign(result)
    log_out = jnp.log(jnp.abs(result)) + maxlogdet
    return sign_out, log_out


class LogSumDet(ReparametrizedModule):
    """
    A module that computes the logarithm of the weighted sum of the determinants of a list of matrices.

    Args:
    - adaptive: A boolean indicating whether the weighting parameters are adative.
    """
    adaptive: bool

    @staticmethod
    def param_spec(adaptive, n_det):
        if not adaptive:
            return {}
        return {
            'w': ParamSpec(ParamType.GLOBAL, (n_det,), 1.0, 0.0)
        }

    @nn.compact
    def __call__(self, xs, params, config: SystemConfigs):
        signs, log_psis = [], []
        if self.adaptive:
            for x, w in zip(xs, group_by_config(config, params['w'], lambda *_: 1)):
                s, l = jax.vmap(log_sum_det)(x, w)
                signs.append(s)
                log_psis.append(l)
        else:
            weight = self.define_parameters(True, xs[0][0].shape[1])['w']
            for x in xs:
                s, l = jax.vmap(log_sum_det, in_axes=(0, None))(x, weight)
                signs.append(s)
                log_psis.append(l)
        return jnp.concatenate(signs, axis=0), jnp.concatenate(log_psis, axis=0)


class Jastrow(ReparametrizedModule):
    """
    A module that computes the Jastrow factor of the wave function.

    Args:
    - mlp: An optional neural network module that computes the Jastrow factor.
    - include_pair: A boolean indicating whether to include the pairwise term in the Jastrow factor.
    - adaptive: A boolean indicating whether the weighting parameters are adaptive.

    Methods:
    - param_spec(adaptive: bool) -> Dict[str, ParamSpec]: Returns the parameter specification for the module.
    - __call__(self, h_one: jax.Array, h_two: jax.Array, config: SystemConfigs, params: Optional[Dict[str, jax.Array]] = None) -> jax.Array:
        Computes the Jastrow factor of the wave function.
    """
    mlp: Optional[nn.Module]
    include_pair: bool
    adaptive: bool

    @staticmethod
    def param_spec(adaptive):
        if not adaptive:
            return {}
        return {
            'mlp_weight': ParamSpec(
                ParamType.GLOBAL,
                shape=(1,),
                mean=0,
                std=0
            ),
            'fixed_weight': ParamSpec(
                ParamType.GLOBAL,
                shape=(2,),
                mean=1e-2,
                std=0
            ),
            'alpha': ParamSpec(
                ParamType.GLOBAL,
                shape=(2,),
                mean=1,
                std=0
            ),
        }
    
    @nn.compact
    def __call__(self, h_one: jax.Array, h_two: jax.Array, config: SystemConfigs, params: ParamTree = None) -> jax.Array:
        """
        Computes the wave function of a molecular system.

        Args:
        - h_one: A tensor of shape (n_elec, n_dim) representing the one-electron features.
        - h_two: A tuple of two tensors representing the two-electron features.
        - config: A `SystemConfigs` object containing the configuration of the system.
        - params: A dictionary containing the parameters of the module. If `None`, the default parameters are used.

        Returns:
        - A tensor of shape `(n_elec,)` representing the wave function of the system.
        """
        if not self.adaptive:
            params = self.define_parameters(True)
        spins = np.array(config.spins)
        n_elec = np.sum(spins, axis=-1)
        out = jnp.zeros((n_elec.size,))
        if self.mlp is not None:
            weight = params['mlp_weight'].squeeze(-1)
            out_before_sum = self.mlp(h_one)
            
            segments = np.repeat(np.arange(n_elec.size), n_elec.reshape(-1))
            out = weight * jraph.segment_sum(
                out_before_sum,
                segments,
                n_elec.size,
                True
            ).reshape(n_elec.size)

        if self.include_pair:
            weight, alpha = params['fixed_weight'], params['alpha']
            a_par_w, a_anti_w = weight[..., 0], weight[..., 1]
            a_par, a_anti = alpha[..., 0], alpha[..., 1]
            if self.adaptive:
                a_par = a_par[..., 0].repeat((spins**2 - spins).sum(-1))
                a_anti = a_anti[..., 1].repeat(np.prod(spins, -1) * 2)
            (same, diff), (same_seg, diff_seg) = h_two
            same, diff = same[:, -1], diff[:, -1]
            
            # Jastrow taken from: https://openreview.net/pdf?id=xveTeHVlF7j
            out += a_par_w * jraph.segment_sum(
                -(1/4) * a_par**2 / (a_par + same),
                same_seg,
                n_elec.size,
                True
            )
            out += a_anti_w * jraph.segment_sum(
                -(1/2) * a_anti**2 / (a_anti + diff),
                diff_seg,
                n_elec.size,
                True
            )
        return out


class WaveFunction(ReparametrizedModule):
    """
    A module that computes the wave function of a molecular system.

    Args:
    - activation: An activation function to use in the neural network.
    - orbital_type: A type of orbital module to use in the wave function.
    - orbital_config: A dictionary of configuration options for the orbital module.
    - jastrow_mlp_layers: The number of layers in the Jastrow factor neural network.
    - jastrow_include_pair: A boolean indicating whether to include the pairwise term in the Jastrow factor.
    - adaptive_jastrow: A boolean indicating whether the weighting parameters in the Jastrow factor are adaptive.
    - adaptive_sum_weights: A boolean indicating whether the weights in the sum of determinants are adaptive.

    Methods:
    - param_spec(out_dim: int, shared_orbitals: bool, full_det: bool, determinants: int, orbital_type: Type[OrbitalModule], orbital_config: dict, adaptive_jastrow: bool, adaptive_sum_weights: bool) -> SpecTree:
        Returns the parameter specification for the module.
    - setup(self):
        Sets up the module.
    - _encode(self, xs: Tuple[jax.Array], params: Dict[str, jax.Array], config: SystemConfigs) -> tuple[jax.Array, jax.Array, jax.Array]:
        Encodes the input into a feature vector. To be implemented by subclasses.
    - encode(self, xs: Tuple[jax.Array], params: Dict[str, jax.Array], config: SystemConfigs) -> tuple[jax.Array, jax.Array, jax.Array]:
        Interface around _encode that handles unified preprocessing.
    - orbitals(self, xs: Tuple[jax.Array], params: Dict[str, jax.Array], config: SystemConfigs) -> tuple[tuple[jax.Array, ...], ...]:
        Computes the orbitals of the wave function.
    - signed(self, xs: Tuple[jax.Array], params: Dict[str, jax.Array], config: SystemConfigs) -> tuple[jax.Array, jax.Array]:
        Computes the sign and log absolute value of the wave function.
    - __call__(self, xs: Tuple[jax.Array], params: Optional[Dict[str, jax.Array]] = None, config: Optional[SystemConfigs] = None, signed: bool = False) -> jax.Array:
        Computes the log absolute value of the wave function of a quantum system.
    """
    activation: Activation = 'silu'
    orbital_type: Type[OrbitalModule] = ProductOrbitals
    orbital_config: dict = field(default_factory=lambda *_: {
        'separate_k': True
    })
    jastrow_mlp_layers: int = 3
    jastrow_include_pair: bool = True
    adaptive_jastrow: bool = False
    adaptive_sum_weights: bool = False

    @staticmethod
    def param_spec(
        out_dim: int,
        shared_orbitals: bool,
        full_det: bool,
        determinants: int,
        orbital_type: Type[OrbitalModule],
        orbital_config: dict,
        adaptive_jastrow: bool,
        adaptive_sum_weights: bool
    ) -> SpecTree:
        return {
            'orbitals': orbital_type.spec(
                shared_orbitals=shared_orbitals,
                full_det=full_det,
                orbital_inp=out_dim,
                determinants=determinants,
                **orbital_config
            ),
            'logsumdet': LogSumDet.spec(adaptive=adaptive_sum_weights, n_det=determinants),
            'jastrow': Jastrow.spec(adaptive=adaptive_jastrow)
        }

    def setup(self):
        self.orbitals = self.orbital_type.create(**self.orbital_config, activation=self.activation)
        self.logsumdet = LogSumDet(self.adaptive_sum_weights)

        self.jastrow = Jastrow(
            AutoMLP(
                1,
                n_layers=self.jastrow_mlp_layers,
                activation=self.activation
            ) if self.jastrow_mlp_layers > 0 else None,
            self.jastrow_include_pair,
            adaptive=self.adaptive_jastrow
        )
    
    def _encode(self, electrons: jax.Array, atoms: jax.Array, config: SystemConfigs, params: ParamTree) -> tuple[jax.Array, jax.Array, jax.Array]:
        raise NotImplementedError()

    def encode(self, electrons: jax.Array, atoms: jax.Array, config: SystemConfigs, params: ParamTree) -> tuple[jax.Array, jax.Array, jax.Array]:
        spins, n_nuclei = config.spins, nuclei_per_graph(config)
        n_elec = np.sum(config.spins, -1)
        n_graph_mask = np.arange(len(n_nuclei)).repeat(n_nuclei)
        e_graph_mask = np.arange(len(n_elec)).repeat(n_elec)
        axes = params['axes']
        # Prepare input
        atoms = atoms.reshape(-1, 3)
        electrons = electrons.reshape(-1, 3)
        assert electrons.shape[0] == np.sum(spins)
        assert atoms.shape[0] == sum(n_nuclei)
        atoms = jnp.einsum('...d,...dk->...k', atoms, axes[n_graph_mask])
        electrons = jnp.einsum('...d,...dk->...k', electrons, axes[e_graph_mask])

        return self._encode(electrons, atoms, config, params)

    def orbitals(self, electrons: jax.Array, atoms: jax.Array, config: SystemConfigs, params: ParamTree) -> tuple[tuple[jax.Array, ...], ...]:
        h_one, _, r_im = self.encode(electrons, atoms, config, params)
        orbits = self.orbitals(params['orbitals'], h_one, r_im, config)
        # Flatten and reconstruct original order
        orbits = [orbs for group in orbits for orbs in zip(*group)]
        result = [orbits[i] for i in inverse_group_idx(config)]
        return result

    def signed(self, electrons: jax.Array, atoms: jax.Array, config: SystemConfigs, params: ParamTree) -> tuple[jax.Array, jax.Array]:
        # Compute orbitals
        h_one, r_ij, r_im = self.encode(electrons, atoms, config, params)
        orbits = self.orbitals(params['orbitals'], h_one, r_im, config)
        # Compute log det
        sign, log_psi = self.logsumdet(orbits, params['logsumdet'], config)
        # Reconstruct original order
        idx = inverse_group_idx(config)
        sign, log_psi = sign[idx], log_psi[idx]

        # Jastrow factor
        log_psi += self.jastrow(h_one, r_ij, config)

        return sign, log_psi

    def __call__(self, electrons: jax.Array, atoms: jax.Array, config: SystemConfigs, params: ParamTree) -> jax.Array:
        return self.signed(electrons, atoms, config, params)[1]
