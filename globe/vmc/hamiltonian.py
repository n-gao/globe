import functools
from typing import Callable
import jax
import jax.numpy as jnp
from jax import lax 
import jraph
import numpy as np


from globe.nn.ferminet import netjit
from globe.utils import flatten, nuclei_per_graph, triu_idx, adj_idx, SystemConfigs, tree_generator_zip
from globe.utils.config import group_by_config, inverse_group_idx


def make_kinetic_energy_function(f, group_parameters_fn, linearize: bool = True) -> Callable:
    """
    Returns a function that computes the kinetic energy of the system.
    
    Args:
    - f: function that computes the wave function
    - group_parameters_fn: function that groups the parameters by configuration
    - linearize: whether to linearize the gradient of f
    Returns:
    - function that computes the kinetic energy of the system
    """
    @netjit
    def laplacian_of_f(params, electrons, atoms, config: SystemConfigs, mol_params):
        @functools.partial(jax.vmap, in_axes=(0, 0, None, 0))
        def _laplacian(elec, atoms, config, mol_params):
            def f_closure(x): return f(params, x, atoms, config, mol_params).squeeze()
            elec = elec.reshape(-1)
            n = elec.shape[0]
            eye = jnp.eye(n)

            grad_f = jax.grad(f_closure)
            if linearize:
                primal, dgrad_f = jax.linearize(grad_f, elec)
                def ith_sum_element(carry: jax.Array, i: int) -> jax.Array:
                    return carry + dgrad_f(eye[i])[i], None
                return -0.5 * (lax.scan(ith_sum_element, jnp.zeros(()), jnp.arange(n))[0] + (primal ** 2).sum())
            else:
                def ith_sum_element(carry: jax.Array, i: int) -> jax.Array:
                    grad, gradgrad = jax.jvp(grad_f, (elec,), (eye[i],))
                    return carry + grad[i]**2 + gradgrad[i], None
                return -0.5 * lax.scan(ith_sum_element, jnp.zeros(()), jnp.arange(n))[0]

        result = []
        for (elec, (spins, charges)), at, pm in tree_generator_zip(
            group_by_config(config, electrons, lambda s, _: np.sum(s), return_config=True),
            group_by_config(config, atoms, lambda _, c: len(c)),
            group_parameters_fn(mol_params, config)
        ):
            conf = SystemConfigs((spins,), (charges,))
            result.append(_laplacian(elec, at, conf, pm))
        idx = inverse_group_idx(config)
        return jnp.concatenate(result)[idx]
    return laplacian_of_f


def potential_energy(electrons: jax.Array, atoms: jax.Array, config: SystemConfigs) -> jax.Array:
    """
    Computes the potential energy of the system.
    
    Args:
    - electrons: (n_electrons, 3) array of electron positions
    - atoms: (n_atoms, 3) array of atom positions
    - config: SystemConfigs object containing the spin and charge of each atom
    Returns:
    - (n_graphs,) array of potential energies
    """
    charges = np.fromiter(flatten(config.charges), np.int32)
    spins, n_nuclei = config.spins, nuclei_per_graph(config)
    electrons = electrons.reshape(-1, 3)
    n_graphs = len(spins)

    i, j, m = triu_idx(np.sum(spins, -1), 1)
    r_ee = jnp.linalg.norm(electrons[i] - electrons[j], axis=-1)
    v_ee = jraph.segment_sum(
        1.0/r_ee,
        m,
        n_graphs,
        True
    )
    
    i, j, m = adj_idx(np.sum(spins, -1), n_nuclei)
    r_ae = jnp.linalg.norm(electrons[i] - atoms[j], axis=-1)
    v_ae = -jraph.segment_sum(
        charges[j] / r_ae,
        m,
        n_graphs,
        True
    )

    i, j, m = triu_idx(n_nuclei, 1)
    r_aa = jnp.linalg.norm(atoms[i] - atoms[j], axis=-1)
    v_aa = jraph.segment_sum(
        charges[i] * charges[j] / r_aa,
        m,
        n_graphs,
        True
    )
    return v_ee + v_ae + v_aa


def make_local_energy_function(f, group_parameters_fn, linearize: bool = True) -> Callable:
    """
    Returns a function that computes the local energy of the system.
    
    Args:
    - f: function that computes the wavefunction
    - group_parameters_fn: function that groups the parameters by graph
    - linearize: whether to linearize the function f
    Returns:
    - function that computes the local energy
    """
    kinetic_energy_fn = make_kinetic_energy_function(f, group_parameters_fn, linearize=linearize)

    @netjit
    def local_energy(
            params,
            electrons,
            atoms,
            config: SystemConfigs,
            mol_params) -> jax.Array:
        potential = potential_energy(electrons, atoms, config)
        kinetic = kinetic_energy_fn(params, electrons, atoms, config, mol_params)
        return potential + kinetic
    return local_energy
