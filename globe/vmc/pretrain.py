import functools
from typing import Callable, List, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from scipy.special import factorial2
from globe.systems.scf import Scf
from globe.nn.parameters import ParamSpec, ParamTree, SpecTree, INVERSE_TRANSFORMS

from globe.utils import iterate_segments
from globe.utils.jax_utils import pmap, pmean_if_pmap


def eval_orbitals(scf_approx: List[Scf], electrons: jax.Array, spins: Tuple[int, int]) -> Tuple[jax.Array, jax.Array]:
    """Returns the molecular orbitals of Hartree Fock calculations.

    Args:
        scf_approx (List[Scf]): Hartree Fock calculations, length H
        electrons ([type]): (B, N*H, 3)
        spins ([type]): list of length H with (spin_up, spin_down)

    Returns:
        List of length H where each element is a tuple of ((B, up, up), (B, down, down))
    """
    assert len(scf_approx) == len(spins)
    assert np.sum(spins) == electrons.shape[-2]
    result = []
    for scf, elec, spins in zip(
        scf_approx,
        iterate_segments(electrons, np.sum(spins, axis=-1), axis=-2),
        spins
    ):
        mos = np.array(scf.eval_molecular_orbitals(elec.reshape(-1, 3)))
        mos = mos.reshape(2, *elec.shape[:-1], mos.shape[-1])
        mo_alpha = mos[0, ..., :spins[0], :spins[0]]
        mo_beta = mos[1, ..., spins[0]:, :spins[1]]
        result.append((mo_alpha, mo_beta))
    return result


def make_mol_param_loss(spec_tree: SpecTree, scale: float, max_moment: int = 4, eps=1e-6) -> Callable[[ParamTree], jax.Array]:
    """
    Computes the loss for the molecular parameters.
    The loss is the difference between the target moments and the observed moments.

    Args:
    - spec_tree: The parameter specification tree
    - scale: The scale of the loss
    - max_moment: The maximum moment to compute
    - eps: A small number to avoid division by zero
    Returns:
    - A function that computes the loss
    """
    p = np.arange(1, max_moment+1)
    # all odd moments are 0
    # https://en.wikipedia.org/wiki/Normal_distribution#Moments:~:text=standard%20normal%20distribution.-,Moments,-See%20also%3A
    target_moments = 1**p * factorial2(p - 1) * (1 - p % 2)
    
    def distr_loss(param: jax.Array, spec: ParamSpec):
        if not spec.keep_distr:
            return 0
        if isinstance(spec.keep_distr, float):
            scale = spec.keep_distr
        else:
            scale = 1
        # We must reverse the transformation applied to the parameters if possible
        if spec.transform in INVERSE_TRANSFORMS:
            param = INVERSE_TRANSFORMS[spec.transform](param)

        p_norm = (param - spec.mean) / (eps + spec.std)
        x = (p_norm[..., None] ** p)
        # average over all but last dim
        observed_moments = x.mean(axis=tuple(range(x.ndim - 1)))
        return scale * ((target_moments - observed_moments) ** 2).sum()

    def loss(mol_params):
        result = jtu.tree_reduce(jnp.add, jtu.tree_map(distr_loss, mol_params, spec_tree))
        return scale * jnp.sum(result)
    return loss


def make_pretrain_step(
    mcmc_step: Callable,
    mol_param_fn: Callable,
    orbital_fn: Callable,
    opt_update: Callable,
    full_det: bool,
    mol_param_aux_loss: Callable = None,
    natgrad_precond: Callable = None
):
    """
    Creates a pretrain step function for the molecular parameters and the orbitals.
    
    Args:
    - mcmc_step: A function that performs a single MCMC step
    - mol_param_fn: A function that computes the molecular parameters
    - orbital_fn: A function that computes the orbitals
    - opt_update: A function that updates the optimizer state
    - full_det: Whether to train with full determinants or not
    - mol_param_aux_loss: An auxiliary loss for the molecular parameters
    - natgrad_precond: A function that computes the preconditioner for the natural gradient
    Returns:
    - A function that performs a single pretrain step
    """
    orbital_fn = jax.vmap(orbital_fn, in_axes=(None, 0, None, None, None))
    orbital_fn = jax.jit(orbital_fn, static_argnums=3)

    @functools.partial(pmap, in_axes=(0, 0, None, None, 0, 0, 0, None, 0), static_broadcasted_argnums=3)
    def pretrain_step(params, electrons, atoms, config, targets, opt_state, key, properties, natgrad_state):
        def loss_fn(p):
            # Orbitals are a list of tuples
            # if full_det=True the tuple has one element of (K, N, N)
            # Otherwise the output is (K, alpha, alpha), (K, beta, beta)
            mol_params = mol_param_fn({**params, 'params': p}, atoms, config)
            orbitals = orbital_fn(
                params,
                electrons,
                atoms,
                config,
                mol_params
            )
            
            if full_det:
                # If we train with full_det=True we only optimize the
                # block diagonals
                orbitals = [
                    (orbs[0][..., :na, :na], orbs[0][..., na:, na:])
                    for orbs, (na,  _) in zip(orbitals, config.spins)
                ]
            
            loss = 0
            # over different molecules
            for targ, orbs in zip(targets, orbitals):
                # over up and down orbitals
                for t, o in zip(targ, orbs):
                    # we multiply the targets with a random matrix with unit determinant
                    # to avoid learning n_det identical orbital functions
                    # since we use numpy here, these matrices are fixed!
                    #n = t.shape[-1]
                    #d = o.shape[-3]
                    #mats = np.eye(n) + np.random.normal(size=(d, n, n)) * 1e-2
                    #mats /= np.linalg.det(mats)[..., None, None] ** (1/n) # normalize determinant
                    # compute new targets
                    #t = jnp.einsum('...nm,...kmo->...kno', t, mats)
                    loss += jnp.mean((t[..., None, :, :] - o)**2)
            loss /= len(orbitals)

            total_loss = loss
            if mol_param_aux_loss is not None:
                total_loss += mol_param_aux_loss(mol_params)
            return total_loss, loss

        ((total_loss, loss_val), grad) = pmean_if_pmap(jax.value_and_grad(loss_fn, has_aux=True)(params['params']))
        if natgrad_precond is not None:
            grad, natgrad_state = natgrad_precond(params, electrons, atoms, config, grad, natgrad_state)
        
        updates, opt_state = opt_update(grad, opt_state, params['params'])
        params['params'] = optax.apply_updates(params['params'], updates)

        key, subkey = jax.random.split(key)
        electrons, pmove = mcmc_step(
            params,
            electrons,
            atoms,
            config,
            subkey,
            properties['mcmc_width']
        )
        return params, electrons, opt_state, natgrad_state, loss_val, pmove
    return pretrain_step
