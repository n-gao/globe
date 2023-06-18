import functools
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax
from globe.nn.parameters import ParamTree
from globe.utils.config import SystemConfigs

from globe.utils.jax_utils import pmap, pmean_if_pmap, pgather
from globe.utils.jnp_utils import tree_dot
from globe.utils.optim import NaturalGradientState


def local_energy_diff(
        e_loc: jax.Array,
        clip_local_energy: float,
        stat: str = 'median'
) -> jax.Array:
    """
    Compute the difference between the local energy and its mean/median.
    Also clips the local energy if requested.
    
    Args:
    - e_loc: local energy
    - clip_local_energy: clipping range for the local energy
    - stat: statistic to use for centering the local energy
    Returns:
    - centered local energy
    """
    if stat == 'mean':
        stat_fn = jnp.mean
    elif stat == 'median':
        stat_fn = jnp.median
    else:
        raise ValueError()
    if clip_local_energy > 0.0:
        full_e = pgather(e_loc)
        clip_center = stat_fn(full_e, axis=(0, 1))
        mad = jnp.abs(full_e - clip_center).mean(axis=(0, 1))
        max_dev = clip_local_energy * mad
        e_loc = jnp.clip(e_loc, clip_center-max_dev, clip_center+max_dev)
    center = pmean_if_pmap(jnp.mean(e_loc, axis=0, keepdims=True))
    e_loc -= center
    return e_loc


def make_loss_and_natural_gradient_fn(
        network,
        natgrad_precond,
        clip_local_energy: float,
        target_std: float = None,
        limit_scaling: bool = True,
        **_):
    """
    Returns a function that computes the loss and the preconditioned natural gradient.

    Args:
    - network: the electronic wave function
    - natgrad_precond: the preconditioner for the natural gradient
    - clip_local_energy: clipping range for the local energy
    - target_std: target standard deviation for the local energy
    - limit_scaling: whether to limit the scaling of the local energy
    Returns:
    - loss_and_grad: function that computes the loss and the preconditioned natural gradient
    """
    network = jax.vmap(network, in_axes=(None, 0, None, None))
    network = jax.jit(network, static_argnums=3)
    
    @functools.partial(jax.jit, static_argnums=3, static_argnames='config')
    def loss_and_grad(
            params: ParamTree,
            electrons: jax.Array,
            atoms: jax.Array,
            config: SystemConfigs,
            e_l: jax.Array,
            std_ema: jax.Array,
            natgrad_state: NaturalGradientState) -> tuple[tuple[jax.Array, jax.Array, NaturalGradientState], dict[str, jax.Array]]:
        aux_data = {}
        n = e_l.size

        diff = local_energy_diff(e_l, clip_local_energy)
        E = pmean_if_pmap(jnp.mean(e_l, axis=-2))

        if target_std is not None:
            scaling = target_std/std_ema
            if limit_scaling:
                scaling = jnp.minimum(jnp.full_like(std_ema, target_std), scaling)
        else:
            scaling = jnp.ones_like(E)

        def loss(p):
            log_psi = network({**params, 'params': p}, electrons, atoms, config)
            return jnp.vdot(log_psi, diff * scaling)/n
        grad = jax.grad(loss)(params['params'])
        grad = pmean_if_pmap(grad)
        
        natgrad, natgrad_state = natgrad_precond(params, electrons, atoms, config, grad, natgrad_state)
        
        # Aux data
        aux_data['grad_norm'] = {
            'final': jnp.sqrt(tree_dot(natgrad, natgrad)),
            'euclidean': jnp.sqrt(tree_dot(grad, grad))
        }
        aux_data['damping'] = natgrad_state.damping

        return (E, natgrad, natgrad_state), aux_data
    return loss_and_grad


def make_std_based_damping_fn(base: float, target_pow: float = 0.5, decay: float = 0.999, **_) -> Callable[[jax.Array, jax.Array], jax.Array]:
    """
    Returns a function that computes the damping based on the standard deviation of the local energy.
    
    Args:
    - base: base damping
    - target_pow: power of the standard deviation to use for the target
    - decay: decay factor for the damping
    Returns:
    - damping_fn: function that computes the damping
    """
    # A simple damping scheme based on the standard deviation of the local energy.
    def data_based(damping, e_l):
        mean = pmean_if_pmap(jnp.mean(e_l, axis=-2))
        E_std = (pmean_if_pmap(jnp.mean((e_l - mean)**2, axis=-2)) ** 0.5).mean()
        target = pmean_if_pmap(base * jnp.power(E_std, target_pow))
        damping = jnp.where(damping < target, damping/decay, damping)
        damping = jnp.where(damping > target, decay*damping, damping)
        damping = jnp.clip(damping, 1e-8, 1e-1)
        return damping
    return data_based


def make_training_step(
        mcmc_step,
        val_and_grad,
        el_fn,
        damping_fn,
        opt_update):
    """
    Returns a function that performs a training step.
    
    Args:
    - mcmc_step: function that performs a MCMC step
    - val_and_grad: function that computes the loss and the preconditioned natural gradient
    - el_fn: function that computes the local energy
    - damping_fn: function that computes the damping
    - opt_update: function that updates the optimizer state
    Returns:
    - step: function that performs a training step
    """
    el_fn = jax.vmap(el_fn, in_axes=(None, 0, None, None))
    el_fn = jax.jit(el_fn, static_argnums=3)

    @functools.partial(pmap, in_axes=(0, 0, None, None, 0, 0, 0, None), static_broadcasted_argnums=3)
    def step(params, electrons, atoms, config, opt_state, key, natgrad_state, properties):
        key, subkey = jax.random.split(key)
        electrons, pmove = mcmc_step(
            params, electrons, atoms, config, subkey, properties['mcmc_width'])

        e_l = el_fn(params, electrons, atoms, config)

        # Compute damping
        natgrad_state = NaturalGradientState(
            damping=pmean_if_pmap(damping_fn(natgrad_state.damping, e_l)),
            last_grad=natgrad_state.last_grad
        )

        aux_data = {}
        
        (E, grads, natgrad_state), aux_data = val_and_grad(
            params=params,
            electrons=electrons,
            atoms=atoms,
            config=config,
            e_l=e_l,
            std_ema=properties['std_ema'],
            natgrad_state=natgrad_state,
        )

        # Compute variance per atom configuration
        E_var = pmean_if_pmap(jnp.mean((e_l - E)**2, axis=-2))
        E_std = E_var ** 0.5
        E_err = E_std / np.sqrt(e_l.shape[-2] * jax.device_count())

        # Optimizer step
        # gradients are already pmeaned
        updates, opt_state = opt_update(grads, opt_state, params['params'])
        params['params'] = optax.apply_updates(params['params'], updates)

        mol_data = dict(
            pmove=pmove,
            E=E,
            E_std=E_std,
            E_var=E_var,
            E_err=E_err
        )
        aux_data.update(mol_data)

        return (electrons, params, opt_state, natgrad_state), mol_data, aux_data
    return step
