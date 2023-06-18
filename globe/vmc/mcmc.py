from typing import Callable
from chex import PRNGKey
import jax
import jax.numpy as jnp
import numpy as np

from globe.nn.ferminet import netjit
from globe.nn.parameters import ParamTree
from globe.utils import SystemConfigs

from globe.utils.jax_utils import pmean_if_pmap


def make_mh_update(
    logprob_fn: Callable[[jax.Array], jax.Array],
    widths: jax.Array,
    n_elecs: int
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, PRNGKey, jax.Array, jax.Array]]:
    """
    Returns a function that performs a Metropolis-Hastings update.
    
    Args:
    - logprob_fn: function that computes the log probability of the system
    - widths: widths of the Gaussian proposal distribution
    - n_elecs: number of electrons
    Returns:
    - function that performs a Metropolis-Hastings update
    """
    def mh_update(
        electrons: jax.Array,
        key: PRNGKey,
        log_prob: jax.Array,
        num_accepts: jax.Array
    ) -> tuple[jax.Array, PRNGKey, jax.Array, jax.Array]:
        key, subkey = jax.random.split(key)
        eps = jax.random.normal(subkey, electrons.shape) * jnp.repeat(widths, n_elecs, axis=-1)[...,None]
        new_electrons = electrons + eps
        log_prob2 = logprob_fn(new_electrons)
        ratio = log_prob2 - log_prob

        key, subkey = jax.random.split(key)
        alpha = jnp.log(jax.random.uniform(subkey, ratio.shape))
        cond = ratio > alpha
        new_electrons = jnp.where(jnp.repeat(cond, n_elecs, axis=-1)[..., None], new_electrons, electrons)
        new_log_prob = jnp.where(cond, log_prob2, log_prob)
        num_accepts += cond

        return new_electrons, key, new_log_prob, num_accepts
    return mh_update


def make_mcmc(
    network: Callable[[ParamTree, jax.Array, jax.Array, SystemConfigs, ParamTree], jax.Array],
    steps: int
) -> Callable[[ParamTree, jax.Array, jax.Array, SystemConfigs, ParamTree, PRNGKey, jax.Array], tuple[jax.Array, jax.Array]]:
    """
    Returns a function that performs MCMC sampling.
    
    Args:
    - network: function that computes the log probability of the system
    - steps: number of MCMC steps
    Returns:
    - function that performs MCMC sampling
    """
    batch_network = jax.vmap(network, in_axes=(None, 0, None, None, None))
    @netjit
    def mcmc_step(
        params: ParamTree,
        electrons: jax.Array,
        atoms: jax.Array,
        config: SystemConfigs,
        mol_params: ParamTree,
        key: PRNGKey,
        widths: jax.Array
    ):
        def logprob_fn(x):
            return 2 * batch_network(params, x, atoms, config, mol_params)
        
        mh_update = make_mh_update(logprob_fn, widths, np.sum(config.spins, -1))
        log_probs = logprob_fn(electrons)
        num_accepts = jnp.zeros(log_probs.shape, dtype=jnp.int32)
        
        electrons, key, _, num_accepts = jax.lax.scan(
            lambda x, _: (mh_update(*x), None),
            (electrons, key, log_probs, num_accepts),
            jnp.arange(steps)
        )[0]

        pmove = jnp.sum(num_accepts, axis=0) / (steps * num_accepts.shape[0])
        
        pmove = pmean_if_pmap(pmove)
        return electrons, pmove
    return mcmc_step
