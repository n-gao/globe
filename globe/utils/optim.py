import functools
import logging
import numbers
from collections import namedtuple
from typing import Any, Callable, NamedTuple

import jax
import jax._src.scipy.sparse.linalg as jssl
import jax.lax as lax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
from optax import EmptyState, GradientTransformation

from globe.nn import ParamTree
from globe.nn.globe import Globe
from globe.utils.config import SystemConfigs
from globe.utils.jax_utils import pmean_if_pmap
from globe.utils.jnp_utils import tree_add, tree_scale


class NaturalGradientState(NamedTuple):
    damping: jax.Array
    last_grad: ParamTree


def make_natural_gradient_preconditioner(
    globe: Globe,
    linearize: bool = True,
    precision: str = 'float32',
    **kwargs) -> Callable[[ParamTree, jax.Array, jax.Array, SystemConfigs, jax.Array, NaturalGradientState], tuple[ParamTree, NaturalGradientState]]:
    """
    Returns a function that computes the natural gradient preconditioner for a given set of parameters.
    
    Args:
    - globe: Globe object
    - linearize: whether to linearize the network
    - precision: precision to use for the CG solver
    - kwargs: keyword arguments for the CG solver
    Returns:
    - function that computes the natural gradient preconditioner
    """
    logging.info(f'CG precision: {precision}')
    network = globe.apply
    network = jax.vmap(network, in_axes=(None, 0, None, None, None))
    n_dev = jax.device_count()

    @functools.partial(jax.jit, static_argnums=3, static_argnames='config')
    def nat_cg(params, electrons, atoms, config: SystemConfigs, grad, natgrad_state: NaturalGradientState):
        struc_params = globe.apply(params, atoms, config, method=globe.structure_params)
        with jax.default_matmul_precision(precision):
            # Remove the last two dimensions of electrons to get the batch size
            # multiply by the number of graphs per config
            n = electrons[..., 0, 0].size * len(config.spins)
            def log_p_closure(p):
                return network({**params, 'params': p}, electrons, atoms, config, struc_params)
            
            _, vjp_fn = jax.vjp(log_p_closure, params['params'])
            if linearize:
                _, jvp_fn = jax.linearize(log_p_closure, params['params'])
            else:
                jvp_fn = lambda x: jax.jvp(log_p_closure, params['params'], x)[1]

            def Fisher_matmul(v, damping):
                w = jvp_fn(v) / n
                uncentered = vjp_fn(w)[0]
                result = tree_add(uncentered, tree_scale(damping, v))
                result = pmean_if_pmap(result)
                return result
            
            # Compute natural gradient
            natgrad = cg(
                A=functools.partial(Fisher_matmul, damping=natgrad_state.damping),
                b=grad,
                x0=natgrad_state.last_grad,
                fixed_iter=n_dev > 1, # if we have multiple GPUs we must do a fixed number of iterations
                **kwargs,
            )[0]
        return natgrad, NaturalGradientState(natgrad_state.damping, natgrad)
    return nat_cg


def cg_solve(A, b, x0=None, *, maxiter, M=jssl._identity, min_lookback=10, lookback_frac=0.1, eps=5e-6):
    steps = jnp.arange(maxiter+1)-1
    gaps = (steps*lookback_frac).astype(jnp.int32)
    gaps = jnp.where(gaps < min_lookback, min_lookback, gaps)
    gaps = jnp.where(gaps > steps, steps, gaps)

    def cond_fun(value):
        x, r, gamma, p, k, cache = value
        gap = gaps[k]
        k = k - 1
        relative_still = jnp.logical_and(
            jnp.abs((cache[k] - cache[k - gap])/cache[k]) < eps*gap, gap >= 1)
        over_max = k >= maxiter
        # We check that we are after the third iteration because the first ones may have close to 0 error.
        converged = jnp.logical_and(k > 2, jnp.abs(cache[k]) < 1e-7)
        return ~(relative_still | over_max | converged)

    def body_fun(value):
        x, r, gamma, p, k, cache = value
        Ap = A(p)
        alpha = gamma / jssl._vdot_real_tree(p, Ap)
        x_ = jssl._add(x, jssl._mul(alpha, p))
        r_ = jssl._sub(r, jssl._mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = jssl._vdot_real_tree(r_, z_)
        beta_ = gamma_ / gamma
        p_ = jssl._add(z_, jssl._mul(beta_, p))

        Ax = jssl._add(r_, b)

        val = jtu.tree_reduce(jnp.add, jtu.tree_map(
            lambda a, b, c: jnp.vdot(a-b, c), Ax, b, x_))
        cache_ = cache.at[k].set(val)
        return x_, r_, gamma_, p_, k + 1, cache_

    r0 = jssl._sub(b, A(x0))
    p0 = z0 = M(r0)
    gamma0 = jssl._vdot_real_tree(r0, z0)
    initial_value = (x0, r0, gamma0, p0, 0, jnp.zeros((maxiter,)))

    x_final, _, _, _, _, _ = lax.while_loop(cond_fun, body_fun, initial_value)

    return x_final


def cg_solve_fixediter(A, b, x0=None, *, maxiter, M=jssl._identity):
    # Implementation of CG-method with a fixed number of iterations
    def body_fun(value, i):
        del i
        x, r, gamma, p = value
        Ap = A(p)
        alpha = gamma / jssl._vdot_real_tree(p, Ap)
        x_ = jssl._add(x, jssl._mul(alpha, p))
        r_ = jssl._sub(r, jssl._mul(alpha, Ap))
        z_ = M(r_)
        gamma_ = jssl._vdot_real_tree(r_, z_)
        beta_ = gamma_ / gamma
        p_ = jssl._add(z_, jssl._mul(beta_, p))

        return (x_, r_, gamma_, p_), None

    r0 = jssl._sub(b, A(x0))
    p0 = z0 = M(r0)
    gamma0 = jssl._vdot_real_tree(r0, z0)
    initial_value = (x0, r0, gamma0, p0)

    x_final, _, _, _ = lax.scan(body_fun, initial_value, jnp.arange(maxiter), maxiter)[0]
    return x_final


def cg(A, b, x0=None, *, maxiter=None, min_lookback=10, lookback_frac=0.1, eps=5e-6, M=None, fixed_iter=False):
    """CG-method with the stopping criterium from Martens 2010.

    Args:
        A (Callable): Matrix A in Ax=b
        b (jax.Array): b
        x0 (jax.Array, optional): Initial value for x. Defaults to None.
        maxiter (int, optional): Maximum number of iterations. Defaults to None.
        min_lookback (int, optional): Minimum lookback distance. Defaults to 10.
        lookback_frac (float, optional): Fraction of iterations to look back. Defaults to 0.1.
        eps (float, optional): An epsilon value. Defaults to 5e-6.
        M (Callable, optional): Preconditioner. Defaults to None.

    Returns:
        jax.Array: b
    """
    if x0 is None:
        x0 = jtu.tree_map(jnp.zeros_like, b)

    b, x0 = jax.device_put((b, x0))

    if maxiter is None:
        size = sum(bi.size for bi in jtu.tree_leaves(b))
        maxiter = 10 * size

    if M is None:
        M = jssl._identity
    A = jssl._normalize_matvec(A)
    M = jssl._normalize_matvec(M)

    if jtu.tree_structure(x0) != jtu.tree_structure(b):
        raise ValueError(
            'x0 and b must have matching tree structure: '
            f'{jtu.tree_structure(x0)} vs {jtu.tree_structure(b)}')

    if jssl._shapes(x0) != jssl._shapes(b):
        raise ValueError(
            'arrays in x0 and b must have matching shapes: '
            f'{jssl._shapes(x0)} vs {jssl._shapes(b)}')

    if fixed_iter:
        solve = functools.partial(
            cg_solve_fixediter,
            x0=x0,
            maxiter=maxiter,
            M=M
        )
    else:
        solve = functools.partial(
            cg_solve,
            x0=x0,
            maxiter=maxiter,
            min_lookback=min_lookback,
            lookback_frac=lookback_frac,
            eps=eps,
            M=M
        )

    # real-valued positive-definite linear operators are symmetric
    def real_valued(x):
        return not issubclass(x.dtype.type, np.complexfloating)
    symmetric = all(map(real_valued, jtu.tree_leaves(b)))
    x = lax.custom_linear_solve(
        A,
        b,
        solve=solve,
        transpose_solve=solve,
        symmetric=symmetric
    )
    info = None
    return x, info


def make_schedule(params: dict) -> Callable[[int], float]:
    """Simple function to create different kind of schedules.

    Args:
        params (dict): Parameters for the schedules.

    Returns:
        Callable[[int], float]: schedule function
    """
    if isinstance(params, numbers.Number):
        def result(t): return params
    elif callable(params):
        result = params
    elif isinstance(params, dict):
        if 'schedule' not in params or params['schedule'] == 'hyperbola':
            assert 'init' in params
            assert 'delay' in params
            init = params['init']
            delay = params['delay']
            decay = params['decay'] if 'decay' in params else 1
            def result(t): return init * jnp.power(1/(1 + t/delay), decay)
        elif params['schedule'] == 'exponential':
            assert 'init' in params
            assert 'delay' in params
            init = params['init']
            delay = params['delay']
            def result(t): return init * jnp.exp(-t/delay)
        else:
            raise ValueError()
        if 'min' in params:
            val_fn = result
            def result(t): return jnp.maximum(val_fn(t), params['min'])
    else:
        raise ValueError()
    return result


def scale_by_trust_ratio_embeddings(
    min_norm: float = 0.0,
    trust_coefficient: float = 1.,
    eps: float = 0.,
) -> optax.GradientTransformation:
    """Scale by trust ratio but for embeddings were we don't want the norm
    over all parameters but just the last dimension.
    """
    def init_fn(params):
        del params
        return optax.ScaleByTrustRatioState()

    def update_fn(updates, state, params):
        if params is None:
            raise ValueError(optax.NO_PARAMS_MSG)

        def _scale_update(update, param):
            # Clip norms to minimum value, by default no clipping.
            param_norm = optax.safe_norm(param, min_norm, axis=-1, keepdims=True)
            update_norm = optax.safe_norm(update, min_norm, axis=-1, keepdims=True)
            trust_ratio = trust_coefficient * param_norm / (update_norm + eps)

            # If no minimum norm clipping is used
            # Set trust_ratio to 1 in case where parameters would never be updated.
            zero_norm = jnp.logical_or(param_norm == 0., update_norm == 0.)
            safe_trust_ratio = jnp.where(
                zero_norm, jnp.array(1.0, dtype=param.dtype), trust_ratio)

            return update * safe_trust_ratio

        updates = jax.tree_util.tree_map(_scale_update, updates, params)
        return updates, state

    return optax.GradientTransformation(init_fn, update_fn)
