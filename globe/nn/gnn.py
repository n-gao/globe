import functools

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from globe.nn import (ActivationWithGain, Dense, Dense_no_bias,
                        FixedScalingFactor, ParamTree, ReparametrizedModule)
from globe.nn.parameters import ParamSpec, ParamType, SpecTree, inverse_softplus
from globe.nn.utils import spherical_coordinates

DEFAULT_SIGMA = 10.0


@functools.partial(jax.jit, static_argnums=1)
def real_spherical_harm(x: jax.Array, l_max: int = 4):
    """
    Computes the real spherical harmonics of the input tensor x.

    Args:
    - x (jax.Array): Input tensor of shape (..., 3).
    - l_max (int): Maximum degree of the spherical harmonics.

    Returns:
    - jax.Array: Tensor of shape (..., n_harmonics), where n_harmonics is the number of spherical harmonics
    up to degree l_max.
    """
    r, phi, theta = spherical_coordinates(x[..., :3])
    theta += np.pi
    l, m = [], []
    for l_ in range(l_max):
        for m_ in range(-l_, l_+1):
            m.append(m_)
            l.append(l_)
    m, l = np.array(m), np.array(l)
    fn = jax.vmap(jax.scipy.special.sph_harm, in_axes=(None, None, 0, 0, None))
    result = fn(m, l, theta.reshape(-1, 1), phi.reshape(-1, 1), l_max).real
    return result.reshape(*r.shape, m.shape[0])


class NormEnvelope(ReparametrizedModule):
    """
    A module that computes the normalization envelope for a graph neural network.

    Args:
    - adaptive (bool): Whether to use adaptive normalization.
    - sigma_init (float): The initial value for the sigma parameter.
    - param_type (ParamType): The type of parameter to use for sigma.

    Methods:
    - param_spec(adaptive, sigma_init, param_type): Returns a dictionary of parameter specifications.
    - __call__(self, edges, idx=None, params=None): Computes the normalization envelope for the given edges.
    """
    adaptive: bool = True
    sigma_init: float = DEFAULT_SIGMA
    param_type: ParamType = ParamType.NUCLEI

    def param_spec(adaptive: bool, sigma_init: float, param_type: ParamType) -> SpecTree:
        """
        Returns a dictionary of parameter specifications for the NormEnvelope module.

        Args:
        - adaptive (bool): Whether to use adaptive normalization.
        - sigma_init (float): The initial value for the sigma parameter.
        - param_type (ParamType): The type of parameter to use for sigma.

        Returns:
        - Dict[str, ParamSpec]: A dictionary of parameter specifications.
        """
        if not adaptive:
            return {}
        param_type = ParamType(param_type)
        return {
            'sigma': ParamSpec(
                param_type,
                shape=(1,),
                mean=sigma_init,
                std=1.0,
                transform=jnn.softplus,
                keep_distr=True,
                group='gnn_sigma'
            )
        }

    @nn.compact
    def __call__(self, edges: jax.Array, idx: jax.Array = None, params: ParamTree = None) -> jax.Array:
        """
        Computes the normalization envelope for the given edges.

        Args:
        - edges (jax.Array): The input tensor of shape (..., 2).
        - idx (Optional[jax.Array]): The indices of the edges to use for adaptive normalization.
        - params (ParamTree): The parameters to use for adaptive normalization.

        Returns:
        - jax.Array: The normalization envelope tensor of shape (...,).
        """
        adaptive = params is not None and len(params) > 0 and idx is not None
        if params is None or len(params) == 0:
            params = self.define_parameters(adaptive=True)
        sigma = params['sigma'].squeeze(-1)
        if adaptive:
            sigma = sigma[idx]
        return jnp.exp(-(edges[..., -1]/sigma)**2)


class MlpRbf(ReparametrizedModule):
    """
    A module that computes MLP basis functions.

    Args:
    - out_dim (int): The output dimension of the module.
    - hidden_dim (int): The hidden dimension of the multilayer perceptron.
    - activation (str): The activation function to use in the multilayer perceptron.

    Methods:
    - param_spec(hidden_dim): Returns a dictionary of parameter specifications.
    - __call__(self, x, idx=None, params=None): Computes the output of the module for the given input.
    """
    out_dim: int = 8
    hidden_dim: int = 16
    activation: str = 'silu'

    @staticmethod
    def param_spec(hidden_dim: int) -> SpecTree:
        return {
            'sigma': ParamSpec(
                ParamType.NUCLEI,
                shape=(hidden_dim,),
                mean=DEFAULT_SIGMA,
                std=1.0,
                transform=jnn.softplus,
                group='gnn_sigma',
                keep_distr=True
            ),
            'bias': ParamSpec(ParamType.NUCLEI, (hidden_dim,), 0.0, 2.0)
        }

    @nn.compact
    def __call__(self, x: jax.Array, idx: jax.Array = None, params: ParamTree = None) -> jax.Array:
        """
        Computes the output of the module for the given input.

        Args:
        - x (jax.Array): The input tensor of shape (..., 4).
        - idx (Optional[jax.Array]): The indices of the edges to use for adaptive normalization.
        - params (ParamTree): The parameters to use for adaptive normalization.

        Returns:
        - jax.Array: The output tensor of shape (..., out_dim).
        """
        adaptive = params is not None and idx is not None
        if params is None:
            params = self.define_parameters()

        sigma, bias = params['sigma'], params['bias']
        if adaptive:
            sigma, bias = sigma[idx], bias[idx]

        env = Dense_no_bias(self.out_dim)(jnp.exp(-(x[..., -1:]/sigma)**2))
        y = Dense_no_bias(
            self.hidden_dim,
            kernel_init=jnn.initializers.variance_scaling(0.1, 'fan_in', 'truncated_normal')
        )(x[..., -1:]) + bias
        y = ActivationWithGain(self.activation)(y)
        y = Dense(self.out_dim)(y)
        return y * env


class BesselRbf(nn.Module):
    """
    A module that computes the Bessel radial basis function.

    Attributes:
    - n_rad (int): The number of radial basis functions to use.
    - bessel_cutoff (float): The cutoff value for the Bessel function.
    - sigma_init (float): The initial value for the sigma parameter.

    Methods:
    - __call__(self, x: jax.Array) -> jax.Array: Computes the Bessel radial basis function for the given input.

    Returns:
    - jax.Array: The output tensor of shape (..., n_rad).
    """
    n_rad: int = 6
    bessel_cutoff: float = DEFAULT_SIGMA*2
    sigma_init: float = DEFAULT_SIGMA

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        Computes the Bessel radial basis function for the given input.

        Args:
        - x (jax.Array): The input tensor of shape (..., 4).

        Returns:
        - jax.Array: The output tensor of shape (..., n_rad).
        """
        f = jnn.softplus(self.param(
            'f',
            lambda *_: inverse_softplus((jnp.arange(self.n_rad, dtype=jnp.float32) + 1) * jnp.pi)
        ))
        c = jnn.softplus(self.param(
            'c',
            jnn.initializers.constant(inverse_softplus(self.bessel_cutoff)),
            ()
        ))
        x_ext = x[..., -1:] + 1e-6
        result = jnp.sqrt(2./c) * jnp.sin(f*x_ext/c)/x_ext
        
        sigma = self.param('sigma', jnn.initializers.constant(self.sigma_init), ())
        env = jnp.exp(-(x[..., -1:]/sigma)**2)
        return result * env


class SphHarmEmbedding(ReparametrizedModule):
    """
    A module that computes the spherical harmonic embedding.

    Attributes:
    - n_rad (int): The number of radial basis functions to use.
    - max_l (int): The maximum degree of the spherical harmonics.

    Methods:
    - param_spec() -> dict[str, ParamSpec]: Returns a dictionary of parameter specifications.
    - __call__(self, edges: jax.Array, nuclei_idx: jax.Array = None, params: dict[str, jax.Array] = None) -> jax.Array: Computes the output of the module for the given input.

    Returns:
    - jax.Array: The output tensor of shape (..., n_rad*max_l*(max_l+1)).
    """
    n_rad: int = 6
    max_l: int = 3

    @staticmethod
    def param_spec() -> SpecTree:
        return {}

    @nn.compact
    def __call__(self, edges: jax.Array, nuclei_idx: jax.Array = None, params: ParamTree = None) -> jax.Array:
        """
        Computes the output of the module for the given input.

        Args:
        - edges (jax.Array): The input tensor of shape (..., 4).
        - nuclei_idx (jax.Array): The indices of the nuclei in the input tensor.
        - params (dict[str, jax.Array]): A dictionary of parameter tensors.

        Returns:
        - jax.Array: The output tensor of shape (..., n_rad*n_sph).
        """
        env = BesselRbf(self.max_l)(edges)
        rad = real_spherical_harm(edges[..., :3], self.max_l)
        result = (env[..., None] * rad[..., None, :]).reshape(*edges.shape[:-1], env.shape[-1]*rad.shape[-1])
        return result


class MLPEdgeEmbedding(ReparametrizedModule):
    """
    A module that computes the edge embedding using a multi-layer perceptron (MLP).

    Attributes:
    - out_dim (int): The output dimension of the module.
    - hidden_dim (int): The hidden dimension of the MLP.
    - activation (str): The activation function to use in the MLP.
    - adaptive (bool): Whether to use adaptive parameters.
    - adaptive_weights (bool): Whether to use adaptive weights.
    - init_sph (bool): Whether to initialize the basis functions spherically.
    - inp_dim (int): The input dimension of the module.
    - sigma_init (float): The initial value for the sigma parameter.
    - param_type (ParamType): The type of parameter to use.

    Methods:
    - param_spec(inp_dim: int, hidden_dim: int, adaptive: bool, adaptive_weights: bool, sigma_init: float, param_type: ParamType) -> dict[str, ParamSpec]: Returns a dictionary of parameter specifications.
    - __call__(self, edges: jax.Array, idx: jax.Array = None, params: dict[str, jax.Array] = None) -> jax.Array: Computes the output of the module for the given input.

    Returns:
    - jax.Array: The output tensor of shape (..., out_dim).
    """
    out_dim: int = 8
    hidden_dim: int = 16
    activation: str = 'silu'
    adaptive: bool = True
    adaptive_weights: bool = False
    init_sph: bool = False
    inp_dim: int = 4 # 3 spatial dimensions + length
    sigma_init: float = DEFAULT_SIGMA
    param_type: ParamType = ParamType.NUCLEI

    @staticmethod
    def param_spec(inp_dim: int, hidden_dim: int, adaptive: bool, adaptive_weights: bool, sigma_init: float, param_type: ParamType) -> SpecTree:
        if not adaptive:
            return {}
        param_type = ParamType(param_type)
        result = {
            'sigma': ParamSpec(
                param_type,
                shape=(hidden_dim,),
                mean=sigma_init,
                std=1.0,
                transform=jnn.softplus,
                group='gnn_sigma'
            ),
            'bias': ParamSpec(param_type, (hidden_dim,), 0, 2.0)
        }
        if adaptive_weights:
            result['weights'] = ParamSpec(
                param_type,
                shape=(inp_dim, hidden_dim,),
                mean=0,
                std=1/(np.sqrt(4.0))
            )
        return result

    @nn.compact
    def __call__(self, edges: jax.Array, idx: jax.Array = None, params: ParamTree = None) -> jax.Array:
        """
        Computes the output of the module for the given input.

        Args:
        - edges (jax.Array): The input tensor of shape (..., 4).
        - idx (jax.Array): The indices of the edges in the input tensor.
        - params (ParamTree): A dictionary of parameter tensors.

        Returns:
        - jax.Array: The output tensor of shape (..., out_dim).
        """
        adaptive = params is not None and len(params) > 0 and idx is not None
        if params is None or len(params) == 0:
            params = self.define_parameters(adaptive=True)

        sigma = params['sigma']
        bias = params['bias']
        weights = params['weights'] if self.adaptive_weights else self.param(
            'kernel',
            jnn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal'),
            (self.inp_dim, self.hidden_dim)
        )
        if adaptive:
            sigma = sigma[idx]
            bias = bias[idx]
            if self.adaptive_weights:
                weights = weights[idx]
        
        coord_mask = self.param(
            'coord_kernel',
            lambda _, x: jnp.array(x, dtype=jnp.float32),
            [0, 0, 0, 1] if self.init_sph else [1, 1, 1, 1]
        )

        env = Dense_no_bias(self.out_dim)(jnp.exp(-(edges[..., -1:]/sigma)**2))
        result = jnp.einsum('...d,...dk->...k', edges * coord_mask, weights) + bias
        result = ActivationWithGain(self.activation)(result)
        result = Dense(self.out_dim)(result)
        return result * env


class DenseEdgeEmbedding(ReparametrizedModule):
    """
    A dense edge embedding module that computes the output of the module for the given input.

    Args:
    - out_dim (int): The output dimension of the module.
    - activation (str): The activation function to use in the MLP.
    - adaptive (bool): Whether to use adaptive parameters.
    - adaptive_weights (bool): Whether to use adaptive weights.
    - init_sph (bool): Whether to initialize the basis functions spherically.
    - inp_dim (int): The input dimension of the module.
    - sigma_init (float): The initial value for the sigma parameter.
    - param_type (ParamType): The type of parameter to use.

    Methods:
    - param_spec(inp_dim: int, out_dim: int, adaptive: bool, adaptive_weights: bool, sigma_init: float, param_type: ParamType) -> dict[str, ParamSpec]: Returns a dictionary of parameter specifications.
    - __call__(self, edges: jax.Array, idx: jax.Array = None, params: dict[str, jax.Array] = None) -> jax.Array: Computes the output of the module for the given input.

    Returns:
    - jax.Array: The output tensor of shape (..., out_dim).
    """
    out_dim: int = 16
    activation: str = 'silu'
    adaptive: bool = True
    adaptive_weights: bool = False
    init_sph: bool = False
    inp_dim: int = 4 # 3 spatial dimensions + length
    sigma_init: float = DEFAULT_SIGMA
    param_type: ParamType = ParamType.NUCLEI

    @staticmethod
    def param_spec(inp_dim: int, out_dim: int, adaptive: bool, adaptive_weights: bool, sigma_init: float, param_type: ParamType) -> SpecTree:
        if not adaptive:
            return {}
        param_type = ParamType(param_type)
        result = {
            'sigma': ParamSpec(
                param_type,
                shape=(out_dim,),
                mean=sigma_init,
                std=1.0,
                transform=jnn.softplus,
                group='gnn_sigma'
            ),
            'bias': ParamSpec(param_type, (out_dim,), 0, 2.0)
        }
        if adaptive_weights:
            result['weights'] = ParamSpec(
                param_type,
                shape=(inp_dim, out_dim,),
                mean=0,
                std=1/(np.sqrt(4.0))
            )
        return result

    @nn.compact
    def __call__(self, edges: jax.Array, idx: jax.Array = None, params: ParamTree = None) -> jax.Array:
        """
        Computes the output of the module for the given input.

        Args:
        - edges (jax.Array): The input tensor of shape (..., 4).
        - idx (jax.Array): The indices of the edges in the input tensor.
        - params (ParamTree): A dictionary of parameter tensors.

        Returns:
        - jax.Array: The output tensor of shape (..., out_dim).
        """
        adaptive = params is not None and len(params) > 0 and idx is not None
        if params is None or len(params) == 0:
            params = self.define_parameters(adaptive=True)

        sigma = params['sigma']
        bias = params['bias']
        weights = params['weights'] if self.adaptive_weights else self.param(
            'kernel',
            jnn.initializers.variance_scaling(1.0, 'fan_in', 'truncated_normal'),
            (self.inp_dim, self.out_dim)
        )
        if adaptive:
            sigma = sigma[idx]
            bias = bias[idx]
            if self.adaptive_weights:
                weights = weights[idx]
        
        coord_mask = self.param(
            'coord_kernel',
            lambda _, x: jnp.array(x, dtype=jnp.float32),
            [0, 0, 0, 1] if self.init_sph else [1, 1, 1, 1]
        )

        env = Dense_no_bias(self.out_dim)(jnp.exp(-(edges[..., -1:]/sigma)**2))
        result = jnp.einsum('...d,...dk->...k', edges * coord_mask, weights) + bias
        result = ActivationWithGain(self.activation)(result)
        result = FixedScalingFactor(element_wise=True)(result * env)
        return result
