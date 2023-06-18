import functools
import inspect
from dataclasses import _MISSING_TYPE, fields
from typing import Any, Callable, Iterable, Mapping, Sequence, Type

import flax.linen as nn
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from chex import ArrayDType, PRNGKey
from jax.nn.initializers import normal, orthogonal, variance_scaling

from globe.nn.parameters import ParamSpec, ParamTree, SpecTree
from globe.utils import safe_call

Activation = str | Callable[[jax.Array], jax.Array]


ACTIVATION_GAINS = {
    nn.silu: 1.7868129431578026,
    nn.tanh: 1.5927812698663606,
    nn.sigmoid: 4.801203511726151
}


class ScalingFactor(nn.Module):
    """
    A Flax module that scales the second input tensor by the ratio of the standard deviation of the first input tensor
    to the standard deviation of the second input tensor.
    """
    @nn.compact
    def __call__(self, x: jax.Array, y: jax.Array, x_weighting: jax.Array = None,
                 y_weighting: jax.Array = None) -> jax.Array:
        """
        Scales the second input tensor by the ratio of the standard deviation of the first input tensor
        to the standard deviation of the second input tensor.

        Args:
            x (jax.Array): The first input tensor.
            y (jax.Array): The second input tensor.
            x_weighting (Optional[jax.Array]): The weighting to apply to the first input tensor.
            y_weighting (Optional[jax.Array]): The weighting to apply to the second input tensor.

        Returns:
            jax.Array: The scaled second input tensor.
        """
        is_initialized = self.has_variable('scaling_factors', 'scale')
        scaling = self.variable('scaling_factors', 'scale', jnp.ones, ())
        if not is_initialized:
            if x.size > 1 and y.size > 1:
                if x_weighting is None:
                    x_weighting = jnp.ones(())
                if y_weighting is None:
                    y_weighting = jnp.ones(())
                x_weighting = jnp.broadcast_to(x_weighting, x.shape)
                y_weighting = jnp.broadcast_to(y_weighting, y.shape)
                x_weighting /= x_weighting.sum()
                y_weighting /= y_weighting.sum()

                # Weighted Std computation
                x_mean = (x * x_weighting).sum()
                y_mean = (y * y_weighting).sum()

                x_std = ((x - x_mean)**2 * x_weighting).sum() ** 0.5
                y_std = ((y - y_mean)**2 * y_weighting).sum() ** 0.5

                scaling.value = x_std / y_std
        return y * scaling.value


class FixedScalingFactor(nn.Module):
    """
    A Flax module that scales the input tensor by a fixed factor to achieve a target standard deviation.

    Attributes:
        target_std (float): The target standard deviation to achieve.
        element_wise (bool): Whether to scale each element of the input tensor independently.
    """
    target_std: float = 1.0
    element_wise: bool = False

    @nn.compact
    def __call__(self, x: jax.Array, weighting: jax.Array = None) -> jax.Array:
        """
        Scales the input tensor by a fixed factor to achieve a target standard deviation.

        Args:
            x (jax.Array): The input tensor.
            weighting (Optional[jax.Array]): The weighting to apply to the input tensor.

        Returns:
            jax.Array: The scaled input tensor.
        """
        is_initialized = self.has_variable('scaling_factors', 'scale')
        if self.element_wise:
            scaling = self.variable('scaling_factors', 'scale', jnp.ones, (x.shape[-1],))
        else:
            scaling = self.variable('scaling_factors', 'scale', jnp.ones, ())
        if not is_initialized:
            if x.size > 1:
                # Sum over all but the last dim
                axes = tuple(range(x.ndim - 1)) if self.element_wise else tuple(range(x.ndim))
                if weighting is None:
                    weighting = jnp.ones(())
                weighting = jnp.broadcast_to(weighting, x.shape)
                weighting /= weighting.sum(axes)

                # Weighted Std computation
                x_mean = (x * weighting).sum(axes)
                
                x_std = ((x - x_mean)**2 * weighting).sum(axes) ** 0.5
                value = self.target_std / x_std
                value = jnp.where(jnp.logical_or(jnp.isnan(value), jnp.isinf(value)), 1, value)
                scaling.value = value
        return x * scaling.value


class ReparametrizedModule(nn.Module):
    @classmethod
    def create(cls, *args, **kwargs):
        return safe_call(cls, *args, **kwargs)

    @classmethod
    def spec(cls, *args, **kwargs) -> SpecTree:
        spec_args = {
            f.name: f.default if not isinstance(f.default, _MISSING_TYPE) else f.default_factory()
            for f in fields(cls)
            if not (isinstance(f.default, _MISSING_TYPE) and isinstance(f.default_factory, _MISSING_TYPE))
        }
        spec_args.update(kwargs)

        spec_params = inspect.signature(cls.param_spec).parameters
        for k, v in zip(spec_params, args):
            spec_args[k] = v
        return cls.param_spec(**{
            k: v for k, v in spec_args.items()
            if k in spec_params
        })
    
    @staticmethod
    def param_spec() -> SpecTree:
        return None
    
    def define_parameters(self, *args, **kwargs) -> ParamTree:
        i = 0
        def define_param(spec: ParamSpec):
            nonlocal i
            result = spec.transform(self.param(f'param_{i}',
                lambda key, shape: jax.random.normal(key, shape) * spec.std + spec.mean,
                spec.shape
            ))
            i += 1
            return result
        return jtu.tree_map(define_param, self.spec(*args, **{**self.__dict__, **kwargs}))


def activation_function(fn: Activation) -> Callable[[jax.Array], jax.Array]:
    """
    Returns the activation function given its name or a callable.

    Args:
        fn (Union[str, Activation]): The name of the activation function or a callable.

    Returns:
        Callable[[jax.Array], jax.Array]: The activation function.

    Raises:
        AttributeError: If the activation function is not found in `nn` or `jnp`.
    """
    if callable(fn):
        return fn
    activations = {f.__name__: f for f in ACTIVATION_GAINS.keys()}
    if fn in activations:
        return activations[fn]
    else:
        try:
            return getattr(nn, fn)
        except:
            return getattr(jnp, fn)

LAYERS = {
    'Dense': nn.Dense,
    'Dense_no_bias': functools.partial(nn.Dense, use_bias=False)
}

Dense = lambda *args, **kwargs: LAYERS['Dense'](*args, **kwargs)
Dense_no_bias = lambda *args, **kwargs: LAYERS['Dense_no_bias'](*args, **kwargs)
def emb_init(key, shape, dtype=jnp.float_):
    return (jax.random.uniform(key, shape, dtype) - 0.5) * 2 * np.sqrt(3)
Embed = functools.partial(nn.Embed, embedding_init=jnn.initializers.normal(1.0))


def glorot_orthogonal(scale: float = 2.0) -> Callable[[PRNGKey, tuple[int, int], ArrayDType], jax.Array]:
    """
    Returns a function that generates a matrix with orthogonal columns and Glorot scaling.

    Args:
        scale (float, optional): Scaling factor. Defaults to 2.0.

    Returns:
        Callable[[PRNGKey, Tuple[int, int], Dtype], Array]: Function that generates a matrix with orthogonal columns and Glorot scaling.

    Raises:
        AssertionError: If the shape of the generated matrix is not 2.
    """
    base = orthogonal()
    def _glorot_orthogonal(key: PRNGKey, shape: tuple[int, int], dtype: ArrayDType = jnp.float32) -> jax.Array:
        assert len(shape) == 2
        W = base(key, shape, dtype)
        W *= jnp.sqrt(scale / ((shape[0] + shape[1]) * jnp.var(W)))
        return W
    return _glorot_orthogonal


def set_init_method(method: str = 'default') -> None:
    """
    Sets the initialization method for the Dense layer.

    Args:
        method (str, optional): The initialization method to use. Defaults to 'default'.

    Raises:
        ValueError: If the provided method is not supported.
    """
    if method == 'default':
        LAYERS['Dense'] = nn.Dense
        LAYERS['Dense_no_bias'] = functools.partial(nn.Dense, use_bias=False)
    elif method == 'ferminet':
        LAYERS['Dense'] = functools.partial(
            nn.Dense,
            kernel_init=variance_scaling(
                1,
                mode="fan_in",
                distribution="truncated_normal"
            ),
            bias_init=normal(1)
        )
        LAYERS['Dense_no_bias'] = functools.partial(nn.Dense, use_bias=False)
    elif method == 'pesnet':
        LAYERS['Dense'] = functools.partial(
            nn.Dense,
            kernel_init=variance_scaling(
                1/2,
                mode="fan_in",
                distribution="truncated_normal"
            ),
            bias_init=normal(1/np.sqrt(2))
        )
        LAYERS['Dense_no_bias'] = functools.partial(nn.Dense, use_bias=False)
    elif method == 'orthogonal':
        LAYERS['Dense'] = functools.partial(
            nn.Dense,
            kernel_init=orthogonal()
        )
        LAYERS['Dense_no_bias'] = functools.partial(Dense, use_bias=False)
    elif method == 'orthogonal_glorot':
        LAYERS['Dense'] = functools.partial(
            nn.Dense,
            kernel_init=glorot_orthogonal()
        )
        LAYERS['Dense_no_bias'] = functools.partial(Dense, use_bias=False)
    else:
        raise ValueError()


def residual(
    x: jax.Array,
    y: jax.Array
) -> jax.Array:
    """Adds a residual connection between input x and output y if possible.

    Args:
        x (jax.Array): input
        y (jax.Array): output

    Returns:
        jax.Array: new output
    """
    if x.shape == y.shape:
        return (x + y) / jnp.sqrt(2.0)
    else:
        return y


def mlp(
    module: nn.Module,
    x: jax.Array,
    hidden_dims: Sequence[int],
    activation: Activation,
    kernel_init: Callable | None = None,
    bias_init: Callable | None = None,
    out_kernel_init: Callable | None = None,
    out_bias_init: Callable | None = None,
    intermediate_bias: bool = True,
    final_bias: bool = True
) -> jax.Array:
    """
    A multi-layer perceptron (MLP) implementation.

    Args:
        module (nn.Module): The parent module.
        x (jax.Array): The input tensor.
        hidden_dims (Sequence[int]): A sequence of integers representing the number of hidden units in each layer.
        activation (Activation): The activation function to use.
        kernel_init (Optional[Callable], optional): The initializer function for the kernel. Defaults to None.
        bias_init (Optional[Callable], optional): The initializer function for the bias. Defaults to None.
        out_kernel_init (Optional[Callable], optional): The initializer function for the output kernel. Defaults to None.
        out_bias_init (Optional[Callable], optional): The initializer function for the output bias. Defaults to None.
        intermediate_bias (bool, optional): Whether to use bias in intermediate layers. Defaults to True.
        final_bias (bool, optional): Whether to use bias in the final layer. Defaults to True.

    Returns:
        jax.Array: The output tensor.
    """
    if len(hidden_dims) == 0:
        return x
    
    if hidden_dims[-1] == 0:
        return jnp.zeros((*x.shape[:-1], 0), dtype=x.dtype)

    Dense_inter = Dense if intermediate_bias else Dense_no_bias
    Dense_out = Dense if final_bias else Dense_no_bias
    if kernel_init is not None:
        Dense_inter = functools.partial(Dense_inter, kernel_init=kernel_init)
    if bias_init is not None:
        Dense_inter = functools.partial(Dense_inter, bias_init=bias_init)
    if out_kernel_init is not None:
        Dense_out = functools.partial(Dense_out, kernel_init=out_kernel_init)
    if out_bias_init is not None:
        Dense_out = functools.partial(Dense_out, bias_init=out_bias_init)

    activation = ActivationWithGain(activation, parent=module)

    y = x
    for hidden_dim in hidden_dims[:-1]:
        y = activation(Dense_inter(hidden_dim, parent=module)(y))
    y = Dense_out(hidden_dims[-1], parent=module)(y)
    return y


class MLP(nn.Module):
    """
    A multi-layer perceptron (MLP) implementation.

    Attributes:
        hidden_dims (Sequence[int]): A sequence of integers representing the number of hidden units in each layer.
        activation (Activation): The activation function to use.
        kernel_init (Optional[Callable], optional): The initializer function for the kernel. Defaults to None.
        bias_init (Optional[Callable], optional): The initializer function for the bias. Defaults to None.
        out_kernel_init (Optional[Callable], optional): The initializer function for the output kernel. Defaults to None.
        out_bias_init (Optional[Callable], optional): The initializer function for the output bias. Defaults to None.
        intermediate_bias (bool, optional): Whether to use bias in intermediate layers. Defaults to True.
        final_bias (bool, optional): Whether to use bias in the final layer. Defaults to True.

    Methods:
        __call__(self, x: jax.Array) -> jax.Array:
            A method that applies the MLP to the input tensor.

        extract_final_linear(params: Any) -> jax.Array:
            A static method that extracts the final linear layer from the MLP's parameters.
    """
    hidden_dims: Sequence[int]
    activation: Activation
    kernel_init: Callable | None = None
    bias_init: Callable | None = None
    out_kernel_init: Callable | None = None
    out_bias_init: Callable | None = None
    intermediate_bias: bool = True
    final_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        A method that applies the MLP to the input tensor.

        Args:
            x (jax.Array): The input tensor.

        Returns:
            jax.Array: The output tensor.
        """
        return mlp(
            self,
            x,
            hidden_dims=self.hidden_dims,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            out_kernel_init=self.out_kernel_init,
            out_bias_init=self.out_bias_init,
            intermediate_bias=self.intermediate_bias,
            final_bias=self.final_bias
        )

    @staticmethod
    def extract_final_linear(params: ParamTree) -> ParamTree:
        """
        A static method that extracts the final linear layer from the MLP's parameters.

        Args:
            params (Any): The MLP's parameters.

        Returns:
            jax.Array: The final linear layer.
        """
        key = list(params)[-1]
        return params[key]


class AutoMLP(nn.Module):
    """
    A module that implements an MLP with automatically determined hidden layer sizes.

    Attributes:
        out_dim (int): The output dimension of the MLP.
        n_layers (int): The number of hidden layers in the MLP.
        activation (Activation): The activation function to use.
        kernel_init (Optional[Callable], optional): The initializer function for the kernel. Defaults to None.
        bias_init (Optional[Callable], optional): The initializer function for the bias. Defaults to None.
        out_kernel_init (Optional[Callable], optional): The initializer function for the output kernel. Defaults to None.
        out_bias_init (Optional[Callable], optional): The initializer function for the output bias. Defaults to None.
        scale (str, optional): The scale to use when determining the hidden layer sizes. Can be 'log' or 'linear'. Defaults to 'log'.
        intermediate_bias (bool, optional): Whether to use bias in intermediate layers. Defaults to True.
        final_bias (bool, optional): Whether to use bias in the final layer. Defaults to True.

    Methods:
        __call__(self, x: jax.Array) -> jax.Array:
            A method that applies the MLP to the input tensor.
    """
    out_dim: int
    n_layers: int
    activation: Activation
    kernel_init: Callable | None = None
    bias_init: Callable | None = None
    out_kernel_init: Callable | None = None
    out_bias_init: Callable | None = None
    scale: str = 'log'
    intermediate_bias: bool = True
    final_bias: bool = True

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        A method that applies the MLP to the input tensor.

        Args:
            x (jax.Array): The input tensor.

        Returns:
            jax.Array: The output tensor.
        """
        inp_dim = x.shape[-1]
        # We use np instead of jnp to ensure that it is static.
        if self.out_dim > 0 and inp_dim > 0:
            if self.scale == 'log':
                hidden_dims = np.round(
                    np.logspace(
                        np.log(inp_dim),
                        np.log(self.out_dim),
                        self.n_layers + 1,
                        base=np.e
                    )
                ).astype(np.int32)[1:]
            elif self.scale == 'linear':
                hidden_dims = np.round(
                    np.linspace(
                        inp_dim,
                        self.out_dim,
                        self.n_layers + 1
                    )
                ).astype(np.int32)[1:]
            else:
                raise ValueError()
        else:
            hidden_dims = [0]
        if inp_dim == 0:
            hidden_dims = [self.out_dim]
        return mlp(
            self,
            x,
            hidden_dims=hidden_dims,
            activation=self.activation,
            kernel_init=self.kernel_init,
            bias_init=self.bias_init,
            out_kernel_init=self.out_kernel_init,
            out_bias_init=self.out_bias_init,
            intermediate_bias=self.intermediate_bias,
            final_bias=self.final_bias
        )


class ActivationWithGain(nn.Module):
    """
    A module that applies an activation function with gain to the input tensor.

    Attributes:
        activation (Activation): The activation function to use.

    Methods:
        __call__(self, x: jax.Array) -> jax.Array:
            A method that applies the activation function with gain to the input tensor.

    """
    activation: Activation

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        """
        A method that applies the activation function with gain to the input tensor.

        Args:
            x (jax.Array): The input tensor.

        Returns:
            jax.Array: The output tensor.
        """
        activation = activation_function(self.activation)
        if isinstance(activation, nn.Module) or activation not in ACTIVATION_GAINS:
            return activation(x)
        else:
            return activation(x) * ACTIVATION_GAINS[activation]

def named(name: str, module: Type[nn.Module], *args: Any, **kwargs: Any) -> nn.Module:
    """
    Creates a new module with a given name.

    Args:
        name (str): The name of the new module.
        module (Type[nn.Module]): The module to be named.
        *args (Any): Positional arguments to be passed to the module constructor.
        **kwargs (Any): Keyword arguments to be passed to the module constructor.

    Returns:
        nn.Module: A new module with the given name.
    """
    return type(name, (module,), {})(*args, **kwargs)


def rename(module: Type[nn.Module], name: str) -> Callable[[Type[nn.Module]], nn.Module]:
    """
    Renames a module with a given name.

    Args:
        module (Type[nn.Module]): The module to be renamed.
        name (str): The new name of the module.

    Returns:
        Callable[[Type[nn.Module]], nn.Module]: A function that renames a module with the given name.
    """
    return functools.partial(named, name, module)


def indexed_matmul(x: jax.Array, kernel: jax.Array, bias: jax.Array, idx: jax.Array, max_idx: int) -> jax.Array:
    """
    Computes a matrix multiplication between a tensor and a kernel, with a bias term added.

    Args:
        x (jax.Array): The input tensor.
        kernel (jax.Array): The kernel tensor.
        bias (jax.Array): The bias tensor.
        idx (jax.Array): The index tensor.
        max_idx (int): The maximum index.

    Returns:
        jax.Array: The output tensor.
    """
    kernel = jnp.broadcast_to(kernel, (max_idx, x.shape[-1], kernel.shape[-1]))
    bias = jnp.broadcast_to(bias, (max_idx, kernel.shape[-1]))
    return jnp.einsum('...a,...ab->...b', x, kernel[idx]) + bias[idx]
